#!/usr/bin/env python
"""
Enhanced entity matching with sophisticated control logic.

This script uses the enhanced heuristic engine with:
- Early decision rules (skip LLM for high/low confidence)
- Dynamic weight adjustment based on data patterns
- Cost optimization through smart LLM call reduction
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import pathlib
import re
import time

from typing import Any, Dict, List, Optional

import pandas as pd

from openai import AsyncOpenAI
from src.entity_matching.constants import MODEL_COSTS
from src.entity_matching.enhanced_heuristic_engine import (
    EnhancedHeuristicEngine,
    PipelineStage,
    load_enhanced_heuristics_for_dataset,
)
from src.entity_matching.hybrid_matcher import (
    SEMANTIC_AVAILABLE,
    CandidateCache,
    Config,
    call_openai_async,
    compute_dataset_embeddings,
    get_top_candidates_cached,
    semantic_similarity_cached,
    slugify_model_name,
    token_count,
    trigram_similarity,
)
from src.prompts.hybrid_matcher_prompt import build_prompt


def create_candidate_cache(dataset: str, B: dict, cfg: Config, max_candidates: int) -> CandidateCache:
    """
    Create or reuse candidate cache with smart cache file selection.
    
    Args:
        dataset: Dataset name
        B: Table B records
        cfg: Configuration object with embedding model info
        max_candidates: Maximum candidates needed
        
    Returns:
        CandidateCache: Ready-to-use candidate cache
    """
    print("🔄 Creating candidate cache...")
    
    # Include model name in cache to avoid conflicts between different embedding models
    safe_model_name = slugify_model_name(cfg.embedding_model_name)
    cache_file = f".candidate_cache/{dataset}_{safe_model_name}_candidates_{max_candidates}.json"

    # Try to find a larger cache file we can use
    cache_dir = pathlib.Path(".candidate_cache")
    cache_pattern = f"{dataset}_{safe_model_name}_candidates_*.json"
    existing_caches = list(cache_dir.glob(cache_pattern))

    # Find the largest cache that has >= max_candidates
    best_cache_file = cache_file
    best_cache_size = max_candidates

    for existing_cache in existing_caches:
        # Extract candidates count from filename
        try:
            cache_candidates = int(existing_cache.stem.split('_candidates_')[1])
            if cache_candidates >= max_candidates and cache_candidates > best_cache_size:
                best_cache_file = str(existing_cache)
                best_cache_size = cache_candidates
        except (ValueError, IndexError):
            continue

    if best_cache_file != cache_file:
        print(f"🔍 Found larger cache with {best_cache_size} candidates, reusing: {best_cache_file}")

    candidate_cache = CandidateCache(B, cache_file=best_cache_file)
    print(f"✅ Candidate cache created for {len(B)} records")
    
    return candidate_cache


def load_dataset_and_pairs(dataset: str, use_validation: bool = False):
    """
    Load dataset tables and pairs with proper ID mapping and validation handling.
    
    Returns:
        tuple: (A, B, pairs) where A and B are either dicts (ID mapping) or lists (indexing)
    """
    # Load data with proper ID mapping
    root = pathlib.Path("data") / "raw" / dataset
    A_df = pd.read_csv(root / "tableA.csv")
    B_df = pd.read_csv(root / "tableB.csv")

    # Check if this dataset has non-sequential IDs (like zomato_yelp)
    if "id" in A_df.columns:
        # Create ID-to-record mappings
        A = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
        B = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}
        print(f"Dataset uses ID mapping: A has {len(A)} records (IDs {min(A.keys())}-{max(A.keys())})")
    else:
        # Use list indexing for datasets without ID column
        A = A_df.to_dict(orient="records")
        B = B_df.to_dict(orient="records")
        print(f"Dataset uses list indexing: A has {len(A)} records")

    # Load pairs data - use validation data if flag is set to prevent data leakage
    if use_validation:
        if (root / "valid.csv").exists():
            all_pairs = pd.read_csv(root / "valid.csv")
            # Use balanced sampling to keep evaluation manageable (~200 pairs)
            from src.entity_matching.balanced_sampling import balanced_train_sample, get_sample_info
            pairs = balanced_train_sample(all_pairs, target_size=200, random_state=42)
            print("✅ Using validation set for evaluation (200-pair balanced sample, no test data leakage)")
            print(get_sample_info(pairs, "validation sample"))
        elif (root / "train.csv").exists():
            train_pairs = pd.read_csv(root / "train.csv")

            # Ensure balanced sampling to avoid all-positive or all-negative slices
            positive_pairs = train_pairs[train_pairs['label'] == 1]
            negative_pairs = train_pairs[train_pairs['label'] == 0]

            # Take up to 50 of each class for balanced evaluation (100 total max)
            max_per_class = 50
            pos_sample = positive_pairs.head(min(max_per_class, len(positive_pairs)))
            neg_sample = negative_pairs.head(min(max_per_class, len(negative_pairs)))

            # Combine and shuffle
            pairs = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

            print(f"✅ Using balanced train sample: {len(pos_sample)} positive + {len(neg_sample)} negative = {len(pairs)} total pairs (no test data leakage)")
        else:
            raise ValueError("No validation or training data available - cannot evaluate without test data leakage")
    else:
        pairs = pd.read_csv(root / "test.csv")
        print("⚠️ Using test set for evaluation")

    return A, B, pairs


async def enhanced_match_single_record(
    left_record: dict,
    candidates: List[tuple],
    cfg: Config,
    client: AsyncOpenAI,
    heuristic_engine: EnhancedHeuristicEngine,
    prompt_data: Optional[Dict[str, Any]] = None,
) -> tuple[int, bool]:
    """Enhanced matching with sophisticated control logic"""
    logger = logging.getLogger(__name__)

    # Early decision check before LLM
    best_candidate = None
    best_score = 0.0
    weight_adjustments = 0

    for idx, candidate_record in candidates:
        # Calculate combined similarity score
        left_str = json.dumps(left_record, ensure_ascii=False).lower()
        right_str = json.dumps(candidate_record, ensure_ascii=False).lower()

        trigram_score = trigram_similarity(left_str, right_str)
        semantic_score = semantic_similarity_cached(left_record, candidate_record, cfg.embeddings) if cfg.use_semantic else 0.0

        # Apply weight rules to potentially adjust semantic weight
        current_weights = {"semantic_weight": cfg.semantic_weight}
        weight_action = heuristic_engine.apply_weight_rules(
            left_record, candidate_record, current_weights, PipelineStage.PRE_SEMANTIC
        )

        effective_semantic_weight = cfg.semantic_weight
        if weight_action and weight_action.semantic_weight is not None:
            effective_semantic_weight = weight_action.semantic_weight
            weight_adjustments += 1

        # Combine scores with potentially adjusted weights
        combined_score = (1 - effective_semantic_weight) * trigram_score + effective_semantic_weight * semantic_score

        # Apply score rules
        score_adjustment = heuristic_engine.apply_score_rules(
            left_record, candidate_record, PipelineStage.CANDIDATE_SELECTION
        )
        final_score = combined_score + score_adjustment

        if final_score > best_score:
            best_score = final_score
            best_candidate = (idx, candidate_record)

    if not best_candidate:
        return -1, False

    best_idx, best_record = best_candidate

    # Summarize weight adjustments to reduce noise
    if weight_adjustments > 0:
        pass  # Applied weight adjustments (removed verbose logging)

    # Apply decision rules before LLM call
    decision = heuristic_engine.apply_decision_rules(left_record, best_record, best_score, PipelineStage.PRE_LLM)

    if decision and decision.terminate_early:
        if decision.skip_llm:
            pass  # Early decision made (removed verbose logging)
        else:
            pass  # Early decision made (removed verbose logging)
        # If rule says accept (1), return the best candidate index; if reject (0), return -1
        return (best_idx if decision.final_result == 1 else -1), True

    # Fall back to LLM if no early decision
    # Proceeding to LLM (removed verbose logging)

    # Build prompt with all candidates (recall@N) - fixes recall@1 architecture flaw
    candidates_lines = []
    for position, (actual_id, record) in enumerate(candidates, 1):
        # Remove the 'id' field from the record to prevent LLM confusion (match baseline behavior)
        cleaned_record = {k: v for k, v in record.items() if k != 'id'}
        candidates_lines.append(f"{position}) {json.dumps(cleaned_record, ensure_ascii=False)}")
    candidates_text = "\n".join(candidates_lines)

    # best_idx is actually a record ID, find its position in the candidates list
    display_best_idx = 1  # Default to first candidate if not found
    for position, (actual_id, record) in enumerate(candidates, 1):
        if actual_id == best_idx:
            display_best_idx = position
            break

    # DEBUG LOGGING: Log candidate display info
    logger.debug(f"🔍 Candidate mapping: best_idx (record_id)={best_idx}, display_best_idx (position)={display_best_idx}")
    logger.debug(f"🔍 Showing {len(candidates)} candidates to LLM, positions 1-{len(candidates)}")

    # Format the complete prompt using structured sections
    prompt = build_prompt(
        left_record=left_record,
        candidates_text=candidates_text,
        best_idx=display_best_idx,
        prompt_data=prompt_data,
        additional_guidance=None
    )

    # Check token count
    total_tokens = token_count(prompt, cfg.model)
    if total_tokens > 1000000:  # 1M token limit
        print(f"  WARNING: Prompt too large ({total_tokens:,} tokens)")
        return -1, False

    # Get LLM response
    response = await call_openai_async(prompt, cfg, client)

    # DEBUG LOGGING: Log LLM interaction
    logger.debug(f"🔍 LLM raw response: '{response}'")

    # Parse response
    if not response:
        print("  WARNING: Empty response from LLM")
        return -1, False

    try:
        # Clean the response - remove whitespace and try to extract number
        response_clean = response.strip()

        # Try direct int conversion first
        try:
            llm_choice = int(response_clean)
        except ValueError:
            # Try to extract number from response if it contains extra text
            numbers = re.findall(r'-?\d+', response_clean)
            if numbers:
                llm_choice = int(numbers[0])  # Take first number found
                print(f"  DEBUG: Extracted number {llm_choice} from response: '{response_clean}'")
            else:
                print(f"  WARNING: Could not parse LLM response: '{response_clean}' (no numbers found)")
                return -1, False

        # Handle LLM response: -1 means no match, 1-N means candidate choice
        if llm_choice == -1:
            logger.debug("🔍 LLM chose no match (-1)")
            return -1, False
        if 1 <= llm_choice <= len(candidates):
            # Convert 1-based LLM response to 0-based candidate index
            candidate_idx = llm_choice - 1
            chosen_right_id, _ = candidates[candidate_idx]
            logger.debug(f"🔍 LLM chose position {llm_choice} → candidate_idx={candidate_idx} → right_id={chosen_right_id}")
            # Return the candidate index (not the ID - calling code expects index)
            return candidate_idx, False
        logger.warning(f"🔍 LLM chose invalid candidate {llm_choice} (valid range: 1-{len(candidates)} or -1)")
        return -1, False

    except Exception as e:
        print(f"  WARNING: Error parsing LLM response: '{response}', error: {e}")
        return -1, False


async def run_enhanced_matching(
    dataset: str,
    max_candidates: int = 50,
    model: str = "gpt-4.1-nano",
    concurrency: int = 10,
    semantic_weight: float = 0.5,
    trigram_weight: float = None,
    syntactic_weight: float = None,
    heuristic_file: str = None,
    use_validation: bool = False,
    embedding_base_url: str = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict:
    """Run enhanced entity matching with sophisticated control logic"""

    print("🚀 ENHANCED ENTITY MATCHING")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Candidates: {max_candidates}")
    print(f"Heuristics: {heuristic_file}")
    print("=" * 80)

    # Load enhanced heuristic engine
    heuristic_engine = load_enhanced_heuristics_for_dataset(dataset, heuristic_file)

    # Load configuration from heuristic file if provided
    prompt_data = None
    if heuristic_file and os.path.exists(heuristic_file):
        try:
            import json
            with open(heuristic_file) as f:
                heuristic_config = json.load(f)

            if "hyperparameters" in heuristic_config:
                hyperparams = heuristic_config["hyperparameters"]

                # Override weights from heuristic file
                file_semantic = hyperparams.get("semantic_weight")
                file_trigram = hyperparams.get("trigram_weight")
                file_syntactic = hyperparams.get("syntactic_weight")

                if file_semantic is not None:
                    semantic_weight = file_semantic
                if file_trigram is not None:
                    trigram_weight = file_trigram
                if file_syntactic is not None:
                    syntactic_weight = file_syntactic

                print(f"✅ Loaded weights from heuristic file: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}")
            else:
                print("⚠️ No hyperparameters found in heuristic file")

            # Load prompt data from heuristic file if available
            if "prompt_data" in heuristic_config:
                prompt_data = heuristic_config["prompt_data"]
                print("✅ Loaded prompt data from heuristic file")

        except Exception as e:
            print(f"⚠️ Could not extract configuration from heuristic file: {e}")

    # Auto-fill missing weights with defaults for 3-weight system
    if trigram_weight is None or syntactic_weight is None:
        remaining_weight = 1.0 - semantic_weight
        trigram_weight = remaining_weight * 0.6  # 60% of remaining to trigram
        syntactic_weight = remaining_weight * 0.4  # 40% of remaining to syntactic
        print(f"🎯 Auto-completed 3-weight system: semantic={semantic_weight}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f}")
    else:
        print(f"🎯 Using 3-weight system: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}")
    print("=" * 80)

    # Initialize configuration
    cfg = Config()
    cfg.model = model
    cfg.use_semantic = True

    # Set embedding configuration
    if embedding_base_url:
        cfg.embedding_base_url = embedding_base_url
        cfg.embedding_model_name = embedding_model
        print(f"🤖 Using embedding API: {embedding_base_url} with model {embedding_model}")
    else:
        cfg.embedding_model_name = embedding_model
        print(f"🤖 Using local embedding model: {embedding_model}")

    # Always use 3-weight system (legacy 2-weight system removed)
    cfg.set_weights(trigram_weight, syntactic_weight, semantic_weight)

    # Initialize embeddings cache if using semantic similarity
    if cfg.use_semantic and SEMANTIC_AVAILABLE:
        cfg.embeddings = compute_dataset_embeddings(dataset, cfg)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise ValueError("Missing OpenAI API key")

    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load data with proper ID mapping
    A, B, pairs = load_dataset_and_pairs(dataset, use_validation)

    # Note: limit parameter removed - always use full evaluation set for reliable results

    print(f"Processing {len(pairs)} pairs with {max_candidates} candidates ({max_candidates / len(B):.1%} of table B) per record")

    # Create candidate cache for massive speed improvement with persistent caching
    candidate_cache = create_candidate_cache(dataset, B, cfg, max_candidates)

    # Embeddings cache already initialized in run_enhanced_matching function

    start_time = time.time()
    all_predictions = {}
    early_decisions = 0
    llm_calls = 0
    early_decision_pairs = set()  # Track which pairs had early decisions

    # Process pairs with async concurrency
    import asyncio

    from tqdm import tqdm

    async def process_single_pair(row):
        """Process a single pair with enhanced matching"""
        left_id = row.ltable_id
        left_record = A[left_id]

        # Use cached candidate generation for massive speed improvement
        top_candidates = get_top_candidates_cached(
            left_record, candidate_cache, max_candidates, cfg, dataset
        )

        # Enhanced matching with control logic
        match_idx, was_early_decision = await enhanced_match_single_record(
            left_record, top_candidates, cfg, client, heuristic_engine, prompt_data
        )

        # Return results for thread-safe aggregation (include candidates for ID mapping)
        return left_id, match_idx, was_early_decision, top_candidates

    # Create batches for concurrent processing using dynamic batching like baseline
    batch_size = max(1, len(pairs) // 20)  # Dynamic batch size for ~20 progress updates
    pair_rows = list(pairs.iterrows())

    # Create batches
    batches = []
    for i in range(0, len(pair_rows), batch_size):
        batches.append(pair_rows[i : i + batch_size])

    # Use semaphore for concurrency control like baseline
    semaphore = asyncio.Semaphore(concurrency)

    async def process_batch_with_semaphore(batch):
        """Process a batch of pairs with semaphore control"""
        async with semaphore:
            tasks = [process_single_pair(row) for _, row in batch]
            return await asyncio.gather(*tasks)

    # Process batches with progress tracking
    with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
        # Process all batches concurrently (limited by semaphore)
        batch_tasks = [process_batch_with_semaphore(batch) for batch in batches]

        for task in asyncio.as_completed(batch_tasks):
            batch_results = await task

            # Aggregate results thread-safely
            for left_id, match_idx, was_early_decision, top_candidates in batch_results:
                if match_idx != -1:
                    # match_idx is a candidate index, convert to right_id for evaluation
                    right_id, _ = top_candidates[match_idx]
                    all_predictions[left_id] = right_id

                if was_early_decision:
                    early_decisions += 1
                    early_decision_pairs.add(left_id)
                else:
                    llm_calls += 1

            pbar.update(1)  # Update by 1 batch completed

    elapsed_time = time.time() - start_time
    matches_found = len(all_predictions)

    print("\n=== ENHANCED MATCHING RESULTS ===")
    print(f"Processed: {len(pairs)} pairs")
    print(f"Matches found: {matches_found}")
    print(f"Early decisions: {early_decisions}")
    print(f"LLM calls: {llm_calls}")
    print(f"LLM call reduction: {(1 - llm_calls / len(pairs)) * 100:.1f}%")
    print(f"Processing time: {elapsed_time:.1f} seconds")

    # Evaluate predictions with content-based duplicate awareness
    from src.entity_matching.duplicate_aware_evaluation import duplicate_aware_evaluate, calculate_metrics
    
    # B is already in the right format - either dict or list of dicts
    if isinstance(B, dict):
        # Dict case (ID -> record)
        B_dict = B
    else:
        # List of dicts case
        B_dict = {r['id']: r for r in B}

    # Prepare pairs data for evaluation
    pairs_data = [(rec.ltable_id, rec.rtable_id, rec.label) for _, rec in pairs.iterrows()]
    
    # Use shared duplicate-aware evaluation
    preds, labels = duplicate_aware_evaluate(all_predictions, pairs_data, B_dict, verbose=True)

    # Calculate metrics using shared function
    metrics = calculate_metrics(preds, labels)
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    precision, recall, f1, accuracy = metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]

    # Calculate cost
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get("gpt-4o-mini", (0.00015, 0.0006))

    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print("\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Cost: ${total_cost:.4f}")
    print(f"Candidate selection: {max_candidates} candidates ({max_candidates / len(B):.1%} of table B)")
    print(f"Tokens: {cfg.total_input_tokens} input, {cfg.total_output_tokens} output")

    # Generate detailed failure analysis for debugging
    false_positives = []
    false_negatives = []
    true_positives = []

    for _, rec in pairs.iterrows():
        left_id = rec.ltable_id
        right_id = rec.rtable_id
        true_label = rec.label

        if left_id in all_predictions:
            pred_right_id = all_predictions[left_id]
            pred_label = 1 if pred_right_id == right_id else 0
        else:
            pred_label = 0
            pred_right_id = None

        left_record = A[left_id]

        if pred_label == 1 and true_label == 0:
            # False Positive - predicted match but shouldn't match
            predicted_record = B[pred_right_id] if pred_right_id else None
            actual_record = B[right_id]

            # Calculate similarities for the false positive prediction
            if predicted_record:
                left_str = json.dumps(left_record, ensure_ascii=False).lower()
                pred_str = json.dumps(predicted_record, ensure_ascii=False).lower()
                actual_str = json.dumps(actual_record, ensure_ascii=False).lower()

                false_positives.append({
                    "left_record": left_record,
                    "predicted_record": predicted_record,
                    "actual_record": actual_record,
                    "predicted_similarity": {
                        "trigram": trigram_similarity(left_str, pred_str),
                        "semantic": semantic_similarity_cached(left_record, predicted_record, cfg.embeddings) if cfg.use_semantic else 0.0
                    },
                    "actual_similarity": {
                        "trigram": trigram_similarity(left_str, actual_str),
                        "semantic": semantic_similarity_cached(left_record, actual_record, cfg.embeddings) if cfg.use_semantic else 0.0
                    }
                })

        elif pred_label == 0 and true_label == 1:
            # False Negative - should match but didn't predict
            actual_record = B[right_id]
            left_str = json.dumps(left_record, ensure_ascii=False).lower()
            actual_str = json.dumps(actual_record, ensure_ascii=False).lower()

            # Check if actual match was in candidates
            candidates = get_top_candidates_cached(left_record, candidate_cache, max_candidates, cfg, dataset)
            found_in_candidates = any(idx == right_id for idx, _ in candidates)
            candidate_rank = None
            if found_in_candidates:
                for rank, (idx, _) in enumerate(candidates, 1):
                    if idx == right_id:
                        candidate_rank = rank
                        break

            false_negatives.append({
                "left_record": left_record,
                "missed_record": actual_record,
                "similarity": {
                    "trigram": trigram_similarity(left_str, actual_str),
                    "semantic": semantic_similarity_cached(left_record, actual_record, cfg.embeddings) if cfg.use_semantic else 0.0
                },
                "candidate_analysis": {
                    "found_in_candidates": found_in_candidates,
                    "rank": candidate_rank,
                    "max_candidates": max_candidates
                }
            })

        elif pred_label == 1 and true_label == 1:
            # True Positive - correctly predicted match
            matched_record = B[right_id]
            left_str = json.dumps(left_record, ensure_ascii=False).lower()
            matched_str = json.dumps(matched_record, ensure_ascii=False).lower()

            true_positives.append({
                "left_record": left_record,
                "matched_record": matched_record,
                "similarity": {
                    "trigram": trigram_similarity(left_str, matched_str),
                    "semantic": semantic_similarity_cached(left_record, matched_record, cfg.embeddings) if cfg.use_semantic else 0.0
                }
            })

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "cost": total_cost,
        "early_decisions": early_decisions,
        "llm_calls": llm_calls,
        "llm_call_reduction": (1 - llm_calls / len(pairs)) * 100,
        "predictions": all_predictions,  # Include predictions for failure analysis
        "failure_analysis": {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "summary": {
                "total_fp": len(false_positives),
                "total_fn": len(false_negatives),
                "total_tp": len(true_positives)
            }
        }
    }


async def main():
    """CLI entry point for enhanced matching"""
    parser = argparse.ArgumentParser(description="Enhanced entity matching with sophisticated control logic")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    # Note: limit parameter removed - always use full evaluation set for reliable results
    parser.add_argument("--max-candidates", type=int, default=50, help="Max candidates per record")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Semantic weight (trigram/syntactic auto-calculated if not provided)")
    parser.add_argument("--trigram-weight", type=float, help="Trigram weight (3-weight system)")
    parser.add_argument("--syntactic-weight", type=float, help="Syntactic weight (3-weight system)")
    parser.add_argument("--heuristic-file", help="Enhanced heuristics JSON file (optional for baseline testing)")
    parser.add_argument("--use-validation", action="store_true", help="Use validation data instead of test data (prevents data leakage)")
    parser.add_argument("--embedding-base-url", help="Base URL for embedding API (e.g., http://localhost:8080 for TEI)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Embedding model name")

    args = parser.parse_args()

    return await run_enhanced_matching(
        dataset=args.dataset,
        max_candidates=args.max_candidates,
        model=args.model,
        concurrency=args.concurrency,
        semantic_weight=args.semantic_weight,
        trigram_weight=args.trigram_weight,
        syntactic_weight=args.syntactic_weight,
        heuristic_file=args.heuristic_file,
        use_validation=args.use_validation,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    results = asyncio.run(main())
