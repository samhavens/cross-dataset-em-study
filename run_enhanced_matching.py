#!/usr/bin/env python
"""
Enhanced entity matching with sophisticated control logic.

This script uses the enhanced heuristic engine with:
- Early decision rules (skip LLM for high/low confidence)
- Dynamic weight adjustment based on data patterns
- Cost optimization through smart LLM call reduction
"""

import asyncio
import argparse
import json
import pathlib
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.entity_matching.enhanced_heuristic_engine import (
    load_enhanced_heuristics_for_dataset, 
    PipelineStage,
    EnhancedHeuristicEngine
)
from src.entity_matching.hybrid_matcher import (
    Config, get_semantic_model, trigram_similarity, semantic_similarity, 
    token_count, call_openai_async
)
from src.entity_matching.constants import MODEL_COSTS
from openai import AsyncOpenAI
import os


async def enhanced_match_single_record(
    left_record: dict, 
    candidates: List[tuple], 
    cfg: Config, 
    client: AsyncOpenAI,
    heuristic_engine: EnhancedHeuristicEngine
) -> tuple[int, bool]:
    """Enhanced matching with sophisticated control logic"""
    
    # Early decision check before LLM
    best_candidate = None
    best_score = 0.0
    weight_adjustments = 0
    
    for idx, candidate_record in candidates:
        # Calculate combined similarity score
        left_str = json.dumps(left_record, ensure_ascii=False).lower()
        right_str = json.dumps(candidate_record, ensure_ascii=False).lower()
        
        trigram_score = trigram_similarity(left_str, right_str)
        semantic_score = semantic_similarity(left_str, right_str, cfg) if cfg.use_semantic else 0.0
        
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
        print(f"    Applied {weight_adjustments} weight adjustments to candidates")
    
    # Apply decision rules before LLM call
    decision = heuristic_engine.apply_decision_rules(
        left_record, best_record, best_score, PipelineStage.PRE_LLM
    )
    
    if decision and decision.terminate_early:
        if decision.skip_llm:
            print(f"    Early decision: {decision.reason} -> {decision.final_result} (skipped LLM)")
        else:
            print(f"    Early decision: {decision.reason} -> {decision.final_result}")
        # If rule says accept (1), return the best candidate index; if reject (0), return -1
        return (best_idx if decision.final_result == 1 else -1), True
    
    # Fall back to LLM if no early decision
    print(f"    Proceeding to LLM with score {best_score:.3f}")
    
    # Build prompt with the best candidate
    candidates_text = f"{best_idx}) {json.dumps(best_record, ensure_ascii=False)}"
    
    prompt = f"""You are an expert at entity matching. Your task is to find the candidate that refers to the same real-world entity as the left record.

Two records match if they refer to the same entity, even if:
- They have different formatting or spellings
- One has more/less information than the other
- They use different abbreviations or representations

LEFT RECORD:
{json.dumps(left_record, ensure_ascii=False)}

CANDIDATES:
{candidates_text}

Compare the left record against the candidate. Look for:
1. Same entity name (allowing for variations in spelling/format)
2. Matching key identifiers (IDs, codes, etc.)
3. Consistent attribute values where they overlap
4. No contradictory information

Think step by step and identify if the candidate represents the same entity.

CRITICAL: Your response must contain ONLY a single number:
- If you find a match, output ONLY the candidate number (e.g., "{best_idx}")
- If no candidate represents the same entity, output ONLY "-1"
- Do NOT include any explanation, reasoning, or other text
- Do NOT use quotes around the number

ANSWER:
"""
    
    # Check token count
    total_tokens = token_count(prompt, cfg.model)
    if total_tokens > 1000000:  # 1M token limit
        print(f"  WARNING: Prompt too large ({total_tokens:,} tokens)")
        return -1, False
    
    # Get LLM response
    response = await call_openai_async(prompt, cfg, client)
    
    # Parse response
    if not response:
        print(f"  WARNING: Empty response from LLM")
        return -1, False
    
    try:
        match_idx = int(response)
        return (match_idx if match_idx == best_idx else -1), False
    except ValueError:
        print(f"  WARNING: Could not parse LLM response: '{response}'")
        return -1, False


async def run_enhanced_matching(
    dataset: str,
    limit: Optional[int] = None,
    max_candidates: int = 50,
    model: str = "gpt-4.1-nano",
    concurrency: int = 10,
    semantic_weight: float = 0.5,
    heuristic_file: str = None
) -> Dict:
    """Run enhanced entity matching with sophisticated control logic"""
    
    print(f"🚀 ENHANCED ENTITY MATCHING")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Candidates: {max_candidates}")
    print(f"Semantic weight: {semantic_weight}")
    print(f"Heuristics: {heuristic_file}")
    print("=" * 80)
    
    # Load enhanced heuristic engine
    heuristic_engine = load_enhanced_heuristics_for_dataset(dataset, heuristic_file)
    
    # Initialize configuration
    cfg = Config()
    cfg.model = model
    cfg.use_semantic = True
    cfg.semantic_weight = semantic_weight
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        raise ValueError("Missing OpenAI API key")
    
    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Load data with proper ID mapping
    root = pathlib.Path('data') / 'raw' / dataset
    A_df = pd.read_csv(root / 'tableA.csv')
    B_df = pd.read_csv(root / 'tableB.csv')
    
    # Check if this dataset has non-sequential IDs (like zomato_yelp)
    if 'id' in A_df.columns:
        # Create ID-to-record mappings
        A = {row['id']: row.to_dict() for _, row in A_df.iterrows()}
        B = {row['id']: row.to_dict() for _, row in B_df.iterrows()}
        print(f"Dataset uses ID mapping: A has {len(A)} records (IDs {min(A.keys())}-{max(A.keys())})")
    else:
        # Use list indexing for datasets without ID column
        A = A_df.to_dict(orient='records')
        B = B_df.to_dict(orient='records')
        print(f"Dataset uses list indexing: A has {len(A)} records")
        
    pairs = pd.read_csv(root / 'test.csv')
    
    if limit:
        pairs = pairs.head(limit)
    
    print(f"Processing {len(pairs)} pairs with enhanced control logic...")
    
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
        
        # Get top candidates using trigram similarity
        candidates = []
        
        # Handle both list and dict access patterns
        if isinstance(B, dict):
            # Dict access (ID-based)
            for record_id, right_record in B.items():
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                left_str = json.dumps(left_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)
                candidates.append((score, record_id, right_record))
        else:
            # List access (index-based)
            for j, right_record in enumerate(B):
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                left_str = json.dumps(left_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)
                candidates.append((score, j, right_record))
        
        # Sort and take top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = [(idx, record) for _, idx, record in candidates[:max_candidates]]
        
        # Enhanced matching with control logic
        match_idx, was_early_decision = await enhanced_match_single_record(
            left_record, top_candidates, cfg, client, heuristic_engine
        )
        
        # Return results for thread-safe aggregation
        return left_id, match_idx, was_early_decision
    
    # Create batches for concurrent processing
    batch_size = concurrency
    pair_rows = list(pairs.iterrows())
    
    # Process in batches with progress tracking
    with tqdm(total=len(pairs), desc="Processing pairs", unit="pair") as pbar:
        for i in range(0, len(pair_rows), batch_size):
            batch = pair_rows[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [process_single_pair(row) for _, row in batch]
            batch_results = await asyncio.gather(*tasks)
            
            # Aggregate results thread-safely
            for left_id, match_idx, was_early_decision in batch_results:
                if match_idx != -1:
                    all_predictions[left_id] = match_idx
                    
                if was_early_decision:
                    early_decisions += 1
                    early_decision_pairs.add(left_id)
                else:
                    llm_calls += 1
            
            pbar.update(len(batch))
    
    elapsed_time = time.time() - start_time
    matches_found = len(all_predictions)
    
    print(f"\n=== ENHANCED MATCHING RESULTS ===")
    print(f"Processed: {len(pairs)} pairs")
    print(f"Matches found: {matches_found}")
    print(f"Early decisions: {early_decisions}")
    print(f"LLM calls: {llm_calls}")
    print(f"LLM call reduction: {(1 - llm_calls/len(pairs))*100:.1f}%")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    
    # Evaluate predictions
    preds = []
    labels = []
    
    for _, rec in pairs.iterrows():
        left_id = rec.ltable_id
        right_id = rec.rtable_id
        true_label = rec.label
        
        if left_id in all_predictions:
            pred_right_id = all_predictions[left_id]
            pred_label = 1 if pred_right_id == right_id else 0
        else:
            pred_label = 0
        
        preds.append(pred_label)
        labels.append(true_label)
    
    # Calculate metrics
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(preds)
    
    # Calculate cost
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get('gpt-4o-mini', (0.00015, 0.0006))
    
    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    
    print(f"\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Cost: ${total_cost:.4f}")
    
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "cost": total_cost,
        "early_decisions": early_decisions,
        "llm_calls": llm_calls,
        "llm_call_reduction": (1 - llm_calls/len(pairs))*100
    }


async def main():
    """CLI entry point for enhanced matching"""
    parser = argparse.ArgumentParser(description="Enhanced entity matching with sophisticated control logic")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--limit', type=int, help='Limit number of test pairs')
    parser.add_argument('--max-candidates', type=int, default=50, help='Max candidates per record')
    parser.add_argument('--model', default='gpt-4.1-nano', help='Model to use')
    parser.add_argument('--concurrency', type=int, default=10, help='Concurrency level')
    parser.add_argument('--semantic-weight', type=float, default=0.5, help='Base semantic weight')
    parser.add_argument('--heuristic-file', required=True, help='Enhanced heuristics JSON file')
    
    args = parser.parse_args()
    
    results = await run_enhanced_matching(
        dataset=args.dataset,
        limit=args.limit,
        max_candidates=args.max_candidates,
        model=args.model,
        concurrency=args.concurrency,
        semantic_weight=args.semantic_weight,
        heuristic_file=args.heuristic_file
    )
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())