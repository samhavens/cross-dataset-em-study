#!/usr/bin/env python
import argparse
import asyncio
import datetime
import json
import os
import pathlib
import pickle
import textwrap
import time

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tiktoken

from tqdm.asyncio import tqdm

from openai import AsyncOpenAI

from .constants import MODEL_COSTS
from .heuristic_engine import load_heuristics_for_dataset

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


# Configuration
class Config:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.temperature = 0
        self.max_tokens = 100
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.use_semantic = True  # Enable semantic similarity by default
        self.semantic_weight = 0.5  # Weight for combining semantic + trigram scores
        self.semantic_model = None  # Will be initialized lazily
        self.use_heuristics = False  # Enable heuristic rules
        self.heuristic_engine = None  # Will be initialized if enabled
        self.heuristic_file = None  # Path to heuristics file
        self.embeddings = None  # Cached embeddings for the dataset


# Token counting
def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text.split()) * 1.3


def report_cost(cfg: Config):
    """Report total cost and token usage"""
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        print(f"WARNING: Model {cfg.model} not found in MODEL_COSTS. Using gpt-4.1-mini instead.")
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS["gpt-4.1-mini"]

    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print(f"≈{cfg.total_input_tokens / 1000:.1f}K in, {cfg.total_output_tokens / 1000:.1f}K out → ${total_cost:.3f}")


MAX = 1_000_000  # token limit for the matching prompt


async def call_openai_async(prompt: str, cfg: Config, client: AsyncOpenAI) -> str:
    """Make async call to OpenAI API"""
    try:
        # o3/o4 models use max_completion_tokens instead of max_tokens and don't support temperature=0
        if cfg.model.startswith(("o3", "o4")):
            response = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max(cfg.max_tokens, 1000),  # Give o4 models more tokens
            )
        else:
            response = await client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )

        # Track usage
        usage = response.usage
        cfg.total_input_tokens += usage.prompt_tokens
        cfg.total_output_tokens += usage.completion_tokens

        content = response.choices[0].message.content
        if content is None:
            print(f"  WARNING: Empty response from OpenAI API for model {cfg.model}")
            print(f"  Response object: {response}")
            return ""

        result = content.strip()
        if cfg.model.startswith(("o3", "o4")) and not result:
            print(f"  WARNING: o4 model returned empty string. Raw content: '{content}'")

        return result

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ""


def trigram_similarity(s1: str, s2: str) -> float:
    """Calculate trigram similarity between two strings"""

    def get_trigrams(s):
        s = s.lower()
        return {s[i : i + 3] for i in range(len(s) - 2)}

    t1, t2 = get_trigrams(s1), get_trigrams(s2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_semantic_model(cfg: Config):
    """Get or initialize the semantic similarity model"""
    if not SEMANTIC_AVAILABLE:
        return None

    if cfg.semantic_model is None:
        print("Loading semantic similarity model (first time only)...")
        cfg.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    return cfg.semantic_model


def get_heuristic_engine(cfg: Config, dataset: str):
    """Get or initialize the heuristic engine"""
    if not cfg.use_heuristics:
        return None

    if cfg.heuristic_engine is None:
        print("Loading heuristic engine (first time only)...")
        cfg.heuristic_engine = load_heuristics_for_dataset(dataset, cfg.heuristic_file)

        if cfg.heuristic_engine.rules:
            print(f"Loaded {len(cfg.heuristic_engine.rules)} heuristic rules for {dataset}")
        else:
            print("No heuristic rules loaded - continuing without heuristics")
            cfg.use_heuristics = False

    return cfg.heuristic_engine


def get_embeddings_cache_path(dataset: str) -> pathlib.Path:
    """Get path for embeddings cache file"""
    cache_dir = pathlib.Path(".embeddings_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{dataset}_embeddings.pkl"


def compute_dataset_embeddings(dataset: str, cfg: Config) -> Dict[str, np.ndarray]:
    """Compute and cache embeddings for entire dataset"""
    cache_path = get_embeddings_cache_path(dataset)

    # Check if cache exists
    if cache_path.exists():
        print(f"📁 Loading cached embeddings from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}, recomputing...")

    print(f"🧮 Computing embeddings for {dataset} (this may take a few minutes)...")

    model = get_semantic_model(cfg)
    if model is None:
        return {}

    # Load dataset
    root = pathlib.Path("data") / "raw" / dataset
    A_df = pd.read_csv(root / "tableA.csv")
    B_df = pd.read_csv(root / "tableB.csv")

    embeddings = {}

    # Convert records to strings and compute embeddings
    print(f"🔄 Computing embeddings for {len(A_df)} records in tableA...")
    A_strings = [json.dumps(row.to_dict(), ensure_ascii=False).lower() for _, row in A_df.iterrows()]
    A_embeddings = model.encode(A_strings, show_progress_bar=True, batch_size=32)

    print(f"🔄 Computing embeddings for {len(B_df)} records in tableB...")
    B_strings = [json.dumps(row.to_dict(), ensure_ascii=False).lower() for _, row in B_df.iterrows()]
    B_embeddings = model.encode(B_strings, show_progress_bar=True, batch_size=32)

    # Store with proper ID mapping
    if "id" in A_df.columns:
        # Use actual IDs
        for i, (_, row) in enumerate(A_df.iterrows()):
            embeddings[f"A_{row['id']}"] = A_embeddings[i]
        for i, (_, row) in enumerate(B_df.iterrows()):
            embeddings[f"B_{row['id']}"] = B_embeddings[i]
    else:
        # Use indices
        for i in range(len(A_df)):
            embeddings[f"A_{i}"] = A_embeddings[i]
        for i in range(len(B_df)):
            embeddings[f"B_{i}"] = B_embeddings[i]

    # Cache the embeddings
    print(f"💾 Caching embeddings to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ Embeddings computed and cached for {len(embeddings)} records")
    return embeddings


def semantic_similarity_cached(left_record: dict, right_record: dict, embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate semantic similarity using cached embeddings"""
    try:
        # Get record IDs
        left_id = left_record.get("id", 0)
        right_id = right_record.get("id", 0)

        left_key = f"A_{left_id}"
        right_key = f"B_{right_id}"

        if left_key not in embeddings or right_key not in embeddings:
            return 0.0

        left_emb = embeddings[left_key]
        right_emb = embeddings[right_key]

        # Calculate cosine similarity
        cos_sim = np.dot(left_emb, right_emb) / (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))
        return float(cos_sim)
    except Exception as e:
        print(f"Warning: Cached semantic similarity calculation failed: {e}")
        return 0.0


def semantic_similarity(s1: str, s2: str, cfg: Config) -> float:
    """Calculate semantic similarity using sentence transformers (fallback for non-cached)"""
    if not SEMANTIC_AVAILABLE or not cfg.use_semantic:
        return 0.0

    model = get_semantic_model(cfg)
    if model is None:
        return 0.0

    try:
        embeddings = model.encode([s1, s2])
        # Calculate cosine similarity
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return float(cos_sim)
    except Exception as e:
        print(f"Warning: Semantic similarity calculation failed: {e}")
        return 0.0


def combined_similarity(s1: str, s2: str, cfg: Config) -> float:
    """Calculate combined trigram + semantic similarity"""
    trigram_score = trigram_similarity(s1, s2)

    if not cfg.use_semantic or not SEMANTIC_AVAILABLE:
        return trigram_score

    semantic_score = semantic_similarity(s1, s2, cfg)

    # Weighted combination
    return (1 - cfg.semantic_weight) * trigram_score + cfg.semantic_weight * semantic_score


def get_top_candidates(
    left_record: dict, right_records, max_candidates: int, cfg: Config, dataset: str = None
) -> List[tuple]:
    """Get top candidates for a left record using combined similarity and heuristics"""
    left_str = json.dumps(left_record, ensure_ascii=False).lower()

    # Get heuristic engine if enabled
    heuristic_engine = get_heuristic_engine(cfg, dataset) if dataset else None

    # Fast trigram filtering first to reduce candidates for semantic similarity
    if cfg.use_semantic and SEMANTIC_AVAILABLE:
        # First pass: Get top candidates using trigram similarity (more candidates than final)
        trigram_candidates = max_candidates * 3  # 3x candidates for semantic reranking
        trigram_scores = []

        # Handle both list and dict access patterns
        if isinstance(right_records, dict):
            # Dict access (ID-based)
            for record_id, right_record in right_records.items():
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)

                # Apply candidate generation heuristics to boost similarity for candidate selection
                if heuristic_engine:
                    try:
                        candidate_action = heuristic_engine.apply_stage_heuristics(
                            "candidate_generation", left_record, right_record
                        )
                        if candidate_action and hasattr(candidate_action, "similarity_boost"):
                            score += candidate_action.similarity_boost * candidate_action.confidence
                            score = min(score, 1.0)  # Cap at 1.0
                    except Exception:
                        # Don't let heuristic failures break candidate generation
                        pass

                trigram_scores.append((score, record_id, right_record, right_str))
        else:
            # List access (index-based)
            for i, right_record in enumerate(right_records):
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)

                # Apply candidate generation heuristics to boost similarity for candidate selection
                if heuristic_engine:
                    try:
                        candidate_action = heuristic_engine.apply_stage_heuristics(
                            "candidate_generation", left_record, right_record
                        )
                        if candidate_action and hasattr(candidate_action, "similarity_boost"):
                            score += candidate_action.similarity_boost * candidate_action.confidence
                            score = min(score, 1.0)  # Cap at 1.0
                    except Exception:
                        # Don't let heuristic failures break candidate generation
                        pass

                trigram_scores.append((score, i, right_record, right_str))

        # Sort by trigram similarity and take top candidates for semantic reranking
        trigram_scores.sort(key=lambda x: x[0], reverse=True)
        top_trigram = trigram_scores[:trigram_candidates]

        # Second pass: Semantic similarity using cached embeddings
        if cfg.embeddings is not None:
            try:
                # Calculate combined scores with cached embeddings
                candidates = []
                for trigram_score, orig_idx, record, right_str in top_trigram:
                    # Use cached semantic similarity
                    semantic_score = semantic_similarity_cached(left_record, record, cfg.embeddings)

                    # Weighted combination
                    combined_score = (1 - cfg.semantic_weight) * trigram_score + cfg.semantic_weight * semantic_score

                    # Apply heuristic adjustments if available
                    if heuristic_engine:
                        try:
                            heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                            combined_score += heuristic_adjustment
                        except Exception:
                            # Don't let heuristic failures break the matching
                            pass

                    candidates.append((combined_score, orig_idx, record))

            except Exception as e:
                print(f"Warning: Semantic similarity failed, falling back to trigram: {e}")
                # Fall back to trigram only with heuristics
                candidates = []
                for score, i, record, _ in top_trigram[:max_candidates]:
                    final_score = score
                    # Apply heuristic adjustments if available
                    if heuristic_engine:
                        try:
                            heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                            final_score += heuristic_adjustment
                        except Exception:
                            pass
                    candidates.append((final_score, i, record))
        else:
            # Fall back to trigram only with heuristics
            candidates = []
            for score, i, record, _ in top_trigram[:max_candidates]:
                final_score = score
                # Apply heuristic adjustments if available
                if heuristic_engine:
                    try:
                        heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                        final_score += heuristic_adjustment
                    except Exception:
                        pass
                candidates.append((final_score, i, record))
    else:
        # Trigram only with heuristics
        candidates = []

        # Handle both list and dict access patterns
        if isinstance(right_records, dict):
            # Dict access (ID-based)
            for record_id, right_record in right_records.items():
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)

                # Apply candidate generation heuristics first
                if heuristic_engine:
                    try:
                        candidate_action = heuristic_engine.apply_stage_heuristics(
                            "candidate_generation", left_record, right_record
                        )
                        if candidate_action and hasattr(candidate_action, "similarity_boost"):
                            score += candidate_action.similarity_boost * candidate_action.confidence
                            score = min(score, 1.0)  # Cap at 1.0
                    except Exception:
                        pass

                # Apply other heuristic adjustments if available
                if heuristic_engine:
                    try:
                        heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, right_record)
                        score += heuristic_adjustment
                    except Exception:
                        pass

                candidates.append((score, record_id, right_record))
        else:
            # List access (index-based)
            for i, right_record in enumerate(right_records):
                right_str = json.dumps(right_record, ensure_ascii=False).lower()
                score = trigram_similarity(left_str, right_str)

                # Apply candidate generation heuristics first
                if heuristic_engine:
                    try:
                        candidate_action = heuristic_engine.apply_stage_heuristics(
                            "candidate_generation", left_record, right_record
                        )
                        if candidate_action and hasattr(candidate_action, "similarity_boost"):
                            score += candidate_action.similarity_boost * candidate_action.confidence
                            score = min(score, 1.0)  # Cap at 1.0
                    except Exception:
                        pass

                # Apply other heuristic adjustments if available
                if heuristic_engine:
                    try:
                        heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, right_record)
                        score += heuristic_adjustment
                    except Exception:
                        pass

                candidates.append((score, i, right_record))

    # Sort by similarity and take top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(idx, record) for _, idx, record in candidates[:max_candidates]]


async def match_single_record(left_record: dict, candidates: List[tuple], cfg: Config, client: AsyncOpenAI) -> int:
    """Match a single left record against its filtered candidates"""

    # Create mapping from position numbers to actual IDs for robust parsing
    id_mapping = {}  # position_number -> actual_database_id

    # Build prompt with sequential position numbers (1, 2, 3...) regardless of actual IDs
    candidates_lines = []
    for position, (actual_id, record) in enumerate(candidates, 1):
        id_mapping[position] = actual_id
        candidates_lines.append(f"{position}) {json.dumps(record, ensure_ascii=False)}")

    candidates_text = "\n".join(candidates_lines)

    # Create prompt
    prompt = textwrap.dedent(f"""
      You are an expert at entity matching. Your task is to find the candidate that refers to the same real-world entity as the left record.

      Two records match if they refer to the same entity, even if:
      - They have different formatting or spellings
      - One has more/less information than the other
      - They use different abbreviations or representations

      LEFT RECORD:
      {json.dumps(left_record, ensure_ascii=False)}

      CANDIDATES:
      {candidates_text}

      Compare the left record against each candidate. Look for:
      1. Same entity name (allowing for variations in spelling/format)
      2. Matching key identifiers (IDs, codes, etc.)
      3. Consistent attribute values where they overlap
      4. No contradictory information

      Think step by step and identify the candidate that represents the same entity.

      CRITICAL: Your response must contain ONLY a single number:
      - If you find a match, output ONLY the position number from the list above (e.g., "1", "2", "3")
      - If no candidate represents the same entity, output ONLY "-1"
      - Do NOT include any explanation, reasoning, or other text
      - Do NOT use quotes around the number
      - Do NOT return record IDs - only return the position number (1, 2, 3, etc.)

      ANSWER:
    """)

    # Check token count
    total_tokens = token_count(prompt, cfg.model)
    if total_tokens > MAX:
        print(f"  WARNING: Prompt too large ({total_tokens:,} tokens)")

    # Log the prompt for o4 debugging
    if cfg.model.startswith(("o3", "o4")):
        print(f"  DEBUG: Sending prompt to {cfg.model}:")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  First 200 chars: {prompt[:200]}")
        print(f"  Last 200 chars: {prompt[-200:]}")

    # Get LLM response
    response = await call_openai_async(prompt, cfg, client)

    # Parse response
    if not response:
        if cfg.model.startswith(("o3", "o4")):
            print(f"  DEBUG: Empty response from {cfg.model} - logging prompt to debug_prompt.txt")
            with open("debug_prompt.txt", "w") as f:
                f.write(f"=== FAILED PROMPT FOR {cfg.model} ===\n")
                f.write(f"Prompt length: {len(prompt)} characters\n")
                f.write(f"Token count: {token_count(prompt, cfg.model)} tokens\n")
                f.write("=== FULL PROMPT START ===\n")
                f.write(prompt)
                f.write("\n=== FULL PROMPT END ===\n")
        raise ValueError("Empty response from LLM")

    try:
        # First try direct integer parsing for position number
        position_number = int(response.strip())
        if position_number == -1:
            return -1  # No match
        if position_number in id_mapping:
            return id_mapping[position_number]  # Convert position to actual database ID
        print(f"Warning: LLM returned position {position_number} but only {len(candidates)} candidates available")
        return -1
    except ValueError:
        # Try to extract number from response using regex
        import re

        numbers = re.findall(r"-?\d+", response.strip())
        if numbers:
            try:
                position_number = int(numbers[0])
                if position_number == -1:
                    return -1  # No match
                if position_number in id_mapping:
                    return id_mapping[position_number]  # Convert position to actual database ID
                print(
                    f"Warning: LLM returned position {position_number} but only {len(candidates)} candidates available"
                )
                return -1
            except ValueError:
                pass

        # If all else fails, return -1 (no match) and log the issue
        print(f"Warning: Could not parse LLM response as integer: '{response}', defaulting to -1 (no match)")
        return -1


async def process_batch(
    batch_pairs: List[tuple],
    semaphore: asyncio.Semaphore,
    A: List[dict],
    B: List[dict],
    max_candidates: int,
    cfg: Config,
    client: AsyncOpenAI,
    dataset: str,
) -> Dict[int, int]:
    """Process a batch of pairs with concurrency control"""
    async with semaphore:
        tasks = []
        for _, row in batch_pairs:
            left_id = row.ltable_id
            left_record = A[left_id]

            # Get top candidates for this left record (with heuristics if enabled)
            candidates = get_top_candidates(left_record, B, max_candidates, cfg, dataset)

            # Create async task for matching
            task = match_single_record(left_record, candidates, cfg, client)
            tasks.append((left_id, task))

        # Execute all tasks in this batch
        batch_results = {}
        for left_id, task in tasks:
            match_idx = await task
            if match_idx != -1:
                batch_results[left_id] = match_idx

        return batch_results


async def run_matching(
    dataset: str,
    limit: Optional[int] = None,
    max_candidates: Optional[int] = None,
    candidate_ratio: Optional[float] = None,
    model: str = "gpt-4.1-nano",
    concurrency: int = 20,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    use_semantic: bool = True,
    semantic_weight: float = 0.5,
    use_heuristics: bool = False,
    heuristic_file: Optional[str] = None,
    embeddings_cache_dataset: Optional[str] = None,
) -> Dict:
    """
    Main function to run entity matching

    Returns:
        Dict with results including metrics, cost, etc.
    """

    # Initialize configuration
    cfg = Config()
    cfg.model = model
    cfg.use_semantic = use_semantic
    cfg.semantic_weight = semantic_weight
    cfg.use_heuristics = use_heuristics
    cfg.heuristic_file = heuristic_file

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        raise ValueError("Missing OpenAI API key")

    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Using OpenAI API with model: {cfg.model}")

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

    # Set default if neither option provided
    if max_candidates is None and candidate_ratio is None:
        DEFAULT_MAX_CANDIDATES = 50
        max_candidates = DEFAULT_MAX_CANDIDATES
        print(f"Using default max candidates: {DEFAULT_MAX_CANDIDATES}")

    # Calculate max_candidates based on parameter choice
    if candidate_ratio is not None:
        if not 0.0 < candidate_ratio <= 1.0:
            raise ValueError(f"candidate_ratio must be between 0.0 and 1.0, got {candidate_ratio}")
        max_candidates = max(1, int(len(B) * candidate_ratio))
        candidate_method = f"{candidate_ratio:.1%} of table B ({max_candidates} candidates)"
    else:
        candidate_method = f"{max_candidates} candidates ({max_candidates / len(B):.1%} of table B)"

    # load test pairs
    pairs = pd.read_csv(root / "test.csv")
    if limit:
        pairs = pairs.head(limit)

    start_time = time.time()

    print(f"Processing {len(pairs)} pairs with {candidate_method} per record")
    print(f"Concurrency: {concurrency} parallel requests")

    # Initialize embeddings cache if using semantic similarity
    if cfg.use_semantic and SEMANTIC_AVAILABLE:
        cache_dataset = embeddings_cache_dataset or dataset
        cfg.embeddings = compute_dataset_embeddings(cache_dataset, cfg)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Split pairs into batches for progress tracking
    batch_size = max(1, len(pairs) // 20)  # 20 progress updates
    batches = []
    for i in range(0, len(pairs), batch_size):
        batch = list(pairs.iloc[i : i + batch_size].iterrows())
        batches.append(batch)

    # Process batches with progress bar
    all_predictions = {}

    # Process batches with progress tracking
    tasks = [process_batch(batch, semaphore, A, B, max_candidates, cfg, client, dataset) for batch in batches]

    # Use tqdm with gather for proper progress tracking
    with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
        for task in asyncio.as_completed(tasks):
            batch_results = await task
            all_predictions.update(batch_results)
            pbar.update(1)

    elapsed_time = time.time() - start_time
    matches_found = len(all_predictions)

    print(f"\nCompleted! Found {matches_found} matches out of {len(pairs)} pairs")
    print(f"Processing time: {elapsed_time:.1f} seconds")

    # Evaluate predictions
    preds = []
    labels = []

    for _, rec in pairs.iterrows():
        left_id = rec.ltable_id
        right_id = rec.rtable_id
        true_label = rec.label

        # Check if we predicted a match and if it's correct
        if left_id in all_predictions:
            pred_right_id = all_predictions[left_id]
            pred_label = 1 if pred_right_id == right_id else 0
        else:
            pred_label = 0  # No match predicted

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

    # Calculate final cost
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        print(f"WARNING: Model {cfg.model} not found in MODEL_COSTS. Using gpt-4o-mini pricing.")
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get("gpt-4o-mini", (0.00015, 0.0006))

    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print("\n=== EVALUATION RESULTS ===")
    print(f"Dataset: {dataset}")
    print(f"Model: {cfg.model}")
    print(f"Processed: {len(pairs)} pairs")
    print(f"Candidate selection: {candidate_method}")
    print(f"Predictions made: {len(all_predictions)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    report_cost(cfg)

    # Create structured results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset,
        "model": cfg.model,
        "candidate_method": candidate_method,
        "max_candidates": max_candidates,
        "candidate_ratio": candidate_ratio if candidate_ratio else max_candidates / len(B),
        "processed_pairs": len(pairs),
        "predictions_made": len(all_predictions),
        "matches_found": matches_found,
        "elapsed_seconds": elapsed_time,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "cost_usd": total_cost,
        "tokens": {
            "input": cfg.total_input_tokens,
            "output": cfg.total_output_tokens,
            "total": cfg.total_input_tokens + cfg.total_output_tokens,
        },
        "table_sizes": {"table_a": len(A), "table_b": len(B)},
        "predictions": all_predictions,  # Include predictions for heuristic analysis
    }

    # Save JSON results if requested
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {output_json}")

    # Append to CSV if requested
    if output_csv:
        import csv

        # Flatten the results for CSV
        csv_row = {
            "timestamp": results["timestamp"],
            "dataset": results["dataset"],
            "model": results["model"],
            "candidate_ratio": results["candidate_ratio"],
            "max_candidates": results["max_candidates"],
            "processed_pairs": results["processed_pairs"],
            "predictions_made": results["predictions_made"],
            "matches_found": results["matches_found"],
            "elapsed_seconds": results["elapsed_seconds"],
            "precision": results["metrics"]["precision"],
            "recall": results["metrics"]["recall"],
            "f1": results["metrics"]["f1"],
            "accuracy": results["metrics"]["accuracy"],
            "tp": results["metrics"]["tp"],
            "fp": results["metrics"]["fp"],
            "fn": results["metrics"]["fn"],
            "tn": results["metrics"]["tn"],
            "cost_usd": results["cost_usd"],
            "input_tokens": results["tokens"]["input"],
            "output_tokens": results["tokens"]["output"],
            "total_tokens": results["tokens"]["total"],
            "table_a_size": results["table_sizes"]["table_a"],
            "table_b_size": results["table_sizes"]["table_b"],
        }

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(output_csv)

        with open(output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

        print(f"Results appended to: {output_csv}")

    return results


async def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=None, help="number of test rows")

    # Candidate selection - mutually exclusive options
    candidate_group = parser.add_mutually_exclusive_group()
    candidate_group.add_argument(
        "--max-candidates", type=int, help="maximum candidates per left record (absolute number)"
    )
    candidate_group.add_argument(
        "--candidate-ratio", type=float, help="candidates as ratio of table B size (e.g., 0.02 = 2%% of table B)"
    )

    parser.add_argument("--model", default="gpt-4.1-nano")
    parser.add_argument("--concurrency", type=int, default=20, help="number of concurrent API calls")
    parser.add_argument("--output-json", type=str, help="save detailed results to JSON file")
    parser.add_argument("--output-csv", type=str, help="append results to CSV file")

    # Semantic similarity options
    parser.add_argument("--no-semantic", action="store_true", help="disable semantic similarity (use only trigram)")
    parser.add_argument(
        "--semantic-weight", type=float, default=0.5, help="weight for semantic similarity (0.0-1.0, default 0.5)"
    )

    # Heuristic options
    parser.add_argument("--use-heuristics", action="store_true", help="enable domain-specific heuristic rules")
    parser.add_argument("--heuristic-file", type=str, help="path to heuristics JSON file")

    args = parser.parse_args()

    return await run_matching(
        dataset=args.dataset,
        limit=args.limit,
        max_candidates=args.max_candidates,
        candidate_ratio=args.candidate_ratio,
        model=args.model,
        concurrency=args.concurrency,
        output_json=args.output_json,
        output_csv=args.output_csv,
        use_semantic=not args.no_semantic,
        semantic_weight=args.semantic_weight,
        use_heuristics=args.use_heuristics,
        heuristic_file=args.heuristic_file,
    )


if __name__ == "__main__":
    results = asyncio.run(main())
