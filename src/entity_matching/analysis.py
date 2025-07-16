"""
Generate rich similarity analysis for Claude-driven optimization.

This module analyzes similarity distributions for matches vs non-matches
and generates structured JSON output that Claude can use to make intelligent
decisions about hyperparameters and rules.
"""

import json
import pathlib

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .hybrid_matcher import (
    CandidateCache,
    Config,
    compute_dataset_embeddings,
    get_top_candidates_cached,
    semantic_similarity,
    syntactic_similarity,
    trigram_similarity,
)


def get_dev_dataset(dataset: str, max_pairs: int = 200) -> Tuple[pd.DataFrame, str]:
    """Get development dataset without test leakage"""
    data_root = pathlib.Path("data/raw") / dataset

    if (data_root / "valid.csv").exists():
        print("✅ Using validation set for analysis (no test leakage)")
        pairs = pd.read_csv(data_root / "valid.csv")
        dataset_type = "validation"
    elif (data_root / "train.csv").exists():
        print("✅ Using slice of training set for analysis (no test leakage)")
        train_pairs = pd.read_csv(data_root / "train.csv")
        dev_slice_size = min(max_pairs, len(train_pairs))
        pairs = train_pairs.head(dev_slice_size)
        print(f"📊 Using {dev_slice_size} pairs from training set for analysis")
        dataset_type = "train_slice"
    else:
        print("⚠️ No validation or training set - using test set for analysis (test won't be clean)")
        pairs = pd.read_csv(data_root / "test.csv")
        dataset_type = "test"

    return pairs, dataset_type


def calculate_similarity_stats(similarities: List[float]) -> Dict[str, float]:
    """Calculate statistical measures for similarity scores"""
    if not similarities:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(similarities)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def get_semantic_similarity_for_records(left_record: dict, right_record: dict, cfg: Config) -> float:
    """Get semantic similarity between two records"""
    try:
        left_str = json.dumps(left_record, ensure_ascii=False).lower()
        right_str = json.dumps(right_record, ensure_ascii=False).lower()
        return semantic_similarity(left_str, right_str, cfg)
    except Exception as e:
        print(f"Warning: Semantic similarity calculation failed: {e}")
        return 0.0


def clean_record_for_json(record: dict) -> dict:
    """Clean a record to ensure JSON serializable (no NaN values)"""
    cleaned = {}
    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = ""  # Replace NaN with empty string
        else:
            cleaned[key] = value
    return cleaned


def generate_concrete_examples(
    positive_pairs: pd.DataFrame,
    negative_pairs: pd.DataFrame,
    A_records: Dict[int, dict],
    B_records: Dict[int, dict],
    candidate_cache: CandidateCache,
    cfg: Config,
    dataset: str,
    max_candidates: int,
    max_examples: int = 50,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Generate concrete examples like dump_matches.py for Claude to see"""

    true_match_examples = []
    false_positive_examples = []

    if verbose:
        print("🎯 Generating concrete examples for Claude analysis...")

    # Disable heuristics during analysis for speed
    original_use_heuristics = cfg.use_heuristics
    cfg.use_heuristics = False

    # Smart sampling strategy for maximum information gain
    if verbose:
        print(f"🎯 Smart sampling for concrete examples (max {max_examples} examples)...")

    # Separate positive and negative pairs for balanced sampling
    positive_sample = positive_pairs.sample(min(max_examples//2, len(positive_pairs)), random_state=42)
    negative_sample = negative_pairs.sample(min(max_examples//2, len(negative_pairs)), random_state=42)

    # Get unique left records from sampled pairs only
    all_left_ids = set(positive_sample['ltable_id'].tolist() + negative_sample['ltable_id'].tolist())
    valid_left_ids = [left_id for left_id in all_left_ids if left_id in A_records]

    if verbose:
        print(f"📦 Pre-computing candidates for {len(valid_left_ids)} unique left records (sampled)...")

    # Pre-compute candidates for all unique left records
    candidates_cache = {}

    # Add progress bar for pre-computing candidates
    if verbose:
        from tqdm import tqdm
        left_id_iterator = tqdm(valid_left_ids, desc="Pre-computing candidates", unit="record")
    else:
        left_id_iterator = valid_left_ids

    for left_id in left_id_iterator:
        try:
            left_record = A_records[left_id]
            candidates = get_top_candidates_cached(left_record, candidate_cache, max_candidates, cfg, dataset, intelligent_boost=False, fast_analysis=True)
            candidate_ids = [c[0] for c in candidates]
            candidates_cache[left_id] = candidate_ids
        except Exception as e:
            if verbose:
                print(f"Warning: Error getting candidates for {left_id}: {e}")
            candidates_cache[left_id] = []

    if verbose:
        print(f"✅ Pre-computed candidates for all {len(valid_left_ids)} unique records")

    # Generate true match examples from sampled data
    processed_positive = 0
    errors_positive = 0

    for _, row in positive_sample.iterrows():
        left_id = row.ltable_id
        right_id = row.rtable_id

        if left_id not in A_records or right_id not in B_records:
            if verbose:
                print(f"⚠️ Skipping positive pair {left_id}-{right_id}: missing records")
            continue

        processed_positive += 1
        if verbose and processed_positive % 10 == 0:
            print(f"   Processing positive pair {processed_positive}/{len(positive_sample)}")

        try:
            left_record = clean_record_for_json(A_records[left_id])
            right_record = clean_record_for_json(B_records[right_id])

            # Get key field for comparison
            left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
            right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

            # Calculate all similarities
            syntactic_sim = syntactic_similarity(left_key, right_key)
            trigram_sim = trigram_similarity(left_key, right_key)
            semantic_sim = get_semantic_similarity_for_records(left_record, right_record, cfg)

            # Use pre-computed candidates
            found_in_candidates = False
            candidate_rank = None
            if left_id in candidates_cache:
                candidate_ids = candidates_cache[left_id]
                if right_id in candidate_ids:
                    found_in_candidates = True
                    candidate_rank = candidate_ids.index(right_id) + 1

            true_match_examples.append({
                "left_record": left_record,
                "right_record": right_record,
                "similarities": {
                    "syntactic": round(syntactic_sim, 3),
                    "trigram": round(trigram_sim, 3),
                    "semantic": round(semantic_sim, 3),
                },
                "candidate_generation": {
                    "found": found_in_candidates,
                    "rank": candidate_rank,
                    "max_candidates": max_candidates,
                },
            })
        except Exception as e:
            errors_positive += 1
            if verbose:
                print(f"Warning: Error processing positive pair {left_id}-{right_id}: {e}")

    if verbose:
        print(f"✅ Generated {len(true_match_examples)} true match examples ({errors_positive} errors)")

    # Generate false positive examples (confusing non-matches) from sampled data
    processed_negative = 0
    errors_negative = 0

    for _, row in negative_sample.iterrows():
        left_id = row.ltable_id
        right_id = row.rtable_id

        if left_id not in A_records or right_id not in B_records:
            continue

        processed_negative += 1
        if verbose and processed_negative % 20 == 0:
            print(f"   Processing negative pair {processed_negative}/{len(negative_sample)}")

        try:
            left_record = clean_record_for_json(A_records[left_id])
            right_record = clean_record_for_json(B_records[right_id])

            # Get key field for comparison
            left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
            right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

            # Calculate all similarities
            syntactic_sim = syntactic_similarity(left_key, right_key)
            trigram_sim = trigram_similarity(left_key, right_key)
            semantic_sim = get_semantic_similarity_for_records(left_record, right_record, cfg)

            # Use pre-computed candidates
            found_in_candidates = False
            candidate_rank = None
            if left_id in candidates_cache:
                candidate_ids = candidates_cache[left_id]
                if right_id in candidate_ids:
                    found_in_candidates = True
                    candidate_rank = candidate_ids.index(right_id) + 1

            false_positive_examples.append({
                "left_record": left_record,
                "right_record": right_record,
                "similarities": {
                    "syntactic": round(syntactic_sim, 3),
                    "trigram": round(trigram_sim, 3),
                    "semantic": round(semantic_sim, 3),
                },
                "candidate_generation": {
                    "found": found_in_candidates,
                    "rank": candidate_rank,
                    "max_candidates": max_candidates,
                },
            })
        except Exception as e:
            errors_negative += 1
            if verbose:
                print(f"Warning: Error processing negative pair {left_id}-{right_id}: {e}")

    # Restore original heuristics setting
    cfg.use_heuristics = original_use_heuristics

    if verbose:
        print(f"✅ Generated {len(false_positive_examples)} confusing non-match examples ({errors_negative} errors)")

    return {"true_matches": true_match_examples, "confusing_non_matches": false_positive_examples}


def analyze_candidate_recall(
    pairs: pd.DataFrame,
    A_records: Dict[int, dict],
    B_records: Dict[int, dict],
    candidate_cache: CandidateCache,
    cfg: Config,
    dataset: str,
    max_candidates: int = 100,
    max_analysis_pairs: int = None,
    test_higher_candidates: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Analyze recall at different candidate thresholds"""
    if verbose:
        print("🎯 Analyzing candidate recall at different thresholds...")

    # Disable heuristics during analysis for speed
    original_use_heuristics = cfg.use_heuristics
    cfg.use_heuristics = False

    # Define thresholds to test
    base_thresholds = [1, 5, 10, 25, 50, 100, 150, 200, 300, 500]

    if test_higher_candidates:
        # Test beyond current max_candidates to see potential improvement
        analysis_max_candidates = max(max_candidates, 500)  # Test up to 500 to see potential
        if verbose:
            print(f"   Testing up to {analysis_max_candidates} candidates (current setting: {max_candidates})")
    else:
        # Only test up to current max_candidates setting
        analysis_max_candidates = max_candidates
        if verbose:
            print(f"   Testing up to {analysis_max_candidates} candidates (current setting)")

    thresholds = [t for t in base_thresholds if t <= analysis_max_candidates]
    if analysis_max_candidates not in thresholds:
        thresholds.append(analysis_max_candidates)
    thresholds = sorted(thresholds)

    # Use the highest threshold for actual candidate generation
    max_threshold = max(thresholds)

    positive_pairs = pairs[pairs.label == 1]

    # Limit analysis pairs for speed if requested
    if max_analysis_pairs and len(positive_pairs) > max_analysis_pairs:
        positive_pairs = positive_pairs.head(max_analysis_pairs)
        if verbose:
            print(f"ℹ️ Limiting analysis to first {max_analysis_pairs} positive pairs for speed")

    total_matches = len(positive_pairs)

    if total_matches == 0:
        # Restore heuristics setting before returning
        cfg.use_heuristics = original_use_heuristics
        return {f"recall_at_{t}": 0.0 for t in thresholds}

    # Pre-compute candidates for unique left records to avoid redundant work
    unique_left_ids = set(positive_pairs['ltable_id'])
    candidates_cache = {}
    errors = 0

    # Add progress bar for candidate computation
    if verbose:
        from tqdm import tqdm
        unique_iterator = tqdm(unique_left_ids, desc="Computing candidates", unit="record")
    else:
        unique_iterator = unique_left_ids

    for left_id in unique_iterator:
        if left_id not in A_records:
            continue

        try:
            left_record = A_records[left_id]
            candidates = get_top_candidates_cached(left_record, candidate_cache, max_threshold, cfg, dataset, intelligent_boost=False, fast_analysis=True)
            candidate_ids = [c[0] for c in candidates]
            candidates_cache[left_id] = candidate_ids
        except Exception as e:
            errors += 1
            candidates_cache[left_id] = []
            if verbose:
                print(f"Warning: Error getting candidates for {left_id}: {e}")

    # Now process all pairs using pre-computed candidates
    candidate_ranks = {}  # left_id -> rank of right_id (or None if not found)

    for _, row in positive_pairs.iterrows():
        left_id = row.ltable_id
        right_id = row.rtable_id

        if left_id not in A_records or right_id not in B_records:
            continue

        candidate_ids = candidates_cache.get(left_id, [])
        if right_id in candidate_ids:
            candidate_ranks[left_id] = candidate_ids.index(right_id) + 1  # 1-indexed rank
        else:
            candidate_ranks[left_id] = None  # Not found

    # Restore original heuristics setting
    cfg.use_heuristics = original_use_heuristics

    # Calculate recall at each threshold by counting ranks <= threshold
    recall_results = {}
    for threshold in thresholds:
        found_matches = sum(
            1 for rank in candidate_ranks.values()
            if rank is not None and rank <= threshold
        )
        recall = found_matches / total_matches if total_matches > 0 else 0.0
        recall_results[f"recall_at_{threshold}"] = recall

        if verbose:
            print(f"   Recall@{threshold}: {recall:.3f} ({found_matches}/{total_matches})")

    if verbose and errors > 0:
        print(f"   Total errors: {errors}")

    return recall_results


def analyze_dataset_for_claude(
    dataset: str,
    max_pairs: int = 200,
    max_candidates: int = 200,
    max_analysis_pairs: int = 500,
    output_file: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis for Claude optimization.

    Args:
        dataset: Dataset name (e.g., 'itunes_amazon', 'beer')
        max_pairs: Maximum pairs to analyze
        max_candidates: Candidates threshold for analysis
        output_file: Optional output JSON file path
        verbose: Whether to print progress messages

    Returns:
        Dictionary containing analysis results
    """
    if verbose:
        print(f"🔍 ANALYZING DATASET: {dataset.upper()}")
        print("=" * 80)

    # Check if cached analysis exists
    if output_file and pathlib.Path(output_file).exists():
        if verbose:
            print(f"📁 Loading cached analysis from {output_file}")
        try:
            with open(output_file) as f:
                cached_result = json.load(f)
            # Verify cache is complete and has at least the requested pairs
            cached_pairs = cached_result.get("metadata", {}).get("total_pairs_analyzed", 0)
            if (cached_pairs >= max_pairs and
                cached_result.get("candidate_analysis", {}).get("recall_at_100") is not None):
                if verbose:
                    print("✅ Using cached analysis (complete)")
                return cached_result
            if verbose:
                print("⚠️ Cached analysis incomplete or parameters changed, regenerating...")
        except (json.JSONDecodeError, KeyError) as e:
            if verbose:
                print(f"⚠️ Cached analysis invalid: {e}, regenerating...")

    # Check if dataset exists
    data_root = pathlib.Path("data/raw") / dataset
    if not data_root.exists():
        raise ValueError(f"Dataset '{dataset}' not found in data/raw/")

    # Load dataset
    pairs, dataset_type = get_dev_dataset(dataset, max_pairs)
    A_df = pd.read_csv(data_root / "tableA.csv")
    B_df = pd.read_csv(data_root / "tableB.csv")

    if verbose:
        print(f"📊 Dataset: {len(A_df)} records in A, {len(B_df)} records in B")
        print(f"📊 Analysis pairs: {len(pairs)} pairs ({dataset_type})")
        print(f"📊 Positive pairs: {len(pairs[pairs.label == 1])} matches")

    # Convert to records
    A_records = {row.id: row.to_dict() for _, row in A_df.iterrows()}
    B_records = {row.id: row.to_dict() for _, row in B_df.iterrows()}

    # Build candidate cache ONCE for fast candidate generation
    if verbose:
        print("📦 Building candidate cache for fast processing...")
    B_records_list = list(B_records.values())
    candidate_cache = CandidateCache(B_records_list)
    if verbose:
        print("✅ Candidate cache ready")

    # Get field names
    field_names = [col for col in A_df.columns if col != "id"]

    # Initialize config
    cfg = Config()
    cfg.use_semantic = True  # Keep semantic similarity for analysis results
    cfg.use_heuristics = False  # Disable heuristics during analysis for speed

    # Try to load embeddings for semantic similarity
    semantic_available = False
    try:
        if verbose:
            print("🧮 Loading/computing embeddings for semantic similarity...")
        cfg.embeddings = compute_dataset_embeddings(dataset, cfg)
        if verbose:
            print("✅ Embeddings ready")
        semantic_available = True
    except Exception as e:
        if verbose:
            print(f"⚠️ Could not load embeddings: {e}")
        semantic_available = False

    # Separate positive and negative pairs
    positive_pairs = pairs[pairs.label == 1]
    negative_pairs = pairs[pairs.label == 0]

    if verbose:
        print(f"\n📈 Analyzing {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs...")

    # Analyze similarities
    true_match_similarities = {"syntactic": [], "trigram": [], "semantic": []}
    false_positive_similarities = {"syntactic": [], "trigram": [], "semantic": []}

    # Sample negative pairs for analysis (smaller sample to avoid hanging)
    negative_sample = negative_pairs.sample(min(len(negative_pairs), 30), random_state=42)

    # Calculate similarities for matches
    for _, row in positive_pairs.iterrows():
        left_id = row.ltable_id
        right_id = row.rtable_id

        if left_id not in A_records or right_id not in B_records:
            continue

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Get key field for comparison
        left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
        right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

        true_match_similarities["syntactic"].append(syntactic_similarity(left_key, right_key))
        true_match_similarities["trigram"].append(trigram_similarity(left_key, right_key))
        if semantic_available:
            true_match_similarities["semantic"].append(
                get_semantic_similarity_for_records(left_record, right_record, cfg)
            )

    # Calculate similarities for non-matches
    for _, row in negative_sample.iterrows():
        left_id = row.ltable_id
        right_id = row.rtable_id

        if left_id not in A_records or right_id not in B_records:
            continue

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Get key field for comparison
        left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
        right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

        false_positive_similarities["syntactic"].append(syntactic_similarity(left_key, right_key))
        false_positive_similarities["trigram"].append(trigram_similarity(left_key, right_key))
        if semantic_available:
            false_positive_similarities["semantic"].append(
                get_semantic_similarity_for_records(left_record, right_record, cfg)
            )

    # Analyze candidate recall
    candidate_analysis = analyze_candidate_recall(
        pairs, A_records, B_records, candidate_cache, cfg, dataset, max_candidates, max_analysis_pairs, True, verbose
    )

    # Generate concrete examples
    concrete_examples = generate_concrete_examples(
        positive_pairs, negative_sample, A_records, B_records, candidate_cache, cfg, dataset, max_candidates, 100, verbose
    )

    # Sample records for dataset characteristics
    sample_records = [clean_record_for_json(record) for record in list(A_records.values())[:3]]

    # Build comprehensive analysis result
    result = {
        "dataset": dataset,
        "analysis_type": dataset_type,
        "metadata": {
            "total_pairs_analyzed": len(pairs),
            "positive_pairs": len(positive_pairs),
            "negative_pairs_sampled": len(negative_sample),
            "semantic_available": semantic_available,
        },
        "similarity_analysis": {
            "true_matches": {
                "syntactic": calculate_similarity_stats(true_match_similarities["syntactic"]),
                "trigram": calculate_similarity_stats(true_match_similarities["trigram"]),
                "semantic": calculate_similarity_stats(true_match_similarities["semantic"])
                if semantic_available
                else None,
            },
            "false_positives": {
                "syntactic": calculate_similarity_stats(false_positive_similarities["syntactic"]),
                "trigram": calculate_similarity_stats(false_positive_similarities["trigram"]),
                "semantic": calculate_similarity_stats(false_positive_similarities["semantic"])
                if semantic_available
                else None,
            },
        },
        "candidate_analysis": candidate_analysis,
        "concrete_examples": concrete_examples,
        "dataset_characteristics": {
            "table_a_size": len(A_df),
            "table_b_size": len(B_df),
            "field_names": field_names,
            "sample_records": sample_records,
        },
    }

    # Save if output file specified
    if output_file:
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\n💾 Analysis saved to: {output_path}")

    if verbose:
        print("\n🎯 KEY INSIGHTS:")
        print(f"   True matches - Syntactic: {result['similarity_analysis']['true_matches']['syntactic']['mean']:.3f} ± {result['similarity_analysis']['true_matches']['syntactic']['std']:.3f}")
        print(f"   True matches - Trigram: {result['similarity_analysis']['true_matches']['trigram']['mean']:.3f} ± {result['similarity_analysis']['true_matches']['trigram']['std']:.3f}")
        if semantic_available:
            print(f"   True matches - Semantic: {result['similarity_analysis']['true_matches']['semantic']['mean']:.3f} ± {result['similarity_analysis']['true_matches']['semantic']['std']:.3f}")
        print(f"   False positives - Syntactic: {result['similarity_analysis']['false_positives']['syntactic']['mean']:.3f} ± {result['similarity_analysis']['false_positives']['syntactic']['std']:.3f}")
        print(f"   False positives - Trigram: {result['similarity_analysis']['false_positives']['trigram']['mean']:.3f} ± {result['similarity_analysis']['false_positives']['trigram']['std']:.3f}")
        if semantic_available:
            print(f"   False positives - Semantic: {result['similarity_analysis']['false_positives']['semantic']['mean']:.3f} ± {result['similarity_analysis']['false_positives']['semantic']['std']:.3f}")

        print("\n🏁 Ready for Claude optimization! Use this file with claude_config_generator.py")

    return result
