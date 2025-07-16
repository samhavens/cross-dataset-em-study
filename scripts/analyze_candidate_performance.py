#!/usr/bin/env python3
"""
Analyze the performance of the analyze_candidate_recall() function to understand slowdowns.
"""
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import pandas as pd

from src.entity_matching.analysis import analyze_candidate_recall
from src.entity_matching.hybrid_matcher import CandidateCache, Config


def analyze_performance():
    """Analyze the performance of candidate recall analysis"""
    dataset = "itunes_amazon"

    # Load dataset
    data_root = pathlib.Path("data/raw") / dataset
    pairs = pd.read_csv(data_root / "test.csv").head(50)  # Small sample for analysis
    A_df = pd.read_csv(data_root / "tableA.csv")
    B_df = pd.read_csv(data_root / "tableB.csv")

    # Convert to records
    A_records = {row.id: row.to_dict() for _, row in A_df.iterrows()}
    B_records = {row.id: row.to_dict() for _, row in B_df.iterrows()}

    print(f"Dataset: {dataset}")
    print(f"Pairs to analyze: {len(pairs)}")
    print(f"Table A size: {len(A_records)}")
    print(f"Table B size: {len(B_records)}")

    # Time candidate cache creation
    print("\n=== CANDIDATE CACHE CREATION ===")
    start_time = time.time()
    B_records_list = list(B_records.values())
    candidate_cache = CandidateCache(B_records_list)
    cache_time = time.time() - start_time
    print(f"Cache creation time: {cache_time:.2f} seconds")

    # Initialize config
    cfg = Config()
    cfg.use_semantic = True
    cfg.use_heuristics = False

    # Time candidate recall analysis
    print("\n=== CANDIDATE RECALL ANALYSIS ===")
    start_time = time.time()

    # Profile individual operations
    positive_pairs = pairs[pairs.label == 1]
    print(f"Positive pairs: {len(positive_pairs)}")

    # Test single candidate generation call
    if len(positive_pairs) > 0:
        first_pair = positive_pairs.iloc[0]
        left_id = first_pair.ltable_id
        left_record = A_records[left_id]

        print("\n=== SINGLE CANDIDATE GENERATION TEST ===")
        print(f"Left record ID: {left_id}")

        # Time single call
        from src.entity_matching.hybrid_matcher import get_top_candidates_cached

        single_start = time.time()
        candidates = get_top_candidates_cached(left_record, candidate_cache, 100, cfg, dataset, intelligent_boost=True)
        single_time = time.time() - single_start

        print(f"Single candidate generation time: {single_time:.4f} seconds")
        print(f"Candidates found: {len(candidates)}")

        # Test multiple calls
        print("\n=== MULTIPLE CANDIDATE GENERATION TEST ===")
        multi_start = time.time()
        for i, (_, row) in enumerate(positive_pairs.head(10).iterrows()):
            left_id = row.ltable_id
            if left_id in A_records:
                left_record = A_records[left_id]
                candidates = get_top_candidates_cached(left_record, candidate_cache, 100, cfg, dataset, intelligent_boost=True)
        multi_time = time.time() - multi_start

        print(f"10 candidate generations time: {multi_time:.4f} seconds")
        print(f"Average per generation: {multi_time / 10:.4f} seconds")

    # Time full analysis
    print("\n=== FULL ANALYSIS TEST ===")
    start_time = time.time()

    results = analyze_candidate_recall(
        pairs, A_records, B_records, candidate_cache, cfg, dataset,
        max_candidates=100, verbose=True
    )

    analysis_time = time.time() - start_time

    print(f"\nFull analysis time: {analysis_time:.2f} seconds")
    print(f"Time per pair: {analysis_time / len(pairs):.4f} seconds")
    print(f"Results: {results}")

    # Profile semantic similarity if enabled
    if cfg.use_semantic:
        print("\n=== SEMANTIC SIMILARITY TEST ===")
        try:
            from src.entity_matching.hybrid_matcher import compute_dataset_embeddings
            embeddings_start = time.time()
            cfg.embeddings = compute_dataset_embeddings(dataset, cfg)
            embeddings_time = time.time() - embeddings_start
            print(f"Embeddings computation time: {embeddings_time:.2f} seconds")

            # Test with embeddings
            print("\n=== ANALYSIS WITH EMBEDDINGS ===")
            start_time = time.time()
            analyze_candidate_recall(
                pairs, A_records, B_records, candidate_cache, cfg, dataset,
                max_candidates=100, verbose=True
            )
            analysis_with_embeddings_time = time.time() - start_time

            print(f"Analysis with embeddings time: {analysis_with_embeddings_time:.2f} seconds")
            print(f"Time per pair with embeddings: {analysis_with_embeddings_time / len(pairs):.4f} seconds")

        except Exception as e:
            print(f"Embeddings test failed: {e}")

    # Check cache effectiveness
    print("\n=== CACHE EFFECTIVENESS TEST ===")
    # Test cache hits/misses by running the same query multiple times
    if len(positive_pairs) > 0:
        first_pair = positive_pairs.iloc[0]
        left_id = first_pair.ltable_id
        left_record = A_records[left_id]

        # First call (cache miss)
        start_time = time.time()
        candidates1 = get_top_candidates_cached(left_record, candidate_cache, 100, cfg, dataset, intelligent_boost=True)
        first_call_time = time.time() - start_time

        # Second call (should be faster if caching works)
        start_time = time.time()
        candidates2 = get_top_candidates_cached(left_record, candidate_cache, 100, cfg, dataset, intelligent_boost=True)
        second_call_time = time.time() - start_time

        print(f"First call time: {first_call_time:.4f} seconds")
        print(f"Second call time: {second_call_time:.4f} seconds")
        print(f"Cache speedup: {first_call_time / second_call_time:.2f}x")
        print(f"Results identical: {candidates1 == candidates2}")

if __name__ == "__main__":
    analyze_performance()
