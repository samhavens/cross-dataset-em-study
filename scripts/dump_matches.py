#!/usr/bin/env python3
"""
Dump semantic and syntactic matches for a dataset to get a vibe for the data.

Usage:
    python scripts/dump_matches.py --dataset itunes_amazon --limit 20
    python scripts/dump_matches.py --dataset beer --limit 10

This script helps you understand:
- How similar actual matches are
- What confusing non-matches look like
- Whether candidate generation is working
- What similarity thresholds make sense
"""

import argparse
import pathlib
import sys

from difflib import SequenceMatcher
from typing import Dict

import pandas as pd

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from entity_matching.hybrid_matcher import Config, get_top_candidates, semantic_similarity, compute_dataset_embeddings


def get_syntactic_similarity(text1: str, text2: str) -> float:
    """Get syntactic similarity using sequence matcher"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def get_trigram_similarity(text1: str, text2: str) -> float:
    """Get trigram similarity"""
    if not text1 or not text2:
        return 0.0

    def get_trigrams(text):
        text = text.lower()
        trigrams = set()
        for i in range(len(text) - 2):
            trigrams.add(text[i : i + 3])
        return trigrams

    tri1 = get_trigrams(text1)
    tri2 = get_trigrams(text2)

    if not tri1 or not tri2:
        return 0.0

    intersection = len(tri1 & tri2)
    union = len(tri1 | tri2)
    return intersection / union if union > 0 else 0.0


def format_record(record: Dict, max_len: int = 80) -> str:
    """Format a record for display"""
    formatted = []
    for key, value in record.items():
        if key == "id":
            continue
        value_str = str(value)
        if len(value_str) > max_len:
            value_str = value_str[: max_len - 3] + "..."
        formatted.append(f"{key}: {value_str}")
    return " | ".join(formatted)


def get_semantic_similarity(left_record: dict, right_record: dict, cfg: Config) -> float:
    """Get semantic similarity between two records using the embedding model"""
    try:
        import json
        left_str = json.dumps(left_record, ensure_ascii=False).lower()
        right_str = json.dumps(right_record, ensure_ascii=False).lower()
        return semantic_similarity(left_str, right_str, cfg)
    except Exception as e:
        print(f"Warning: Semantic similarity calculation failed: {e}")
        return 0.0


def dump_matches(dataset: str, limit: int = 20, max_candidates: int = 100, semantic_weight: float = 0.5):
    """Dump semantic and syntactic matches for analysis"""
    print(f"🔍 ANALYZING MATCHES FOR {dataset.upper()}")
    print("=" * 80)

    # Load data
    data_root = pathlib.Path("data/raw") / dataset
    A_df = pd.read_csv(data_root / "tableA.csv")
    B_df = pd.read_csv(data_root / "tableB.csv")

    # Load test pairs to get ground truth
    test_pairs = pd.read_csv(data_root / "test.csv")

    print(f"📊 Dataset: {len(A_df)} records in A, {len(B_df)} records in B")
    print(f"📊 Test pairs: {len(test_pairs)} pairs")
    print(f"📊 Positive pairs: {len(test_pairs[test_pairs.label == 1])} matches")
    print()

    # Convert to records dict
    A_records = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
    B_records = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}

    # Get some positive and negative examples
    positive_pairs = test_pairs[test_pairs.label == 1].head(limit // 2)
    negative_pairs = test_pairs[test_pairs.label == 0].head(limit // 2)

    # Initialize config for candidate generation
    cfg = Config()
    cfg.use_semantic = True
    cfg.semantic_weight = semantic_weight
    
    # Try to compute embeddings for semantic similarity (may take time on first run)
    try:
        print("🧮 Loading/computing embeddings for semantic similarity...")
        cfg.embeddings = compute_dataset_embeddings(dataset, cfg)
        print("✅ Embeddings ready")
    except Exception as e:
        print(f"⚠️ Could not load embeddings: {e}")
        print("   Semantic similarity will use real-time computation (slower)")
        cfg.embeddings = None

    print("🎯 POSITIVE MATCHES (Ground Truth)")
    print("-" * 80)

    for i, (_, row) in enumerate(positive_pairs.iterrows(), 1):
        left_id = row.ltable_id
        right_id = row.rtable_id

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Get key fields for comparison
        left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
        right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

        syntactic_sim = get_syntactic_similarity(left_key, right_key)
        trigram_sim = get_trigram_similarity(left_key, right_key)
        semantic_sim = get_semantic_similarity(left_record, right_record, cfg)

        print(f"\n{i}. MATCH (IDs: {left_id} → {right_id})")
        print(f"   LEFT:  {format_record(left_record)}")
        print(f"   RIGHT: {format_record(right_record)}")
        print(f"   SIMILARITY: Syntactic={syntactic_sim:.3f}, Trigram={trigram_sim:.3f}, Semantic={semantic_sim:.3f}")

        # Check if this would be found by candidate generation
        try:
            candidates = get_top_candidates(left_record, B_records, max_candidates, cfg, dataset)
            candidate_ids = [c[0] for c in candidates]  # c[0] is the id, c[1] is the record
            if right_id in candidate_ids:
                rank = candidate_ids.index(right_id) + 1
                print(f"   CANDIDATES: ✅ Found at rank {rank}/{max_candidates}")
            else:
                print(f"   CANDIDATES: ❌ Not in top {max_candidates}")
        except Exception as e:
            print(f"   CANDIDATES: ⚠️ Error: {e}")

    print("\n\n🚫 NEGATIVE PAIRS (Non-matches)")
    print("-" * 80)

    for i, (_, row) in enumerate(negative_pairs.iterrows(), 1):
        left_id = row.ltable_id
        right_id = row.rtable_id

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Get key fields for comparison
        left_key = next((str(v) for k, v in left_record.items() if k != "id" and v), "")
        right_key = next((str(v) for k, v in right_record.items() if k != "id" and v), "")

        syntactic_sim = get_syntactic_similarity(left_key, right_key)
        trigram_sim = get_trigram_similarity(left_key, right_key)
        semantic_sim = get_semantic_similarity(left_record, right_record, cfg)

        print(f"\n{i}. NON-MATCH (IDs: {left_id} → {right_id})")
        print(f"   LEFT:  {format_record(left_record)}")
        print(f"   RIGHT: {format_record(right_record)}")
        print(f"   SIMILARITY: Syntactic={syntactic_sim:.3f}, Trigram={trigram_sim:.3f}, Semantic={semantic_sim:.3f}")

        # Check if this would be found by candidate generation
        try:
            candidates = get_top_candidates(left_record, B_records, max_candidates, cfg, dataset)
            candidate_ids = [c[0] for c in candidates]  # c[0] is the id, c[1] is the record
            if right_id in candidate_ids:
                rank = candidate_ids.index(right_id) + 1
                print(f"   CANDIDATES: ⚠️ Found at rank {rank}/{max_candidates} (potential false positive)")
            else:
                print(f"   CANDIDATES: ✅ Not in top {max_candidates} (correctly filtered)")
        except Exception as e:
            print(f"   CANDIDATES: ⚠️ Error: {e}")

    # Sample some high-similarity candidates that might be confusing
    print("\n\n🎲 SAMPLE HIGH-SIMILARITY CANDIDATES")
    print("-" * 80)

    # Take first record from A and show its top candidates
    sample_left = A_records[next(iter(A_records.keys()))]
    print(f"\nSAMPLE LEFT RECORD: {format_record(sample_left)}")
    print("TOP CANDIDATES:")

    try:
        candidates = get_top_candidates(sample_left, B_records, 10, cfg, dataset)
        for i, (candidate_id, candidate_record) in enumerate(candidates[:5], 1):
            left_key = next((str(v) for k, v in sample_left.items() if k != "id" and v), "")
            right_key = next((str(v) for k, v in candidate_record.items() if k != "id" and v), "")
            syntactic_sim = get_syntactic_similarity(left_key, right_key)
            trigram_sim = get_trigram_similarity(left_key, right_key)
            semantic_sim = get_semantic_similarity(sample_left, candidate_record, cfg)

            print(f"   {i}. {format_record(candidate_record)}")
            print(f"      SIMILARITY: Syntactic={syntactic_sim:.3f}, Trigram={trigram_sim:.3f}, Semantic={semantic_sim:.3f}")
    except Exception as e:
        print(f"   Error getting candidates: {e}")

    print("\n🏁 ANALYSIS COMPLETE")
    print("Use this data to understand:")
    print("- How similar actual matches are across all three similarity measures")
    print("- What confusing non-matches look like")
    print("- Whether candidate generation is working")
    print("- What similarity thresholds make sense for each measure")
    print("- How syntactic vs trigram vs semantic similarities compare")


def main():
    parser = argparse.ArgumentParser(description="Dump semantic and syntactic matches for analysis")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. itunes_amazon, beer)")
    parser.add_argument("--limit", type=int, default=20, help="Number of examples to show")
    parser.add_argument("--max-candidates", type=int, default=100, help="Number of candidates to generate")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Weight for semantic similarity (0.0-1.0)")

    args = parser.parse_args()

    # Check if dataset exists
    data_root = pathlib.Path("data/raw") / args.dataset
    if not data_root.exists():
        print(f"❌ Dataset '{args.dataset}' not found in data/raw/")
        return

    if not (data_root / "tableA.csv").exists():
        print(f"❌ tableA.csv not found in {data_root}")
        return

    if not (data_root / "tableB.csv").exists():
        print(f"❌ tableB.csv not found in {data_root}")
        return

    if not (data_root / "test.csv").exists():
        print(f"❌ test.csv not found in {data_root}")
        return

    dump_matches(args.dataset, args.limit, args.max_candidates, args.semantic_weight)


if __name__ == "__main__":
    main()
