#!/usr/bin/env python
"""
Duplicate-aware evaluation for entity matching that accepts any duplicate from the correct group.

This addresses the dataset quality issue where identical records have different IDs,
causing algorithms to be penalized for choosing the "wrong" copy of a duplicate.
"""

import json
import pathlib

from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd


def normalize_record(record: dict) -> tuple:
    """Convert record to normalized tuple for comparison (excluding ID)"""
    # Get all fields except 'id'
    fields = [(k, v) for k, v in record.items() if k != 'id']
    # Sort by key for consistent ordering
    fields.sort(key=lambda x: x[0])
    # Convert to tuple, handling NaN values
    normalized = []
    for k, v in fields:
        if pd.isna(v):
            normalized.append((k, None))
        else:
            normalized.append((k, str(v).strip()))
    return tuple(normalized)


def find_duplicate_groups(df: pd.DataFrame, table_name: str) -> Dict[str, List[int]]:
    """Find groups of records that are identical in all non-ID fields"""
    print(f"🔍 Finding duplicate groups in {table_name}...")

    # Group records by their normalized content
    content_groups = defaultdict(list)

    for _, row in df.iterrows():
        record = row.to_dict()
        normalized = normalize_record(record)
        content_groups[normalized].append(row.id)

    # Keep only groups with multiple records (duplicates)
    duplicate_groups = {}
    total_duplicates = 0

    for content_hash, ids in content_groups.items():
        if len(ids) > 1:
            # Use the first field's value as a readable key
            first_field = next(iter(dict(content_hash).values()), "unknown")
            key = f"{table_name}_{first_field}_{len(ids)}records"
            duplicate_groups[key] = sorted(ids)
            total_duplicates += len(ids)

    print(f"✅ Found {len(duplicate_groups)} duplicate groups with {total_duplicates} total records")
    return duplicate_groups


def generate_evaluation_mapping(
    tableA_groups: Dict[str, List[int]],
    tableB_groups: Dict[str, List[int]]
) -> Dict[str, Dict[str, List[int]]]:
    """Generate mapping for modified evaluation: any ID in group -> all valid IDs in group"""
    print("🔧 Generating evaluation mapping for duplicate-aware evaluation...")

    # For tableA: map each ID to all IDs in its group
    id_to_valid_ids_A = {}
    for group_key, ids in tableA_groups.items():
        for id_val in ids:
            id_to_valid_ids_A[str(id_val)] = ids

    # For tableB: map each ID to all IDs in its group
    id_to_valid_ids_B = {}
    for group_key, ids in tableB_groups.items():
        for id_val in ids:
            id_to_valid_ids_B[str(id_val)] = ids

    return {
        "tableA_mapping": id_to_valid_ids_A,
        "tableB_mapping": id_to_valid_ids_B
    }


def generate_duplicate_analysis(dataset: str) -> Dict:
    """Generate duplicate analysis for a dataset"""
    data_root = pathlib.Path("data/raw") / dataset
    if not data_root.exists():
        print(f"❌ Dataset '{dataset}' not found in data/raw/")
        return None

    # Load tables
    tableA = pd.read_csv(data_root / "tableA.csv")
    tableB = pd.read_csv(data_root / "tableB.csv")

    # Find duplicate groups
    tableA_groups = find_duplicate_groups(tableA, "tableA")
    tableB_groups = find_duplicate_groups(tableB, "tableB")

    # Generate evaluation mapping
    evaluation_mapping = generate_evaluation_mapping(tableA_groups, tableB_groups)

    return {
        "dataset": dataset,
        "evaluation_mapping": evaluation_mapping,
        "duplicate_groups": {
            "tableA": tableA_groups,
            "tableB": tableB_groups
        }
    }


def ensure_duplicate_mapping(dataset: str) -> bool:
    """Check if duplicate mapping is available for the dataset"""
    mapping_file = pathlib.Path(f"results/{dataset}_duplicate_analysis.json")
    return mapping_file.exists()


def load_duplicate_mapping(dataset: str) -> Dict[str, Dict[str, List[int]]]:
    """Load or generate duplicate group mapping"""
    # Try to load from cache first
    mapping_file = pathlib.Path(f"results/{dataset}_duplicate_analysis.json")
    if mapping_file.exists():
        with open(mapping_file) as f:
            analysis = json.load(f)
        return analysis["evaluation_mapping"]
    
    # Generate on-demand if not cached
    print(f"🔄 Generating duplicate analysis for {dataset} on-demand...")
    analysis = generate_duplicate_analysis(dataset)
    if analysis:
        return analysis["evaluation_mapping"]
    else:
        return {"tableA_mapping": {}, "tableB_mapping": {}}


def evaluate_with_duplicates(
    predictions: List[Tuple[int, int]],
    ground_truth: List[Tuple[int, int]],
    duplicate_mapping: Dict[str, Dict[str, List[int]]]
) -> Dict[str, float]:
    """
    FIXED: Evaluate predictions allowing any duplicate from the correct group.

    Key fix: Don't expand ground truth size - just check if predictions match duplicate groups.
    This prevents the Cartesian product explosion that caused negative TN values.

    Args:
        predictions: List of (left_id, right_id) predicted matches
        ground_truth: List of (left_id, right_id) ground truth matches
        duplicate_mapping: Mapping from duplicate analysis

    Returns:
        Dictionary with precision, recall, f1 scores
    """
    tableA_mapping = {int(k): set(v) for k, v in duplicate_mapping["tableA_mapping"].items()}
    tableB_mapping = {int(k): set(v) for k, v in duplicate_mapping["tableB_mapping"].items()}

    # Helper function to check if two IDs are in the same duplicate group
    def are_duplicates(id1: int, id2: int, mapping: Dict[int, set]) -> bool:
        """Check if two IDs are in the same duplicate group"""
        group1 = mapping.get(id1, {id1})
        group2 = mapping.get(id2, {id2})
        return bool(group1 & group2)  # True if groups overlap

    # Track which predictions are correct and which ground truth are covered
    correct_predictions = set()
    covered_ground_truth = set()

    # Check each prediction against ground truth
    for i, (pred_left, pred_right) in enumerate(predictions):
        # Check if this prediction matches any ground truth (considering duplicates)
        for j, (gt_left, gt_right) in enumerate(ground_truth):
            left_match = are_duplicates(pred_left, gt_left, tableA_mapping)
            right_match = are_duplicates(pred_right, gt_right, tableB_mapping)

            if left_match and right_match:
                correct_predictions.add(i)
                covered_ground_truth.add(j)
                break  # One match per prediction is enough

    # Calculate metrics
    true_positives = len(correct_predictions)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - len(covered_ground_truth)

    precision = true_positives / len(predictions) if len(predictions) > 0 else 0.0
    recall = len(covered_ground_truth) / len(ground_truth) if len(ground_truth) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predictions": len(predictions),
        "total_ground_truth": len(ground_truth),
        "fixed_version": True  # Flag to indicate this is the fixed version
    }


def compare_evaluations(
    predictions: List[Tuple[int, int]],
    ground_truth: List[Tuple[int, int]],
    duplicate_mapping: Dict[str, Dict[str, List[int]]]
) -> Dict[str, any]:
    """Compare standard vs duplicate-aware evaluation"""

    # Standard evaluation
    pred_set = set(predictions)
    gt_set = set(ground_truth)

    std_tp = len(pred_set & gt_set)
    std_fp = len(pred_set - gt_set)
    std_fn = len(gt_set - pred_set)

    std_precision = std_tp / (std_tp + std_fp) if (std_tp + std_fp) > 0 else 0.0
    std_recall = std_tp / (std_tp + std_fn) if (std_tp + std_fn) > 0 else 0.0
    std_f1 = 2 * std_precision * std_recall / (std_precision + std_recall) if (std_precision + std_recall) > 0 else 0.0

    # Duplicate-aware evaluation
    dup_metrics = evaluate_with_duplicates(predictions, ground_truth, duplicate_mapping)

    return {
        "standard_evaluation": {
            "precision": std_precision,
            "recall": std_recall,
            "f1": std_f1,
            "true_positives": std_tp,
            "false_positives": std_fp,
            "false_negatives": std_fn
        },
        "duplicate_aware_evaluation": dup_metrics,
        "improvement": {
            "precision_gain": dup_metrics["precision"] - std_precision,
            "recall_gain": dup_metrics["recall"] - std_recall,
            "f1_gain": dup_metrics["f1"] - std_f1
        }
    }


def load_pipeline_results(results_file: str, dataset: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Load predictions and ground truth from pipeline results file"""
    with open(results_file) as f:
        results = json.load(f)

    # Use the better result (baseline vs enhanced) - both should now have predictions
    baseline_f1 = results.get("baseline_results", {}).get("f1", 0)
    enhanced_f1 = results.get("enhanced_results", {}).get("f1", 0)
    
    if baseline_f1 > enhanced_f1:
        print(f"📊 Using baseline results (F1: {baseline_f1:.4f} > {enhanced_f1:.4f})")
        predictions_source = "baseline_results"
    else:
        print(f"📊 Using enhanced results (F1: {enhanced_f1:.4f})")
        predictions_source = "enhanced_results"

    # Extract predictions
    predictions = []
    if predictions_source in results and "predictions" in results[predictions_source]:
        for left_id_str, right_id in results[predictions_source]["predictions"].items():
            predictions.append((int(left_id_str), int(right_id)))
    else:
        print(f"⚠️ No predictions found in {predictions_source} - cannot do duplicate-aware evaluation")
        return [], []

    # Load ground truth from test.csv
    data_root = pathlib.Path("data/raw") / dataset
    test_pairs = pd.read_csv(data_root / "test.csv")
    ground_truth = []

    for _, row in test_pairs.iterrows():
        if row.label == 1:
            ground_truth.append((int(row.ltable_id), int(row.rtable_id)))

    return predictions, ground_truth


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Duplicate-aware evaluation for entity matching")
    parser.add_argument("dataset", help="Dataset name (e.g., 'itunes_amazon')")
    parser.add_argument("--results-file", help="Pipeline results JSON file")
    parser.add_argument("--output", help="Output file for comparison results")

    args = parser.parse_args()

    # Load duplicate mapping
    duplicate_mapping = load_duplicate_mapping(args.dataset)

    if args.results_file:
        # Load and re-evaluate specific results file
        predictions, ground_truth = load_pipeline_results(args.results_file, args.dataset)

        print(f"📊 Re-evaluating {args.results_file}")
        print(f"   Predictions: {len(predictions)} matches")
        print(f"   Ground truth: {len(ground_truth)} matches")

        comparison = compare_evaluations(predictions, ground_truth, duplicate_mapping)

        print("\n📈 EVALUATION COMPARISON:")
        print("=" * 50)
        print("Standard Evaluation:")
        print(f"   Precision: {comparison['standard_evaluation']['precision']:.4f}")
        print(f"   Recall: {comparison['standard_evaluation']['recall']:.4f}")
        print(f"   F1: {comparison['standard_evaluation']['f1']:.4f}")

        print("\nDuplicate-Aware Evaluation:")
        print(f"   Precision: {comparison['duplicate_aware_evaluation']['precision']:.4f}")
        print(f"   Recall: {comparison['duplicate_aware_evaluation']['recall']:.4f}")
        print(f"   F1: {comparison['duplicate_aware_evaluation']['f1']:.4f}")

        print("\nImprovement:")
        print(f"   Precision gain: {comparison['improvement']['precision_gain']:+.4f}")
        print(f"   Recall gain: {comparison['improvement']['recall_gain']:+.4f}")
        print(f"   F1 gain: {comparison['improvement']['f1_gain']:+.4f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")

    else:
        # Just load and display duplicate mapping info
        print(f"📊 Duplicate mapping for {args.dataset}:")
        print(f"   Table A: {len(duplicate_mapping['tableA_mapping'])} records with duplicates")
        print(f"   Table B: {len(duplicate_mapping['tableB_mapping'])} records with duplicates")


if __name__ == "__main__":
    main()
