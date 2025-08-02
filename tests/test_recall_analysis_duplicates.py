#!/usr/bin/env python3
"""
Test to identify and validate the duplicate-awareness issue in recall@N analysis.

The recall@N analysis currently does NOT use duplicate-aware evaluation, which means:
- If ground truth says A matches B_1
- But candidates contain B_2 (duplicate of B_1) instead of B_1
- It counts as a false miss, even though B_2 is effectively correct

This underestimates recall and leads to suboptimal candidate thresholds.
"""

import json
import pathlib


def simulate_recall_analysis_issue():
    """Simulate the duplicate-awareness issue in recall analysis"""

    print("🧪 Testing recall@N duplicate-awareness issue")

    # Simulate a scenario
    ground_truth_pairs = [
        (1, 101),  # Record A.1 should match B.101
        (2, 102),  # Record A.2 should match B.102
        (3, 103),  # Record A.3 should match B.103
    ]

    # Simulate candidate generation that finds duplicates instead of exact IDs
    candidate_results = {
        1: [101, 999, 888],  # Found exact match B.101 at rank 1
        2: [201, 102, 777],  # Found exact match B.102 at rank 2
        3: [301, 302, 303],  # Didn't find B.103, but B.301 might be duplicate of B.103
    }

    # Simulate duplicate mapping (B.301 is duplicate of B.103)
    duplicate_mapping = {
        "tableB_mapping": {
            "103": [103, 301],  # B.103 and B.301 are duplicates
            "301": [103, 301],
        }
    }

    print("\n📊 Simulation Setup:")
    print(f"   Ground truth: {ground_truth_pairs}")
    print(f"   Candidates: {candidate_results}")
    print("   Duplicates: B.301 is duplicate of B.103")

    # Current recall analysis (without duplicate awareness)
    def current_recall_analysis(pairs, candidates, threshold=3):
        found = 0
        total = len(pairs)

        for left_id, right_id in pairs:
            candidate_list = candidates.get(left_id, [])
            if right_id in candidate_list:
                rank = candidate_list.index(right_id) + 1
                if rank <= threshold:
                    found += 1

        return found / total if total > 0 else 0.0

    # Proposed duplicate-aware recall analysis
    def duplicate_aware_recall_analysis(pairs, candidates, duplicate_mapping, threshold=3):
        found = 0
        total = len(pairs)
        tableB_mapping = {int(k): set(v) for k, v in duplicate_mapping["tableB_mapping"].items()}

        for left_id, right_id in pairs:
            candidate_list = candidates.get(left_id, [])

            # Check if any candidate is a duplicate of the ground truth
            right_group = tableB_mapping.get(right_id, {right_id})

            for rank, candidate_id in enumerate(candidate_list, 1):
                if candidate_id in right_group and rank <= threshold:
                    found += 1
                    break

        return found / total if total > 0 else 0.0

    # Test both approaches
    current_recall = current_recall_analysis(ground_truth_pairs, candidate_results, threshold=3)
    duplicate_aware_recall = duplicate_aware_recall_analysis(
        ground_truth_pairs, candidate_results, duplicate_mapping, threshold=3
    )

    print("\n📈 Results at Recall@3:")
    print(f"   Current analysis: {current_recall:.3f} (2/3 = 66.7%)")
    print(f"   Duplicate-aware:  {duplicate_aware_recall:.3f} (3/3 = 100%)")
    print(f"   Improvement:      {duplicate_aware_recall - current_recall:+.3f}")

    if duplicate_aware_recall > current_recall:
        print("\n✅ Duplicate-aware analysis finds more matches!")
        print("   The current recall analysis underestimates performance.")
        return True
    print("\n❌ No improvement found")
    return False

def check_dataset_for_duplicates(dataset: str) -> bool:
    """Check if a dataset has duplicate analysis available"""
    mapping_file = pathlib.Path(f"results/{dataset}_duplicate_analysis.json")

    if mapping_file.exists():
        with open(mapping_file) as f:
            analysis = json.load(f)

        tableB_count = len(analysis["evaluation_mapping"]["tableB_mapping"])
        print(f"✅ {dataset}: {tableB_count} records with duplicates")
        return True
    print(f"⚠️ {dataset}: No duplicate analysis found")
    return False

def test_real_datasets():
    """Test which datasets have duplicate mappings available"""
    datasets = [
        "beer", "itunes_amazon", "amazon_google", "abt_buy",
        "fodors_zagat", "walmart_amazon", "zomato_yelp"
    ]

    print("\n🗂️ Checking real datasets for duplicate analysis:")
    available_datasets = []

    for dataset in datasets:
        if check_dataset_for_duplicates(dataset):
            available_datasets.append(dataset)

    print(f"\n📊 Summary: {len(available_datasets)}/{len(datasets)} datasets have duplicate analysis")

    if available_datasets:
        print(f"   Available: {', '.join(available_datasets)}")
        print("\n💡 Recommendation: Update recall analysis to use duplicate-aware evaluation")
        print("   for these datasets to get accurate candidate thresholds.")

    return available_datasets

if __name__ == "__main__":
    print("🔍 Testing Recall@N Duplicate-Awareness Issue")
    print("=" * 60)

    # Test simulation
    simulation_ok = simulate_recall_analysis_issue()

    # Test real datasets
    available_datasets = test_real_datasets()

    if simulation_ok and available_datasets:
        print("\n🎯 CONCLUSION:")
        print("   The recall@N analysis should be updated to use duplicate-aware evaluation")
        print("   for more accurate candidate threshold selection.")
        print(f"   This affects {len(available_datasets)} datasets with known duplicates.")
    else:
        print("\n✅ No issues found or no datasets with duplicates available")
