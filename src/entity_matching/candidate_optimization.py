#!/usr/bin/env python
"""
Candidate count optimization based on recall plateau analysis.

Finds the optimal number of candidates where recall plateaus to avoid
diminishing returns from using too many candidates.
"""

import json
import pathlib

from typing import Dict, List, Optional, Tuple


def _optimize_candidates_backwards(candidates_recalls: List[Tuple[int, float]],
                                 start_candidates: int, start_recall: float) -> Tuple[int, float]:
    """
    Walk backwards from plateau point to find smallest N with same recall.

    Args:
        candidates_recalls: List of (candidate_count, recall) tuples, sorted by candidate_count
        start_candidates: Starting candidate count from plateau analysis
        start_recall: Starting recall value

    Returns:
        Tuple of (optimal_candidates, recall) - smallest N with same recall
    """
    # Create lookup dict for easy access
    dict(candidates_recalls)

    # Start from the plateau point and walk backwards
    best_candidates = start_candidates
    target_recall = start_recall

    # Walk backwards through all tested candidate counts smaller than start_candidates
    for candidates, recall in reversed(candidates_recalls):
        if candidates >= start_candidates:
            continue  # Skip candidates >= start point

        # If this smaller N achieves the same recall, use it
        if recall >= target_recall:
            best_candidates = candidates
            print(f"   📉 Can achieve same recall ({recall:.3f}) with {candidates} candidates (vs {start_candidates})")
        else:
            # Recall dropped - stop here
            break

    return best_candidates, target_recall


def find_recall_plateau(recall_results: Dict[str, float], plateau_threshold: float = 0.05) -> Tuple[int, float]:
    """
    Find the candidate count where recall plateaus (best balance of recall vs efficiency).

    Args:
        recall_results: Dict with keys like "recall_at_25" and values as recall scores
        plateau_threshold: Maximum improvement rate to consider a plateau (default 5%)

    Returns:
        Tuple of (optimal_candidates, recall_at_optimal)
    """
    # Extract candidate counts and recalls, sorted by candidate count
    candidates_recalls = []
    for key, recall in recall_results.items():
        if key.startswith("recall_at_"):
            candidate_count = int(key.split("_")[-1])
            candidates_recalls.append((candidate_count, recall))

    candidates_recalls.sort()  # Sort by candidate count

    if not candidates_recalls:
        return 100, 0.0  # Default fallback

    if len(candidates_recalls) == 1:
        return candidates_recalls[0]

    # Find the point with best efficiency (recall gain per candidate added)
    best_candidates = candidates_recalls[0][0]
    best_recall = candidates_recalls[0][1]

    # Look for the sweet spot: good recall with diminishing returns beyond this point
    for i in range(1, len(candidates_recalls)):
        current_candidates, current_recall = candidates_recalls[i]

        # Calculate remaining improvement potential
        max_recall = max(r for _, r in candidates_recalls)
        remaining_improvement = max_recall - current_recall

        # MINIMUM THRESHOLD: Never go below 50 candidates to avoid LLM performance issues
        # If we have good recall (>85%) and remaining improvement is small (<5%), this is optimal
        if (current_candidates >= 1 and
            current_recall >= 0.87 and
            remaining_improvement <= plateau_threshold):
            # Found plateau - now optimize backwards to find smallest N with same recall
            optimal_candidates, optimal_recall = _optimize_candidates_backwards(
                candidates_recalls, current_candidates, current_recall
            )
            return optimal_candidates, optimal_recall

        # Otherwise, keep the best point so far
        if current_recall > best_recall:
            best_candidates = current_candidates
            best_recall = current_recall

    # If no clear plateau found, return the point with best recall, but minimum 10 candidates
    fallback_candidates = max(best_candidates, 10)
    # Still optimize backwards from fallback point
    optimal_candidates, optimal_recall = _optimize_candidates_backwards(
        candidates_recalls, fallback_candidates, best_recall
    )
    return optimal_candidates, optimal_recall


def get_optimal_candidates_for_dataset(dataset: str, plateau_threshold: float = 0.05) -> Optional[int]:
    """
    Get optimal candidate count for a dataset from its analysis file.

    Args:
        dataset: Dataset name (e.g., "itunes_amazon")
        plateau_threshold: Maximum improvement to consider a plateau

    Returns:
        Optimal candidate count, or None if analysis not found
    """
    analysis_file = pathlib.Path(f"results/{dataset}_claude_analysis.json")

    if not analysis_file.exists():
        print(f"⚠️ No analysis file found for {dataset}: {analysis_file}")
        return None

    try:
        with open(analysis_file) as f:
            analysis = json.load(f)

        candidate_analysis = analysis.get("candidate_analysis", {})
        if not candidate_analysis:
            print(f"⚠️ No candidate analysis found in {analysis_file}")
            return None

        optimal_candidates, optimal_recall = find_recall_plateau(candidate_analysis, plateau_threshold)

        print(f"📊 {dataset}: Optimal candidates = {optimal_candidates} (recall = {optimal_recall:.3f})")

        # Show improvement vs max candidates tested
        max_candidates_key = max(
            (k for k in candidate_analysis if k.startswith("recall_at_")),
            key=lambda k: int(k.split("_")[-1])
        )
        max_candidates = int(max_candidates_key.split("_")[-1])
        max_recall = candidate_analysis[max_candidates_key]

        recall_loss = max_recall - optimal_recall
        efficiency_gain = max_candidates / optimal_candidates if optimal_candidates > 0 else 1

        print(f"   vs {max_candidates} candidates: {recall_loss:.1%} recall loss, {efficiency_gain:.1f}x efficiency gain")

        return optimal_candidates

    except Exception as e:
        print(f"❌ Error reading analysis for {dataset}: {e}")
        return None


def get_all_optimal_candidates() -> Dict[str, int]:
    """
    Get optimal candidate counts for all datasets that have analysis files.

    Returns:
        Dict mapping dataset names to optimal candidate counts
    """
    results_dir = pathlib.Path("results")
    optimal_candidates = {}

    # Find all analysis files
    for analysis_file in results_dir.glob("*_claude_analysis.json"):
        dataset = analysis_file.stem.replace("_claude_analysis", "")
        optimal = get_optimal_candidates_for_dataset(dataset)
        if optimal:
            optimal_candidates[dataset] = optimal

    return optimal_candidates


def print_candidate_optimization_summary():
    """Print a summary of optimal candidate counts for all datasets."""
    print("🎯 CANDIDATE COUNT OPTIMIZATION SUMMARY")
    print("=" * 60)

    optimal_candidates = get_all_optimal_candidates()

    if not optimal_candidates:
        print("❌ No datasets with candidate analysis found")
        return

    print(f"{'Dataset':<20} {'Optimal':<10} {'Notes'}")
    print("-" * 60)

    for dataset, candidates in sorted(optimal_candidates.items()):
        efficiency_note = ""
        if candidates <= 50:
            efficiency_note = "🚀 Very efficient"
        elif candidates <= 100:
            efficiency_note = "✅ Efficient"
        elif candidates <= 200:
            efficiency_note = "⚡ Good"
        else:
            efficiency_note = "⚠️ High"

        print(f"{dataset:<20} {candidates:<10} {efficiency_note}")

    print("\n💡 Usage: Use these optimal candidate counts to reduce computation")
    print("   while maintaining recall performance.")


if __name__ == "__main__":
    print_candidate_optimization_summary()
