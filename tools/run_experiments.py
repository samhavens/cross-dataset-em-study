#!/usr/bin/env python
"""
Run comprehensive entity matching experiments with hyperparameter sweeping.
Handles multiple datasets, automatic test evaluation, and results aggregation.
"""

import argparse
import json
import pathlib
import subprocess
import time

from datetime import datetime
from typing import Dict

import pandas as pd

from dataset_info import get_competitive_f1_threshold


def run_full_experiment(
    dataset: str,
    model: str = "gpt-4.1-nano",
    num_points: int = 8,
    limit: int = None,
    results_dir: str = "experiment_results",
) -> Dict:
    """
    Run complete experiment: sweep validation → test best config.

    Returns:
        Dict with sweep and test results
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING FULL EXPERIMENT: {dataset.upper()}")
    print(f"{'=' * 60}")

    # Create results directory
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(exist_ok=True)

    # File paths
    csv_file = results_path / "all_results.csv"
    json_file = results_path / f"{dataset}_{model.replace('/', '_')}_experiment.json"

    print("Results will be saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")

    # Run sweep with auto-test
    cmd = [
        "python",
        "tools/sweep_candidates.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--num-points",
        str(num_points),
        "--output",
        str(json_file),
        "--output-csv",
        str(csv_file),
        "--auto-test",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\nRunning: {' '.join(cmd)}")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time

        print(f"✅ Experiment completed in {elapsed:.1f}s")

        # Load and return results
        if json_file.exists():
            with open(json_file) as f:
                experiment_data = json.load(f)

            # Add metadata
            experiment_data["experiment_metadata"] = {
                "dataset": dataset,
                "model": model,
                "num_points": num_points,
                "limit": limit,
                "elapsed_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            }

            # Re-save with metadata
            with open(json_file, "w") as f:
                json.dump(experiment_data, f, indent=2)

            return experiment_data
        return {"error": "Results file not created"}

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        error_msg = f"Experiment failed after {elapsed:.1f}s: {e.stderr if e.stderr else str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


def analyze_results(results_dir: str = "experiment_results") -> pd.DataFrame:
    """
    Analyze all experimental results and create summary.

    Returns:
        DataFrame with experiment summary
    """
    results_path = pathlib.Path(results_dir)
    csv_file = results_path / "all_results.csv"

    if not csv_file.exists():
        print(f"No results file found at {csv_file}")
        return pd.DataFrame()

    # Load all results
    df = pd.read_csv(csv_file)

    print("\n📊 EXPERIMENT ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Total runs: {len(df)}")
    print(f"Datasets: {df['dataset'].nunique()}")
    print(f"Models: {df['model'].nunique()}")

    # Group by dataset and find best results
    best_results = []

    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]

        # Separate validation and test runs
        # Validation runs typically have smaller processed_pairs (due to --limit)
        # or are the first runs for each candidate_ratio
        validation_df = dataset_df.drop_duplicates(subset=["candidate_ratio"], keep="first")

        if len(validation_df) > 0:
            best_val = validation_df.loc[validation_df["f1"].idxmax()]

            # Find corresponding test run (same candidate_ratio, but likely more pairs)
            test_runs = dataset_df[
                (dataset_df["candidate_ratio"] == best_val["candidate_ratio"])
                & (dataset_df["processed_pairs"] >= best_val["processed_pairs"])
            ]

            if len(test_runs) > 0:
                test_result = test_runs.iloc[-1]  # Take the last/largest run
            else:
                test_result = best_val  # Fallback to validation result

            try:
                competitive_f1 = get_competitive_f1_threshold(dataset)
            except:
                competitive_f1 = None

            best_results.append(
                {
                    "dataset": dataset,
                    "best_candidate_ratio": best_val["candidate_ratio"],
                    "validation_f1": best_val["f1"],
                    "test_f1": test_result["f1"],
                    "test_precision": test_result["precision"],
                    "test_recall": test_result["recall"],
                    "predictions_made": test_result["predictions_made"],
                    "processed_pairs": test_result["processed_pairs"],
                    "cost_usd": test_result["cost_usd"],
                    "competitive_f1": competitive_f1,
                    "is_competitive": test_result["f1"] >= competitive_f1 if competitive_f1 else None,
                    "elapsed_seconds": test_result["elapsed_seconds"],
                }
            )

    summary_df = pd.DataFrame(best_results)

    if len(summary_df) > 0:
        print("\n📈 BEST RESULTS PER DATASET:")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            competitive_status = ""
            if row["is_competitive"] is not None:
                competitive_status = "🎉" if row["is_competitive"] else "⚠️"

            print(
                f"{row['dataset']:15} | F1: {row['test_f1']:5.1f} | "
                f"Ratio: {row['best_candidate_ratio']:5.1%} | "
                f"Cost: ${row['cost_usd']:6.2f} | {competitive_status}"
            )

        # Save summary
        summary_file = results_path / "experiment_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to: {summary_file}")

        # Overall stats
        competitive_count = summary_df["is_competitive"].sum() if "is_competitive" in summary_df else 0
        total_cost = summary_df["cost_usd"].sum()
        avg_f1 = summary_df["test_f1"].mean()

        print("\n🎯 OVERALL PERFORMANCE:")
        print(f"   Average F1: {avg_f1:.1f}")
        print(f"   Competitive results: {competitive_count}/{len(summary_df)}")
        print(f"   Total cost: ${total_cost:.2f}")

    return summary_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Single dataset to run (default: run all small datasets)")
    parser.add_argument("--datasets", nargs="+", help="List of datasets to run")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--num-points", type=int, default=8, help="Number of candidate ratios to test")
    parser.add_argument("--limit", type=int, help="Limit test pairs for faster debugging")
    parser.add_argument("--results-dir", default="experiment_results", help="Directory for results")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing results")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.results_dir)
        return

    # Determine which datasets to run
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = args.datasets
    else:
        # Default: run small/fast datasets
        datasets = ["fodors_zagat", "beer", "zomato_yelp", "rotten_imdb"]
        print(f"No datasets specified, running default small datasets: {datasets}")

    print("🚀 STARTING EXPERIMENTS")
    print(f"Datasets: {datasets}")
    print(f"Model: {args.model}")
    print(f"Results directory: {args.results_dir}")

    # Run experiments
    all_results = {}
    total_start = time.time()

    for i, dataset in enumerate(datasets):
        print(f"\n📍 Progress: {i + 1}/{len(datasets)} datasets")

        try:
            result = run_full_experiment(dataset, args.model, args.num_points, args.limit, args.results_dir)
            all_results[dataset] = result

            if "error" not in result:
                print(f"✅ {dataset} completed successfully")
            else:
                print(f"❌ {dataset} failed: {result['error']}")

        except Exception as e:
            print(f"❌ {dataset} crashed: {e}")
            all_results[dataset] = {"error": str(e)}

    total_elapsed = time.time() - total_start

    print("\n🏁 ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Successful: {sum(1 for r in all_results.values() if 'error' not in r)}/{len(datasets)}")

    # Analyze results
    analyze_results(args.results_dir)


if __name__ == "__main__":
    main()
