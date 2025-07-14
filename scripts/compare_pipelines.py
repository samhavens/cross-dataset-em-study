#!/usr/bin/env python3
"""
Compare original hyperparameter sweep vs new analysis-driven approach.

Usage:
    python scripts/compare_pipelines.py --dataset itunes_amazon
    python scripts/compare_pipelines.py --dataset beer --quick
"""

import argparse
import json
import subprocess
import sys
import time

from pathlib import Path


def run_pipeline(dataset: str, use_analysis_driven: bool, output_suffix: str) -> dict:
    """Run a pipeline and return the results"""

    cmd = [
        "python", "run_complete_pipeline.py",
        "--dataset", dataset,
        "--model", "gpt-4.1-nano",
        "--use-agentic-rules"
    ]

    if use_analysis_driven:
        cmd.append("--use-analysis-driven")

    print(f"🚀 Running {'analysis-driven' if use_analysis_driven else 'sweep-based'} pipeline...")
    print(f"   Command: {' '.join(cmd)}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=False, capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Pipeline completed in {elapsed_time:.1f}s")

            # Try to load the results file
            results_file = f"results/{dataset}_complete_pipeline.json"
            if Path(results_file).exists():
                with open(results_file) as f:
                    results = json.load(f)

                # Add timing info
                results["total_wall_time"] = elapsed_time
                results["approach"] = "analysis_driven" if use_analysis_driven else "sweep_based"

                # Save with suffix
                comparison_file = f"results/{dataset}_pipeline_{output_suffix}.json"
                with open(comparison_file, "w") as f:
                    json.dump(results, f, indent=2)

                print(f"📊 Results saved to: {comparison_file}")
                return results
            print(f"⚠️ Results file not found: {results_file}")
            return {"error": "Results file not found", "stderr": result.stderr}
        print(f"❌ Pipeline failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return {"error": f"Pipeline failed: {result.stderr}", "returncode": result.returncode}

    except subprocess.TimeoutExpired:
        print("⏰ Pipeline timed out after 30 minutes")
        return {"error": "Pipeline timed out"}
    except Exception as e:
        print(f"❌ Pipeline failed with exception: {e}")
        return {"error": str(e)}


def compare_results(sweep_results: dict, analysis_results: dict, dataset: str):
    """Compare and summarize the results"""

    print(f"\n🏆 COMPARISON RESULTS FOR {dataset.upper()}")
    print("=" * 60)

    # Check if both succeeded
    if "error" in sweep_results:
        print(f"❌ Sweep-based approach failed: {sweep_results['error']}")
        sweep_success = False
    else:
        sweep_success = True

    if "error" in analysis_results:
        print(f"❌ Analysis-driven approach failed: {analysis_results['error']}")
        analysis_success = False
    else:
        analysis_success = True

    if not sweep_success and not analysis_success:
        print("❌ Both approaches failed!")
        return
    if not sweep_success:
        print("⚠️ Only analysis-driven approach succeeded")
        print_single_results(analysis_results, "Analysis-driven")
        return
    if not analysis_success:
        print("⚠️ Only sweep-based approach succeeded")
        print_single_results(sweep_results, "Sweep-based")
        return

    # Both succeeded - compare them
    print("📊 PERFORMANCE COMPARISON:")

    # Extract key metrics
    sweep_summary = sweep_results.get("summary", {})
    analysis_summary = analysis_results.get("summary", {})

    sweep_f1 = sweep_summary.get("enhanced_f1", 0)
    analysis_f1 = analysis_summary.get("enhanced_f1", 0)
    f1_diff = analysis_f1 - sweep_f1

    sweep_cost = sweep_summary.get("total_cost_usd", 0)
    analysis_cost = analysis_summary.get("total_cost_usd", 0)
    cost_diff = analysis_cost - sweep_cost

    sweep_time = sweep_results.get("total_wall_time", 0)
    analysis_time = analysis_results.get("total_wall_time", 0)
    time_diff = analysis_time - sweep_time

    print("F1 Score:")
    print(f"  Sweep-based:      {sweep_f1:.4f}")
    print(f"  Analysis-driven:  {analysis_f1:.4f}")
    print(f"  Difference:       {f1_diff:+.4f}")

    print("\nCost (USD):")
    print(f"  Sweep-based:      ${sweep_cost:.3f}")
    print(f"  Analysis-driven:  ${analysis_cost:.3f}")
    print(f"  Difference:       ${cost_diff:+.3f}")

    print("\nWall Time (seconds):")
    print(f"  Sweep-based:      {sweep_time:.1f}s")
    print(f"  Analysis-driven:  {analysis_time:.1f}s")
    print(f"  Difference:       {time_diff:+.1f}s")

    # Determine winner
    print("\n🎯 WINNER:")
    if abs(f1_diff) < 0.001:  # Essentially the same
        if cost_diff < -0.01:  # Analysis is cheaper
            print("🥇 Analysis-driven (same F1, lower cost)")
        elif time_diff < -10:  # Analysis is faster
            print("🥇 Analysis-driven (same F1, faster)")
        else:
            print("🤝 Tie (similar performance across all metrics)")
    elif f1_diff > 0.001:  # Analysis is better
        print(f"🥇 Analysis-driven (F1 improved by {f1_diff:.4f})")
    else:  # Sweep is better
        print(f"🥇 Sweep-based (F1 improved by {-f1_diff:.4f})")

    # Additional insights
    print("\n💡 INSIGHTS:")
    sweep_rule_cost = sweep_summary.get("rule_generation_cost_usd", 0)
    analysis_rule_cost = analysis_summary.get("rule_generation_cost_usd", 0)

    print(f"Rule generation cost: Sweep ${sweep_rule_cost:.3f} vs Analysis ${analysis_rule_cost:.3f}")

    if time_diff < -60:
        print(f"⚡ Analysis-driven is {-time_diff/60:.1f} minutes faster")
    elif time_diff > 60:
        print(f"🐌 Analysis-driven is {time_diff/60:.1f} minutes slower")

    # Save comparison summary
    comparison_summary = {
        "dataset": dataset,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep_results": sweep_summary,
        "analysis_results": analysis_summary,
        "comparison": {
            "f1_difference": f1_diff,
            "cost_difference": cost_diff,
            "time_difference": time_diff,
            "winner": "analysis_driven" if f1_diff > 0.001 or (abs(f1_diff) < 0.001 and (cost_diff < -0.01 or time_diff < -10)) else "sweep_based"
        }
    }

    summary_file = f"results/{dataset}_pipeline_comparison.json"
    with open(summary_file, "w") as f:
        json.dump(comparison_summary, f, indent=2)
    print(f"\n📋 Comparison summary saved to: {summary_file}")


def print_single_results(results: dict, approach: str):
    """Print results for a single approach"""
    summary = results.get("summary", {})

    print(f"📊 {approach} Results:")
    print(f"  F1 Score:     {summary.get('enhanced_f1', 0):.4f}")
    print(f"  Cost:         ${summary.get('total_cost_usd', 0):.3f}")
    print(f"  Wall Time:    {results.get('total_wall_time', 0):.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Compare sweep vs analysis-driven pipelines")
    parser.add_argument("--dataset", required=True, help="Dataset to test (e.g., itunes_amazon, beer)")
    parser.add_argument("--quick", action="store_true", help="Run analysis-driven only (faster)")
    parser.add_argument("--sweep-only", action="store_true", help="Run sweep-based only")
    parser.add_argument("--analysis-only", action="store_true", help="Run analysis-driven only")

    args = parser.parse_args()

    print(f"🔬 PIPELINE COMPARISON FOR {args.dataset.upper()}")
    print("=" * 60)

    # Determine what to run
    run_sweep = not args.analysis_only and not args.quick
    run_analysis = not args.sweep_only

    if args.quick:
        print("⚡ Quick mode: Running analysis-driven approach only")

    sweep_results = None
    analysis_results = None

    # Run sweep-based pipeline
    if run_sweep:
        print("\n1️⃣ Running sweep-based pipeline...")
        sweep_results = run_pipeline(args.dataset, use_analysis_driven=False, output_suffix="sweep")

    # Run analysis-driven pipeline
    if run_analysis:
        print("\n2️⃣ Running analysis-driven pipeline...")
        analysis_results = run_pipeline(args.dataset, use_analysis_driven=True, output_suffix="analysis")

    # Compare results
    if sweep_results and analysis_results:
        compare_results(sweep_results, analysis_results, args.dataset)
    elif analysis_results:
        print_single_results(analysis_results, "Analysis-driven")
    elif sweep_results:
        print_single_results(sweep_results, "Sweep-based")
    else:
        print("❌ No successful runs to analyze")
        sys.exit(1)


if __name__ == "__main__":
    main()
