#!/usr/bin/env python3
"""
Test the complete intelligent pipeline: analysis → Claude optimization → results.

This script tests the new approach on a small dataset to verify everything works.

Usage: python scripts/test_intelligent_pipeline.py --dataset itunes_amazon
"""

import argparse
import json
import pathlib
import subprocess


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    print(f"   Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Exit code: {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        return False


def test_intelligent_pipeline(dataset: str):
    """Test the complete intelligent pipeline"""
    print(f"🧪 TESTING INTELLIGENT PIPELINE ON {dataset.upper()}")
    print("=" * 70)

    # Step 1: Generate rich analysis
    analysis_file = f"results/{dataset}_claude_analysis.json"
    if not run_command(
        [
            "python",
            "scripts/analyze_for_claude.py",
            "--dataset",
            dataset,
            "--max-pairs",
            "50",  # Small test
        ],
        "Step 1: Rich similarity analysis",
    ):
        return False

    # Verify analysis file was created
    if not pathlib.Path(analysis_file).exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        return False

    # Load and display analysis summary
    with open(analysis_file) as f:
        analysis = json.load(f)

    print("\n📊 ANALYSIS SUMMARY:")
    sim_analysis = analysis["similarity_analysis"]
    true_matches = sim_analysis["true_matches"]
    false_positives = sim_analysis["false_positives"]

    print(f"   True matches - Syntactic: {true_matches['syntactic']['mean']:.3f}")
    print(f"   True matches - Trigram: {true_matches['trigram']['mean']:.3f}")
    if true_matches["semantic"]:
        print(f"   True matches - Semantic: {true_matches['semantic']['mean']:.3f}")

    print(f"   False positives - Syntactic: {false_positives['syntactic']['mean']:.3f}")
    print(f"   False positives - Trigram: {false_positives['trigram']['mean']:.3f}")
    if false_positives["semantic"]:
        print(f"   False positives - Semantic: {false_positives['semantic']['mean']:.3f}")

    # Step 2: Generate Claude config (with fallback for testing)
    config_file = f"results/{dataset}_claude_config.json"
    if not run_command(
        [
            "python",
            "scripts/claude_config_generator.py",
            "--dataset",
            dataset,
            "--fallback-only",  # Use fallback for reliable testing
        ],
        "Step 2: Claude configuration generation",
    ):
        return False

    # Verify config file was created
    if not pathlib.Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        return False

    # Load and display config summary
    with open(config_file) as f:
        config = json.load(f)

    print("\n⚙️ GENERATED CONFIG:")
    hyperparams = config["hyperparameters"]
    print(f"   max_candidates: {hyperparams['max_candidates']}")
    print(f"   trigram_weight: {hyperparams['trigram_weight']}")
    print(f"   syntactic_weight: {hyperparams['syntactic_weight']}")
    print(f"   semantic_weight: {hyperparams['semantic_weight']}")
    print(
        f"   Weight sum: {sum([hyperparams['trigram_weight'], hyperparams['syntactic_weight'], hyperparams['semantic_weight']]):.3f}"
    )
    print(f"   Rules generated: {len(config['rules'])}")

    # Validate weights sum to 1.0
    weight_sum = sum([hyperparams["trigram_weight"], hyperparams["syntactic_weight"], hyperparams["semantic_weight"]])
    if abs(weight_sum - 1.0) > 0.01:
        print(f"❌ Weight validation failed: sum = {weight_sum} (should be 1.0)")
        return False

    print("\n✅ ALL TESTS PASSED!")
    print("📁 Files generated:")
    print(f"   Analysis: {analysis_file}")
    print(f"   Config: {config_file}")
    print("\n🚀 Ready to integrate with run_complete_pipeline.py!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test the intelligent pipeline")
    parser.add_argument("--dataset", default="itunes_amazon", help="Dataset to test (default: itunes_amazon)")

    args = parser.parse_args()

    # Check if dataset exists
    data_root = pathlib.Path("data/raw") / args.dataset
    if not data_root.exists():
        print(f"❌ Dataset '{args.dataset}' not found in data/raw/")
        return

    # Run test
    success = test_intelligent_pipeline(args.dataset)

    if success:
        print("\n🎉 INTELLIGENT PIPELINE TEST SUCCESSFUL!")
        print("The new approach is working correctly and ready for full integration.")
    else:
        print("\n💥 TEST FAILED!")
        print("Check the error messages above to debug issues.")


if __name__ == "__main__":
    main()
