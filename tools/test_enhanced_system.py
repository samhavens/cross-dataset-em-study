#!/usr/bin/env python
"""
Test the enhanced experimental system with structured output and auto-testing.
"""

import json
import os
import pathlib
import subprocess


def test_structured_output():
    """Test llm_em_hybrid.py with structured output"""
    print("🧪 Testing structured output...")

    # Ensure mock mode
    env = os.environ.copy()
    if "OPENAI_API_KEY" in env:
        del env["OPENAI_API_KEY"]

    # Test JSON output
    cmd = [
        "python",
        "llm_em_hybrid.py",
        "--dataset",
        "beer",
        "--limit",
        "5",
        "--candidate-ratio",
        "0.05",
        "--output-json",
        "test_results.json",
        "--output-csv",
        "test_results.csv",
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, env=env, timeout=60)

        if result.returncode == 0:
            print("✅ Command completed successfully")

            # Check JSON output
            if pathlib.Path("test_results.json").exists():
                with open("test_results.json") as f:
                    data = json.load(f)

                print("✅ JSON output created")
                print(f"   Dataset: {data.get('dataset')}")
                print(f"   F1: {data.get('metrics', {}).get('f1', 'N/A')}")
                print(f"   Cost: ${data.get('cost_usd', 0):.3f}")

                # Check required fields
                required_fields = ["timestamp", "dataset", "model", "metrics", "candidate_ratio"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    print(f"❌ Missing JSON fields: {missing}")
                else:
                    print("✅ All required JSON fields present")
            else:
                print("❌ JSON file not created")

            # Check CSV output
            if pathlib.Path("test_results.csv").exists():
                import pandas as pd

                df = pd.read_csv("test_results.csv")
                print(f"✅ CSV output created with {len(df)} rows")
                print(f"   Columns: {list(df.columns)}")
            else:
                print("❌ CSV file not created")

        else:
            print(f"❌ Command failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_sweep_with_autotest():
    """Test sweep with auto-test functionality"""
    print("\n🧪 Testing sweep with auto-test...")

    env = os.environ.copy()
    if "OPENAI_API_KEY" in env:
        del env["OPENAI_API_KEY"]

    cmd = [
        "python",
        "tools/sweep_candidates.py",
        "--dataset",
        "beer",
        "--limit",
        "5",
        "--num-points",
        "2",
        "--output",
        "test_sweep.json",
        "--output-csv",
        "test_sweep.csv",
        "--auto-test",
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False, env=env, timeout=120)

        if result.returncode == 0:
            print("✅ Sweep with auto-test completed")

            if pathlib.Path("test_sweep.json").exists():
                with open("test_sweep.json") as f:
                    data = json.load(f)

                print("✅ Sweep results JSON created")
                if "sweep_results" in data and "test_results" in data:
                    print("✅ Contains both sweep and test results")
                    sweep_count = len(data["sweep_results"])
                    test_f1 = data["test_results"].get("f1", 0)
                    print(f"   Sweep trials: {sweep_count}")
                    print(f"   Test F1: {test_f1}")
                else:
                    print("❌ Missing sweep_results or test_results")

        else:
            print(f"❌ Sweep failed with return code {result.returncode}")

    except subprocess.TimeoutExpired:
        print("❌ Sweep timed out")
    except Exception as e:
        print(f"❌ Error: {e}")


def cleanup():
    """Clean up test files"""
    test_files = ["test_results.json", "test_results.csv", "test_sweep.json", "test_sweep.csv"]

    for file in test_files:
        if pathlib.Path(file).exists():
            pathlib.Path(file).unlink()


if __name__ == "__main__":
    print("🚀 TESTING ENHANCED EXPERIMENTAL SYSTEM")
    print("=" * 50)

    test_structured_output()
    test_sweep_with_autotest()

    print("\n🧹 Cleaning up test files...")
    cleanup()

    print("\n✅ All tests completed!")
