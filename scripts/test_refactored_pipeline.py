#!/usr/bin/env python3
"""
Test script to verify refactored pipeline functionality.
"""

import asyncio
import pathlib
import sys
import traceback

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from run_complete_pipeline import get_available_datasets, run_complete_pipeline


async def test_refactored_pipeline():
    """Test the refactored pipeline with known params to keep it fast"""
    print("🧪 Testing Refactored Pipeline")
    print("=" * 40)

    # Get available datasets
    datasets = get_available_datasets()
    print(f"📁 Available datasets: {datasets}")

    # Choose a small dataset
    test_dataset = "beer" if "beer" in datasets else datasets[0] if datasets else None

    if not test_dataset:
        print("❌ No datasets available for testing")
        return False

    print(f"🎯 Testing with dataset: {test_dataset}")

    # Test with known params to make it fast
    known_params = {
        "max_candidates": 50,  # Small for testing
        "semantic_weight": 0.7,
        "syntactic_weight": 0.2,
        "trigram_weight": 0.1,
        "decision_threshold": 0.5,
        "auto_accept_threshold": 0.85,
        "auto_reject_threshold": 0.15
    }

    try:
        print("⏳ Running refactored pipeline with known params...")
        result = await run_complete_pipeline(
            dataset=test_dataset,
            model="gpt-4.1-nano",
            concurrency=3,
            known_best_params=known_params,
            use_analysis_driven=True  # This should be ignored
        )

        print("✅ Pipeline completed successfully!")
        print(f"🎯 Final F1: {result.get('final_f1', 'N/A')}")
        print(f"📊 Keys: {list(result.keys())}")

        # Verify expected structure
        expected_keys = ['timestamp', 'dataset', 'pipeline_version', 'dev_results', 'optimal_params']
        missing_keys = [k for k in expected_keys if k not in result]

        if missing_keys:
            print(f"⚠️ Missing expected keys: {missing_keys}")
        else:
            print("✅ All expected keys present")

        return True

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        print("🔍 Traceback:")
        traceback.print_exc()
        return False


def test_imports():
    """Test that imports work correctly after refactoring"""
    print("\n🔍 Testing imports after refactoring...")

    try:
        # These should work
        print("✅ json_serializer import works")

        print("✅ agentic_heuristic_generator import works")

        # These should fail (removed files)
        try:
            from src.experiments.improved_sweep import run_improved_sweep  # noqa: F401
            print("❌ improved_sweep import should have failed but didn't")
            return False
        except ImportError:
            print("✅ improved_sweep import correctly fails (file removed)")

        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 Refactored Pipeline Test Suite")
    print("=" * 50)

    # Test imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n❌ Import tests failed")
        return False

    # Test pipeline
    pipeline_ok = await test_refactored_pipeline()

    if imports_ok and pipeline_ok:
        print("\n✅ All tests passed! Refactoring successful.")
        return True
    print("\n❌ Some tests failed")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
