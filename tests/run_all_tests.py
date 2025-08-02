#!/usr/bin/env python3
"""
Run all tests in the test suite and provide a summary.
"""

import subprocess
import sys

from pathlib import Path


def run_test(test_file: Path) -> tuple[bool, str]:
    """Run a single test and return (success, output)"""
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            check=False, capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 30 seconds"
    except Exception as e:
        return False, f"Failed to run test: {e}"

def main():
    """Run all tests and summarize results"""
    print("🧪 Running Entity Matching Test Suite")
    print("=" * 50)

    tests_dir = Path(__file__).parent
    test_files = sorted(tests_dir.glob("test_*.py"))

    if not test_files:
        print("❌ No test files found!")
        return 1

    results = []

    for test_file in test_files:
        test_name = test_file.stem
        print(f"\n🏃 Running {test_name}...")

        success, output = run_test(test_file)
        results.append((test_name, success, output))

        if success:
            print(f"   ✅ {test_name}: PASSED")
        else:
            print(f"   ❌ {test_name}: FAILED")
            # Show first few lines of error
            error_lines = output.split('\n')[:5]
            for line in error_lines:
                if line.strip():
                    print(f"      {line}")

    # Summary
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\n📊 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for use.")
        return 0
    print(f"❌ {total - passed} test(s) failed. Check issues before running pipeline.")

    # Show failed test details
    print("\n🔍 Failed test details:")
    for test_name, success, output in results:
        if not success:
            print(f"\n--- {test_name} ---")
            print(output[:500] + "..." if len(output) > 500 else output)

    return 1

if __name__ == "__main__":
    sys.exit(main())
