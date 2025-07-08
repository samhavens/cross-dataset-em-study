#!/usr/bin/env python
"""
Test script for LLM entity matching benchmark.
Tests the mock mode to ensure the pipeline works without API keys.
"""

import os
import subprocess

from pathlib import Path


def test_mock_llm_bench():
    """Test that the LLM bench works in mock mode without API keys."""
    # Ensure no API key is set for this test
    env = os.environ.copy()
    if "OPENAI_API_KEY" in env:
        del env["OPENAI_API_KEY"]

    # Run the benchmark with a small limit
    result = subprocess.run(
        ["./bin/quick_llm_bench.sh", "abt_buy", "10", "gpt-4.1-nano"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    # Check that it completed successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}"

    # Check that it processed some pairs
    assert "processed" in result.stdout, "Expected 'processed' in output"

    # Check that mock cost was reported
    assert "K in" in result.stdout and "K out" in result.stdout, "Expected cost reporting"

    print("✅ Mock LLM benchmark test passed!")


def test_keys_file_exists():
    """Test that keys file is created and can be loaded."""
    keys_file = Path("data/abt_buy_llmkeys.pkl")
    assert keys_file.exists(), f"Keys file not found at {keys_file}"

    # Try to load the keys file
    import pickle

    with open(keys_file, "rb") as f:
        keys = pickle.load(f)

    assert isinstance(keys, dict), "Keys should be a dictionary"
    assert len(keys) > 0, "Keys should not be empty"

    # Check key structure
    first_key = next(iter(keys.values()))
    assert "key" in first_key, "Each key entry should have a 'key' field"
    assert "row" in first_key, "Each key entry should have a 'row' field"

    print("✅ Keys file test passed!")


if __name__ == "__main__":
    test_mock_llm_bench()
    test_keys_file_exists()
    print("🎉 All tests passed!")
