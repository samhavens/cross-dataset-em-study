#!/usr/bin/env python3
"""Test the async fixes to see if SDK scope conflicts are resolved"""

import asyncio
import os
import pathlib
import sys

# Fix tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.entity_matching.hybrid_matcher import run_matching


async def test_async_fix():
    """Test with the async fixes"""
    print("🧪 Testing async fixes for SDK scope conflicts...")

    try:
        result = await run_matching(
            dataset="beer",
            limit=20,  # Small test
            max_candidates=25,  # Small number
            model="gpt-4o-mini",
            semantic_weight=0.5,
            concurrency=2  # Low concurrency
        )

        print("✅ Async fixes successful!")
        print(f"Results: F1={result.get('f1', 0):.3f}")
        return True

    except Exception as e:
        print(f"❌ Async issues still present: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_async_fix())
    if success:
        print("\n🎉 SDK scope conflicts resolved!")
    else:
        print("\n⚠️ More async fixes needed")
