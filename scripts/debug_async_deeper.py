#!/usr/bin/env python3
"""Debug the deeper async issue - why 383 tasks hang but 10 tasks work"""

import asyncio
import os
import pathlib
import sys
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.entity_matching.hybrid_matcher import run_matching


async def test_scaling():
    """Test with different numbers of pairs to find the breaking point"""

    test_cases = [10, 50, 100, 200, 500]

    for limit in test_cases:
        print(f"\n🔬 Testing with {limit} pairs...")

        try:
            start_time = time.time()
            await asyncio.wait_for(
                run_matching(
                    dataset="dblp_scholar",
                    limit=limit,
                    max_candidates=50,  # Keep small for speed
                    model="gpt-4o-mini",
                    semantic_weight=0.5,
                    concurrency=5  # Reduce concurrency
                ),
                timeout=180  # 3 minute timeout
            )

            elapsed = time.time() - start_time
            print(f"✅ {limit} pairs completed in {elapsed:.1f}s")

        except asyncio.TimeoutError:
            print(f"❌ {limit} pairs timed out after 3 minutes")
            break
        except Exception as e:
            print(f"❌ {limit} pairs failed: {e}")
            break

if __name__ == "__main__":
    asyncio.run(test_scaling())
