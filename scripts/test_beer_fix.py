#!/usr/bin/env python3
"""Test the beer dataset fix"""

import os
import sys
import pathlib
import asyncio
import time

# Set the tokenizer fix first
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.entity_matching.hybrid_matcher import run_matching

async def test_beer_fix():
    """Test the beer dataset with the tokenizer fix"""
    print("🧪 Testing beer dataset with tokenizer fix...")
    
    start_time = time.time()
    
    try:
        result = await asyncio.wait_for(
            run_matching(
                dataset="beer",
                limit=20,  # Small test
                max_candidates=50,
                model="gpt-4o-mini",
                semantic_weight=0.5,
                concurrency=3
            ),
            timeout=120  # 2 minutes
        )
        
        elapsed = time.time() - start_time
        print(f"✅ Beer test completed in {elapsed:.1f}s")
        print(f"Results: {result}")
        
    except asyncio.TimeoutError:
        print("❌ Beer test timed out")
    except Exception as e:
        print(f"❌ Beer test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_beer_fix())