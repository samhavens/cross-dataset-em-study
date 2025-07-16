#!/usr/bin/env python3
"""
Quick fix for the async batch processing hang by adding timeouts and better error handling.
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.entity_matching.hybrid_matcher import run_matching
import asyncio
import os

# Set environment variable to prevent tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def test_with_timeout():
    """Test with very low concurrency and timeout"""
    print("🔧 Testing with timeout and low concurrency...")
    
    try:
        # Use very conservative settings
        result = await asyncio.wait_for(
            run_matching(
                dataset="dblp_scholar",
                limit=10,  # Very small limit for testing
                max_candidates=20,  # Small number of candidates
                model="gpt-4o-mini",
                semantic_weight=0.5,
                concurrency=2  # Very low concurrency
            ),
            timeout=120  # 2 minutes timeout
        )
        
        print("✅ Test completed successfully!")
        return result
        
    except asyncio.TimeoutError:
        print("❌ Test timed out - there's definitely a deadlock issue")
        return None
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    result = asyncio.run(test_with_timeout())
    if result:
        print(f"Results: {result}")