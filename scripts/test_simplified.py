#!/usr/bin/env python3
"""
Test the simplified MCP-only generator
"""
import asyncio


async def test_simplified_generator():
    """Test simplified generator with minimal dev results"""

    # Create minimal test data
    test_dev_results = {
        "metadata": {"f1": 0.5, "precision": 0.6, "recall": 0.4},
        "dev_pairs": [
            {"left_record": {"name": "Test Beer 1"}, "right_record": {"name": "Test Beer 1"}, "match": 1, "category": "true_positive"},
            {"left_record": {"name": "Beer A"}, "right_record": {"name": "Beer B"}, "match": 0, "category": "false_positive"}
        ]
    }

    from src.experiments.simplified_agentic_generator import generate_simplified_heuristics

    print("🧪 Testing simplified agentic generator...")

    try:
        result_file, cost_info = await generate_simplified_heuristics(
            "beer",
            test_dev_results,
            "results/temp/test_simplified_rules.json"
        )

        print("✅ Test successful!")
        print(f"📁 Rules file: {result_file}")
        print(f"💰 Cost: ${cost_info.get('total_cost_usd', 0):.4f}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simplified_generator())
