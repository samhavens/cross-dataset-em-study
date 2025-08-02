#!/usr/bin/env python3
"""
Test MCP server functionality directly
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, '.')

async def test_get_baseline():
    """Test the get_baseline_tool function directly"""
    from src.mcp_servers.entity_matching_server import get_baseline_tool

    print("🧪 Testing get_baseline_tool...")
    result = await get_baseline_tool("beer")

    for content in result:
        print(f"Result: {content.text}")

    return result

async def test_read_sample_data():
    """Test the read_sample_data_tool function directly"""
    from src.mcp_servers.entity_matching_server import read_sample_data_tool

    print("\n🧪 Testing read_sample_data_tool...")

    # First create sample data
    sample_data = {
        "dataset": "beer",
        "target_f1": 95.3,
        "dev_metrics": {"f1": 0.5, "precision": 0.6, "recall": 0.4},
        "dev_pairs": [
            {"left_record": {"Beer_Name": "Test Beer 1"}, "right_record": {"Beer_Name": "Test Beer 1"}, "match": 1, "category": "true_positive"}
        ]
    }

    sample_file = "results/temp/sample_data_beer.json"
    os.makedirs(os.path.dirname(sample_file), exist_ok=True)
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"📄 Created sample file: {sample_file}")

    result = await read_sample_data_tool("beer")

    for content in result:
        print(f"Result: {content.text}")

    return result

if __name__ == "__main__":
    async def main():
        await test_get_baseline()
        await test_read_sample_data()

    asyncio.run(main())
