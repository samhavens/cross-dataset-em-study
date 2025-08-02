#!/usr/bin/env python3
"""
Test WriteRules validation
"""
import asyncio
import sys

sys.path.insert(0, '.')

async def test_write_rules_validation():
    from src.mcp_servers.entity_matching_server import write_rules_tool

    print("🧪 Testing WriteRules validation...")

    # Test with invalid English description (should fail)
    invalid_rules = [{
        "rule_name": "test_rule",
        "implementation": "if beer name matches exactly, boost by 0.1",
        "stage": "post_semantic"
    }]

    try:
        result = await write_rules_tool(
            semantic_weight=0.7,
            trigram_weight=0.2,
            syntactic_weight=0.1,
            score_rules=invalid_rules
        )
        print("❌ Should have failed validation!")
        for content in result:
            print(f"Result: {content.text}")
    except ValueError as e:
        print(f"✅ Correctly caught invalid rule: {e}")

    # Test with valid Python code (should succeed)
    valid_rules = [{
        "rule_name": "exact_name_boost",
        "implementation": "0.1 if left_record.get('name', '').lower() == right_record.get('name', '').lower() else 0",
        "stage": "post_semantic"
    }]

    try:
        result = await write_rules_tool(
            semantic_weight=0.7,
            trigram_weight=0.2,
            syntactic_weight=0.1,
            score_rules=valid_rules
        )
        print("✅ Valid rule accepted!")
        for content in result:
            print(f"Result: {content.text}")
    except Exception as e:
        print(f"❌ Valid rule failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_write_rules_validation())
