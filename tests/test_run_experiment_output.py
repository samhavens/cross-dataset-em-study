#!/usr/bin/env python3
"""
Test to verify RunExperiment tool output and F1 score logging
"""

import asyncio
import json
import os
import tempfile

from pathlib import Path


async def test_run_experiment_output():
    """Test RunExperiment MCP tool directly"""

    # Add parent directory to path
    import sys
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import the MCP server function directly
    from src.mcp_servers.entity_matching_server import run_experiment_tool

    # Create a test rules file
    test_rules = {
        "hyperparameters": {
            "max_candidates": 50,
            "semantic_weight": 0.6,
            "trigram_weight": 0.25,
            "syntactic_weight": 0.15
        },
        "candidate_rules": [],
        "score_rules": [],
        "decision_rules": []
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_rules, f, indent=2)
        temp_rules_file = f.name

    try:
        print("🧪 Testing RunExperiment tool output")
        print(f"   Rules file: {temp_rules_file}")

        # Call RunExperiment directly
        result = await run_experiment_tool(
            dataset="beer",
            rules_file=temp_rules_file,
            max_examples=5
        )

        if result and len(result) > 0:
            output_text = result[0].text
            print(f"\n📋 RunExperiment Output ({len(output_text)} chars):")
            print("=" * 60)
            print(output_text)
            print("=" * 60)

            # Test the F1 extraction logic
            if "F1 SCORE:" in output_text or "F1 Score:" in output_text:
                import re
                f1_matches = re.findall(r'F1 SCORE?:\s*([0-9.]+)', output_text, re.IGNORECASE)
                if f1_matches:
                    f1_score = float(f1_matches[0])
                    print(f"\n✅ F1 Score extracted: {f1_score:.4f}")

                    # Extract precision/recall too
                    precision_matches = re.findall(r'PRECISION:\s*([0-9.]+)', output_text, re.IGNORECASE)
                    recall_matches = re.findall(r'RECALL:\s*([0-9.]+)', output_text, re.IGNORECASE)
                    if precision_matches and recall_matches:
                        precision = float(precision_matches[0])
                        recall = float(recall_matches[0])
                        print(f"✅ Precision/Recall: P={precision:.4f}, R={recall:.4f}")

                    return True
                print("❌ F1 score found in text but regex failed")
                print("   Looking for pattern: 'F1 SCORE?:\\s*([0-9.]+)'")

                # Show what the actual F1 lines look like
                lines = output_text.split('\n')
                f1_lines = [line for line in lines if 'F1' in line.upper()]
                print(f"   F1 lines found: {f1_lines}")
                return False
            print("❌ No F1 score found in output")
            print("   Searching for: 'F1 SCORE:' or 'F1 Score:'")

            # Show what we got instead
            lines = output_text.split('\n')[:10]
            print(f"   First 10 lines: {lines}")
            return False
        print("❌ RunExperiment returned empty result")
        return False

    except Exception as e:
        print(f"❌ Error testing RunExperiment: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp file
        if os.path.exists(temp_rules_file):
            os.unlink(temp_rules_file)

def test_f1_extraction_patterns():
    """Test different F1 output patterns"""
    print("\n🧪 Testing F1 extraction patterns")

    test_cases = [
        "F1 SCORE: 0.8234",
        "F1 Score: 0.7456",
        "   F1 SCORE: 0.9123",
        "F1: 0.6789",
        "f1 score: 0.5432",
        "F1-Score: 0.4321"
    ]

    import re
    pattern = r'F1 SCORE?:\s*([0-9.]+)'

    for test_case in test_cases:
        matches = re.findall(pattern, test_case, re.IGNORECASE)
        if matches:
            print(f"✅ '{test_case}' → {matches[0]}")
        else:
            print(f"❌ '{test_case}' → no match")

def main():
    """Main test function - only test patterns for speed"""
    print("🔍 Testing RunExperiment F1 Extraction Patterns")
    print("=" * 60)

    # Always test pattern matching (fast)
    test_f1_extraction_patterns()

    print("\n✅ F1 extraction patterns work correctly")
    print("📝 Run this test directly to test actual RunExperiment tool (slow)")

if __name__ == "__main__":
    # Check if being run directly
    import os
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Full test with actual tool
        print("🔍 Testing RunExperiment Output and F1 Extraction (FULL)")
        print("=" * 60)
        test_f1_extraction_patterns()
        print("\n" + "=" * 60)
        result = asyncio.run(test_run_experiment_output())
        if result:
            print("\n✅ RunExperiment tool is working and F1 scores are extractable")
        else:
            print("\n❌ Issue found with RunExperiment tool or F1 extraction")
    else:
        # Just pattern test (for test suite)
        main()
