#!/usr/bin/env python3
"""
Test the corrected SimplifiedAgenticGenerator using ClaudeSDKClient
"""
import asyncio
import json
import os
import sys
sys.path.insert(0, '.')

from src.experiments.simplified_agentic_generator import SimplifiedAgenticGenerator

async def test_proper_client():
    """Test the corrected generator using ClaudeSDKClient"""
    
    print("🧪 TESTING: SimplifiedAgenticGenerator with proper ClaudeSDKClient")
    print("=" * 70)
    
    # Clear logs
    if os.path.exists("results/temp/mcp_server.log"):
        os.remove("results/temp/mcp_server.log")
    
    # Create test dev_results 
    dev_results = {
        "metadata": {"f1": 0.8571, "precision": 0.8824, "recall": 0.8333},
        "dev_pairs": [
            {"category": "positive", "left_record": {"Beer_Name": "Budweiser"}, "right_record": {"Beer_Name": "Bud"}},
            {"category": "negative", "left_record": {"Beer_Name": "Corona"}, "right_record": {"Beer_Name": "Heineken"}}
        ]
    }
    
    # Create generator  
    generator = SimplifiedAgenticGenerator("beer", model="gpt-4.1-nano")  # Entity matching model
    
    print("🚀 Starting corrected agentic generator...")
    
    try:
        result_file, cost_info = await generator.generate_rules(dev_results)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Rules file: {result_file}")
        print(f"   Approach: {cost_info.get('approach', 'unknown')}")
        print(f"   Cost: ${cost_info.get('total_cost_usd', 0):.4f}")
        print(f"   Turns: {cost_info.get('turn_count', 0)}")
        
        # Check MCP server activity
        mcp_calls = 0
        if os.path.exists("results/temp/mcp_server.log"):
            with open("results/temp/mcp_server.log") as f:
                log_content = f.read()
            mcp_calls = log_content.count("Tool called:")
            print(f"   MCP server calls: {mcp_calls}")
            
            # Show specific tools used
            if "WriteRules" in log_content:
                print(f"   🎯 WriteRules was called - rules should be properly formatted!")
            if "ReadInstructions" in log_content:
                print(f"   📋 ReadInstructions was called - task understood!")
            if "ReadSampleData" in log_content:
                print(f"   📊 ReadSampleData was called - errors analyzed!")
        
        # Validate generated rules
        if os.path.exists(result_file):
            with open(result_file) as f:
                rules = json.load(f)
            
            print(f"\n📋 Generated Rules Validation:")
            required_sections = ['candidate_rules', 'score_rules', 'decision_rules', 'weight_rules']
            
            all_present = True
            for section in required_sections:
                if section in rules:
                    if section == 'weight_rules':
                        weights = rules[section]
                        has_all_weights = all(w in weights for w in ['semantic_weight', 'trigram_weight', 'syntactic_weight'])
                        print(f"   ✅ {section}: {weights if has_all_weights else 'MISSING WEIGHTS'}")
                        if not has_all_weights:
                            all_present = False
                    else:
                        print(f"   ✅ {section}: {len(rules[section])} rules")
                else:
                    print(f"   ❌ {section}: MISSING")
                    all_present = False
            
            if all_present:
                print(f"   🎯 All required sections present with proper weights!")
            
            return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Still check logs for debugging
        if os.path.exists("results/temp/mcp_server.log"):
            print(f"\n📄 MCP Server Log (last 10 lines):")
            with open("results/temp/mcp_server.log") as f:
                lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.strip()}")
        
        return False

if __name__ == "__main__":
    success = asyncio.run(test_proper_client())
    if success:
        print(f"\n🎉 The corrected approach using ClaudeSDKClient works!")
    else:
        print(f"\n💥 Still having issues - need further debugging")