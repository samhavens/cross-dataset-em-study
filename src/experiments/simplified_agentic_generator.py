#!/usr/bin/env python3
"""
Simplified agentic heuristic generator with only MCP tools + Read
"""

import asyncio
import json
import os
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    McpServerConfig,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)


def get_leaderboard_target_f1(dataset: str) -> float:
    """Get the top F1 score from leaderboard.md for the given dataset"""
    leaderboard_targets = {
        "abt_buy": 92.4,
        "beer": 95.3,
        "itunes_amazon": 85.0,
        "amazon_google": 75.0,  # Updated from internal_leaderboard.md
        "dblp_acm": 96.5,       # Updated from internal_leaderboard.md  
        "walmart_amazon": 85.1,  # Updated from internal_leaderboard.md
        "dblp_scholar": 89.8,    # Updated from internal_leaderboard.md
        "fodors_zagat": 99.6,
        "rotten_imdb": 97.2,
        "zomato_yelp": 98.2
    }
    return leaderboard_targets.get(dataset, 85.0)


@dataclass
class SampleData:
    dataset: str
    target_f1: float
    dev_metrics: Dict[str, float]
    dev_pairs: List[Dict[str, Any]]
    analysis_insights: Optional[Dict] = None


class SimplifiedAgenticGenerator:
    """Simplified agentic generator using only MCP tools + Read"""
    
    def __init__(self, dataset: str, model: str = "claude-3-5-sonnet-20241022", no_cache: bool = False, mode: str = "heuristics"):
        self.dataset = dataset
        self.model = model
        self.no_cache = no_cache
        self.mode = mode
        self.target_f1 = get_leaderboard_target_f1(dataset)

    async def generate_rules(self, dev_results: Dict[str, Any], output_file: Optional[str] = None, analysis_data: Optional[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate rules using proper ClaudeSDKClient for interactive sessions"""
        
        if output_file is None:
            output_file = f"results/generated_rules/{self.dataset}_mcp_generated_config.json"
        
        print(f"🚀 SIMPLIFIED AGENTIC HEURISTIC GENERATION FOR {self.dataset}")
        print("============================================================")
        print(f"🎯 Target: F1 > {self.target_f1}")
        
        # Create sample data and save it for MCP server to use
        sample_data = self._create_sample_data(dev_results, analysis_data)
        self._save_sample_data_for_mcp(sample_data)
        
        # Configure with proper MCP tools and restrictions
        # McpServerConfig already imported above
        from claude_code_sdk import ClaudeSDKClient
        import json
        
        # Load MCP server config from file
        with open('mcp_config.json') as f:
            mcp_config = json.load(f)
        
        # Use the config directly as a dict (per SDK guide)
        mcp_servers = mcp_config['mcpServers']
        
        options = ClaudeCodeOptions(
            mcp_servers=mcp_servers,
            # Allow MCP tools + essential planning/analysis tools
            allowed_tools=[
                "Read",  # Allow reading code files for analysis
                "Task",  # Allow planning and task management
                "LS",    # Allow listing directories for navigation
                "Grep",  # Allow searching through files
                "mcp__entity-matching__ReadInstructions",
                "mcp__entity-matching__ReadSampleData", 
                "mcp__entity-matching__GetBaseline",
                "mcp__entity-matching__WriteRules",
                "mcp__entity-matching__TestRules",
                "mcp__entity-matching__AnalyzePerformance",
                "mcp__entity-matching__ReportIssue"
            ],
            # Explicitly disallow system tools (except Read, Task, LS, Grep which are allowed above)
            disallowed_tools=[
                "Bash", "Write", "Edit", "MultiEdit", "Glob", 
                "NotebookEdit", "NotebookRead", "TodoWrite",
                "WebSearch", "WebFetch"  # Block web access
            ],
            permission_mode="acceptEdits",
            cwd=os.getcwd(),
            max_turns=30
            # Note: ClaudeSDKClient uses its own model, self.model is for reference
            # Claude Code SDK uses Sonnet 4 (the model running this conversation)
        )

        # Initial prompt for the interactive session - different for each mode
        if self.mode == "weights-only":
            initial_prompt = f"""WEIGHTS-ONLY OPTIMIZATION MODE: Focus only on optimizing semantic/trigram/syntactic weights.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset weights for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
3. mcp__entity-matching__WriteRules with mode="weights-only" - ONLY change weights, NO rules
4. mcp__entity-matching__TestRules - Test on full dev set
5. Iterate quickly on weight combinations

FOCUS: Find optimal semantic_weight, trigram_weight, syntactic_weight (must sum to 1.0).
DO NOT generate candidate_rules, score_rules, decision_rules, or prompt_rules - weights only!"""
        
        elif self.mode == "prompt-modification":
            initial_prompt = f"""PROMPT-MODIFICATION OPTIMIZATION MODE: Optimize weights AND generate prompt modification rules.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
3. mcp__entity-matching__WriteRules with mode="prompt-modification" - Optimize weights AND create prompt rules
4. mcp__entity-matching__TestRules - Test on full dev set
5. Iterate on both weights and prompt modifications

FOCUS: 
- Find optimal semantic_weight, trigram_weight, syntactic_weight (must sum to 1.0)
- Generate ONLY prompt_rules that dynamically improve LLM instructions based on record patterns
- DO NOT generate candidate_rules, score_rules, decision_rules, or weight_rules
- Prompt rules should contain conditions and prompt additions to guide LLM decision-making"""
        
        else:  # mode == "heuristics"
            initial_prompt = f"""HEURISTICS OPTIMIZATION MODE: Full rule generation with traditional heuristic rules.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__ReadSampleData (dataset: "{self.dataset}") - Analyze error patterns
3. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
4. mcp__entity-matching__WriteRules - Generate optimized weights AND custom rules
5. mcp__entity-matching__TestRules - Test on full dev set

FOCUS: Optimize weights AND create custom rules based on data analysis."""

        # Track session metrics
        turn_count = 0
        total_cost_usd = 0.0
        session_id = None
        start_time = time.time()
        rules_generated = False

        print(f"🚀 Starting interactive ClaudeSDKClient session...")
        mcp_tools = [t for t in options.allowed_tools if 'mcp__' in t]
        print(f"   MCP tools: {len(mcp_tools)}")
        print(f"   MCP tool list: {mcp_tools}")
        print(f"   All allowed: {len(options.allowed_tools)}")
        print(f"   Disallowed: {len(options.disallowed_tools)}")
        
        print("🔍 MCP tools configured and ready")

        try:
            # Use ClaudeSDKClient for proper interactive session
            async with ClaudeSDKClient(options=options) as client:
                # Send initial prompt
                await client.query(initial_prompt)
                
                # Receive and process messages interactively
                async for message in client.receive_messages():
                    if isinstance(message, AssistantMessage):
                        turn_count += 1
                        print(f"   💭 Turn {turn_count}: Claude is working...")
                        
                        # Log tool calls for debugging
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                print(f"      🔧 Tool: {block.name}")
                                
                                # CRITICAL DEBUG: Check if disallowed tools are being used  
                                disallowed = ["Bash", "Write", "Edit", "MultiEdit", "Glob", 
                                            "NotebookEdit", "NotebookRead", "TodoWrite", "WebSearch", "WebFetch"]
                                if block.name in disallowed:
                                    print(f"      ⚠️  ERROR: DISALLOWED TOOL USED: {block.name}")
                                    print(f"         Tool restrictions are NOT working properly!")
                                    print(f"         All args: {json.dumps(block.input, indent=8)}")
                                
                                if hasattr(block, 'input') and block.input:
                                    # Show ALL args for debugging the "shit"
                                    print(f"         Args: {json.dumps(block.input, indent=8)}")
                                    
                            elif isinstance(block, ToolResultBlock):
                                # Show tool results summary for debugging - CRITICAL for MCP debugging
                                print(f"      📋 Tool Result for tool use ID: {getattr(block, 'tool_use_id', 'unknown')}")
                                if hasattr(block, 'content'):
                                    content_str = str(block.content)
                                    print(f"      📋 Result content ({len(content_str)} chars): {content_str[:500]}...")
                                    
                                    # CRITICAL: Check for MCP errors
                                    if "error" in content_str.lower() or "failed" in content_str.lower():
                                        print(f"      🚨 TOOL ERROR DETECTED: {content_str}")
                                else:
                                    print(f"      📋 No content in tool result")
                                    
                            elif isinstance(block, TextBlock):
                                # Show reasoning snippets
                                if len(block.text) > 150:
                                    preview = block.text[:150] + "..."
                                else:
                                    preview = block.text
                                print(f"      💭 {preview}")
                    
                    elif isinstance(message, ResultMessage):
                        # Session complete
                        total_cost_usd = message.total_cost_usd or 0.0
                        session_id = message.session_id
                        print(f"   ✅ Session complete: {turn_count} turns, ${total_cost_usd:.4f}")
                        break
                    
                    # Check if rules were generated
                    if os.path.exists("results/temp/generated_rules.json"):
                        rules_generated = True
        
        except Exception as e:
            print(f"❌ Error during interactive session: {e}")
            raise

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Check for generated rules
        rules_file = "results/temp/generated_rules.json"
        if os.path.exists(rules_file):
            print(f"✅ Rules generated: {rules_file}")
            
            # Copy to final location
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(rules_file, 'r') as src:
                rules_data = json.load(src)
            
            # Add generation info
            rules_data['generation_cost_info'] = {
                'total_cost_usd': total_cost_usd,
                'session_id': session_id,
                'duration_ms': duration_ms,
                'turn_count': turn_count,
                'model': self.model,
                'approach': 'interactive_client'
            }
            
            with open(output_file, 'w') as dst:
                json.dump(rules_data, dst, indent=2)
            
            cost_info = rules_data['generation_cost_info']
            print(f"✅ Final rules saved: {output_file}")
            print(f"📊 {turn_count} turns, {duration_ms/1000:.1f}s, ${total_cost_usd:.4f}")
            
            return output_file, cost_info
        else:
            raise RuntimeError("No rules were generated - interactive session completed but no output file")

    def _create_sample_data(self, dev_results: Dict[str, Any], analysis_data: Optional[Dict] = None) -> SampleData:
        """Create sample data for MCP tools"""
        
        dev_metrics = dev_results.get("metadata", {})
        dev_pairs = dev_results.get("dev_pairs", [])
        
        return SampleData(
            dataset=self.dataset,
            target_f1=self.target_f1,
            dev_metrics=dev_metrics,
            dev_pairs=dev_pairs,
            analysis_insights=analysis_data
        )

    def _save_sample_data_for_mcp(self, sample_data: SampleData):
        """Save sample data to file for MCP server to read"""
        sample_file = f"results/temp/sample_data_{self.dataset}.json"
        os.makedirs(os.path.dirname(sample_file), exist_ok=True)
        
        # Convert to JSON serializable format
        data = {
            "dataset": sample_data.dataset,
            "target_f1": sample_data.target_f1,
            "dev_metrics": sample_data.dev_metrics,
            "dev_pairs": sample_data.dev_pairs,
            "analysis_insights": sample_data.analysis_insights
        }
        
        with open(sample_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Sample data saved: {sample_file}")


async def generate_simplified_heuristics(
    dataset: str, dev_results: Dict[str, Any], output_file: Optional[str] = None, 
    analysis_data: Optional[Dict] = None, model: str = "gpt-4.1-nano", no_cache: bool = False,
    mode: str = "heuristics"
) -> Tuple[str, Dict[str, Any]]:
    """Main entry point for simplified agentic heuristic generation"""
    generator = SimplifiedAgenticGenerator(dataset, model, no_cache, mode)
    return await generator.generate_rules(dev_results, output_file, analysis_data)