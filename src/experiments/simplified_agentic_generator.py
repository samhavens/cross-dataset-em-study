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
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    query,
)


def get_leaderboard_target_f1(dataset: str) -> float:
    """Get the top F1 score from leaderboard.md for the given dataset"""
    leaderboard_targets = {
        "beer": 95.3,
        "itunes_amazon": 85.0,
        "amazon_google": 77.0,
        "dblp_acm": 95.0,
        "walmart_amazon": 85.0,
        "dblp_scholar": 95.0
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
    
    def __init__(self, dataset: str, model: str = "gpt-4.1-nano", no_cache: bool = False):
        self.dataset = dataset
        self.model = model
        self.no_cache = no_cache
        self.target_f1 = get_leaderboard_target_f1(dataset)

    async def generate_rules(self, dev_results: Dict[str, Any], output_file: Optional[str] = None, analysis_data: Optional[Dict] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate rules using clean MCP-based approach"""
        
        if output_file is None:
            output_file = f"results/generated_rules/{self.dataset}_mcp_generated_config.json"
        
        print(f"🚀 SIMPLIFIED AGENTIC HEURISTIC GENERATION FOR {self.dataset}")
        print("============================================================")
        print(f"🎯 Target: F1 > {self.target_f1}")
        
        # Create sample data and save it for MCP server to use
        sample_data = self._create_sample_data(dev_results, analysis_data)
        self._save_sample_data_for_mcp(sample_data)
        
        # Clean prompt
        prompt = f"""You are an entity matching expert optimizing the {self.dataset} dataset.

**Goal**: Achieve F1 > {self.target_f1}

**IMPORTANT**: You have access to specialized MCP tools for this task. Use them instead of generic tools.

**MCP Tools Available**:
- mcp__entity-matching__ReadInstructions - Get task details for {self.dataset}
- mcp__entity-matching__ReadSampleData - See error patterns for {self.dataset}
- mcp__entity-matching__GetBaseline - Check current performance for {self.dataset}
- mcp__entity-matching__WriteRules - Save optimized rules (enforces correct format)
- mcp__entity-matching__TestRules - Test your rules on {self.dataset}
- mcp__entity-matching__AnalyzePerformance - Get improvement suggestions

**Other Tools**:
- Read - Read any files you need
- Task, Grep, Glob, LS - For searching/navigation if needed

**Workflow**: 
1. Use mcp__entity-matching__ReadInstructions to understand the task
2. Use mcp__entity-matching__ReadSampleData to see what's failing
3. Use mcp__entity-matching__GetBaseline to see current metrics
4. Create rules with mcp__entity-matching__WriteRules (must include semantic_weight, trigram_weight, syntactic_weight)
5. Use mcp__entity-matching__TestRules to check performance
6. Repeat until F1 > {self.target_f1}

Start with mcp__entity-matching__ReadInstructions.
"""

        # Configure with only MCP tools + Read
        from claude_code_sdk.types import McpStdioServerConfig
        
        options = ClaudeCodeOptions(
            mcp_servers={
                "entity-matching": McpStdioServerConfig(
                    command="python",
                    args=["src/mcp_servers/entity_matching_server.py"],
                    env={"PYTHONPATH": "."}
                )
            },
            # Explicitly allow MCP tools + basic tools, disallow dangerous ones
            allowed_tools=[
                "Read",
                "Task", 
                "Grep", 
                "Glob",
                "LS",
                "mcp__entity-matching__WriteRules",
                "mcp__entity-matching__TestRules", 
                "mcp__entity-matching__ReadSampleData",
                "mcp__entity-matching__ReadInstructions",
                "mcp__entity-matching__AnalyzePerformance",
                "mcp__entity-matching__GetBaseline"
            ],
            disallowed_tools=[
                "Bash",
                "Write", 
                "Edit",
                "MultiEdit",
                "NotebookEdit",
                "NotebookRead"
            ],
            permission_mode="acceptEdits",
            cwd=os.getcwd(),
        )

        # Run the session
        turn_count = 0
        total_cost_usd = 0.0
        session_id = None
        start_time = time.time()

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    turn_count += 1
                    print(f"   💭 Turn {turn_count}: Claude is thinking and taking actions...")
                    
                    # Detailed logging of tool calls for debugging
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            print(f"      🔧 TOOL CALL: {block.name}")
                            if hasattr(block, 'input') and block.input:
                                # Show full input for debugging
                                print(f"         📝 Args: {json.dumps(block.input, indent=4)}")
                        
                        elif isinstance(block, TextBlock):
                            # Show Claude's reasoning
                            if len(block.text) > 200:
                                preview = block.text[:200] + "..."
                                print(f"      🔤 {preview}")
                            else:
                                print(f"      🔤 {block.text}")

                elif isinstance(message, ResultMessage):
                    # Capture final results
                    total_cost_usd = message.total_cost_usd or 0.0
                    session_id = message.session_id
                    break

        except Exception as e:
            print(f"❌ Error during session: {e}")
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
                'model': self.model
            }
            
            with open(output_file, 'w') as dst:
                json.dump(rules_data, dst, indent=2)
            
            cost_info = rules_data['generation_cost_info']
            print(f"✅ Final rules saved: {output_file}")
            print(f"📊 {turn_count} turns, {duration_ms/1000:.1f}s, ${total_cost_usd:.4f}")
            
            return output_file, cost_info
        else:
            raise RuntimeError("No rules were generated")

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
    analysis_data: Optional[Dict] = None, model: str = "gpt-4.1-nano", no_cache: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """Main entry point for simplified agentic heuristic generation"""
    generator = SimplifiedAgenticGenerator(dataset, model, no_cache)
    return await generator.generate_rules(dev_results, output_file, analysis_data)