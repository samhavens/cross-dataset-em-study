#!/usr/bin/env python3
"""
Simplified agentic heuristic generator with only MCP tools + Read
"""

import json
import os
import re
import sys
import time

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from src.entity_matching.candidate_optimization import get_optimal_candidates_for_dataset
from src.utils.patch_claude_sdk_transport import apply_patch


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

    def __init__(self, dataset: str, model: str = "claude-3-5-sonnet-20241022", no_cache: bool = False, mode: str = "heuristics", embedding_model: str = "all-MiniLM-L6-v2", embedding_base_url: str = None):
        self.dataset = dataset
        self.model = model
        self.no_cache = no_cache
        self.mode = mode
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.target_f1 = get_leaderboard_target_f1(dataset)

    async def generate_rules(self, dev_results: Dict[str, Any], output_file: Optional[str] = None, analysis_data: Optional[Dict] = None, optimal_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate rules using proper ClaudeSDKClient for interactive sessions"""

        if output_file is None:
            output_file = f"results/generated_rules/{self.dataset}_mcp_generated_config.json"

        print(f"🚀 SIMPLIFIED AGENTIC HEURISTIC GENERATION FOR {self.dataset}")
        print("============================================================")
        print(f"🎯 Target: F1 > {self.target_f1}")

        # Create sample data and save it for MCP server to use
        sample_data = self._create_sample_data(dev_results, analysis_data)
        self._save_sample_data_for_mcp(sample_data)

        # Initialize baseline rules file with optimal parameters
        self._initialize_baseline_rules(optimal_params)

        mcp_mode = self.mode
        # Configure with proper MCP tools and restrictions
        # McpServerConfig already imported above

        # Load MCP server config from file
        with open('mcp_config.json') as f:
            mcp_config = json.load(f)

        # Update MCP config to include the server mode environment variable and embedding config
        mcp_servers = mcp_config['mcpServers']
        if 'entity-matching' in mcp_servers:
            if 'env' not in mcp_servers['entity-matching']:
                mcp_servers['entity-matching']['env'] = {}
            mcp_servers['entity-matching']['env']['MCP_SERVER_MODE'] = mcp_mode
            mcp_servers['entity-matching']['env']['PYTHONPATH'] = '.'
            mcp_servers['entity-matching']['env']['EMBEDDING_MODEL'] = self.embedding_model
            if self.embedding_base_url:
                mcp_servers['entity-matching']['env']['EMBEDDING_BASE_URL'] = self.embedding_base_url

        allowed_tools=[
            "Read",  # Allow reading code files for analysis
            "Task",  # Allow planning and task management
            "LS",    # Allow listing directories for navigation
            "Grep",  # Allow searching through files
            "mcp__entity-matching__ReadInstructions",
            "mcp__entity-matching__ReadSampleData",
            "mcp__entity-matching__GetBaseline",
            "mcp__entity-matching__RunExperiment",
            "mcp__entity-matching__ReportIssue",
            "mcp__entity-matching__WriteWeights",
        ]

        # Initial prompt for the interactive session - different for each mode
        initial_prompt = "You are optimizing an entity matching system. It works by finding potential matching records by embedding, sequence-diff, and trigram similarity, " + \
        "then sending max_candidates to an LLM to select the best match by an integer match index. You will be shown various metrics and example true and false positive matches. " + \
        "You are iterating on the dev set, but the target is on the test set. So the scores may not perfectly align.\n\n"

        if self.mode == "weights-only":
            initial_prompt += f"""WEIGHTS-ONLY OPTIMIZATION MODE: Focus only on optimizing semantic/trigram/syntactic weights.

⚠️ CRITICAL: If ANY MCP tool fails or returns an error, IMMEDIATELY call mcp__entity-matching__ReportIssue to report the problem. Do not continue without reporting tool failures.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset weights for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
3. mcp__entity-matching__WriteWeights - ONLY change weights, NO rules
4. mcp__entity-matching__RunExperiment - Test on full dev set
5. Iterate quickly on weight combinations

FOCUS: Find optimal semantic_weight, trigram_weight, syntactic_weight (must sum to 1.0).
DO NOT generate candidate_rules, score_rules, decision_rules, or prompt_rules - weights only!"""

        elif self.mode == "prompt-modification":
            allowed_tools.append("mcp__entity-matching__WritePrompt")
            allowed_tools.append("mcp__entity-matching__ReadPrompt")

            initial_prompt += f"""PROMPT-MODIFICATION OPTIMIZATION MODE: Optimize weights AND modify the prompt structure.

⚠️ CRITICAL: If ANY MCP tool fails or returns an error, IMMEDIATELY call mcp__entity-matching__ReportIssue to report the problem. Do not continue without reporting tool failures.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
3a. mcp__entity-matching__WriteWeights - Optimize similarity weights (semantic_weight, trigram_weight, syntactic_weight must sum to 1.0)
3b.1. mcp__entity-matching__ReadPrompt - Read current prompt structure
3b.2. mcp__entity-matching__WritePrompt - Modify prompt structure for better LLM guidance
4. mcp__entity-matching__RunExperiment - Test on dev set and analyze detailed failure modes
5. Repeat steps 3 (a or b1 & 2) and 4 to iterate on both weights and prompt modifications, but only do one or the other at a time to be a good scientist.

FOCUS:
- Find optimal semantic_weight, trigram_weight, syntactic_weight (must sum to 1.0) using WriteWeights
- Modify the prompt structure using WritePrompt to give better LLM guidance for specific matching scenarios
- Use WriteWeights for weight changes, WritePrompt for prompt modifications - these are separate tools!"""

        else:  # mode == "heuristics"
            allowed_tools.append("mcp__entity-matching__WriteRules")

            initial_prompt += f"""HEURISTICS OPTIMIZATION MODE: Full rule generation with traditional heuristic rules.

⚠️ CRITICAL: If ANY MCP tool fails or returns an error, IMMEDIATELY call mcp__entity-matching__ReportIssue to report the problem. Do not continue without reporting tool failures.

START by calling: mcp__entity-matching__ReadInstructions with dataset "{self.dataset}"

Goal: Optimize {self.dataset} dataset for F1 > {self.target_f1}

WORKFLOW:
1. mcp__entity-matching__ReadInstructions (dataset: "{self.dataset}") - START HERE
2. mcp__entity-matching__ReadSampleData (dataset: "{self.dataset}") - Analyze error patterns
3. mcp__entity-matching__GetBaseline (dataset: "{self.dataset}") - Check current performance
4. mcp__entity-matching__WriteRules - Generate optimized weights AND custom rules
5. mcp__entity-matching__RunExperiment - Test on full dev set

FOCUS: Optimize weights AND create custom rules based on data analysis."""

        # Track session metrics
        turn_count = 0
        total_cost_usd = 0.0
        session_id = None
        start_time = time.time()

        options = ClaudeCodeOptions(
            mcp_servers=mcp_servers,
            # Allow MCP tools + essential planning/analysis tools
            allowed_tools=allowed_tools,
            # Explicitly disallow system tools (except Read, Task, LS, Grep which are allowed above)
            disallowed_tools=[
                "Bash", "Write", "Edit", "MultiEdit", "Glob",
                "NotebookEdit", "NotebookRead", "TodoWrite",
                "WebSearch", "WebFetch"  # Block web access
            ],
            permission_mode="acceptEdits",
            cwd=os.getcwd(),
            max_turns=30
        )

        # Environment variable is already set via MCP config above
        os.environ["MCP_SERVER_MODE"] = mcp_mode

        print("🚀 Starting interactive ClaudeSDKClient session...")
        print(f"  MCP_SERVER_MODE: {os.environ.get('MCP_SERVER_MODE')}")
        mcp_tools = [t for t in options.allowed_tools if 'mcp__' in t]
        print(f"   MCP tools: {len(mcp_tools)}")
        print(f"   MCP tool list: {mcp_tools}")
        print(f"   All allowed: {len(options.allowed_tools)}")
        print(f"   Disallowed: {len(options.disallowed_tools)}")
        print(f"   MCP server mode: {os.environ.get('MCP_SERVER_MODE', 'full')}")

        print("🔍 MCP tools configured and ready")

        # Apply SDK transport patch to prevent stderr deadlock
        apply_patch(redirect_stderr_to_parent=True, remove_verbose_flag=False)
        # Use ClaudeSDKClient for proper interactive session
        async with ClaudeSDKClient(options=options) as client:
            print("Starting ClaudeSDKClient session...")
            # Send initial prompt
            await client.query(initial_prompt)
            print("Initial prompt sent")

            # Receive and process messages interactively with debugging
            print("Starting receive_messages loop...")
            # Claude is working (visible in MCP server logs) but messages aren't reaching parent process
            try:
                message_received = False
                async for message in client.receive_messages():
                    message_received = True
                    print(f"Received message type: {type(message).__name__}")
                    if isinstance(message, AssistantMessage):
                        turn_count += 1
                        print(f"   💭 Turn {turn_count}: Claude is working...")
                        sys.stdout.flush()

                        # Log tool calls for debugging
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                print(f"      🔧 Tool: {block.name}")
                                sys.stdout.flush()

                                # CRITICAL DEBUG: Check if disallowed tools are being used
                                disallowed = ["Bash", "Write", "Edit", "MultiEdit", "Glob",
                                            "NotebookEdit", "NotebookRead", "TodoWrite", "WebSearch", "WebFetch"]
                                if block.name in disallowed:
                                    print(f"      ⚠️  ERROR: DISALLOWED TOOL USED: {block.name}")
                                    print("         Tool restrictions are NOT working properly!")
                                    print(f"         All args: {json.dumps(block.input, indent=8)}")

                                if hasattr(block, 'input') and block.input:
                                    # Show ALL args for debugging the "shit"
                                    print(f"         Args: {json.dumps(block.input, indent=8)}")

                                    # Highlight weight changes for easy tracking
                                    if block.name == "mcp__entity-matching__WriteWeights":
                                        sem = block.input.get('semantic_weight')
                                        tri = block.input.get('trigram_weight')
                                        syn = block.input.get('syntactic_weight')
                                        candidates = block.input.get('max_candidates')
                                        if sem and tri and syn:
                                            print(f"         💫 WEIGHTS: sem={sem}, tri={tri}, syn={syn}, candidates={candidates}")

                            elif isinstance(block, ToolResultBlock):
                                # Show tool results summary for debugging - CRITICAL for MCP debugging
                                print(f"      📋 Tool Result for tool use ID: {getattr(block, 'tool_use_id', 'unknown')}")
                                if hasattr(block, 'content'):
                                    content_str = str(block.content)
                                    print(f"      📋 Result content ({len(content_str)} chars): {content_str[:500]}...")

                                    # Extract and highlight F1 scores from RunExperiment results
                                    if "F1 SCORE:" in content_str or "F1 Score:" in content_str:
                                        f1_matches = re.findall(r'F1 SCORE?:\s*([0-9.]+)', content_str, re.IGNORECASE)
                                        if f1_matches:
                                            f1_score = float(f1_matches[0])
                                            print(f"      🎯 F1 SCORE: {f1_score:.4f}")

                                        # Also extract precision/recall if available
                                        precision_matches = re.findall(r'PRECISION:\s*([0-9.]+)', content_str, re.IGNORECASE)
                                        recall_matches = re.findall(r'RECALL:\s*([0-9.]+)', content_str, re.IGNORECASE)
                                        if precision_matches and recall_matches:
                                            precision = float(precision_matches[0])
                                            recall = float(recall_matches[0])
                                            print(f"      📊 P={precision:.4f}, R={recall:.4f}")

                                    # CRITICAL: Check for MCP errors
                                    if "error" in content_str.lower() or "failed" in content_str.lower():
                                        print(f"      🚨 TOOL ERROR DETECTED: {content_str}")
                                else:
                                    print("      📋 No content in tool result")

                            elif isinstance(block, TextBlock):
                                # Show reasoning snippets
                                preview = block.text
                                print(f"      💭 {preview}")

                    elif isinstance(message, ResultMessage):
                        # Session complete
                        total_cost_usd = message.total_cost_usd or 0.0
                        session_id = message.session_id
                        print(f"   ✅ Session complete: {turn_count} turns, ${total_cost_usd:.4f}")
                        break

                if not message_received:
                    print("⚠️  No messages received from Claude, but MCP server shows activity")
                    print("   This suggests a message streaming issue in ClaudeSDKClient")

            except Exception as e:
                print(f"🚨 Error in message loop: {e}")
                import traceback
                traceback.print_exc()
                raise


        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Check for generated rules
        rules_file = "results/temp/generated_rules.json"
        if os.path.exists(rules_file):
            print(f"✅ Rules generated: {rules_file}")

            # Copy to final location
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(rules_file) as src:
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
            print(f"📊 Claude optimization complete: {turn_count} turns, {duration_ms/1000:.1f}s, ${total_cost_usd:.4f}")

            return output_file, cost_info
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

    def _initialize_baseline_rules(self, optimal_params: Optional[Dict[str, Any]]):
        """Initialize baseline rules file with optimal parameters from pipeline"""
        rules_file = "results/temp/generated_rules.json"
        os.makedirs(os.path.dirname(rules_file), exist_ok=True)

        # Use provided optimal_params or sensible defaults
        if optimal_params:
            max_candidates = optimal_params.get("max_candidates", 150)
            semantic_weight = optimal_params.get("semantic_weight", 0.6)
            trigram_weight = optimal_params.get("trigram_weight", 0.2)
            syntactic_weight = optimal_params.get("syntactic_weight", 0.2)
        else:
            # Fallback to 3-weight system defaults

            optimal_candidates = get_optimal_candidates_for_dataset(self.dataset)
            max_candidates = optimal_candidates if optimal_candidates else 150
            semantic_weight = 0.6
            trigram_weight = 0.2
            syntactic_weight = 0.2

        baseline_rules = {
            "hyperparameters": {
                "max_candidates": max_candidates,
                "semantic_weight": semantic_weight,
                "trigram_weight": trigram_weight,
                "syntactic_weight": syntactic_weight,
                "decision_threshold": 0.5,
                "auto_accept_threshold": 0.9,
                "auto_reject_threshold": 0.1
            },
            "candidate_rules": [],
            "score_rules": [],
            "decision_rules": [],
            "weight_rules": [],
            "prompt_rules": [],
            "pipeline_rules": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generation_method": "baseline_initialization",
            "mode": self.mode
        }

        with open(rules_file, 'w') as f:
            json.dump(baseline_rules, f, indent=2)

        print(f"📝 Baseline rules initialized: candidates={max_candidates}, semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}")


async def generate_simplified_heuristics(
    dataset: str, dev_results: Dict[str, Any], output_file: Optional[str] = None,
    analysis_data: Optional[Dict] = None, model: str = "gpt-4.1-nano", no_cache: bool = False,
    mode: str = "heuristics", optimal_params: Optional[Dict[str, Any]] = None,
    embedding_model: str = "all-MiniLM-L6-v2", embedding_base_url: str = None
) -> Tuple[str, Dict[str, Any]]:
    """Main entry point for simplified agentic heuristic generation"""
    generator = SimplifiedAgenticGenerator(dataset, model, no_cache, mode, embedding_model, embedding_base_url)
    return await generator.generate_rules(dev_results, output_file, analysis_data, optimal_params)
