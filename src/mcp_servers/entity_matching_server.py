#!/usr/bin/env python3
"""
Entity Matching MCP Server

Custom MCP server that provides specialized tools for entity matching rule generation.
This ensures Claude follows the correct workflow and format requirements.
"""

import asyncio
import difflib
import json
import logging
import os
import pathlib
import sys

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
import pandas as pd

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Local imports
from src.entity_matching.candidate_optimization import find_recall_plateau
from src.entity_matching.experiment_config import ExperimentConfig
from src.entity_matching.experiment_registry import ExperimentRegistry
from src.entity_matching.hybrid_matcher import (
    CandidateCache,
    compute_dataset_embeddings,
    get_top_candidates_cached,
    run_enhanced_matching,
)
from src.experiments.simplified_agentic_generator import get_leaderboard_target_f1
from src.prompts.hybrid_matcher_prompt import build_prompt, get_prompt_data, update_prompt_data
from src.utils.json_serializer import json_serialize

# Set up logging
os.makedirs('results/temp', exist_ok=True)  # Ensure log directory exists

# Get mode from environment variable
SERVER_MODE = os.getenv("MCP_SERVER_MODE", "full")  # full, weights-only, prompt-only, heuristics-only
print(f"🔧 SERVER_MODE detected: {SERVER_MODE} (from env: {os.getenv('MCP_SERVER_MODE')})")
# Enhanced logging configuration for chained command visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/temp/mcp_server.log'),
        logging.StreamHandler(sys.stderr)
    ],
    force=True  # Force reconfiguration to ensure visibility in chained commands
)

# Ensure stdout is line-buffered for immediate output
logger = logging.getLogger("entity-matching-server")

# Log server startup
logger.info("🚀 MCP Entity Matching Server starting up...")
logger.info(f"📁 Working directory: {os.getcwd()}")
print(f"📁 Working directory: {os.getcwd()}")

# Initialize the MCP server
server = Server("entity-matching-server")
logger.info(f"✅ MCP Server initialized in {SERVER_MODE} mode")
print(f"✅ MCP Server initialized in {SERVER_MODE} mode")

def get_tools_for_mode(mode: str) -> List[Tool]:
    """Get tools based on server mode to avoid confusion and focus Claude's attention."""
    all_tools = {
        "WriteRules": Tool(
            name="WriteRules",
            description=(
                "Write traditional heuristic entity matching rules only (candidate, score, decision, weight rules). "
                "Only supports heuristics mode. Use TestWeights for weight changes, WritePrompt for prompt modifications. "
                "Automatically gets baseline configuration from active pipeline registry."
                "Automatically saves to results/temp/generated_rules.json with correct structure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_name": {"type": "string"},
                                "description": {"type": "string"},
                                "implementation": {"type": "string"},
                                "confidence": {"type": "number"},
                                "stage": {"type": "string", "enum": ["candidate_selection", "pre_semantic", "post_semantic", "pre_llm", "post_llm"]}
                            },
                            "required": ["rule_name", "description", "implementation", "confidence", "stage"]
                        },
                        "default": [],
                        "description": "Candidate generation rules"
                    },
                    "score_rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_name": {"type": "string"},
                                "description": {"type": "string"},
                                "implementation": {"type": "string"},
                                "confidence": {"type": "number"},
                                "stage": {"type": "string", "enum": ["candidate_selection", "pre_semantic", "post_semantic", "pre_llm", "post_llm"]}
                            },
                            "required": ["rule_name", "description", "implementation", "confidence", "stage"]
                        },
                        "default": [],
                        "description": "Score adjustment rules"
                    },
                    "decision_rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_name": {"type": "string"},
                                "description": {"type": "string"},
                                "implementation": {"type": "string"},
                                "confidence": {"type": "number"},
                                "stage": {"type": "string", "enum": ["candidate_selection", "pre_semantic", "post_semantic", "pre_llm", "post_llm"]}
                            },
                            "required": ["rule_name", "description", "implementation", "confidence", "stage"]
                        },
                        "default": [],
                        "description": "Decision rules"
                    },
                    "weight_rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_name": {"type": "string"},
                                "description": {"type": "string"},
                                "implementation": {"type": "string"},
                                "confidence": {"type": "number"},
                                "stage": {"type": "string", "enum": ["candidate_selection", "pre_semantic", "post_semantic", "pre_llm", "post_llm"]}
                            },
                            "required": ["rule_name", "description", "implementation", "confidence", "stage"]
                        },
                        "default": [],
                        "description": "Weight adjustment rules"
                    }
                },
                "required": []
            }
        ),
        "TestWeights": Tool(
            name="TestWeights",
            description="""Preview candidate rankings with different similarity weights.

            WORKFLOW: Use this to test different weight combinations until you find what works, then use those weights in RunExperiment.

            RETURNS: Candidate comparison showing how the weights affect the top_n candidates for a sample record_id.

            FEATURES:
            - Shows individual similarity scores (trigram, syntactic, semantic, combined)
            - Pure preview tool - makes NO changes to system state
            - Compare with baseline weights to see impact

            Example: {"semantic_weight": 0.6, "trigram_weight": 0.3, "syntactic_weight": 0.1}""",
            inputSchema={
                "type": "object",
                "properties": {
                    "semantic_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for semantic similarity (required)"
                    },
                    "trigram_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for trigram similarity (required)"
                    },
                    "syntactic_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Weight for syntactic similarity (required)"
                    },
                    "top_n": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 200,
                        "description": "Number of top candidates to show"
                    },
                    "record_id": {
                        "type": "integer",
                        "default": 0,
                        "description": "Record ID to analyze (defaults to 0)"
                    },
                },
                "required": ["semantic_weight", "trigram_weight", "syntactic_weight", "top_n"]
            }
        ),
        "WritePrompt": Tool(
            name="WritePrompt",
            description="""Modify the entity matching prompt structure.

            RETURNS: Success message with section count and COMPLETE UNIFIED DIFF showing exactly what changed.

            DIFF OUTPUT FORMAT:
            - Shows before/after prompt text with unified diff format
            - Lines starting with '-' were removed
            - Lines starting with '+' were added
            - Context lines show unchanged parts
            - NO TRUNCATION - shows all changes for complete visibility

            Use this to see exactly how your prompt modifications affect the actual LLM prompt.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_data": {
                        "type": "object",
                        "properties": {
                            "sections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "description": {"type": "string"},
                                        "ordered_list": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "unordered_list": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    },
                                    "required": ["title"]
                                }
                            }
                        },
                        "required": ["sections"]
                    }
                },
                "required": ["prompt_data"]
            }
        ),
        "ReadPrompt": Tool(
            name="ReadPrompt",
            description="Read the current entity matching prompt structure for modification.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        "RunExperiment": Tool(
            name="RunExperiment",
            description=(
                "Run complete entity matching experiment with full configuration. "
                "Saves experiment config and returns detailed results. "
                "This is the main tool for testing complete configurations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "semantic_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Semantic similarity weight"
                    },
                    "trigram_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Trigram similarity weight (auto-calculated if not provided)"
                    },
                    "syntactic_weight": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Syntactic similarity weight (auto-calculated if not provided)"
                    },
                    "max_candidates": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": "Maximum candidates to consider"
                    },
                    "prompt_data": {
                        "type": "object",
                        "description": "Custom prompt structure (optional, use current prompt if not provided)"
                    },
                    "max_examples": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum FP/FN examples to return"
                    }
                },
                "required": []
            }
        ),
        "ReadSampleData": Tool(
            name="ReadSampleData",
            description="Read and return structured sample data for analysis from the active pipeline.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        "ReadInstructions": Tool(
            name="ReadInstructions",
            description="Get clear, structured instructions for the entity matching task from the active pipeline.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        "ReportIssue": Tool(
            name="ReportIssue",
            description="Report issues with MCP tools or optimization process for debugging help.",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_description": {
                        "type": "string",
                        "description": "Detailed description of the issue or error encountered"
                    }
                },
                "required": ["issue_description"]
            }
        )
    }

    # Define tool sets for different modes
    if mode == "weights-only":
        selected_tools = ["TestWeights", "RunExperiment", "ReadSampleData", "ReadInstructions", "ReportIssue"]
        logger.info("🎯 Mode: weights-only - showing only weight optimization tools")
    elif mode == "prompt-modification":
        selected_tools = ["TestWeights", "WritePrompt", "ReadPrompt", "RunExperiment", "ReadSampleData", "ReadInstructions", "ReportIssue"]
        logger.info("📝 Mode: prompt-modification - showing weight optimization AND prompt modification tools")
    elif mode == "heuristics":
        selected_tools = ["WriteRules", "RunExperiment", "ReadSampleData", "ReadInstructions", "ReportIssue"]
        logger.info("🔧 Mode: heuristics - showing only traditional rule tools")
    else:  # mode == "full"
        selected_tools = list(all_tools.keys())
        logger.info("🌟 Mode: full - showing all available tools")

    return [all_tools[tool_name] for tool_name in selected_tools if tool_name in all_tools]

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available entity matching tools based on server mode."""
    tools = get_tools_for_mode(SERVER_MODE)
    logger.info(f"📋 Tools list requested - returning {len(tools)} tools for {SERVER_MODE} mode")
    return tools

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""

    logger.info(f"🔧 Tool called: {name}")
    logger.info(f"📝 Arguments: {json.dumps(arguments, indent=2)}")

    try:
        if name == "WriteRules":
            result = await write_rules_tool(**arguments)
        elif name == "TestWeights":
            result = await test_weights_tool(**arguments)
        elif name == "RunExperiment":
            result = await run_experiment_tool(**arguments)
        elif name == "ReadSampleData":
            result = await read_sample_data_tool()
        elif name == "ReadInstructions":
            result = await read_instructions_tool()
        elif name == "ReadPrompt":
            result = await read_prompt_tool(**arguments)
        elif name == "WritePrompt":
            result = await write_prompt_tool(**arguments)
        elif name == "ReportIssue":
            issue_description = arguments.get("issue_description", "No description provided")

            # Log to console and logger
            logger.error(f"🚨 CLAUDE REPORTED ISSUE: {issue_description}")

            # Also log to results file
            os.makedirs("results", exist_ok=True)
            issues_file = "results/claude_reported_issues.json"

            issue_entry = {
                "timestamp": datetime.now().isoformat(),
                "issue_description": issue_description,
                "server_mode": os.environ.get("MCP_SERVER_MODE", "unknown"),
                "tool_call_context": name
            }

            # Load existing issues or create new list
            try:
                if os.path.exists(issues_file):
                    with open(issues_file) as f:
                        issues = json.load(f)
                else:
                    issues = []
            except Exception:
                issues = []

            issues.append(issue_entry)

            # Save updated issues
            try:
                with open(issues_file, 'w') as f:
                    json.dump(issues, f, indent=2)
                logger.info(f"📝 Issue logged to: {issues_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to log issue to file: {e}")

            result = [TextContent(type="text", text=f"✅ Issue reported and logged to {issues_file}: {issue_description}")]
        else:
            error_msg = f"Unknown tool: {name}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        logger.info(f"✅ Tool {name} completed successfully")
        logger.info(f"📤 Response: {result[0].text[:200]}..." if result and result[0].text else "📤 Empty response")
        return result

    except Exception as e:
        logger.error(f"❌ Tool {name} failed: {e!s}")
        logger.error(f"💥 Exception type: {type(e).__name__}")

        # Return error to Claude instead of letting it fail silently
        return [TextContent(
            type="text",
            text=f"❌ MCP Tool Error in {name}: {e!s}\n\n"
                 f"Arguments received: {json.dumps(arguments, indent=2)}\n\n"
                 f"Please check the parameters and try again, or use mcp__entity-matching__ReportIssue to get help."
        )]

async def generate_candidate_comparison(
    dataset: str, record_id: int, top_n: int,
    old_semantic: float, old_trigram: float, old_syntactic: float,
    new_semantic: float, new_trigram: float, new_syntactic: float,
    max_candidates: int,
) -> str:
    """Generate candidate comparison showing how weights affect candidate ranking."""
    try:
        # Load dataset
        root = pathlib.Path("data") / "raw" / dataset
        if not root.exists():
            return f"\n⚠️ Dataset '{dataset}' not found\n"

        A_df = pd.read_csv(root / "tableA.csv")
        B_df = pd.read_csv(root / "tableB.csv")

        # Handle ID mapping or list indexing
        if "id" in A_df.columns:
            A = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
            B = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}
        else:
            A = A_df.to_dict(orient="records")
            B = B_df.to_dict(orient="records")

        # Check if record_id exists
        if record_id not in A:
            return f"\n⚠️ Record ID {record_id} not found in dataset '{dataset}'\n"

        left_record = A[record_id]

        # Get embedding model from active pipeline (just like RunExperiment does)
        registry = ExperimentRegistry.find_active_pipeline_registry()
        if not registry or not registry.experiments:
            return "\n❌ No active pipeline found. TestWeights needs an active pipeline to inherit embedding model.\n"
        
        latest_config = registry.experiments[-1].experiment_config
        
        # Create config inheriting embedding model from pipeline
        exp_config = ExperimentConfig(
            dataset=dataset,
            embedding_model=latest_config.embedding_model,
            embedding_base_url=latest_config.embedding_base_url,
            max_candidates=max_candidates
        )
        logger.info(f"📄 TestWeights using config: {exp_config}")

        # Create candidate cache with model-specific path
        cache_key = exp_config.get_cache_key()
        cache_file = f".candidate_cache/{cache_key}.json"
        candidate_cache = CandidateCache(B, cache_file=cache_file)

        # Function to get top candidates with specific weights
        def get_candidates_with_weights(sem_w, tri_w, syn_w):
            # Create config based on experiment config with updated weights
            test_config = ExperimentConfig(
                dataset=exp_config.dataset,
                embedding_model=exp_config.embedding_model,
                embedding_base_url=exp_config.embedding_base_url,
                llm_model=exp_config.llm_model,
                max_candidates=max_candidates,
                semantic_weight=sem_w,
                trigram_weight=tri_w,
                syntactic_weight=syn_w
            )
            cfg = test_config.to_legacy_config()

            # Compute embeddings for semantic similarity if needed
            if cfg.use_semantic:
                cfg.embeddings = compute_dataset_embeddings(dataset, cfg)
            else:
                cfg.embeddings = None

            candidates = get_top_candidates_cached(
                left_record, candidate_cache, max_candidates, cfg, dataset
            )

            # Use CandidateCache's pre-computed similarity scores
            scored_candidates = []
            for idx, candidate_record in candidates[:top_n]:
                # Use the CandidateCache's efficient similarity calculation
                combined_score = candidate_cache.compute_combined_similarity(left_record, json.dumps(left_record, ensure_ascii=False).lower(), idx, cfg)

                # Get individual scores for display
                left_str = json.dumps(left_record, ensure_ascii=False).lower()
                trigram_score = candidate_cache.compute_trigram_similarity(left_str, idx)
                syntactic_score = candidate_cache.compute_syntactic_similarity(left_str, idx)
                semantic_score = candidate_cache.compute_semantic_similarity(left_record, idx, cfg.embeddings) if cfg.embeddings else 0.0

                scored_candidates.append({
                    'idx': idx,
                    'record': candidate_record,
                    'trigram': trigram_score,
                    'syntactic': syntactic_score,
                    'semantic': semantic_score,
                    'combined': combined_score
                })

            return scored_candidates

        # Get candidates with old and new weights
        old_candidates = get_candidates_with_weights(old_semantic, old_trigram, old_syntactic)
        new_candidates = get_candidates_with_weights(new_semantic, new_trigram, new_syntactic)

        # Build comparison text
        comparison = f"\n\n🔍 **CANDIDATE COMPARISON** (Record ID {record_id} from {dataset})\n"
        comparison += f"📄 **Left Record**: {json.dumps(left_record, ensure_ascii=False)[:100]}...\n\n"

        comparison += f"**Old Weights** (semantic={old_semantic:.3f}, trigram={old_trigram:.3f}, syntactic={old_syntactic:.3f}):\n"
        for i, cand in enumerate(old_candidates, 1):
            cand_str = json.dumps(cand['record'], ensure_ascii=False)[:80]
            comparison += f"{i}. Score {cand['combined']:.3f} (trig:{cand['trigram']:.3f}, syn:{cand['syntactic']:.3f}, sem:{cand['semantic']:.3f}) → {cand_str}...\n"

        comparison += f"\n**New Weights** (semantic={new_semantic:.3f}, trigram={new_trigram:.3f}, syntactic={new_syntactic:.3f}):\n"
        for i, cand in enumerate(new_candidates, 1):
            cand_str = json.dumps(cand['record'], ensure_ascii=False)[:80]
            comparison += f"{i}. Score {cand['combined']:.3f} (trig:{cand['trigram']:.3f}, syn:{cand['syntactic']:.3f}, sem:{cand['semantic']:.3f}) → {cand_str}...\n"

        # Show ranking changes
        old_ids = [c['idx'] for c in old_candidates]
        new_ids = [c['idx'] for c in new_candidates]

        if old_ids != new_ids:
            comparison += "\n📊 **Ranking Changes**: "
            changes = []
            for new_pos, cand_id in enumerate(new_ids, 1):
                if cand_id in old_ids:
                    old_pos = old_ids.index(cand_id) + 1
                    if old_pos != new_pos:
                        changes.append(f"ID{cand_id}: {old_pos}→{new_pos}")
                else:
                    changes.append(f"ID{cand_id}: new")

            if changes:
                comparison += ", ".join(changes[:5])  # Show first 5 changes
            else:
                comparison += "Order unchanged"
        else:
            comparison += "\n📊 **Ranking**: Order unchanged, but scores adjusted"

        return comparison

    except Exception as e:
        return f"\n⚠️ Error generating candidate comparison: {e}\n"

async def test_weights_tool(
    semantic_weight: float,
    trigram_weight: float,
    syntactic_weight: float,
    top_n: int,
    record_id: int = 0,
) -> List[TextContent]:
    """Preview candidate rankings with given weights using current active pipeline."""

    try:
        # Get current active pipeline from registry
        registry = ExperimentRegistry.find_active_pipeline_registry()
        if not registry:
            return [TextContent(
                type="text",
                text="❌ No active pipeline found. Use RunExperiment to start a pipeline first."
            )]

        # Get dataset from the active pipeline
        dataset = registry.dataset
        if not dataset:
            return [TextContent(
                type="text",
                text="❌ No dataset found in active pipeline. Run an experiment first to establish dataset."
            )]


        if not registry.experiments:
            return [TextContent(
                type="text",
                text="❌ No experiments found in active pipeline. This is an error."
            )]

        # Get weights from the most recent experiment
        latest_experiment = registry.experiments[-1]
        experiment_config = latest_experiment.experiment_config
        baseline_weights = (
            experiment_config.semantic_weight,
            experiment_config.trigram_weight,
            experiment_config.syntactic_weight
        )
        baseline_max_candidates = experiment_config.max_candidates

        # Show weight comparison
        comparison_text = await generate_candidate_comparison(
            dataset, record_id, top_n,
            baseline_weights[0], baseline_weights[1], baseline_weights[2],
            semantic_weight, trigram_weight, syntactic_weight,
            baseline_max_candidates,
        )

        return [TextContent(
            type="text",
            text=f"📊 WEIGHT PREVIEW for {dataset.upper()}\n"
                 f"Testing: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f}\n"
                 f"Baseline: semantic={baseline_weights[0]:.3f}, trigram={baseline_weights[1]:.3f}, syntactic={baseline_weights[2]:.3f}\n"
                 f"Record: {record_id}, Showing top {top_n} candidates\n\n"
                 f"{comparison_text}\n\n"
                 f"💡 Use these weights in RunExperiment when ready to test full performance."
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ TestWeights failed: {e!s}"
        )]



async def write_rules_tool(
    candidate_rules: List[Dict] = None,
    score_rules: List[Dict] = None,
    decision_rules: List[Dict] = None,
    weight_rules: List[Dict] = None
) -> List[TextContent]:
    """Write traditional heuristic entity matching rules only.

    Only supports heuristics mode. Preserves existing weights and prompt_rules from file.
    Use TestWeights/RunExperiment for weight changes, WritePrompt for prompt modifications.
    Gets baseline configuration from active pipeline registry.

    Args:
        candidate_rules: Python code to generate additional candidates
        score_rules: Python code to boost/penalize similarity scores
        decision_rules: Python code for early accept/reject decisions
        weight_rules: Python code to dynamically adjust similarity weights
    """

    try:
        logger.info("🎯 WriteRules called - heuristics mode only")

        # Try to get baseline configuration from registry
        baseline_config = None
        try:
            registry = ExperimentRegistry.find_active_pipeline_registry()
            if registry and registry.experiments:
                latest_experiment = registry.experiments[-1]
                baseline_config = latest_experiment.experiment_config
                logger.info(f"📊 Using baseline config from active pipeline: {baseline_config.dataset}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get baseline config from registry: {e}")

        # Load existing rules file to preserve weights and prompt_rules
        output_path = "results/temp/generated_rules.json"
        existing_rules = {}

        if os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing_rules = json.load(f)
                logger.info("📁 Loaded existing rules file to preserve weights and prompt_rules")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing rules file: {e}")
                # Use defaults if file is corrupted

        # Extract and preserve existing weights and other settings
        hyperparams = existing_rules.get("hyperparameters", {})
        semantic_weight = hyperparams.get("semantic_weight",
                                         baseline_config.semantic_weight if baseline_config else 0.5)
        trigram_weight = hyperparams.get("trigram_weight",
                                        baseline_config.trigram_weight if baseline_config else 0.25)
        syntactic_weight = hyperparams.get("syntactic_weight",
                                          baseline_config.syntactic_weight if baseline_config else 0.25)
        max_candidates = hyperparams.get("max_candidates",
                                        baseline_config.max_candidates if baseline_config else 100)
        decision_threshold = hyperparams.get("decision_threshold", 0.5)
        auto_accept_threshold = hyperparams.get("auto_accept_threshold", 0.9)
        auto_reject_threshold = hyperparams.get("auto_reject_threshold", 0.1)

        # Preserve existing prompt_rules (don't modify them)
        existing_prompt_rules = existing_rules.get("prompt_rules", [])

        logger.info(f"✅ Preserving weights: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}")
        logger.info(f"✅ Preserving {len(existing_prompt_rules)} existing prompt_rules")
        if baseline_config:
            logger.info(f"✅ Used baseline config from pipeline: {baseline_config.experiment_id}")

        # Calculate weight sum for display
        weight_sum = semantic_weight + trigram_weight + syntactic_weight

        # Validate rule implementations contain Python code, not English
        def validate_rule_implementation(rules, _rule_type):
            if not rules:
                return
            for rule in rules:
                if 'implementation' in rule:
                    impl = rule['implementation']
                    # Check for obvious English phrases that shouldn't be in Python code
                    english_phrases = ['if beer name', 'boost matches where', 'boost by',
                                     'when', 'and are not empty', 'values exactly match']
                    if any(phrase in impl.lower() for phrase in english_phrases):
                        raise ValueError(f"Rule '{rule.get('rule_name', 'unknown')}' has English description instead of Python code: {impl}")

                    # Try to compile as Python (basic syntax check)
                    try:
                        # For code with 'return' statements, wrap in a function
                        if impl.strip().startswith('return'):
                            # Handle multi-line returns by proper indentation
                            indented_impl = '\n'.join('    ' + line for line in impl.split('\n'))
                            test_code = f"def rule_func():\n{indented_impl}"
                            compile(test_code, '<string>', 'exec')
                        else:
                            # Try exec mode first, then eval mode for expressions
                            try:
                                compile(impl, '<string>', 'exec')
                            except SyntaxError:
                                compile(impl, '<string>', 'eval')
                    except SyntaxError as e:
                        raise ValueError(f"Rule '{rule.get('rule_name', 'unknown')}' has invalid Python syntax: {e}")

        # Parse rule arrays if they come as JSON strings
        try:
            if isinstance(candidate_rules, str):
                candidate_rules = json.loads(candidate_rules) if candidate_rules else []
            if isinstance(score_rules, str):
                score_rules = json.loads(score_rules) if score_rules else []
            if isinstance(decision_rules, str):
                decision_rules = json.loads(decision_rules) if decision_rules else []
            if isinstance(weight_rules, str):
                weight_rules = json.loads(weight_rules) if weight_rules else []
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rule arrays: {e}")

        # Set defaults for None values
        candidate_rules = candidate_rules or []
        score_rules = score_rules or []
        decision_rules = decision_rules or []
        weight_rules = weight_rules or []

        # Validate all rule types
        validate_rule_implementation(candidate_rules, "candidate")
        validate_rule_implementation(score_rules, "score")
        validate_rule_implementation(decision_rules, "decision")
        validate_rule_implementation(weight_rules, "weight")

        # Create rules data structure, preserving existing weights and prompt_rules
        rules_data = {
            "hyperparameters": {
                "max_candidates": max_candidates,
                "semantic_weight": semantic_weight,
                "trigram_weight": trigram_weight,
                "syntactic_weight": syntactic_weight,
                "decision_threshold": decision_threshold,
                "auto_accept_threshold": auto_accept_threshold,
                "auto_reject_threshold": auto_reject_threshold
            },
            "candidate_rules": candidate_rules,
            "score_rules": score_rules,
            "decision_rules": decision_rules,
            "weight_rules": weight_rules,
            "prompt_rules": existing_prompt_rules,  # Preserve existing prompt_rules
            "pipeline_rules": existing_rules.get("pipeline_rules", []),  # Preserve other existing rules
            "timestamp": datetime.now().isoformat(),
            "generation_method": "mcp_heuristics_only",
            "mode": "heuristics"
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write file
        with open(output_path, 'w') as f:
            json.dump(rules_data, f, indent=2)

        # Generate response message
        total_rules = len(candidate_rules) + len(score_rules) + len(decision_rules) + len(weight_rules)

        return [TextContent(
            type="text",
            text=f"✅ Heuristic rules saved to {output_path}\n"
                 f"📊 New rules: {len(candidate_rules)} candidate, {len(score_rules)} score, {len(decision_rules)} decision, {len(weight_rules)} weight (total: {total_rules})\n"
                 f"⚖️ Preserved weights: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f} (sum={weight_sum:.3f})\n"
                 f"🧠 Preserved {len(existing_prompt_rules)} existing prompt_rules\n"
                 f"🔧 Mode: Heuristics only - use TestWeights for weights, WritePrompt for prompts"
        )]

    except Exception as e:
        logger.error(f"❌ Error in write_rules_tool: {e}")
        return [TextContent(type="text", text=f"❌ Error writing rules: {e}")]

async def run_experiment_tool(
    semantic_weight: float = None,  # Changed to None so we can detect unspecified
    trigram_weight: float = None,
    syntactic_weight: float = None,
    max_candidates: int = None,  # Changed to None so we can detect unspecified
    prompt_data: dict = None,
    max_examples: int = 20
) -> List[TextContent]:
    """Run complete experiment with full config, inherit from previous experiments when parameters not specified."""

    try:
        # Step 0: Get dataset and config from registry (required - no fallback)
        registry = ExperimentRegistry.find_active_pipeline_registry()
        if not registry:
            return [TextContent(
                type="text",
                text="❌ No active pipeline found. This tool requires an active pipeline registry."
            )]

        if not registry.experiments:
            return [TextContent(
                type="text",
                text="❌ No experiments found in pipeline registry. Cannot inherit configuration."
            )]

        latest_experiment = registry.experiments[-1]
        latest_config = latest_experiment.experiment_config
        logger.info(f"📊 Inheriting configuration from experiment {latest_config.experiment_id}")

        # Step 2: Simple parameter resolution with basic inheritance
        # Use provided values or fall back to base config defaults

        # If no prompt_data provided, use the current prompt data from WritePrompt updates
        if prompt_data is None:
            try:
                final_prompt_data = get_prompt_data()
                if final_prompt_data:
                    logger.info("📝 Using updated prompt data from WritePrompt")
                else:
                    final_prompt_data = None
                    logger.info("📝 No custom prompt data available, using default")
            except Exception as e:
                logger.warning(f"⚠️ Could not get prompt data: {e}")
                final_prompt_data = None
        else:
            final_prompt_data = prompt_data


        # Step 4: Create experiment config with resolved values
        experiment_config = ExperimentConfig(
            # Inherit from base config
            dataset=latest_config.dataset,
            embedding_model=latest_config.embedding_model,  # INHERITED - Claude cannot change
            embedding_base_url=latest_config.embedding_base_url,  # INHERITED - Claude cannot change
            llm_model=latest_config.llm_model,
            concurrency=latest_config.concurrency,
            mode=latest_config.mode,
            semantic_weight=semantic_weight if semantic_weight is not None else latest_config.semantic_weight,
            trigram_weight=trigram_weight if trigram_weight is not None else latest_config.trigram_weight,
            syntactic_weight=syntactic_weight if syntactic_weight is not None else latest_config.syntactic_weight,
            max_candidates=max_candidates if max_candidates is not None else latest_config.max_candidates,
            prompt_data=final_prompt_data,
        )

        logger.info(f"🔧 Resolved parameters: semantic={experiment_config.semantic_weight}, trigram={experiment_config.trigram_weight}, syntactic={experiment_config.syntactic_weight}, candidates={experiment_config.max_candidates}")
        # Step 3: Save experiment config
        config_save_path = experiment_config.save_experiment()
        logger.info(f"💾 Saved experiment config: {config_save_path}")

        # Step 4: Save prompt data if provided
        if prompt_data:
            os.makedirs("results/temp", exist_ok=True)
            with open("results/temp/prompt_data.json", "w") as f:
                json.dump(prompt_data, f, indent=2)
            logger.info("💾 Saved custom prompt data")

        # Step 5: Run the experiment
        logger.info(f"🚀 Running experiment: {experiment_config}")

        # Call run_enhanced_matching with experiment config settings
        results = await run_enhanced_matching(
            experiment=experiment_config,
        )

        # Extract core metrics from structured results
        f1_score = results["f1"]
        precision = results["precision"]
        recall = results["recall"]
        tp = results["failure_analysis"]["summary"]["total_tp"]
        fp = results["failure_analysis"]["summary"]["total_fp"]
        fn = results["failure_analysis"]["summary"]["total_fn"]

        # Get failure analysis data
        false_positives = results["failure_analysis"]["false_positives"]
        false_negatives = results["failure_analysis"]["false_negatives"]
        true_positives = results["failure_analysis"]["true_positives"]

        registry.register_experiment(
            experiment_config,
            "claude_optimization",
            {
                "f1": f1_score,
                "precision": precision,
                "recall": recall,
                "cost_usd": results.get("cost", 0.0),
                "tp": tp,
                "fp": fp,
                "fn": fn
            },
            f"Claude optimization experiment with weights {semantic_weight:.2f}/{trigram_weight:.2f}/{syntactic_weight:.2f}"
        )
        logger.info(f"📋 Registered experiment {experiment_config.experiment_id} with active pipeline registry {registry.pipeline_run_id}")

        # Save experiment results with full tracking
        experiment_results = {
            "experiment_id": experiment_config.experiment_id,
            "config_path": str(config_save_path),
            "results": results,
            "summary": {
                "f1": f1_score,
                "precision": precision,
                "recall": recall,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        }

        # Save to experiment directory using proper JSON serialization
        results_file = experiment_config.get_experiment_dir() / "results.json"
        with open(results_file, "w") as f:
            # Use json_serialize to handle pandas/numpy types properly
            clean_results = json_serialize(experiment_results)
            json.dump(clean_results, f, indent=2)

        # Log complete experiment results
        logger.info(f"🧪 EXPERIMENT {experiment_config.experiment_id} COMPLETE")
        logger.info(f"   📊 Results: F1={f1_score:.4f}, P={precision:.4f}, R={recall:.4f} (TP={tp}, FP={fp}, FN={fn})")
        logger.info(f"   📁 Saved: {config_save_path}")
        logger.info(f"   📊 Results: {results_file}")
        
        if fp == 0 and fn == 0:
            logger.info("   ✅ Perfect results - no errors!")

        result_text = f"🧪 Enhanced Test Results for {experiment_config.dataset} [Experiment {experiment_config.experiment_id}]:\n"

        # Show performance metrics
        result_text += "\n📊 PERFORMANCE METRICS:\n"
        result_text += f"   F1 Score: {f1_score:.4f}\n"
        result_text += f"   Precision: {precision:.4f}\n"
        result_text += f"   Recall: {recall:.4f}\n"

        # Show confusion matrix with interpretation
        result_text += "\n🔍 DETAILED ANALYSIS:\n"
        result_text += f"   Confusion Matrix: TP={tp}, FP={fp}, FN={fn}\n"

        if fp > 0:
            result_text += f"   ⚠️  FALSE POSITIVES: {fp} pairs incorrectly marked as matches\n"

        if fn > 0:
            result_text += f"   ⚠️  FALSE NEGATIVES: {fn} pairs missed (should have matched)\n"

        if tp == 0 and fn > 0:
            result_text += "   🚨 CRITICAL: No true positives found! All matches were missed.\n"

        # Add detailed failure analysis with record details
        if false_positives and len(false_positives) > 0:
            result_text += f"\n🔴 FALSE POSITIVES ({len(false_positives)} cases - showing up to {max_examples}):\n"
            for i, fp in enumerate(false_positives[:max_examples]):
                result_text += f"\n   [{i+1}] Left Record: {json.dumps(fp['left_record'], ensure_ascii=False)}\n"
                result_text += f"       Predicted Match: {json.dumps(fp['predicted_record'], ensure_ascii=False)}\n"
                result_text += f"       Should Match: {json.dumps(fp['actual_record'], ensure_ascii=False)}\n"
                result_text += f"       Predicted Similarity: semantic={fp['predicted_similarity']['semantic']:.3f}, trigram={fp['predicted_similarity']['trigram']:.3f}, syntactic={fp['predicted_similarity']['syntactic']:.3f}\n"
                result_text += f"       Actual Similarity: semantic={fp['actual_similarity']['semantic']:.3f}, trigram={fp['actual_similarity']['trigram']:.3f}, syntactic={fp['actual_similarity']['syntactic']:.3f}\n"

        if false_negatives and len(false_negatives) > 0:
            result_text += f"\n🔴 FALSE NEGATIVES ({len(false_negatives)} cases - showing up to {max_examples}):\n"
            for i, fn_case in enumerate(false_negatives[:max_examples]):
                result_text += f"\n   [{i+1}] Left Record: {json.dumps(fn_case['left_record'], ensure_ascii=False)}\n"
                result_text += f"       Missed Match: {json.dumps(fn_case['missed_record'], ensure_ascii=False)}\n"
                result_text += f"       Similarity: trigram={fn_case['similarity']['trigram']:.3f}, semantic={fn_case['similarity']['semantic']:.3f}\n"
                result_text += f"       Candidate Analysis: found={fn_case['candidate_analysis']['found_in_candidates']}, rank={fn_case['candidate_analysis']['rank']}, max_candidates={fn_case['candidate_analysis']['max_candidates']}\n"
                if not fn_case['candidate_analysis']['found_in_candidates']:
                    result_text += f"       ⚠️ Issue: Match not in top {fn_case['candidate_analysis']['max_candidates']} candidates - increase max_candidates or improve similarity\n"
                elif fn_case['candidate_analysis']['rank'] and fn_case['candidate_analysis']['rank'] > 10:
                    result_text += f"       ⚠️ Issue: Match ranked {fn_case['candidate_analysis']['rank']} - candidate selection working but LLM/rules rejecting\n"

        if true_positives and len(true_positives) > 0:
            result_text += f"\n✅ TRUE POSITIVES ({len(true_positives)} cases - showing sample):\n"
            for i, tp_case in enumerate(true_positives[:min(3, max_examples)]):  # Show fewer TPs since they're working
                result_text += f"\n   [{i+1}] Left Record: {json.dumps(tp_case['left_record'], ensure_ascii=False)}\n"
                result_text += f"       Matched Record: {json.dumps(tp_case['matched_record'], ensure_ascii=False)}\n"
                result_text += f"       Similarity: trigram={tp_case['similarity']['trigram']:.3f}, semantic={tp_case['similarity']['semantic']:.3f}\n"

        if fp == 0 and fn == 0:
            result_text += "   🎉 PERFECT RESULTS - No errors!\n"
        else:
            result_text += "Reflect on these results and improve the weighting and/or prompt."

        logger.info(f"📊 Result text: {result_text}")
        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        logger.error(f"Error in run_experiment_tool: {e}")
        return [TextContent(type="text", text=f"❌ Error running enhanced experiment: {e}")]

async def read_sample_data_tool() -> List[TextContent]:
    """Read and return sample data from active pipeline."""

    try:
        # Get current active pipeline from registry
        registry = ExperimentRegistry.find_active_pipeline_registry()
        if not registry:
            return [TextContent(type="text", text="❌ No active pipeline found. Use RunExperiment to start a pipeline first.")]

        # Get dataset from the active pipeline
        dataset = registry.dataset
        if not dataset:
            return [TextContent(type="text", text="❌ No dataset found in active pipeline.")]

        sample_file = f"results/temp/sample_data_{dataset}.json"

        if not os.path.exists(sample_file):
            return [TextContent(type="text", text=f"❌ Sample data file not found: {sample_file}")]

        with open(sample_file) as f:
            data = json.load(f)

        # Format the data nicely
        summary = f"📊 **SAMPLE DATA ANALYSIS FOR {dataset.upper()}**\n\n"

        # Basic info
        summary += f"🎯 **TARGET**: F1 > {data.get('target_f1', 'unknown')}\n"
        summary += f"📊 **DATASET**: {data.get('dataset', 'unknown')}\n"

        # Analysis insights from the rich sample data
        insights = data.get('analysis_insights', {})
        if insights:
            metadata = insights.get('metadata', {})
            summary += f"📈 **ANALYSIS SCOPE**: {metadata.get('total_pairs_analyzed', 'unknown')} pairs analyzed\n"
            summary += f"✅ **POSITIVE PAIRS**: {metadata.get('positive_pairs', 'unknown')} matches\n"
            summary += f"❌ **NEGATIVE PAIRS**: {metadata.get('negative_pairs_sampled', 'unknown')} non-matches\n\n"

            # Similarity analysis
            true_matches = insights.get('similarity_analysis', {}).get('true_matches', {})
            false_positives = insights.get('similarity_analysis', {}).get('false_positives', {})

            summary += "🔍 **SIMILARITY PATTERNS**:\n"
            for sim_type in ['semantic', 'syntactic', 'trigram']:
                if sim_type in true_matches and sim_type in false_positives:
                    tm_mean = true_matches[sim_type].get('mean', 0)
                    fp_mean = false_positives[sim_type].get('mean', 0)
                    gap = tm_mean - fp_mean
                    summary += f"  • {sim_type.title()}: True matches {tm_mean:.3f} vs False positives {fp_mean:.3f} (gap: {gap:.3f})\n"

            # Candidate recall with optimization analysis
            candidate_analysis = insights.get('candidate_analysis', {})
            if candidate_analysis:
                summary += "\n🎯 **CANDIDATE RECALL & OPTIMIZATION**:\n"

                # Use existing optimal candidate calculation function
                try:
                    optimal_candidates, optimal_recall = find_recall_plateau(candidate_analysis)

                    # Find max candidates for efficiency comparison
                    max_candidates_key = max(
                        [k for k in candidate_analysis if k.startswith('recall_at_')],
                        key=lambda k: int(k.split("_")[-1])
                    )
                    max_candidates = int(max_candidates_key.split("_")[-1])
                    max_recall = candidate_analysis[max_candidates_key]

                    recall_loss = max_recall - optimal_recall
                    efficiency_gain = max_candidates / optimal_candidates if optimal_candidates > 0 else 1

                    summary += f"  🎯 **OPTIMAL**: {optimal_candidates} candidates (recall = {optimal_recall:.1%})\n"
                    summary += f"  📊 **TRADEOFF**: vs {max_candidates} candidates → {recall_loss:.1%} recall loss, {efficiency_gain:.1f}x efficiency gain\n\n"

                except ImportError:
                    optimal_candidates = None
                    summary += "  ⚠️ Could not calculate optimal candidates\n\n"

                # Show detailed recall curve
                for k, v in sorted(candidate_analysis.items()):
                    if k.startswith('recall_at_'):
                        threshold = k.split('_')[-1]
                        marker = " ← OPTIMAL" if optimal_candidates and k == f"recall_at_{optimal_candidates}" else ""
                        summary += f"  • Recall@{threshold}: {v:.1%}{marker}\n"

            # Show concrete examples
            examples = insights.get('concrete_examples', {})
            true_examples = examples.get('true_matches', [])[:3]
            false_examples = examples.get('confusing_non_matches', [])[:3]

            if true_examples:
                summary += "\n✅ **TRUE MATCH EXAMPLES**:\n"
                for i, ex in enumerate(true_examples, 1):
                    left = ex.get('left_record', {})
                    right = ex.get('right_record', {})
                    sims = ex.get('similarities', {})
                    # Get first non-id field for display
                    left_display = next((str(v) for k, v in left.items() if k != 'id' and v), 'N/A')[:50]
                    right_display = next((str(v) for k, v in right.items() if k != 'id' and v), 'N/A')[:50]
                    summary += f"  {i}. Similarities: sem={sims.get('semantic', 0):.3f}, syn={sims.get('syntactic', 0):.3f}, tri={sims.get('trigram', 0):.3f}\n"
                    summary += f"     Left: {left_display}...\n"
                    summary += f"     Right: {right_display}...\n\n"

            if false_examples:
                summary += "❌ **CONFUSING NON-MATCH EXAMPLES**:\n"
                for i, ex in enumerate(false_examples, 1):
                    left = ex.get('left_record', {})
                    right = ex.get('right_record', {})
                    sims = ex.get('similarities', {})
                    left_display = next((str(v) for k, v in left.items() if k != 'id' and v), 'N/A')[:50]
                    right_display = next((str(v) for k, v in right.items() if k != 'id' and v), 'N/A')[:50]
                    summary += f"  {i}. Similarities: sem={sims.get('semantic', 0):.3f}, syn={sims.get('syntactic', 0):.3f}, tri={sims.get('trigram', 0):.3f}\n"
                    summary += f"     Left: {left_display}...\n"
                    summary += f"     Right: {right_display}...\n\n"

        else:
            summary += "⚠️ No analysis insights available\n"

        logger.info(f"📊 Sample data analysis for {dataset}: {summary}")

        return [TextContent(type="text", text=summary)]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error reading sample data: {e}")]

async def read_instructions_tool() -> List[TextContent]:
    """Provide clear task instructions from active pipeline."""

    # Get current active pipeline from registry
    registry = ExperimentRegistry.find_active_pipeline_registry()
    if not registry:
        return [TextContent(type="text", text="❌ No active pipeline found. Use RunExperiment to start a pipeline first.")]

    # Get dataset from the active pipeline
    dataset = registry.dataset
    if not dataset:
        return [TextContent(type="text", text="❌ No dataset found in active pipeline.")]

    # Get target F1 from leaderboard - COMPLETE SET
    leaderboard_targets = {
        "abt_buy": 92.4,
        "amazon_google": 75.0,
        "beer": 95.3,
        "dblp_acm": 96.5,
        "dblp_scholar": 89.8,
        "fodors_zagat": 99.6,
        "itunes_amazon": 85.0,
        "rotten_imdb": 97.2,
        "walmart_amazon": 85.1,
        "zomato_yelp": 98.2
    }

    target_f1 = leaderboard_targets.get(dataset)

    # Define tool instructions based on mode
    if SERVER_MODE == "prompt-modification":
        tools_section = """🔧 **AVAILABLE TOOLS**:
1. **ReadSampleData** - Examine error patterns and current performance
2. **TestWeights** - Preview how weight changes affect candidate ranking
3. **WritePrompt** - Modify the entity matching prompt structure
4. **ReadPrompt** - View current prompt structure
5. **RunExperiment** - Test your changes on validation data

📊 **WORKFLOW**:
1. ReadSampleData → identify issues
2. TestWeights to find good weight combinations or WritePrompt
3. RunExperiment with chosen weights and prompt to test improvements
4. Iterate until F1 > {target_f1}"""
    elif SERVER_MODE == "weights-only":
        tools_section = """🔧 **AVAILABLE TOOLS**:
1. **ReadSampleData** - Examine error patterns and current performance
2. **TestWeights** - Preview how weight changes affect candidate ranking
4. **RunExperiment** - Test your changes on validation data

📊 **WORKFLOW**:
1. ReadSampleData → identify issues
2. TestWeights to find optimal weight combinations
3. RunExperiment with chosen weights to test improvements
4. Iterate until F1 > {target_f1}"""
    else:  # full or heuristics-only
        tools_section = """🔧 **AVAILABLE TOOLS**:
1. **ReadSampleData** - Examine error patterns and current performance
2. **WriteRules** - Create rules (ENFORCES correct format with 3 weights)
3. **TestWeights** - Test weight combinations
4. **RunExperiment** - Test your changes on validation data

📊 **WORKFLOW**:
1. ReadSampleData → identify issues
2. WriteRules to create matching heuristics
3. RunExperiment to test improvements
4. Iterate until F1 > {target_f1}"""

    instructions = f"""🎯 Entity Matching Task for {dataset}

📋 **OBJECTIVE**: Optimize entity matching rules to achieve F1 > {target_f1}
⚠️ **If baseline already exceeds target**: Goal is to MAINTAIN performance, not improve it

{tools_section}

🎯 **OPTIMIZATION STRATEGY**:
- **If baseline EXCEEDS target**: Make minimal/no changes, maintain performance
- **If baseline is CLOSE to target** (within 1-2%): Make only 1 conservative adjustment, don't over-optimize
- **If baseline is NEAR target** (within 5%): Moderate optimization, focus on obvious improvements
- **If baseline is FAR from target**: More aggressive optimization needed

⚠️ **CRITICAL REQUIREMENTS**:
- ALL 3 WEIGHTS REQUIRED: semantic_weight, trigram_weight, syntactic_weight must sum to ~1.0
- VALIDATION ONLY: Always use validation data to prevent test leakage"""

    return [TextContent(type="text", text=instructions)]


async def get_baseline_tool() -> List[TextContent]:
    """Get baseline performance from the active pipeline."""

    try:
        # Get current active pipeline from registry
        registry = ExperimentRegistry.find_active_pipeline_registry()
        if not registry:
            return [TextContent(type="text", text="❌ No active pipeline found. Use RunExperiment to start a pipeline first.")]
        # Get dataset from the active pipeline
        dataset = registry.dataset
        if not dataset:
            return [TextContent(type="text", text="❌ No dataset found in active pipeline.")]
        # Try to find recent dev results
        dev_files = [
            f"results/temp/{dataset}_dev_predictions.json",
            f"results/{dataset}_complete_pipeline.json"
        ]

        for dev_file in dev_files:
            if os.path.exists(dev_file):
                with open(dev_file) as f:
                    data = json.load(f)

                # Try both metadata and metrics keys for compatibility
                metadata = data.get('metadata', {})
                metrics = data.get('metrics', {})

                f1 = metadata.get('f1') or metrics.get('f1', 0.0)
                precision = metadata.get('precision') or metrics.get('precision', 0.0)
                recall = metadata.get('recall') or metrics.get('recall', 0.0)

                baseline = f"📊 Baseline Performance for {dataset}:\n"
                baseline += f"F1: {f1:.4f}\n"
                baseline += f"Precision: {precision:.4f}\n"
                baseline += f"Recall: {recall:.4f}\n"

                # Show what dataset this was evaluated on
                eval_dataset = data.get('dataset', 'unknown')
                baseline += f"Evaluated on: {eval_dataset}\n"

                # Add hyperparameters if available
                max_candidates = data.get('max_candidates', 50)
                baseline += f"Max Candidates: {max_candidates}\n"

                # Extract weights if available (default to typical values if not found)
                semantic_weight = data.get('semantic_weight', 0.5)
                trigram_weight = data.get('trigram_weight', 0.25)
                syntactic_weight = data.get('syntactic_weight', 0.25)

                baseline += f"Baseline Weights: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}\n"
                baseline += "📋 MAKE ONE CHANGE AT A TIME: Either adjust weights OR add prompt rules, not both\n"
                baseline += f"Source: {dev_file}\n"

                # Get target and check if already close
                target_f1 = get_leaderboard_target_f1(dataset) / 100.0  # Convert to decimal
                gap = target_f1 - f1

                if gap <= 0:  # Baseline already exceeds target!
                    baseline += f"\n🎉 BASELINE ALREADY EXCEEDS TARGET ({target_f1:.3f})!\n"
                    baseline += f"Current: {f1:.3f}, Target: {target_f1:.3f} (surplus: {-gap:.3f})\n"
                    baseline += "Strategy: Make MINIMAL changes to maintain performance. Try just 1 iteration.\n"
                    baseline += "For prompt-modification mode: Use very conservative prompt additions only.\n"
                    if eval_dataset != dataset:
                        baseline += f"\n⚠️ WARNING: Baseline was on '{eval_dataset}' but TestRules uses '{dataset}' - results may differ!\n"
                elif gap <= 0.01:  # Within 1% of target
                    baseline += f"\n✅ BASELINE IS CLOSE TO TARGET ({target_f1:.3f})!\n"
                    baseline += f"Gap: {gap:.3f} - only minor optimization needed.\n"
                    baseline += "Strategy: Try 1-2 conservative adjustments, don't over-optimize.\n"
                elif gap <= 0.05:  # Within 5% of target
                    baseline += f"\n⚡ BASELINE IS NEAR TARGET ({target_f1:.3f})\n"
                    baseline += f"Gap: {gap:.3f} - moderate optimization should work.\n"
                else:
                    baseline += f"\n🎯 TARGET: {target_f1:.3f} (gap: {gap:.3f})\n"
                    baseline += "Significant optimization needed.\n"

                return [TextContent(type="text", text=baseline)]

        return [TextContent(type="text", text=f"❌ No baseline results found for {dataset}")]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error getting baseline: {e}")]

async def read_prompt_tool() -> List[TextContent]:
    """Read the current prompt data structure."""
    try:
        prompt_data = get_prompt_data()

        response = "📄 **Current Entity Matching Prompt Structure**\n\n"
        response += "```json\n"
        response += json.dumps(prompt_data, indent=2)
        response += "\n```\n\n"

        response += "**Structure Explanation**:\n"
        response += "- `sections`: Array of prompt sections\n"
        response += "- Each section has:\n"
        response += "  - `title`: Section heading\n"
        response += "  - `description`: Optional explanatory text\n"
        response += "  - `ordered_list`: Optional numbered list (1, 2, 3...)\n"
        response += "  - `unordered_list`: Optional bullet points (-)\n\n"

        response += "**Current Sections**:\n"
        for i, section in enumerate(prompt_data["sections"], 1):
            response += f"{i}. **{section['title']}**\n"
            if section.get("description"):
                response += f"   Description: {section['description']}\n"
            if section.get("ordered_list"):
                response += f"   Ordered list: {len(section['ordered_list'])} items\n"
            if section.get("unordered_list"):
                response += f"   Unordered list: {len(section['unordered_list'])} items\n"

        response += "\n**Usage**: Use WritePrompt to modify this structure."

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error reading prompt: {e}")]

async def write_prompt_tool(prompt_data: dict) -> List[TextContent]:
    """Write/update the prompt data structure with diff output."""
    try:
        # Validate the structure
        if "sections" not in prompt_data:
            return [TextContent(type="text", text="❌ Missing 'sections' key in prompt_data")]

        sections = prompt_data["sections"]
        if not isinstance(sections, list):
            return [TextContent(type="text", text="❌ 'sections' must be an array")]

        # Validate each section
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                return [TextContent(type="text", text=f"❌ Section {i} must be an object")]
            if "title" not in section:
                return [TextContent(type="text", text=f"❌ Section {i} missing required 'title'")]

        # Get current prompt data and build old prompt text
        old_prompt_data = get_prompt_data()

        # Build example prompts to show diff (using sample data)
        sample_left = {"name": "Example Product", "brand": "Example Brand"}
        sample_candidates = "0) {\"name\": \"Similar Product\", \"brand\": \"Example Brand\"}"

        try:
            old_prompt_text = build_prompt(
                left_record=sample_left,
                candidates_text=sample_candidates,
                best_idx=0,
                prompt_data=None,  # Use default behavior for MCP server
                additional_guidance=None
            )
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Failed to build old prompt for comparison: {e}\n\nThis indicates an issue with the current prompt structure. Please check the prompt data integrity before making changes.")]

        # Update the prompt data
        update_prompt_data(prompt_data)

        # Build new prompt text with same sample data
        try:
            new_prompt_text = build_prompt(
                left_record=sample_left,
                candidates_text=sample_candidates,
                best_idx=0,
                prompt_data=None,  # Use default behavior for MCP server
                additional_guidance=None
            )
        except Exception as e:
            # Revert the change since new prompt is broken
            update_prompt_data(old_prompt_data)
            return [TextContent(type="text", text=f"❌ New prompt structure is invalid: {e}\n\nReverted to previous prompt structure. Please fix the prompt data and try again.")]

        # Generate textual diff
        old_lines = old_prompt_text.splitlines(keepends=True)
        new_lines = new_prompt_text.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="Old Prompt",
            tofile="New Prompt",
            lineterm=""
        ))

        # Build response with diff
        response = "✅ **Prompt Structure Updated Successfully**\n\n"
        response += f"**Updated Sections**: {len(sections)}\n"
        for i, section in enumerate(sections, 1):
            response += f"{i}. {section['title']}\n"

        # Add diff output if there are changes
        if diff_lines:
            response += "\n**📝 COMPLETE TEXTUAL DIFF (what changed)**:\n"
            response += "```diff\n"
            # Show ALL diff lines - no truncation for complete visibility
            for line in diff_lines:
                response += line.rstrip() + "\n"
            response += "```\n"
        else:
            response += "\n**📝 No changes detected** (prompt structure identical to previous version)\n"

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error updating prompt: {e}")]

async def report_issue_tool(
    issue_type: str,
    description: str,
    attempted_action: str,
    error_message: str = None
) -> List[TextContent]:
    """Report issues and get debugging help."""

    report = f"🚨 ISSUE REPORT - {issue_type.upper()}\n\n"
    report += f"**Problem**: {description}\n\n"
    report += f"**Attempted Action**: {attempted_action}\n\n"

    if error_message:
        report += f"**Error Message**: {error_message}\n\n"

    report += "📋 **Issue logged for debugging**."

    # Also log to file for debugging
    try:
        log_file = "results/temp/claude_issues.log"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] {issue_type}: {description}\n")
            f.write(f"Attempted: {attempted_action}\n")
            if error_message:
                f.write(f"Error: {error_message}\n")
            f.write("---\n")
    except:
        pass  # Don't fail if logging fails

    return [TextContent(type="text", text=report)]

async def main():
    """Run the MCP server."""

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="entity-matching-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
