#!/usr/bin/env python3
"""
Entity Matching MCP Server

Custom MCP server that provides specialized tools for entity matching rule generation.
This ensures Claude follows the correct workflow and format requirements.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent
from datetime import datetime

# Set up logging
os.makedirs('results/temp', exist_ok=True)  # Ensure log directory exists
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/temp/mcp_server.log'),
        logging.StreamHandler(sys.stderr)  # Also log to stderr so it shows up in Claude Code SDK logs
    ]
)
logger = logging.getLogger("entity-matching-server")

# Log server startup
logger.info("🚀 MCP Entity Matching Server starting up...")
logger.info(f"📁 Working directory: {os.getcwd()}")

# Initialize the MCP server
server = Server("entity-matching-server")
logger.info("✅ MCP Server initialized")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List all available entity matching tools."""
    logger.info("📋 Tools list requested - returning 7 entity matching tools")
    return [
        Tool(
            name="WriteRules",
            description=(
                "Write entity matching rules in the enforced correct format. "
                "REQUIRES all 3 weights: semantic_weight, trigram_weight, syntactic_weight. "
                "Rule types available depend on mode: weights-only (no rules), prompt-modification (only prompt_rules), heuristics (traditional rules). "
                "Automatically saves to results/temp/generated_rules.json with correct structure."
            ),
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
                    "max_candidates": {
                        "type": "integer", 
                        "minimum": 10, 
                        "maximum": 500,
                        "default": 100,
                        "description": "Maximum number of candidates to consider"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["weights-only", "prompt-modification", "heuristics"],
                        "default": "heuristics",
                        "description": "Optimization mode: weights-only (weights only), prompt-modification (weights + prompt rules), heuristics (traditional rules)"
                    },
                    "prompt_rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "rule_name": {"type": "string"},
                                "description": {"type": "string"},
                                "condition": {
                                    "type": "string", 
                                    "description": "Python expression that evaluates to True/False. Available variables: left_record (dict), right_record (dict), candidate_record (dict). Examples: \"'brand' in left_record\", \"left_record.get('name', '').lower().startswith('sony')\", \"True\" (always apply)"
                                },
                                "prompt_addition": {"type": "string"},
                                "stage": {"type": "string", "enum": ["pre_llm"], "default": "pre_llm"}
                            },
                            "required": ["rule_name", "description", "condition", "prompt_addition"]
                        },
                        "default": [],
                        "description": "Prompt modification rules (only for prompt-modification mode). Each rule needs a Python condition that evaluates to True/False using available variables: left_record, right_record, candidate_record (all dicts)."
                    },
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
                        "description": "Candidate generation rules (only for heuristics mode)"
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
                        "description": "Score adjustment rules (only for heuristics mode)"
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
                        "description": "Decision rules (only for heuristics mode)"
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
                        "description": "Weight adjustment rules (only for heuristics mode)"
                    }
                },
                "required": ["semantic_weight", "trigram_weight", "syntactic_weight"]
            }
        ),
        Tool(
            name="TestRules",
            description=(
                "Test entity matching rules and return structured performance results. "
                "Runs the pipeline with the generated rules on the full dev set and returns F1, precision, recall."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name (e.g., 'beer', 'itunes_amazon')"
                    },
                    "rules_file": {
                        "type": "string",
                        "default": "results/temp/generated_rules.json",
                        "description": "Path to rules file to test"
                    },
                    "validation_only": {
                        "type": "boolean",
                        "default": True,
                        "description": "Use validation set to prevent test leakage"
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="ReadSampleData",
            description="Read and return structured sample data for analysis.",
            inputSchema={
                "type": "object", 
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name"
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="ReadInstructions", 
            description="Get clear, structured instructions for the entity matching task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name"
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="AnalyzePerformance",
            description="Analyze current performance vs target and suggest improvements.",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_f1": {
                        "type": "number",
                        "description": "Current F1 score"
                    },
                    "target_f1": {
                        "type": "number", 
                        "description": "Target F1 score"
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name"
                    }
                },
                "required": ["current_f1", "target_f1", "dataset"]
            }
        ),
        Tool(
            name="GetBaseline",
            description="Get current baseline performance metrics for the dataset.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Dataset name"
                    }
                },
                "required": ["dataset"]
            }
        ),
        Tool(
            name="ReportIssue",
            description="Report problems or confusion to help debug the rule generation process.",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "enum": ["tool_confusion", "parameter_error", "validation_failure", "unclear_requirements", "other"],
                        "description": "Type of issue you're experiencing"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the problem"
                    },
                    "attempted_action": {
                        "type": "string",
                        "description": "What you were trying to do when the issue occurred"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Any error message received (optional)"
                    }
                },
                "required": ["issue_type", "description", "attempted_action"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    
    logger.info(f"🔧 Tool called: {name}")
    logger.info(f"📝 Arguments: {json.dumps(arguments, indent=2)}")
    
    try:
        if name == "WriteRules":
            result = await write_rules_tool(**arguments)
        elif name == "TestRules":
            result = await test_rules_tool(**arguments)
        elif name == "ReadSampleData":
            result = await read_sample_data_tool(**arguments)
        elif name == "ReadInstructions":
            result = await read_instructions_tool(**arguments)
        elif name == "AnalyzePerformance":
            result = await analyze_performance_tool(**arguments)
        elif name == "GetBaseline":
            result = await get_baseline_tool(**arguments)
        elif name == "ReportIssue":
            result = await report_issue_tool(**arguments)
        else:
            error_msg = f"Unknown tool: {name}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"✅ Tool {name} completed successfully")
        logger.info(f"📤 Response: {result[0].text[:200]}..." if result and result[0].text else "📤 Empty response")
        return result
        
    except Exception as e:
        logger.error(f"❌ Tool {name} failed: {str(e)}")
        logger.error(f"💥 Exception type: {type(e).__name__}")
        
        # Return error to Claude instead of letting it fail silently
        error_response = [TextContent(
            type="text", 
            text=f"❌ MCP Tool Error in {name}: {str(e)}\n\n"
                 f"Arguments received: {json.dumps(arguments, indent=2)}\n\n"
                 f"Please check the parameters and try again, or use mcp__entity-matching__ReportIssue to get help."
        )]
        return error_response

async def write_rules_tool(
    semantic_weight: float,
    trigram_weight: float,
    syntactic_weight: float,
    max_candidates: int = 100,
    candidate_rules: List[Dict] = None,
    score_rules: List[Dict] = None,
    decision_rules: List[Dict] = None,
    weight_rules: List[Dict] = None,
    prompt_rules: List[Dict] = None,
    mode: str = "heuristics"
) -> List[TextContent]:
    """Write rules in enforced correct format. 
    
    Args:
        mode: Optimization mode - "weights-only", "prompt-modification", or "heuristics"
              - weights-only: Only update weights, keep all rule arrays empty
              - prompt-modification: Update weights + prompt_rules only, keep other rules empty  
              - heuristics: Full rule generation (traditional mode)
    """
    
    try:
        logger.info(f"🎯 WriteRules called with semantic_weight={semantic_weight}, trigram_weight={trigram_weight}, syntactic_weight={syntactic_weight}")
        
        # Convert string parameters to proper types if needed
        try:
            semantic_weight = float(semantic_weight)
            trigram_weight = float(trigram_weight) 
            syntactic_weight = float(syntactic_weight)
            max_candidates = int(max_candidates)
            logger.info(f"✅ Parameters converted successfully")
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Parameter conversion failed: {e}")
            raise ValueError(f"Parameter type error: {e}. semantic_weight, trigram_weight, syntactic_weight must be numbers 0.0-1.0, max_candidates must be integer")
        
        # Validate weights
        if not (0.0 <= semantic_weight <= 1.0):
            raise ValueError(f"semantic_weight must be 0.0-1.0, got {semantic_weight}")
        if not (0.0 <= trigram_weight <= 1.0):
            raise ValueError(f"trigram_weight must be 0.0-1.0, got {trigram_weight}")
        if not (0.0 <= syntactic_weight <= 1.0):
            raise ValueError(f"syntactic_weight must be 0.0-1.0, got {syntactic_weight}")
        
        # Check weights sum approximately to 1.0
        weight_sum = semantic_weight + trigram_weight + syntactic_weight
        if not (0.95 <= weight_sum <= 1.05):
            raise ValueError(f"Weights should sum to ~1.0, got {weight_sum:.3f}")
        
        # Validate rule implementations contain Python code, not English
        def validate_rule_implementation(rules, rule_type):
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
                        # Use exec mode for statements (like "return 0.1"), fallback to eval for expressions
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
            if isinstance(prompt_rules, str):
                prompt_rules = json.loads(prompt_rules) if prompt_rules else []
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rule arrays: {e}")
        
        # MODE-SPECIFIC VALIDATION: Prevent inappropriate rule types
        if mode == "weights-only":
            # Only weights allowed, no rules at all
            if candidate_rules or score_rules or decision_rules or weight_rules or prompt_rules:
                raise ValueError(f"weights-only mode does not allow any rules. Remove all rule parameters and provide only semantic_weight, trigram_weight, syntactic_weight.")
        elif mode == "prompt-modification":
            # Only prompt_rules allowed, no traditional rules
            if candidate_rules or score_rules or decision_rules or weight_rules:
                raise ValueError(f"prompt-modification mode only allows prompt_rules. Remove candidate_rules, score_rules, decision_rules, and weight_rules. Use prompt_rules to provide dynamic LLM guidance.")
        elif mode == "heuristics":
            # Traditional rules allowed, no prompt_rules
            if prompt_rules:
                raise ValueError(f"heuristics mode does not allow prompt_rules. Use candidate_rules, score_rules, decision_rules, and weight_rules instead.")
        
        # Validate all rule types
        validate_rule_implementation(candidate_rules, "candidate")
        validate_rule_implementation(score_rules, "score") 
        validate_rule_implementation(decision_rules, "decision")
        validate_rule_implementation(weight_rules, "weight")
        
        # Validate prompt rules (different validation - needs valid Python condition)
        if prompt_rules:
            for rule in prompt_rules:
                if not rule.get('condition') or not rule.get('prompt_addition'):
                    raise ValueError(f"Prompt rule '{rule.get('rule_name', 'unknown')}' missing required condition or prompt_addition")
                
                # Validate that condition is valid Python
                try:
                    # Test compile the condition
                    compile(rule['condition'], '<string>', 'eval')
                except SyntaxError as e:
                    raise ValueError(f"Prompt rule '{rule.get('rule_name', 'unknown')}' has invalid Python condition: '{rule['condition']}'. Error: {e}. Examples: \"'brand' in left_record\", \"True\"")
                
                # Check for unsafe eval patterns
                condition = rule['condition'].lower()
                unsafe_patterns = ['import', '__', 'exec', 'eval', 'open', 'file']
                if any(pattern in condition for pattern in unsafe_patterns):
                    raise ValueError(f"Prompt rule '{rule.get('rule_name', 'unknown')}' condition contains unsafe code: '{rule['condition']}'")
        
        # Create rules in correct format based on mode
        if mode == "weights-only":
            # Only weights, no rules of any kind
            candidate_rules_final = []
            score_rules_final = []
            decision_rules_final = []
            weight_rules_final = []
            prompt_rules_final = []
        elif mode == "prompt-modification":
            # Only weights and prompt rules
            candidate_rules_final = []
            score_rules_final = []
            decision_rules_final = []
            weight_rules_final = []
            prompt_rules_final = prompt_rules or []
        else:  # mode == "heuristics"
            # Traditional rules (no prompt rules)
            candidate_rules_final = candidate_rules or []
            score_rules_final = score_rules or []
            decision_rules_final = decision_rules or []
            weight_rules_final = weight_rules or []
            prompt_rules_final = []
        
        rules_data = {
            "hyperparameters": {
                "max_candidates": max_candidates,
                "semantic_weight": semantic_weight,
                "trigram_weight": trigram_weight,
                "syntactic_weight": syntactic_weight,
                "decision_threshold": 0.5,
                "auto_accept_threshold": 0.9,
                "auto_reject_threshold": 0.1
            },
            "candidate_rules": candidate_rules_final,
            "score_rules": score_rules_final,
            "decision_rules": decision_rules_final,
            "weight_rules": weight_rules_final,
            "prompt_rules": prompt_rules_final,
            "pipeline_rules": [],
            "timestamp": datetime.now().isoformat(),
            "generation_method": "mcp_enforced",
            "mode": mode
        }
        
        # Ensure directory exists
        output_path = "results/temp/generated_rules.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file
        with open(output_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        # Generate mode-specific response message
        weight_sum = semantic_weight + trigram_weight + syntactic_weight
        
        if mode == "weights-only":
            return [TextContent(
                type="text",
                text=f"✅ Weights-only configuration saved to {output_path}\n"
                     f"⚖️ Weights: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f} (sum={weight_sum:.3f})\n"
                     f"🚀 Fast mode: No rules generated, ready for quick testing"
            )]
        elif mode == "prompt-modification":
            return [TextContent(
                type="text",
                text=f"✅ Prompt-modification configuration saved to {output_path}\n"
                     f"⚖️ Weights: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f} (sum={weight_sum:.3f})\n"
                     f"🧠 Prompt rules: {len(prompt_rules_final)} dynamic LLM guidance rules\n"
                     f"🎯 Mode: Optimized for prompt engineering + weight tuning"
            )]
        else:  # mode == "heuristics"
            return [TextContent(
                type="text",
                text=f"✅ Heuristics configuration saved to {output_path}\n"
                     f"📊 Rules: {len(candidate_rules_final)} candidate, {len(score_rules_final)} score, {len(decision_rules_final)} decision, {len(weight_rules_final)} weight\n"
                     f"⚖️ Weights: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f} (sum={weight_sum:.3f})\n"
                     f"🔧 Mode: Full heuristic rule generation"
            )]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error writing rules: {e}")]

async def test_rules_tool(
    dataset: str,
    rules_file: str = "results/temp/generated_rules.json",
    validation_only: bool = True
) -> List[TextContent]:
    """Test rules and return performance metrics on full dev set."""
    
    try:
        # Check if rules file exists
        if not os.path.exists(rules_file):
            return [TextContent(type="text", text=f"❌ Rules file not found: {rules_file}")]
        
        # Build command - validation sampling handled internally (~200 balanced pairs)
        cmd = [
            "python", "run_enhanced_matching.py", 
            "--dataset", dataset,
            "--heuristic-file", rules_file
        ]
        
        if validation_only:
            cmd.append("--use-validation")
        
        # Run test
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return [TextContent(
                type="text", 
                text=f"❌ Test failed with return code {result.returncode}\n"
                     f"STDOUT: {result.stdout}\n"
                     f"STDERR: {result.stderr}"
            )]
        
        # Parse results from output
        output = result.stdout
        
        # Extract metrics and additional information
        lines = output.split('\n')
        f1_score = None
        precision = None
        recall = None
        tp = fn = fp = tn = None
        actual_weights = {"semantic": None, "trigram": None, "syntactic": None}
        weight_system = "unknown"
        
        for line in lines:
            # Extract metrics
            if 'F1-Score:' in line:
                try:
                    f1_score = float(line.split('F1-Score:')[1].strip())
                except:
                    pass
            elif 'Precision:' in line:
                try:
                    precision = float(line.split('Precision:')[1].strip())
                except:
                    pass
            elif 'Recall:' in line:
                try:
                    recall = float(line.split('Recall:')[1].strip())
                except:
                    pass
            elif 'TP:' in line and 'FP:' in line and 'FN:' in line and 'TN:' in line:
                # Extract confusion matrix: "TP: 0, FP: 5, FN: 1, TN: 4"
                try:
                    parts = line.split(',')
                    tp = int(parts[0].split('TP:')[1].strip())
                    fp = int(parts[1].split('FP:')[1].strip()) 
                    fn = int(parts[2].split('FN:')[1].strip())
                    tn = int(parts[3].split('TN:')[1].strip())
                except:
                    pass
            elif 'Using 3-weight system:' in line:
                # Extract actual weights: "🎯 Using 3-weight system: semantic=0.2, trigram=0.7, syntactic=0.1"
                weight_system = "3-weight"
                try:
                    weight_part = line.split('Using 3-weight system:')[1].strip()
                    for pair in weight_part.split(','):
                        if 'semantic=' in pair:
                            actual_weights["semantic"] = float(pair.split('semantic=')[1].strip())
                        elif 'trigram=' in pair:
                            actual_weights["trigram"] = float(pair.split('trigram=')[1].strip())
                        elif 'syntactic=' in pair:
                            actual_weights["syntactic"] = float(pair.split('syntactic=')[1].strip())
                except:
                    pass
            elif 'Using legacy 2-weight system:' in line:
                weight_system = "2-weight"
                try:
                    actual_weights["semantic"] = float(line.split('semantic=')[1].strip())
                except:
                    pass
        
        eval_data = "validation set" if validation_only else "test set"
        result_text = f"🧪 Test Results for {dataset} ({eval_data}):\n"
        
        # Show weight verification - compare requested vs actual
        try:
            with open(rules_file, 'r') as f:
                rules_config = json.load(f)
            requested_weights = rules_config.get("hyperparameters", {})
            req_sem = requested_weights.get("semantic_weight")
            req_tri = requested_weights.get("trigram_weight") 
            req_syn = requested_weights.get("syntactic_weight")
            
            result_text += f"\n🎯 WEIGHT VERIFICATION:\n"
            result_text += f"   Requested: semantic={req_sem}, trigram={req_tri}, syntactic={req_syn}\n"
            result_text += f"   Actually used: semantic={actual_weights['semantic']}, trigram={actual_weights['trigram']}, syntactic={actual_weights['syntactic']}\n"
            
            # Check if weights match
            weights_match = (
                abs(req_sem - actual_weights['semantic']) < 0.01 if req_sem and actual_weights['semantic'] else False
            )
            if weights_match:
                result_text += f"   ✅ Weights loaded correctly from rules file\n"
            else:
                result_text += f"   ⚠️  Weight mismatch - rules file not being used properly\n"
        except:
            result_text += f"\n⚠️ Could not verify weights (rules file read error)\n"
        
        # Show performance metrics
        if f1_score is not None:
            result_text += f"\n📊 PERFORMANCE METRICS:\n"
            result_text += f"   F1 Score: {f1_score:.4f}\n"
            if precision is not None:
                result_text += f"   Precision: {precision:.4f}\n"
            if recall is not None:
                result_text += f"   Recall: {recall:.4f}\n"
        
        # Show confusion matrix with interpretation
        if tp is not None and fp is not None and fn is not None and tn is not None:
            total = tp + fp + fn + tn
            result_text += f"\n🔍 DETAILED ANALYSIS:\n"
            result_text += f"   Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn} (total={total})\n"
            
            if fp > 0:
                result_text += f"   ⚠️  FALSE POSITIVES: {fp} pairs incorrectly marked as matches\n"
                result_text += f"      → Need to be more conservative (higher thresholds)\n"
            
            if fn > 0:
                result_text += f"   ⚠️  FALSE NEGATIVES: {fn} pairs missed (should have matched)\n"  
                result_text += f"      → Need to be more sensitive (lower thresholds or better similarity)\n"
            
            if tp == 0 and fn > 0:
                result_text += f"   🚨 CRITICAL: No true positives found! All matches were missed.\n"
                result_text += f"      → System is too conservative or similarity weights are wrong\n"
        
        # Baseline comparison - error if missing
        if f1_score is not None:
            baseline_file = f"results/temp/{dataset}_dev_predictions.json"
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                baseline_f1 = baseline_data.get('metrics', {}).get('f1')
                if not baseline_f1:
                    result_text += f"\n❌ ERROR: No baseline F1 found in {baseline_file}\n"
                else:
                    change = f1_score - baseline_f1
                    result_text += f"\n📊 Baseline F1: {baseline_f1:.4f}, Current F1: {f1_score:.4f} (change: {change:+.4f})\n"
            except FileNotFoundError:
                result_text += f"\n❌ ERROR: Baseline file not found: {baseline_file}\n"
            except Exception as e:
                result_text += f"\n❌ ERROR: Could not read baseline: {e}\n"
        
        result_text += f"\n📝 Full Output:\n{output}"
        
        return [TextContent(type="text", text=result_text)]
        
    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text="❌ Test timed out after 5 minutes")]
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error running test: {e}")]

async def read_sample_data_tool(dataset: str) -> List[TextContent]:
    """Read and return sample data."""
    
    sample_file = f"results/temp/sample_data_{dataset}.json"
    
    if not os.path.exists(sample_file):
        return [TextContent(type="text", text=f"❌ Sample data file not found: {sample_file}")]
    
    try:
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
            
            summary += f"🔍 **SIMILARITY PATTERNS**:\n"
            for sim_type in ['semantic', 'syntactic', 'trigram']:
                if sim_type in true_matches and sim_type in false_positives:
                    tm_mean = true_matches[sim_type].get('mean', 0)
                    fp_mean = false_positives[sim_type].get('mean', 0)
                    gap = tm_mean - fp_mean
                    summary += f"  • {sim_type.title()}: True matches {tm_mean:.3f} vs False positives {fp_mean:.3f} (gap: {gap:.3f})\n"
            
            # Candidate recall with optimization analysis
            candidate_analysis = insights.get('candidate_analysis', {})
            if candidate_analysis:
                summary += f"\n🎯 **CANDIDATE RECALL & OPTIMIZATION**:\n"
                
                # Use existing optimal candidate calculation function
                try:
                    from src.entity_matching.candidate_optimization import find_recall_plateau
                    optimal_candidates, optimal_recall = find_recall_plateau(candidate_analysis)
                    
                    # Find max candidates for efficiency comparison
                    max_candidates_key = max(
                        [k for k in candidate_analysis.keys() if k.startswith('recall_at_')],
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
                    summary += f"  ⚠️ Could not calculate optimal candidates\n\n"
                
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
                summary += f"\n✅ **TRUE MATCH EXAMPLES**:\n"
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
                summary += f"❌ **CONFUSING NON-MATCH EXAMPLES**:\n"
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
        
        return [TextContent(type="text", text=summary)]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error reading sample data: {e}")]

async def read_instructions_tool(dataset: str) -> List[TextContent]:
    """Provide clear task instructions."""
    
    # Get target F1 from leaderboard
    leaderboard_targets = {
        "beer": 95.3,
        "itunes_amazon": 85.0,
        "amazon_google": 77.0,
        "dblp_acm": 95.0
    }
    
    target_f1 = leaderboard_targets.get(dataset, 85.0)
    
    instructions = f"""🎯 Entity Matching Task for {dataset}

📋 **OBJECTIVE**: Optimize entity matching rules to achieve F1 > {target_f1}
   ⚠️ **If baseline already exceeds target**: Goal is to MAINTAIN performance, not improve it

🔧 **AVAILABLE TOOLS**:
1. **ReadSampleData** - Examine error patterns and current performance
2. **GetBaseline** - Check current baseline metrics  
3. **WriteRules** - Create rules (ENFORCES correct format with 3 weights)
4. **TestRules** - Test your rules on validation data
5. **AnalyzePerformance** - Get suggestions for improvement

📊 **WORKFLOW**:
1. Use ReadSampleData to understand current errors
2. Use GetBaseline to see current performance and baseline strategy
3. Create rules with WriteRules (must include semantic_weight, trigram_weight, syntactic_weight)
4. Test with TestRules 
5. Iterate until F1 > {target_f1}

⚠️ **ONE CHANGE AT A TIME**: Either adjust weights OR add prompt rules, not both in same iteration

🎯 **OPTIMIZATION STRATEGY**:
- **If baseline EXCEEDS target**: Make minimal/no changes, maintain performance
- **If baseline is CLOSE to target** (within 1-2%): Make only 1 conservative adjustment, don't over-optimize
- **If baseline is NEAR target** (within 5%): Moderate optimization, focus on obvious improvements  
- **If baseline is FAR from target**: More aggressive optimization needed

⚠️ **CRITICAL REQUIREMENTS**:
- ALL 3 WEIGHTS REQUIRED: semantic_weight, trigram_weight, syntactic_weight must sum to ~1.0
- CORRECT FORMAT: Tools enforce "candidate_rules", "score_rules", "decision_rules" format
- VALIDATION ONLY: Always use validation data to prevent test leakage
- PROMPT CONDITIONS: In prompt-modification mode, condition field must be valid Python expressions like:
  * "True" (always apply)  
  * "'brand' in left_record" (check if brand field exists)
  * "left_record.get('name', '').lower().startswith('sony')" (Sony products)
  * "left_record.get('price', 0) > 100" (expensive items)
- PROMPT LOGIC: Low recall means missing matches → prompt should help find MORE matches, not fewer

🎯 **SUCCESS CRITERIA**: 
- F1 > {target_f1} on validation set
- Rules saved in correct format via WriteRules tool
- All 3 weight parameters properly specified
"""

    return [TextContent(type="text", text=instructions)]

async def analyze_performance_tool(current_f1: float, target_f1: float, dataset: str) -> List[TextContent]:
    """Analyze performance gap and suggest improvements."""
    
    gap = target_f1 - current_f1
    
    analysis = f"📊 Performance Analysis for {dataset}:\n\n"
    analysis += f"Current F1: {current_f1:.4f}\n"
    analysis += f"Target F1: {target_f1:.1f}\n"
    analysis += f"Gap: {gap:.3f} ({gap/target_f1*100:.1f}%)\n\n"
    
    if gap <= 0:
        analysis += "🎉 **SUCCESS!** Target achieved! Use WriteRules to save your final configuration."
    elif gap < 0.05:
        analysis += "🔧 **SMALL GAP**: Try hyperparameter tuning:\n"
        analysis += "- Adjust max_candidates (50, 100, 150, 250)\n"
        analysis += "- Fine-tune weight balance (semantic vs trigram vs syntactic)\n"
        analysis += "- Consider simple decision rules (auto-accept/reject thresholds)"
    elif gap < 0.15:
        analysis += "⚙️ **MEDIUM GAP**: Need targeted rules:\n"
        analysis += "- Add candidate_rules to capture missed matches\n"
        analysis += "- Add score_rules to boost/penalize specific patterns\n"
        analysis += "- Use ReadSampleData to identify error patterns"
    else:
        analysis += "🛠️ **LARGE GAP**: Need comprehensive rule engineering:\n"
        analysis += "- Extensive candidate_rules for recall improvement\n"
        analysis += "- Multiple score_rules for precision/ranking\n"
        analysis += "- Decision_rules for early termination\n"
        analysis += "- Analyze false negatives and false positives systematically"
    
    analysis += f"\n\n💡 **NEXT STEPS**:\n"
    analysis += f"1. Use ReadSampleData to understand error patterns\n"
    analysis += f"2. Create targeted rules with WriteRules\n"
    analysis += f"3. Test with TestRules\n"
    analysis += f"4. Iterate until gap is closed"
    
    return [TextContent(type="text", text=analysis)]

async def get_baseline_tool(dataset: str) -> List[TextContent]:
    """Get baseline performance for the dataset."""
    
    try:
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
                baseline += f"📋 MAKE ONE CHANGE AT A TIME: Either adjust weights OR add prompt rules, not both\n"
                baseline += f"Source: {dev_file}\n"
                
                # Get target and check if already close
                from src.experiments.simplified_agentic_generator import get_leaderboard_target_f1
                target_f1 = get_leaderboard_target_f1(dataset) / 100.0  # Convert to decimal
                gap = target_f1 - f1
                
                if gap <= 0:  # Baseline already exceeds target!
                    baseline += f"\n🎉 BASELINE ALREADY EXCEEDS TARGET ({target_f1:.3f})!\n"
                    baseline += f"Current: {f1:.3f}, Target: {target_f1:.3f} (surplus: {-gap:.3f})\n"
                    baseline += f"Strategy: Make MINIMAL changes to maintain performance. Try just 1 iteration.\n"
                    baseline += f"For prompt-modification mode: Use very conservative prompt additions only.\n"
                    if eval_dataset != dataset:
                        baseline += f"\n⚠️ WARNING: Baseline was on '{eval_dataset}' but TestRules uses '{dataset}' - results may differ!\n"
                elif gap <= 0.01:  # Within 1% of target
                    baseline += f"\n✅ BASELINE IS CLOSE TO TARGET ({target_f1:.3f})!\n"
                    baseline += f"Gap: {gap:.3f} - only minor optimization needed.\n"
                    baseline += f"Strategy: Try 1-2 conservative adjustments, don't over-optimize.\n"
                elif gap <= 0.05:  # Within 5% of target
                    baseline += f"\n⚡ BASELINE IS NEAR TARGET ({target_f1:.3f})\n"
                    baseline += f"Gap: {gap:.3f} - moderate optimization should work.\n"
                else:
                    baseline += f"\n🎯 TARGET: {target_f1:.3f} (gap: {gap:.3f})\n"
                    baseline += f"Significant optimization needed.\n"
                
                return [TextContent(type="text", text=baseline)]
        
        return [TextContent(type="text", text=f"❌ No baseline results found for {dataset}")]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error getting baseline: {e}")]

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
    
    # Provide specific guidance based on issue type
    if issue_type == "tool_confusion":
        report += "💡 **HELP**: Use exact MCP tool names:\n"
        report += "- mcp__entity-matching__WriteRules (NOT Write tool)\n"
        report += "- mcp__entity-matching__TestRules\n"
        report += "- mcp__entity-matching__ReadSampleData\n\n"
        
    elif issue_type == "parameter_error":
        report += "💡 **HELP**: WriteRules requires:\n"
        report += "- semantic_weight: float (0.0-1.0)\n"
        report += "- trigram_weight: float (0.0-1.0)\n" 
        report += "- syntactic_weight: float (0.0-1.0)\n"
        report += "- Optional: candidate_rules, score_rules, decision_rules arrays\n\n"
        
    elif issue_type == "validation_failure":
        report += "💡 **HELP**: Common validation issues:\n"
        report += "- Rules must have Python code, not English descriptions\n"
        report += "- All 3 weights must be provided and sum to ~1.0\n"
        report += "- Use simple Python expressions in 'implementation' field\n\n"
        
    elif issue_type == "unclear_requirements":
        report += "💡 **HELP**: Follow the workflow:\n"
        report += "1. ReadInstructions → ReadSampleData → GetBaseline\n"
        report += "2. WriteRules with proper weights\n"
        report += "3. TestRules to check performance\n"
        report += "4. Repeat until target F1 achieved\n\n"
    
    report += "📋 **Issue logged for debugging**. Try the suggested approach above."
    
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
    from mcp.server.stdio import stdio_server
    
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