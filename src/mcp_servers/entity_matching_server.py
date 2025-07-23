#!/usr/bin/env python3
"""
Entity Matching MCP Server

Custom MCP server that provides specialized tools for entity matching rule generation.
This ensures Claude follows the correct workflow and format requirements.
"""

import asyncio
import json
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

# Initialize the MCP server
server = Server("entity-matching-server")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List all available entity matching tools."""
    return [
        Tool(
            name="WriteRules",
            description=(
                "Write entity matching rules in the enforced correct format. "
                "REQUIRES all 3 weights: semantic_weight, trigram_weight, syntactic_weight. "
                "Rules must have 'implementation' field with valid Python code, not English descriptions. "
                "Example rule: {'rule_name': 'exact_name_boost', 'implementation': 'return 0.1 if left_record.get(\"name\", \"\").lower() == right_record.get(\"name\", \"\").lower() else 0', 'stage': 'post_semantic'} "
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
                "required": ["semantic_weight", "trigram_weight", "syntactic_weight"]
            }
        ),
        Tool(
            name="TestRules",
            description=(
                "Test entity matching rules and return structured performance results. "
                "Runs the pipeline with the generated rules and returns F1, precision, recall."
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
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Number of validation examples to test on"
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
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    
    if name == "WriteRules":
        return await write_rules_tool(**arguments)
    elif name == "TestRules":
        return await test_rules_tool(**arguments)
    elif name == "ReadSampleData":
        return await read_sample_data_tool(**arguments)
    elif name == "ReadInstructions":
        return await read_instructions_tool(**arguments)
    elif name == "AnalyzePerformance":
        return await analyze_performance_tool(**arguments)
    elif name == "GetBaseline":
        return await get_baseline_tool(**arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def write_rules_tool(
    semantic_weight: float,
    trigram_weight: float,
    syntactic_weight: float,
    max_candidates: int = 100,
    candidate_rules: List[Dict] = None,
    score_rules: List[Dict] = None,
    decision_rules: List[Dict] = None,
    weight_rules: List[Dict] = None
) -> List[TextContent]:
    """Write rules in enforced correct format."""
    
    try:
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
                        compile(impl, '<string>', 'eval')
                    except SyntaxError as e:
                        raise ValueError(f"Rule '{rule.get('rule_name', 'unknown')}' has invalid Python syntax: {e}")
        
        # Validate all rule types
        validate_rule_implementation(candidate_rules, "candidate")
        validate_rule_implementation(score_rules, "score") 
        validate_rule_implementation(decision_rules, "decision")
        validate_rule_implementation(weight_rules, "weight")
        
        # Create rules in correct format
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
            "candidate_rules": candidate_rules or [],
            "score_rules": score_rules or [],
            "decision_rules": decision_rules or [],
            "weight_rules": weight_rules or [],
            "pipeline_rules": [],
            "timestamp": datetime.now().isoformat(),
            "generation_method": "mcp_enforced"
        }
        
        # Ensure directory exists
        output_path = "results/temp/generated_rules.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file
        with open(output_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        return [TextContent(
            type="text",
            text=f"✅ Rules successfully saved to {output_path}\n"
                 f"📊 Rules: {len(candidate_rules or [])} candidate, {len(score_rules or [])} score, {len(decision_rules or [])} decision\n"
                 f"⚖️ Weights: semantic={semantic_weight:.3f}, trigram={trigram_weight:.3f}, syntactic={syntactic_weight:.3f} (sum={weight_sum:.3f})"
        )]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error writing rules: {e}")]

async def test_rules_tool(
    dataset: str,
    rules_file: str = "results/temp/generated_rules.json",
    validation_only: bool = True,
    limit: int = 20
) -> List[TextContent]:
    """Test rules and return performance metrics."""
    
    try:
        # Check if rules file exists
        if not os.path.exists(rules_file):
            return [TextContent(type="text", text=f"❌ Rules file not found: {rules_file}")]
        
        # Build command
        cmd = [
            "python", "run_enhanced_matching.py",
            "--dataset", dataset,
            "--use-agentic-rules", rules_file,
            "--limit", str(limit)
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
        
        # Extract metrics (this is a simplified parser - we'd need to make it more robust)
        lines = output.split('\n')
        f1_score = None
        precision = None
        recall = None
        
        for line in lines:
            if 'F1:' in line:
                try:
                    f1_score = float(line.split('F1:')[1].strip().split()[0])
                except:
                    pass
            if 'Precision:' in line:
                try:
                    precision = float(line.split('Precision:')[1].strip().split()[0])
                except:
                    pass
            if 'Recall:' in line:
                try:
                    recall = float(line.split('Recall:')[1].strip().split()[0])
                except:
                    pass
        
        result_text = f"🧪 Test Results for {dataset}:\n"
        if f1_score is not None:
            result_text += f"📊 F1 Score: {f1_score:.4f}\n"
        if precision is not None:
            result_text += f"📊 Precision: {precision:.4f}\n"
        if recall is not None:
            result_text += f"📊 Recall: {recall:.4f}\n"
        
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
        summary = f"📊 Sample Data for {dataset}:\n"
        summary += f"Dataset: {data.get('dataset', 'unknown')}\n"
        summary += f"Target F1: {data.get('target_f1', 'unknown')}\n"
        summary += f"Current F1: {data.get('dev_metrics', {}).get('f1', 'unknown'):.4f}\n"
        summary += f"Sample pairs: {len(data.get('dev_pairs', []))}\n"
        
        # Show a few examples
        pairs = data.get('dev_pairs', [])[:3]
        summary += f"\n📝 Example pairs:\n"
        for i, pair in enumerate(pairs, 1):
            summary += f"{i}. {pair.get('category', 'unknown')}: {pair.get('left_record', {}).get('Beer_Name', 'N/A')} vs {pair.get('right_record', {}).get('Beer_Name', 'N/A')}\n"
        
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

🔧 **AVAILABLE TOOLS**:
1. **ReadSampleData** - Examine error patterns and current performance
2. **GetBaseline** - Check current baseline metrics  
3. **WriteRules** - Create rules (ENFORCES correct format with 3 weights)
4. **TestRules** - Test your rules on validation data
5. **AnalyzePerformance** - Get suggestions for improvement

📊 **WORKFLOW**:
1. Use ReadSampleData to understand current errors
2. Use GetBaseline to see current performance
3. Create rules with WriteRules (must include semantic_weight, trigram_weight, syntactic_weight)
4. Test with TestRules 
5. Iterate until F1 > {target_f1}

⚠️ **CRITICAL REQUIREMENTS**:
- ALL 3 WEIGHTS REQUIRED: semantic_weight, trigram_weight, syntactic_weight must sum to ~1.0
- CORRECT FORMAT: Tools enforce "candidate_rules", "score_rules", "decision_rules" format
- VALIDATION ONLY: Always use validation data to prevent test leakage

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
                
                metadata = data.get('metadata', {})
                f1 = metadata.get('f1', 0.0)
                precision = metadata.get('precision', 0.0) 
                recall = metadata.get('recall', 0.0)
                
                baseline = f"📊 Baseline Performance for {dataset}:\n"
                baseline += f"F1: {f1:.4f}\n"
                baseline += f"Precision: {precision:.4f}\n"
                baseline += f"Recall: {recall:.4f}\n"
                baseline += f"Source: {dev_file}\n"
                
                return [TextContent(type="text", text=baseline)]
        
        return [TextContent(type="text", text=f"❌ No baseline results found for {dataset}")]
        
    except Exception as e:
        return [TextContent(type="text", text=f"❌ Error getting baseline: {e}")]

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