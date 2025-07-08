#!/usr/bin/env python3
"""
Generate optimal hyperparameters and rules using Claude SDK based on rich similarity analysis.

This script takes the output from analyze_for_claude.py and sends it to Claude
to get back both optimal hyperparameters AND heuristic rules in one shot.

Usage:
    python scripts/claude_config_generator.py --analysis results/itunes_amazon_claude_analysis.json
    python scripts/claude_config_generator.py --dataset beer  # Auto-finds analysis file
"""

import argparse
import json
import pathlib
import sys

from typing import Any, Dict

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

try:
    from experiments.claude_sdk_heuristic_generator import ClaudeSDKHeuristicGenerator

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    print("Warning: Claude SDK not available")


def create_claude_prompt(analysis: Dict[str, Any]) -> str:
    """Create a comprehensive prompt for Claude optimization"""

    dataset = analysis["dataset"]
    sim_analysis = analysis["similarity_analysis"]
    candidate_analysis = analysis["candidate_analysis"]
    dataset_chars = analysis["dataset_characteristics"]

    # Extract key statistics
    true_matches = sim_analysis["true_matches"]
    false_positives = sim_analysis["false_positives"]

    semantic_available = analysis["metadata"]["semantic_available"]

    prompt = f"""You are an expert at entity matching optimization. Based on this comprehensive similarity analysis, determine the optimal hyperparameters AND generate heuristic rules that leverage all three similarity measures.

DATASET: {dataset}
Analysis Type: {analysis["analysis_type"]}
Semantic Similarity Available: {semantic_available}

SIMILARITY DISTRIBUTIONS:

True Matches:
- Syntactic: mean={true_matches["syntactic"]["mean"]:.3f}, std={true_matches["syntactic"]["std"]:.3f}, median={true_matches["syntactic"]["median"]:.3f}
- Trigram: mean={true_matches["trigram"]["mean"]:.3f}, std={true_matches["trigram"]["std"]:.3f}, median={true_matches["trigram"]["median"]:.3f}"""

    if semantic_available and true_matches["semantic"]:
        prompt += f"\n- Semantic: mean={true_matches['semantic']['mean']:.3f}, std={true_matches['semantic']['std']:.3f}, median={true_matches['semantic']['median']:.3f}"

    prompt += f"""

False Positives:
- Syntactic: mean={false_positives["syntactic"]["mean"]:.3f}, std={false_positives["syntactic"]["std"]:.3f}, median={false_positives["syntactic"]["median"]:.3f}
- Trigram: mean={false_positives["trigram"]["mean"]:.3f}, std={false_positives["trigram"]["std"]:.3f}, median={false_positives["trigram"]["median"]:.3f}"""

    if semantic_available and false_positives["semantic"]:
        prompt += f"\n- Semantic: mean={false_positives['semantic']['mean']:.3f}, std={false_positives['semantic']['std']:.3f}, median={false_positives['semantic']['median']:.3f}"

    prompt += """

CANDIDATE GENERATION PERFORMANCE:"""
    for metric, value in candidate_analysis.items():
        prompt += f"\n- {metric}: {value:.3f}"

    prompt += f"""

DATASET CHARACTERISTICS:
- Table A size: {dataset_chars["table_a_size"]} records
- Table B size: {dataset_chars["table_b_size"]} records
- Fields: {", ".join(dataset_chars["field_names"])}

CONCRETE EXAMPLES:

True Match Examples:"""

    # Add concrete true match examples
    concrete_examples = analysis.get("concrete_examples", {})
    true_matches_examples = concrete_examples.get("true_matches", [])

    for i, example in enumerate(true_matches_examples[:3], 1):  # Show first 3 examples
        similarities = example["similarities"]
        candidate_gen = example["candidate_generation"]
        prompt += f"""

Example {i} - TRUE MATCH:
- Left: {example["left_record"]}
- Right: {example["right_record"]}
- Similarities: Syntactic={similarities["syntactic"]}, Trigram={similarities["trigram"]}, Semantic={similarities["semantic"]}
- Candidate Generation: Found={candidate_gen["found"]}, Rank={candidate_gen["rank"]}/{candidate_gen["max_candidates"]}"""

    prompt += """

Confusing Non-Match Examples:"""

    # Add confusing non-match examples
    confusing_examples = concrete_examples.get("confusing_non_matches", [])

    for i, example in enumerate(confusing_examples[:3], 1):  # Show first 3 examples
        similarities = example["similarities"]
        candidate_gen = example["candidate_generation"]
        prompt += f"""

Example {i} - NON-MATCH (but potentially confusing):
- Left: {example["left_record"]}
- Right: {example["right_record"]}
- Similarities: Syntactic={similarities["syntactic"]}, Trigram={similarities["trigram"]}, Semantic={similarities["semantic"]}
- Candidate Generation: Found={candidate_gen["found"]}, Rank={candidate_gen["rank"]}/{candidate_gen["max_candidates"]}"""

    prompt += """

TASK:
Analyze these patterns and provide:

1. **Optimal Hyperparameters**: Choose max_candidates and weights (trigram_weight, syntactic_weight, semantic_weight) that sum to 1.0
   - Consider the recall@X metrics to choose max_candidates
   - Consider the similarity distributions to weight the measures appropriately
   - Higher weights should go to measures that better separate true matches from false positives

2. **Heuristic Rules**: Generate rules that leverage all three similarity measures to improve precision
   - Rules should target patterns where one measure might mislead but others provide clarity
   - Consider edge cases like formatting differences, abbreviations, extra words
   - Use the actual similarity distributions to set meaningful thresholds

3. **Reasoning**: Explain your choices based on the data patterns

IMPORTANT CONSTRAINTS:
- Weights must sum to 1.0 exactly
- max_candidates should balance recall vs computational cost
- Rules should be conservative (prefer precision over recall)
- Consider all three similarity measures in your rule logic

Output format (valid JSON):
{{
  "hyperparameters": {{
    "max_candidates": 100,
    "trigram_weight": 0.2,
    "syntactic_weight": 0.3,
    "semantic_weight": 0.5
  }},
  "rules": [
    {{
      "stage": "matching",
      "action": "boost_score",
      "conditions": [
        {{"field": "syntactic_similarity", "operator": ">", "value": 0.85}},
        {{"field": "semantic_similarity", "operator": ">", "value": 0.8}}
      ],
      "score_adjustment": 0.15,
      "reasoning": "High syntactic + semantic indicates strong match despite potential trigram differences"
    }}
  ],
  "reasoning": "Based on the similarity distributions, I chose these hyperparameters and rules because..."
}}"""

    return prompt


def generate_config_with_claude(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Use Claude SDK to generate optimal config and rules"""

    if not CLAUDE_SDK_AVAILABLE:
        raise ValueError("Claude SDK not available. Install anthropic package.")

    # Create prompt
    prompt = create_claude_prompt(analysis)

    print("🤖 Sending analysis to Claude for optimization...")
    print(f"📝 Prompt length: {len(prompt)} characters")

    # Initialize Claude generator (this will use existing config)
    generator = ClaudeSDKHeuristicGenerator(
        dataset=analysis["dataset"],
        model="claude-3-5-sonnet-20241022",  # Use best model for optimization
    )

    try:
        # Get Claude's response
        response = generator._call_claude_sync(prompt)
        print(f"✅ Received response from Claude ({len(response)} characters)")

        # Try to parse as JSON
        try:
            config = json.loads(response)

            # Validate the response structure
            if "hyperparameters" not in config:
                raise ValueError("Response missing 'hyperparameters' section")
            if "rules" not in config:
                raise ValueError("Response missing 'rules' section")

            # Validate weights sum to 1.0
            weights = config["hyperparameters"]
            total_weight = (
                weights.get("trigram_weight", 0)
                + weights.get("syntactic_weight", 0)
                + weights.get("semantic_weight", 0)
            )
            if abs(total_weight - 1.0) > 0.01:
                print(f"⚠️ Warning: Weights sum to {total_weight}, not 1.0. Normalizing...")
                # Normalize weights
                for key in ["trigram_weight", "syntactic_weight", "semantic_weight"]:
                    if key in weights:
                        weights[key] = weights[key] / total_weight

            return config

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Claude response as JSON: {e}")
            print(f"Raw response: {response[:500]}...")

            # Fallback: create a basic config based on analysis
            return create_fallback_config(analysis)

    except Exception as e:
        print(f"❌ Error calling Claude: {e}")
        return create_fallback_config(analysis)


def create_fallback_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Create a reasonable fallback config based on simple heuristics"""
    print("🔄 Creating fallback configuration based on analysis...")

    candidate_analysis = analysis["candidate_analysis"]
    true_matches = analysis["similarity_analysis"]["true_matches"]
    false_positives = analysis["similarity_analysis"]["false_positives"]

    # Choose max_candidates based on recall
    max_candidates = 100  # Default
    for threshold in [50, 100, 150, 200]:
        key = f"recall_at_{threshold}"
        if key in candidate_analysis and candidate_analysis[key] > 0.85:
            max_candidates = threshold
            break

    # Choose weights based on separation power
    semantic_available = analysis["metadata"]["semantic_available"]

    if semantic_available and true_matches["semantic"]:
        # Calculate separation power for each measure
        syntactic_sep = true_matches["syntactic"]["mean"] - false_positives["syntactic"]["mean"]
        trigram_sep = true_matches["trigram"]["mean"] - false_positives["trigram"]["mean"]
        semantic_sep = true_matches["semantic"]["mean"] - false_positives["semantic"]["mean"]

        total_sep = syntactic_sep + trigram_sep + semantic_sep

        if total_sep > 0:
            trigram_weight = max(0.1, trigram_sep / total_sep)
            syntactic_weight = max(0.1, syntactic_sep / total_sep)
            semantic_weight = max(0.1, semantic_sep / total_sep)

            # Normalize
            total = trigram_weight + syntactic_weight + semantic_weight
            trigram_weight /= total
            syntactic_weight /= total
            semantic_weight /= total
        else:
            trigram_weight, syntactic_weight, semantic_weight = 0.2, 0.3, 0.5
    else:
        # No semantic similarity available
        trigram_weight, syntactic_weight, semantic_weight = 0.5, 0.5, 0.0

    # Create basic rules
    rules = []

    # Rule: High syntactic + semantic boost
    if semantic_available:
        syntactic_threshold = true_matches["syntactic"]["median"]
        semantic_threshold = true_matches["semantic"]["median"]

        rules.append(
            {
                "stage": "matching",
                "action": "boost_score",
                "conditions": [
                    {"field": "syntactic_similarity", "operator": ">", "value": syntactic_threshold},
                    {"field": "semantic_similarity", "operator": ">", "value": semantic_threshold},
                ],
                "score_adjustment": 0.1,
                "reasoning": "High syntactic + semantic indicates strong match (fallback rule)",
            }
        )

    return {
        "hyperparameters": {
            "max_candidates": max_candidates,
            "trigram_weight": round(trigram_weight, 3),
            "syntactic_weight": round(syntactic_weight, 3),
            "semantic_weight": round(semantic_weight, 3),
        },
        "rules": rules,
        "reasoning": "Fallback configuration generated from statistical analysis of similarity distributions",
        "fallback": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate optimal config using Claude based on similarity analysis")
    parser.add_argument("--analysis", help="Path to analysis JSON file")
    parser.add_argument("--dataset", help="Dataset name (will look for results/{dataset}_claude_analysis.json)")
    parser.add_argument("--output", help="Output JSON file (default: results/{dataset}_claude_config.json)")
    parser.add_argument("--fallback-only", action="store_true", help="Skip Claude and use fallback config only")

    args = parser.parse_args()

    # Determine analysis file
    if args.analysis:
        analysis_file = pathlib.Path(args.analysis)
    elif args.dataset:
        analysis_file = pathlib.Path(f"results/{args.dataset}_claude_analysis.json")
    else:
        print("❌ Must provide either --analysis or --dataset")
        return

    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        print("Run analyze_for_claude.py first to generate the analysis file")
        return

    # Load analysis
    with open(analysis_file) as f:
        analysis = json.load(f)

    dataset = analysis["dataset"]
    print(f"🔧 Generating optimized config for dataset: {dataset}")

    # Generate config
    if args.fallback_only:
        config = create_fallback_config(analysis)
    else:
        config = generate_config_with_claude(analysis)

    # Save results
    output_file = args.output or f"results/{dataset}_claude_config.json"
    output_path = pathlib.Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n💾 Optimized config saved to: {output_path}")

    # Display results
    hyperparams = config["hyperparameters"]
    print("\n🎯 OPTIMIZED HYPERPARAMETERS:")
    print(f"   max_candidates: {hyperparams['max_candidates']}")
    print(f"   trigram_weight: {hyperparams['trigram_weight']}")
    print(f"   syntactic_weight: {hyperparams['syntactic_weight']}")
    print(f"   semantic_weight: {hyperparams['semantic_weight']}")
    print(
        f"   Weight sum: {sum([hyperparams['trigram_weight'], hyperparams['syntactic_weight'], hyperparams['semantic_weight']]):.3f}"
    )

    print(f"\n📋 GENERATED RULES: {len(config['rules'])} rules")
    for i, rule in enumerate(config["rules"], 1):
        print(f"   {i}. {rule['action']} - {rule.get('reasoning', 'No reasoning provided')}")

    if "reasoning" in config:
        print("\n💭 REASONING:")
        print(f"   {config['reasoning']}")

    print("\n🚀 Ready to test! Use this config with the enhanced pipeline.")


if __name__ == "__main__":
    main()
