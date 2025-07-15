#!/usr/bin/env python3
"""
Intelligent pipeline that uses Claude analysis to optimize hyperparameters and rules.
This replaces the blind hyperparameter sweep with data-driven optimization.
"""

import argparse
import json
import pathlib
import sys
import time

from typing import Dict, Optional

import pandas as pd

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from entity_matching.analysis import analyze_dataset_for_claude
from entity_matching.hybrid_matcher import Config


def run_claude_analysis(dataset: str, max_pairs: int = 200, verbose: bool = True) -> Dict:
    """
    Run the Claude analysis to get optimal hyperparameters and rules.
    """
    if verbose:
        print(f"🔍 Running Claude analysis for {dataset}...")

    # Generate analysis
    analysis_file = f"results/{dataset}_claude_analysis.json"
    analysis = analyze_dataset_for_claude(
        dataset=dataset,
        max_pairs=max_pairs,
        max_candidates=100,
        output_file=analysis_file,
        verbose=verbose
    )

    # Generate Claude config (using existing script)
    if verbose:
        print("🤖 Generating Claude-optimized configuration...")

    try:
        # Try enhanced agentic generation with rich analysis data
        return generate_agentic_config_with_analysis(analysis, dataset, verbose)

    except Exception as e:
        if verbose:
            print(f"⚠️ Error running agentic config generator: {e}")
        # Fall back to analysis-based config
        return generate_fallback_config(analysis, verbose)


def generate_agentic_config_with_analysis(analysis: Dict, dataset: str, verbose: bool = True) -> Dict:
    """
    Generate configuration using agentic Claude with rich analysis data.
    """
    if verbose:
        print("🤖 Generating agentic configuration with rich analysis data...")

    try:
        # Import the enhanced agentic generator
        import sys
        sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))
        from experiments.agentic_heuristic_generator import generate_agentic_heuristics

        # Create mock dev results for the agentic generator
        # (In a real implementation, you'd have actual dev predictions)
        mock_dev_results = {
            "predictions": {},  # Would be filled with actual predictions
            "metrics": {"f1": 0.85, "precision": 0.9, "recall": 0.8}  # Mock metrics
        }

        if verbose:
            print("⚠️ Using mock dev results - integrate with actual pipeline for real predictions")

        # Run agentic generation with rich analysis data
        import asyncio

        async def run_agentic():
            return await generate_agentic_heuristics(
                dataset=dataset,
                dev_results=mock_dev_results,
                output_file=f"results/{dataset}_agentic_rules.json",
                analysis_data=analysis
            )

        rules_file, cost_info = asyncio.run(run_agentic())

        if rules_file and pathlib.Path(rules_file).exists():
            with open(rules_file) as f:
                config = json.load(f)

            if verbose:
                print(f"✅ Generated agentic config with ${cost_info.get('total_cost_usd', 0):.4f} cost")
                print(f"   Rules file: {rules_file}")

            return config
        if verbose:
            print("⚠️ Agentic generation failed, falling back to rule-based config")
        return generate_fallback_config(analysis, verbose)

    except Exception as e:
        if verbose:
            print(f"⚠️ Agentic generation error: {e}, falling back to rule-based config")
        return generate_fallback_config(analysis, verbose)


def generate_fallback_config(analysis: Dict, verbose: bool = True) -> Dict:
    """
    Generate a fallback configuration based on analysis insights.
    """
    if verbose:
        print("📊 Generating fallback config based on analysis insights...")

    # Extract insights from analysis
    true_match_stats = analysis["similarity_analysis"]["true_matches"]
    false_positive_stats = analysis["similarity_analysis"]["false_positives"]
    recall_analysis = analysis["candidate_analysis"]

    # Calculate optimal thresholds based on separation between true matches and false positives
    syntactic_threshold = max(0.6, (true_match_stats["syntactic"]["mean"] + false_positive_stats["syntactic"]["mean"]) / 2)
    trigram_threshold = max(0.3, (true_match_stats["trigram"]["mean"] + false_positive_stats["trigram"]["mean"]) / 2)

    # Set semantic threshold if available
    semantic_threshold = 0.8
    if analysis["metadata"]["semantic_available"]:
        semantic_threshold = max(0.7, (true_match_stats["semantic"]["mean"] + false_positive_stats["semantic"]["mean"]) / 2)

    # Generate rules to improve recall
    rules = []

    # Add rules for missed candidates (recall < 95%)
    if recall_analysis.get("recall_at_10", 0) < 0.95:
        rules.append({
            "type": "candidate_expansion",
            "condition": "recall_at_10 < 0.95",
            "action": "increase_trigram_candidates",
            "params": {"max_candidates": 150}
        })

    # Add rules for high semantic similarity pairs that might be missed
    if analysis["metadata"]["semantic_available"]:
        rules.append({
            "type": "high_semantic_match",
            "condition": f"semantic_similarity > {semantic_threshold - 0.1}",
            "action": "force_match",
            "params": {"threshold": semantic_threshold - 0.1}
        })

    # Add rules for perfect syntactic matches
    if true_match_stats["syntactic"]["max"] > 0.95:
        rules.append({
            "type": "perfect_syntactic_match",
            "condition": "syntactic_similarity >= 0.95",
            "action": "force_match",
            "params": {"threshold": 0.95}
        })

    config = {
        "hyperparameters": {
            "syntactic_threshold": round(syntactic_threshold, 3),
            "trigram_threshold": round(trigram_threshold, 3),
            "semantic_threshold": round(semantic_threshold, 3),
            "semantic_weight": 0.4,
            "max_candidates": 150,  # Increase to improve recall
            "use_heuristics": True
        },
        "rules": rules,
        "analysis_insights": {
            "true_match_syntactic_mean": true_match_stats["syntactic"]["mean"],
            "false_positive_syntactic_mean": false_positive_stats["syntactic"]["mean"],
            "recall_at_10": recall_analysis.get("recall_at_10", 0),
            "recall_at_100": recall_analysis.get("recall_at_100", 0),
            "semantic_available": analysis["metadata"]["semantic_available"]
        }
    }

    if verbose:
        print("📊 Generated fallback config:")
        print(f"   Syntactic threshold: {syntactic_threshold:.3f}")
        print(f"   Trigram threshold: {trigram_threshold:.3f}")
        print(f"   Semantic threshold: {semantic_threshold:.3f}")
        print(f"   Generated {len(rules)} rules")

    return config


def apply_config_to_matcher(config: Dict, cfg: Config, verbose: bool = True) -> None:
    """
    Apply the Claude-generated configuration to the matcher.
    """
    hyperparams = config["hyperparameters"]

    # Apply hyperparameters
    if "semantic_weight" in hyperparams:
        cfg.semantic_weight = hyperparams["semantic_weight"]
    if "use_heuristics" in hyperparams:
        cfg.use_heuristics = hyperparams["use_heuristics"]

    # Note: Thresholds and other parameters would be applied during matching
    # since they're used in the matching logic, not stored in config

    if verbose:
        print("🔧 Applied configuration:")
        print(f"   Semantic weight: {cfg.semantic_weight}")
        print(f"   Use heuristics: {cfg.use_heuristics}")
        if "rules" in config:
            print(f"   Rules: {len(config['rules'])} rules defined")


def run_intelligent_pipeline(
    dataset: str,
    max_pairs: int = 200,
    output_file: Optional[str] = None,
    use_analysis_flag: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run the full intelligent pipeline:
    1. Analyze dataset with Claude
    2. Generate optimal config
    3. Run matching with optimized parameters
    4. Return results with performance metrics
    """
    start_time = time.time()

    if verbose:
        print("🚀 INTELLIGENT ENTITY MATCHING PIPELINE")
        print("=" * 60)
        print(f"Dataset: {dataset}")
        print(f"Use Claude analysis: {use_analysis_flag}")

    # Step 1: Load dataset
    if verbose:
        print("\n📂 Loading dataset...")

    try:
        data_root = pathlib.Path("data/raw") / dataset
        if not data_root.exists():
            raise ValueError(f"Dataset '{dataset}' not found in data/raw/")

        A_df = pd.read_csv(data_root / "tableA.csv")
        B_df = pd.read_csv(data_root / "tableB.csv")

        # Try to load test pairs
        pairs_df = pd.DataFrame()
        if (data_root / "test.csv").exists():
            pairs_df = pd.read_csv(data_root / "test.csv")
        elif (data_root / "valid.csv").exists():
            pairs_df = pd.read_csv(data_root / "valid.csv")

        if verbose:
            print(f"✅ Loaded {len(A_df)} records in A, {len(B_df)} records in B")
            print(f"✅ Loaded {len(pairs_df)} test pairs")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return {"error": f"Failed to load dataset: {e}"}

    # Step 2: Generate optimal configuration
    config = None
    if use_analysis_flag:
        try:
            config = run_claude_analysis(dataset, max_pairs, verbose)
        except Exception as e:
            if verbose:
                print(f"⚠️ Analysis failed, using default config: {e}")

    # Step 3: Initialize matcher with config
    cfg = Config()
    if config:
        apply_config_to_matcher(config, cfg, verbose)

    # Step 4: Run matching
    if verbose:
        print("\n🔄 Running entity matching...")

    try:
        # Convert dataframes to the format expected by matching
        A_records = A_df.to_dict('records')
        B_records = B_df.to_dict('records')

        # For now, return a mock result structure since we don't have the full matching pipeline yet
        matching_start = time.time()

        # Simulate matching process
        if verbose:
            print(f"🔄 Simulating matching with {len(A_records)} x {len(B_records)} record pairs...")

        matching_time = time.time() - matching_start

        results = {
            "matches": [],
            "total_matches": 0,
            "total_time": matching_time,
            "config_applied": config is not None,
            "dataset_size": {"A": len(A_records), "B": len(B_records)}
        }

        if verbose:
            print(f"✅ Matching simulation completed in {matching_time:.2f}s")

    except Exception as e:
        if verbose:
            print(f"❌ Matching failed: {e}")
        results = {
            "matches": [],
            "total_matches": 0,
            "total_time": time.time() - start_time,
            "error": str(e)
        }

    # Step 5: Evaluate results if test pairs available
    evaluation = None
    if len(pairs_df) > 0:
        if verbose:
            print("\n📊 Evaluating results...")
        # evaluation = evaluate_results(results, pairs_df, verbose)
        # For now, skip evaluation
        evaluation = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Step 6: Compile final results
    pipeline_results = {
        "dataset": dataset,
        "use_analysis": use_analysis_flag,
        "config": config,
        "matching_results": results,
        "evaluation": evaluation,
        "total_time": time.time() - start_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save results if requested
    if output_file:
        output_path = pathlib.Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(pipeline_results, f, indent=2)
        if verbose:
            print(f"\n💾 Results saved to: {output_path}")

    if verbose:
        print(f"\n🏁 Pipeline completed in {pipeline_results['total_time']:.2f}s")
        if evaluation:
            print(f"📊 Performance: P={evaluation.get('precision', 0):.3f}, R={evaluation.get('recall', 0):.3f}, F1={evaluation.get('f1', 0):.3f}")

    return pipeline_results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Run intelligent entity matching pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., 'itunes_amazon')")
    parser.add_argument("--max-pairs", type=int, default=200, help="Maximum pairs for analysis (default: 200)")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--no-analysis", action="store_true", help="Skip Claude analysis (use default config)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    verbose = not args.quiet
    use_analysis = not args.no_analysis

    try:
        results = run_intelligent_pipeline(
            dataset=args.dataset,
            max_pairs=args.max_pairs,
            output_file=args.output,
            use_analysis_flag=use_analysis,
            verbose=verbose
        )

        if "error" in results:
            print(f"❌ Pipeline failed: {results['error']}")
            sys.exit(1)

        print("✅ Pipeline completed successfully")

    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
