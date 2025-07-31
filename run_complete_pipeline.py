#!/usr/bin/env python
"""
SINGLE ENTRYPOINT for complete entity matching pipeline.

This script does exactly one thing:
1. Run dev set with high max-candidates and cheap model
2. Create ACTUAL RULES based on dev set analysis using Claude SDK
3. Run test set with enhanced matching using those rules and record the answer

Usage:
    python run_complete_pipeline.py --dataset beer
    python run_complete_pipeline.py --dataset walmart_amazon
"""

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import time

# Fix tokenizer fork warnings in async processing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set environment variables for output visibility in chained commands
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from src.entity_matching.analysis import analyze_dataset_for_claude
from src.entity_matching.candidate_optimization import get_optimal_candidates_for_dataset
from src.entity_matching.experiment_registry import ExperimentRegistry
from src.entity_matching.hybrid_matcher import run_enhanced_matching
from src.experiments.simplified_agentic_generator import generate_simplified_heuristics, get_leaderboard_target_f1
from src.prompts.hybrid_matcher_prompt import get_prompt_data
from src.utils.json_serializer import json_serialize


def get_available_datasets() -> list[str]:
    """Get list of all available datasets from data/raw directory"""
    data_dir = pathlib.Path("data/raw")
    if not data_dir.exists():
        return []

    datasets = []
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            # Check if it has the required files
            required_files = ["tableA.csv", "tableB.csv", "test.csv"]
            if all((dataset_dir / file).exists() for file in required_files):
                datasets.append(dataset_dir.name)

    return sorted(datasets)


async def run_dev_only_analysis_with_params(
    dataset: str,
    params: Dict[str, Any],
    model: str = "gpt-4.1-nano",
    concurrency: int = 3,
    embedding_base_url: str = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Run dev set analysis with specific hyperparameters - direct call to run_enhanced_matching"""
    print(f"🎯 Running dev analysis on {dataset} with use_validation=True")

    # Call run_enhanced_matching directly on the original dataset
    # use_validation=True makes it automatically sample 200 records from validation data
    raw_results = await run_enhanced_matching(
        dataset=dataset,  # Use original dataset name
        max_candidates=params.get("max_candidates", 150),
        model=model,
        semantic_weight=params.get("semantic_weight", 0.5),
        trigram_weight=params.get("trigram_weight"),
        syntactic_weight=params.get("syntactic_weight"),
        use_validation=True,  # Let run_enhanced_matching handle validation sampling internally
        concurrency=concurrency,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    )

    # Transform format to match caller expectations
    # run_enhanced_matching returns: {'f1': 0.85, 'cost': 1.2, ...}
    # Callers expect: {'metrics': {'f1': 0.85}, 'cost_usd': 1.2, ...}
    return {
        "metrics": {
            "f1": raw_results["f1"],
            "precision": raw_results["precision"],
            "recall": raw_results["recall"],
            "accuracy": raw_results["accuracy"],
        },
        "cost_usd": raw_results["cost"],
        "predictions": raw_results.get("predictions", {}),
        "failure_analysis": raw_results.get("failure_analysis", {}),
        "early_decisions": raw_results.get("early_decisions", 0),
        "llm_calls": raw_results.get("llm_calls", 0),
        "llm_call_reduction": raw_results.get("llm_call_reduction", 0.0),
    }


async def run_dev_only_analysis(dataset: str, model: str = "gpt-4.1-nano", concurrency: int = 3) -> Dict[str, Any]:
    """Run dev set analysis without test set leakage - NO FILE SWAPPING"""
    # Get optimal candidate count from recall analysis if available
    optimal_candidates = get_optimal_candidates_for_dataset(dataset)
    default_candidates = optimal_candidates if optimal_candidates else 150

    # Use 3-weight system: 0.6 semantic, 0.2 trigram, 0.2 syntactic
    params = {
        "max_candidates": default_candidates,
        "semantic_weight": 0.6,
        "trigram_weight": 0.2,
        "syntactic_weight": 0.2,
        "use_semantic": True,
    }

    print(f"🎯 Dev analysis parameters: candidates={default_candidates}, semantic=0.6, trigram=0.2, syntactic=0.2")

    return await run_dev_only_analysis_with_params(
        dataset=dataset,
        params=params,
        model=model,
        concurrency=concurrency,
    )


async def run_complete_pipeline(
    dataset: str,
    resume: bool = False,
    concurrency: int = 3,
    model: str = "gpt-4.1-nano",
    known_best_params: Optional[Dict[str, Any]] = None,
    use_train_for_rules: bool = False,
    mode: str = "prompt-modification",
    no_cache: bool = False,
    embedding_base_url: str = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Complete pipeline: dev analysis -> ACTUAL rule generation -> test with enhanced matching"""

    print("🚀 COMPLETE ENTITY MATCHING PIPELINE", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    if resume:
        print("🔄 RESUME MODE: Will skip completed steps", flush=True)
    print("=" * 60, flush=True)

    # Create pipeline registry and experiment configuration
    from src.entity_matching.experiment_config import ExperimentConfig

    # Initialize experiment registry for this pipeline run
    registry = ExperimentRegistry()
    print(f"📋 Initialized experiment registry: {registry.pipeline_run_id}")

    # Create baseline experiment configuration for this pipeline run
    base_experiment_config = ExperimentConfig(
        dataset=dataset,
        llm_model=model,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
        concurrency=concurrency,
        mode=mode,
        use_train_for_rules=use_train_for_rules,
        no_cache=no_cache,
        max_candidates=known_best_params.get("max_candidates", 50) if known_best_params else 50,
        semantic_weight=known_best_params.get("semantic_weight", 0.5) if known_best_params else 0.5,
        trigram_weight=known_best_params.get("trigram_weight") if known_best_params else None,
        syntactic_weight=known_best_params.get("syntactic_weight") if known_best_params else None,
    )

    print(f"🔬 Pipeline Registry ID: {registry.pipeline_run_id}")
    print(f"⚙️ Base Config: {base_experiment_config}")
    print("=" * 60, flush=True)

    # Check for existing checkpoint
    checkpoint_file = f"results/{dataset}_pipeline_checkpoint.json"
    checkpoint = {}
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        print(f"📁 Loaded checkpoint: {list(checkpoint.keys())}", flush=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "pipeline_version": "complete_v3_working_rules",
    }

    # Initialize default_params for later use (will be overridden in different branches)
    optimal_candidates_default = get_optimal_candidates_for_dataset(dataset) or 100
    default_params = {
        "max_candidates": optimal_candidates_default,
        "semantic_weight": 0.6,
        "trigram_weight": 0.2,
        "syntactic_weight": 0.2,
        "model": model,
        "use_semantic": True,
    }

    # STEP 1: Analysis-driven optimization (default) OR known params
    if known_best_params:
        print(f"✅ STEP 1: Using provided hyperparameters: {known_best_params}")
        print("⏳ Running single dev evaluation to get predictions for rule generation...")

        # Create dev experiment configuration
        # Load current prompt data
        current_prompt_data = get_prompt_data()

        dev_config = ExperimentConfig(
            dataset=dataset,
            llm_model=model,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
            use_validation=True,  # Key difference for dev stage
            max_candidates=known_best_params.get("max_candidates", get_optimal_candidates_for_dataset(dataset) or 150),
            semantic_weight=known_best_params.get("semantic_weight", 0.6),
            trigram_weight=known_best_params.get("trigram_weight", 0.2),
            syntactic_weight=known_best_params.get("syntactic_weight", 0.2),
            prompt_data=current_prompt_data,
            concurrency=concurrency,
            mode=mode,
            no_cache=no_cache,
        )

        start_time = time.time()
        dev_results = await run_dev_only_analysis_with_params(
            dataset, known_best_params, model, concurrency, embedding_base_url, embedding_model
        )
        dev_time = time.time() - start_time

        # Register dev experiment
        registry.register_experiment(
            dev_config,
            "dev",
            {
                "f1": dev_results["metrics"]["f1"],
                "precision": dev_results["metrics"]["precision"],
                "recall": dev_results["metrics"]["recall"],
                "cost_usd": dev_results["cost_usd"],
                "processing_time": dev_time,
            },
            "Development stage with known parameters and validation data",
        )

        # Get optimal candidate count from recall analysis if available
        optimal_candidates = get_optimal_candidates_for_dataset(dataset)
        default_candidates = optimal_candidates if optimal_candidates else 150

        # Ensure optimal_params has all required fields
        optimal_params = {
            "max_candidates": known_best_params.get("max_candidates", default_candidates),
            "semantic_weight": known_best_params.get("semantic_weight", 0.6),  # Default to 3-weight system
            "trigram_weight": known_best_params.get("trigram_weight", 0.2),
            "syntactic_weight": known_best_params.get("syntactic_weight", 0.2),
            "model": known_best_params.get("model", model),  # Use dev model if not specified
            "use_semantic": known_best_params.get("use_semantic", True),
        }

        print(
            f"🎯 Known params dev run: candidates={optimal_params['max_candidates']}, "
            f"semantic={optimal_params['semantic_weight']}, "
            f"trigram={optimal_params['trigram_weight']}, "
            f"syntactic={optimal_params['syntactic_weight']}"
        )

        print(
            f"✅ Dev Results with known params: F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}"
        )

        # Generate rules based on the dev results
        print("🤖 Generating rules based on known hyperparameters...", flush=True)
        print("    📡 Starting Claude optimization session (progress will be shown below)...", flush=True)

        analysis_file = f"results/{dataset}_claude_analysis.json"
        analysis_data = analyze_dataset_for_claude(
            dataset=dataset,
            max_pairs=200,
            max_candidates=known_best_params.get("max_candidates", 50) if known_best_params else 50,
            output_file=analysis_file,
            verbose=True,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
        )

        try:
            heuristics_file, rule_cost_info = await generate_simplified_heuristics(
                dataset,
                dev_results,
                f"results/generated_rules/{dataset}_known_params_config.json",
                analysis_data,
                model,
                no_cache,
                mode=mode,
                optimal_params=optimal_params,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
            )
        except Exception as e:
            print(f"❌ Known params rule generation failed: {e}")
            if "cancel scope" in str(e) or "task group" in str(e):
                print("   🔧 Detected async scope error - trying to continue...")
                # Try to find any generated files anyway
                potential_files = [
                    f"results/generated_rules/{dataset}_known_params_config.json",
                    "results/temp/generated_rules.json",
                    "results/temp/final_rules.json",
                    f"{dataset}_rules_final.json",
                    f"{dataset}_rules.json",
                ]
                heuristics_file = None
                for pf in potential_files:
                    if os.path.exists(pf):
                        heuristics_file = pf
                        print(f"   🔍 Found generated file: {pf}")
                        break
                if not heuristics_file:
                    print("   ❌ No generated files found")
                rule_cost_info = {"error": str(e), "method": "async_error"}
            else:
                raise

        if heuristics_file and os.path.exists(heuristics_file):
            checkpoint["heuristics_file"] = heuristics_file
            checkpoint["rule_generation_cost"] = rule_cost_info
            print(f"✅ Rules generated: {heuristics_file}", flush=True)
        else:
            raise RuntimeError("Rule generation failed for known parameters")
    elif "dev_results" in checkpoint and "optimal_params" in checkpoint:
        print("✅ STEP 1: Using cached dev results from checkpoint")
        dev_results = checkpoint["dev_results"]
        optimal_params = checkpoint["optimal_params"]
        dev_time = checkpoint.get("dev_time", 0)

        print(f"📊 Loaded: F1={dev_results['metrics']['f1']:.4f}, Config={optimal_params}")

        # Ensure heuristics file exists in checkpoint
        if "heuristics_file" not in checkpoint:
            raise RuntimeError("Checkpoint missing heuristics file - may need to regenerate")
    else:
        # Default: Analysis-driven optimization
        print("🔬 STEP 1: ANALYSIS-DRIVEN OPTIMIZATION (Default approach)")
        print("⏳ Running rich analysis and joint hyperparameter + rule optimization...")

        from src.entity_matching.analysis import analyze_dataset_for_claude

        start_time = time.time()

        # Generate rich analysis
        analysis_file = f"results/{dataset}_claude_analysis.json"
        analysis_data = analyze_dataset_for_claude(
            dataset=dataset,
            max_pairs=200,
            max_candidates=known_best_params.get("max_candidates", 50) if known_best_params else 50,
            output_file=analysis_file,
            verbose=True,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
        )

        # Get optimal candidate count from recall analysis if available
        optimal_candidates = get_optimal_candidates_for_dataset(dataset)
        if optimal_candidates:
            print(f"🎯 Using optimal candidate count: {optimal_candidates} (from recall analysis)")
            default_candidates = optimal_candidates
        else:
            print(f"⚠️ No recall analysis found for {dataset}, using default: 100 candidates")
            default_candidates = 100

        # Run a quick evaluation with 3-weight system to get real predictions for rule generation
        default_params = {
            "max_candidates": default_candidates,
            "semantic_weight": 0.6,
            "trigram_weight": 0.2,  # Use 3-weight system
            "syntactic_weight": 0.2,
            "model": model,
            "use_semantic": True,
        }

        print(f"🎯 Rule generation dev run: candidates={default_candidates}, semantic=0.6, trigram=0.2, syntactic=0.2")

        # Use the requested concurrency level
        analysis_concurrency = concurrency

        dev_cache_file = f"results/temp/{dataset}_dev_predictions.json"
        if os.path.exists(dev_cache_file):
            print("📁 Loading cached dev predictions...")
            with open(dev_cache_file) as f:
                dev_results = json.load(f)
            print(
                f"✅ Using cached dev predictions: F1={dev_results['metrics']['f1']:.4f}, {len(dev_results.get('predictions', {}))} predictions"
            )
            
            # Create dev experiment config for registry (needed for Step 3A)
            current_prompt_data = get_prompt_data()
            dev_config = ExperimentConfig(
                dataset=dataset,
                llm_model=model,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                use_validation=True,  # Dev stage uses validation data
                max_candidates=default_candidates,
                semantic_weight=0.6,
                trigram_weight=0.2,
                syntactic_weight=0.2,
                prompt_data=current_prompt_data,
                concurrency=analysis_concurrency,
                mode=mode,
                no_cache=no_cache,
            )

            # Register cached dev experiment for Step 3A consistency
            registry.register_experiment(
                dev_config,
                "dev",
                {
                    "f1": dev_results["metrics"]["f1"],
                    "precision": dev_results["metrics"]["precision"],
                    "recall": dev_results["metrics"]["recall"],
                    "cost_usd": dev_results.get("cost_usd", 0),
                    "processing_time": 0,  # Cached result
                },
                dev_results.get("predictions", {}),
            )
        else:
            # Create dev experiment configuration for analysis-driven approach
            # Load current prompt data
            current_prompt_data = get_prompt_data()

            dev_config = ExperimentConfig(
                dataset=dataset,
                llm_model=model,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
                use_validation=True,  # Dev stage uses validation data
                max_candidates=default_candidates,
                semantic_weight=0.6,
                trigram_weight=0.2,
                syntactic_weight=0.2,
                prompt_data=current_prompt_data,
                concurrency=analysis_concurrency,
                mode=mode,
                no_cache=no_cache,
            )

            print("🔄 Running quick evaluation to get predictions for rule generation...")
            dev_results = await run_dev_only_analysis_with_params(
                dataset, default_params, model, analysis_concurrency, embedding_base_url, embedding_model
            )
            print(
                f"✅ Quick evaluation: F1={dev_results['metrics']['f1']:.4f}, {len(dev_results.get('predictions', {}))} predictions"
            )

            # Register dev experiment
            registry.register_experiment(
                dev_config,
                "dev",
                {
                    "f1": dev_results["metrics"]["f1"],
                    "precision": dev_results["metrics"]["precision"],
                    "recall": dev_results["metrics"]["recall"],
                    "cost_usd": dev_results["cost_usd"],
                    "processing_time": 0,  # Will be updated with actual time later
                },
                "Development stage with analysis-driven optimization and validation data",
            )

            # Cache dev predictions
            os.makedirs(os.path.dirname(dev_cache_file), exist_ok=True)

            # Clean dev_results for JSON serialization
            cleaned_dev_results = json_serialize(dev_results)

            with open(dev_cache_file, "w") as f:
                json.dump(cleaned_dev_results, f, indent=2)
            print(f"💾 Cached dev predictions to {dev_cache_file}")

        print("🤖 Generating joint hyperparameter + rule optimization with Claude...", flush=True)
        print("    📡 Starting Claude optimization session (progress will be shown below)...", flush=True)
        try:
            heuristics_file, rule_cost_info = await generate_simplified_heuristics(
                dataset,
                dev_results,
                f"results/generated_rules/{dataset}_analysis_driven_config.json",
                analysis_data,
                model,
                no_cache,
                mode=mode,
                optimal_params=default_params,
                embedding_model=embedding_model,
                embedding_base_url=embedding_base_url,
            )
        except Exception as e:
            print(f"❌ Analysis-driven rule generation failed: {e}")
            if "cancel scope" in str(e) or "task group" in str(e):
                print("   🔧 Detected async scope error - trying to continue...")
                # Try to find any generated files anyway
                potential_files = [
                    f"results/generated_rules/{dataset}_analysis_driven_config.json",
                    "results/temp/generated_rules.json",
                    "results/temp/final_rules.json",
                    f"{dataset}_rules_final.json",
                    f"{dataset}_rules.json",
                ]
                heuristics_file = None
                for pf in potential_files:
                    if os.path.exists(pf):
                        heuristics_file = pf
                        print(f"   🔍 Found generated file: {pf}")
                        break
                if not heuristics_file:
                    print("   ❌ No generated files found")
                rule_cost_info = {"error": str(e), "method": "async_error"}
            else:
                raise

        analysis_time = time.time() - start_time

        if heuristics_file and os.path.exists(heuristics_file):
            # Load the generated configuration to extract hyperparameters
            with open(heuristics_file) as f:
                generated_config = json.load(f)

            # Extract hyperparameters from Claude's output
            if "hyperparameters" in generated_config:
                hyperparams = generated_config["hyperparameters"]
                # Get optimal candidate count for Claude's hyperparameters
                optimal_candidates = get_optimal_candidates_for_dataset(dataset)
                default_candidates = optimal_candidates if optimal_candidates else 150

                claude_params = {
                    "max_candidates": hyperparams.get("max_candidates")
                    or hyperparams.get("n_candidates", default_candidates),
                    "semantic_weight": hyperparams.get("semantic_weight", 0.5),
                    "trigram_weight": hyperparams.get("trigram_weight"),
                    "syntactic_weight": hyperparams.get("syntactic_weight"),
                    "model": model,
                    "use_semantic": True,
                }
                print(f"✅ Claude chose hyperparameters: {claude_params}")

                optimal_params = claude_params
            else:
                # Fallback if no hyperparameters in output
                print("⚠️ No hyperparameters in Claude output, using analysis-based defaults")
                optimal_candidates = get_optimal_candidates_for_dataset(dataset)
                default_candidates = optimal_candidates if optimal_candidates else 150

                optimal_params = {
                    "max_candidates": default_candidates,
                    "semantic_weight": 0.7,  # Higher for analysis-driven approach
                    "trigram_weight": 0.18,  # Balanced 3-weight system
                    "syntactic_weight": 0.12,  # 0.7 + 0.18 + 0.12 = 1.0
                    "model": model,
                    "use_semantic": True,
                }

            # Claude's experiments are already tracked via MCP server, no need for additional dev run
            print("✅ Using dev evaluation results (Claude experiments tracked separately via MCP)")

            dev_time = analysis_time

            print(
                f"✅ Analysis-driven approach: F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}"
            )
            print(f"💰 Analysis + rule generation cost: ${rule_cost_info.get('total_cost_usd', 0):.4f}")

            # Save the heuristics file info for later use
            checkpoint["heuristics_file"] = heuristics_file
            checkpoint["rule_generation_cost"] = rule_cost_info

        else:
            raise RuntimeError("Analysis-driven optimization failed - no fallback available")

    # NOW refresh registry after Claude optimization completes - this avoids the hanging issue
    print("🔄 Claude optimization complete, now refreshing registry...")
    registry.reload_from_disk()  # Explicit reload
    best_claude_config_for_3b = registry.get_best_claude_experiment()
    claude_experiments_for_3b = registry.get_claude_experiments()
    print(f"🔍 Found {len(claude_experiments_for_3b)} Claude experiments for Step 3B")

    # Get Claude's best results for accurate logging
    best_claude_config = registry.get_best_claude_experiment()
    if best_claude_config:
        claude_experiments = registry.get_claude_experiments()
        # Find the best experiment to get its F1 score
        best_claude_entry = None
        best_f1 = 0.0
        for exp in claude_experiments:
            f1 = registry._extract_f1_score(exp.results)
            if f1 and f1 > best_f1:
                best_f1 = f1
                best_claude_entry = exp
        
        if best_claude_entry:
            print(f"✅ Best Dev Results (Claude): F1={best_f1:.4f}, Cost=${best_claude_entry.results.get('cost_usd', 0):.3f}")
        else:
            print(f"✅ Best Dev Results (initial): F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}")
    else:
        print(f"✅ Best Dev Results (initial): F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}")
    # Display Claude's optimal parameters if available
    if best_claude_config:
        print(
            f"🎯 Optimal Parameters (Claude): {best_claude_config.max_candidates} candidates, {best_claude_config.semantic_weight:.2f} semantic, {best_claude_config.trigram_weight:.2f} trigram, {best_claude_config.syntactic_weight:.2f} syntactic, {best_claude_config.llm_model}"
        )
    else:
        print(
            f"🎯 Optimal Parameters (initial): {optimal_params['max_candidates']} candidates, {optimal_params['semantic_weight']:.2f} semantic weight, {optimal_params['model']}"
        )

    results["dev_results"] = {
        "f1": dev_results["metrics"]["f1"],
        "precision": dev_results["metrics"]["precision"],
        "recall": dev_results["metrics"]["recall"],
        "cost_usd": dev_results["cost_usd"],
        "processing_time": dev_time,
    }
    results["optimal_params"] = optimal_params

    # STEP 2: Rules are already generated in analysis-driven approach or use cached
    if "heuristics_file" in checkpoint:
        print("✅ STEP 2: Using rules from analysis-driven optimization (already generated)")
        heuristics_file = checkpoint["heuristics_file"]
        rule_generation_cost = checkpoint.get(
            "rule_generation_cost", {"total_cost_usd": 0.0, "method": "analysis_driven"}
        )
    else:
        # This shouldn't happen with the new simplified flow, but handle gracefully
        raise RuntimeError("No heuristics file found - analysis-driven optimization should have generated rules")

    if heuristics_file:
        results["heuristics_file"] = heuristics_file
        results["rule_generation"] = "claude_sdk_success"
        print(f"✅ Rules available: {heuristics_file}")
    else:
        results["rule_generation"] = "failed"
        print("❌ Rule generation failed")
        return results

    # Pipeline execution logic based on mode
    if mode == "weights-only":
        print("\n💨 WEIGHTS-ONLY MODE: Skipping both 3A and 3B (identical without rules)")
        # Use dev results as final results since no additional testing needed
        enhanced_results = {
            "f1": dev_results["metrics"]["f1"],
            "precision": dev_results["metrics"]["precision"],
            "recall": dev_results["metrics"]["recall"],
            "cost": 0.0,  # No additional cost for skipped steps
            "method": "weights_only_dev_results",
        }
        baseline_results = {"metrics": enhanced_results, "cost_usd": 0.0}
        enhanced_time = 0
        baseline_time = 0

        print(f"✅ Final Results (dev optimization): F1={enhanced_results['f1']:.4f}")

        # Skip to results compilation
        results["test_results"] = {
            "baseline": baseline_results,
            "enhanced": enhanced_results,
            "baseline_time": baseline_time,
            "enhanced_time": enhanced_time,
            "target_f1": get_leaderboard_target_f1(dataset),
            "mode": mode,
        }

        return results


    # STEP 3A: Test set evaluation WITHOUT rules (baseline with optimal params)
    # Always run 3A - prompt-modification mode needs it for before/after comparison
    print("\n🎯 STEP 3A: FINAL TEST EVALUATION WITHOUT rules (optimal params baseline)")
    print("⏳ Running baseline matching on FULL TEST SET (this is the real evaluation)...")

    # Get dev experiment config for baseline consistency
    dev_experiment = registry.get_dev_experiment()
    if not dev_experiment:
        raise RuntimeError("No dev experiment found - cannot ensure consistency between dev and 3A")

    start_time = time.time()
    # Extract learned weights from heuristics file for baseline evaluation
    baseline_semantic_weight = optimal_params["semantic_weight"]  # default
    baseline_trigram_weight = optimal_params.get("trigram_weight")  # default
    baseline_syntactic_weight = optimal_params.get("syntactic_weight")  # default

    if heuristics_file and os.path.exists(heuristics_file):
        try:
            with open(heuristics_file) as f:
                heuristics_config = json.load(f)
                if "hyperparameters" in heuristics_config:
                    hyperparams = heuristics_config["hyperparameters"]
                    baseline_semantic_weight = hyperparams.get("semantic_weight", baseline_semantic_weight)
                    baseline_trigram_weight = hyperparams.get("trigram_weight", baseline_trigram_weight)
                    baseline_syntactic_weight = hyperparams.get("syntactic_weight", baseline_syntactic_weight)
        except Exception:
            pass  # Use default if can't load

    # Apply same fix: Always use 3-weight system with proper defaults for baseline too
    if baseline_trigram_weight is None or baseline_syntactic_weight is None:
        remaining_weight = 1.0 - baseline_semantic_weight
        baseline_trigram_weight = remaining_weight * 0.6  # 60% of remaining to trigram
        baseline_syntactic_weight = remaining_weight * 0.4  # 40% of remaining to syntactic

    # Display baseline parameters for comparison
    print("📊 Baseline Parameters:")
    print(f"   Max Candidates: {optimal_params['max_candidates']}")
    print(
        f"   Weights: semantic={baseline_semantic_weight:.3f}, trigram={baseline_trigram_weight:.3f}, syntactic={baseline_syntactic_weight:.3f}"
    )
    print(f"   Weight Sum: {baseline_semantic_weight + baseline_trigram_weight + baseline_syntactic_weight:.3f}")

    # Create 3A baseline experiment config (identical to dev but with test data)
    baseline_config = ExperimentConfig(
        dataset=dev_experiment.dataset,
        llm_model=dev_experiment.llm_model,
        embedding_model=dev_experiment.embedding_model,
        embedding_base_url=dev_experiment.embedding_base_url,
        use_validation=False,  # Key difference - use test data
        max_candidates=dev_experiment.max_candidates,
        semantic_weight=baseline_semantic_weight,
        trigram_weight=baseline_trigram_weight,
        syntactic_weight=baseline_syntactic_weight,
        prompt_data=dev_experiment.prompt_data,  # Keep same prompt data
        concurrency=concurrency,
        mode=dev_experiment.mode,
        no_cache=dev_experiment.no_cache,
    )

    # Calculate max candidates needed for cache efficiency (before any stages run)
    # Use existing registry that was created at the beginning of the function
    best_claude_config = registry.get_best_claude_experiment()
    max_candidates_needed = baseline_config.max_candidates
    if best_claude_config:
        max_candidates_needed = max(baseline_config.max_candidates, best_claude_config.max_candidates)
        print(f"🔧 Using max_candidates={max_candidates_needed} for cache efficiency (3A:{baseline_config.max_candidates}, Claude:{best_claude_config.max_candidates})")
    else:
        print(f"🔧 Using max_candidates={max_candidates_needed} (no Claude experiments found)")

    raw_baseline_results = await run_enhanced_matching(
        dataset=dataset,
        max_candidates=max_candidates_needed,
        model="gpt-4.1-nano",  # Use cheaper model for test
        semantic_weight=baseline_semantic_weight,
        trigram_weight=baseline_trigram_weight,
        syntactic_weight=baseline_syntactic_weight,
        concurrency=concurrency,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    )

    # Transform format to match caller expectations
    baseline_results = {
        "metrics": {
            "f1": raw_baseline_results["f1"],
            "precision": raw_baseline_results["precision"],
            "recall": raw_baseline_results["recall"],
        },
        "cost_usd": raw_baseline_results["cost"],
        "predictions": raw_baseline_results.get("predictions", {}),
    }

    baseline_time = time.time() - start_time

    # Register 3A baseline experiment
    registry.register_experiment(
        baseline_config,
        "3A_baseline",
        {
            "f1": baseline_results["metrics"]["f1"],
            "precision": baseline_results["metrics"]["precision"],
            "recall": baseline_results["metrics"]["recall"],
            "cost_usd": baseline_results["cost_usd"],
            "processing_time": baseline_time,
        },
        "Baseline test evaluation without rules, using dev stage parameters",
    )

    print(
        f"✅ Baseline Results (no rules): F1={baseline_results['metrics']['f1']:.4f}, Cost=${baseline_results['cost_usd']:.3f}"
    )

    # STEP 3B: Test set evaluation WITH generated rules
    print("\n🎯 STEP 3B: FINAL TEST EVALUATION WITH rules (enhanced approach)")

    # Check if baseline already exceeds target - if so, skip expensive rules test
    # (Skip this check for prompt-modification mode where baseline_results is None)
    target_f1 = get_leaderboard_target_f1(dataset)

    if baseline_results and baseline_results["metrics"]["f1"] >= target_f1:
        baseline_f1 = baseline_results["metrics"]["f1"]
        print(f"🎉 BASELINE ALREADY EXCEEDS TARGET! ({baseline_f1:.4f} >= {target_f1:.1f})")
        print("   Skipping expensive rules test since baseline is already excellent.")
        print("   Using baseline results as final results.")

        # Use baseline results as enhanced results
        enhanced_results = {
            "f1": baseline_results["metrics"]["f1"],
            "precision": baseline_results["metrics"]["precision"],
            "recall": baseline_results["metrics"]["recall"],
            "cost": baseline_results["cost_usd"],
            "method": "baseline_skip_rules",
        }
        enhanced_time = 0  # No additional time needed

        print(
            f"✅ Final Results (baseline used): F1={enhanced_results['f1']:.4f}, Cost=${enhanced_results['cost']:.3f}"
        )
    else:
        print("⏳ Running enhanced matching with rules on FULL TEST SET (this is the real evaluation)...")

        start_time = time.time()

        # Extract learned parameters from the heuristics file
        learned_max_candidates = optimal_params["max_candidates"]  # default
        learned_semantic_weight = optimal_params["semantic_weight"]  # default
        learned_trigram_weight = optimal_params.get("trigram_weight")  # default
        learned_syntactic_weight = optimal_params.get("syntactic_weight")  # default

        if heuristics_file and os.path.exists(heuristics_file):
            try:
                with open(heuristics_file) as f:
                    heuristics_config = json.load(f)
                    if "hyperparameters" in heuristics_config:
                        hyperparams = heuristics_config["hyperparameters"]
                        learned_max_candidates = hyperparams.get("max_candidates", learned_max_candidates)
                        learned_semantic_weight = hyperparams.get("semantic_weight", learned_semantic_weight)
                        learned_trigram_weight = hyperparams.get("trigram_weight", learned_trigram_weight)
                        learned_syntactic_weight = hyperparams.get("syntactic_weight", learned_syntactic_weight)
            except Exception as e:
                print(f"⚠️ Could not load parameters from {heuristics_file}: {e}")

        # Force use of Claude's optimized weights - no fallback allowed
        if learned_trigram_weight is None or learned_syntactic_weight is None:
            raise ValueError(
                f"Claude must provide complete weight optimization! Got: semantic={learned_semantic_weight}, trigram={learned_trigram_weight}, syntactic={learned_syntactic_weight}"
            )

        print(
            f"✅ Using Claude's optimized parameters: candidates={learned_max_candidates}, semantic={learned_semantic_weight:.3f}, trigram={learned_trigram_weight:.3f}, syntactic={learned_syntactic_weight:.3f}"
        )

        if best_claude_config_for_3b:
            print(f"🏆 Using winning Claude experiment: {best_claude_config_for_3b.experiment_id}")
            # Create 3B config using Claude's winning settings but with same base settings as 3A
            enhanced_config = ExperimentConfig(
                dataset=baseline_config.dataset,
                llm_model=baseline_config.llm_model,
                embedding_model=baseline_config.embedding_model,
                embedding_base_url=baseline_config.embedding_base_url,
                use_validation=False,  # Test data like 3A
                max_candidates=max_candidates_needed,  # Use max for cache efficiency
                semantic_weight=best_claude_config_for_3b.semantic_weight,
                trigram_weight=best_claude_config_for_3b.trigram_weight,
                syntactic_weight=best_claude_config_for_3b.syntactic_weight,
                prompt_data=best_claude_config_for_3b.prompt_data,
                concurrency=concurrency,
                mode=baseline_config.mode,
                no_cache=baseline_config.no_cache,
                heuristic_file=heuristics_file,
            )
        else:
            print("⚠️ No Claude experiments found - using file-based parameters")
            # Fallback: create enhanced config based on file parameters
            enhanced_config = ExperimentConfig(
                dataset=baseline_config.dataset,
                llm_model=baseline_config.llm_model,
                embedding_model=baseline_config.embedding_model,
                embedding_base_url=baseline_config.embedding_base_url,
                use_validation=False,  # Test data like 3A
                max_candidates=learned_max_candidates,
                semantic_weight=learned_semantic_weight,
                trigram_weight=learned_trigram_weight,
                syntactic_weight=learned_syntactic_weight,
                concurrency=concurrency,
                mode=baseline_config.mode,
                no_cache=baseline_config.no_cache,
                heuristic_file=heuristics_file,
            )

        enhanced_results = await run_enhanced_matching(
            dataset=dataset,
            max_candidates=enhanced_config.max_candidates,
            model="gpt-4.1-nano",  # Use cheaper model for test
            semantic_weight=enhanced_config.semantic_weight,
            trigram_weight=enhanced_config.trigram_weight,
            syntactic_weight=enhanced_config.syntactic_weight,
            heuristic_file=heuristics_file,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
        )

        # Calculate enhanced processing time
        enhanced_time = time.time() - start_time

        # Register 3B enhanced experiment
        registry.register_experiment(
            enhanced_config,
            "3B_enhanced",
            {
                "f1": enhanced_results["f1"],
                "precision": enhanced_results["precision"],
                "recall": enhanced_results["recall"],
                "cost_usd": enhanced_results["cost"],
                "processing_time": enhanced_time,
            },
            f"Enhanced test evaluation with rules{' using winning Claude experiment' if best_claude_config else ' using file-based parameters'}",
        )

        print(
            f"✅ Enhanced Results (with rules): F1={enhanced_results['f1']:.4f}, Cost=${enhanced_results['cost']:.3f}"
        )

    f1_improvement = enhanced_results["f1"] - baseline_results["metrics"]["f1"]
    cost_change = enhanced_results["cost"] - baseline_results["cost_usd"]

    print("\n📊 A/B COMPARISON:")
    print(
        f"F1 Change: {f1_improvement:+.4f} ({'✅ IMPROVED' if f1_improvement > 0 else '❌ WORSE' if f1_improvement < 0 else '➡️ NO CHANGE'})"
    )
    print(f"Cost Change: ${cost_change:+.3f}")
    print(
        f"Rules {'✅ HELPED' if f1_improvement > 0.01 else '❌ DID NOT HELP' if f1_improvement < -0.01 else '➡️ NEUTRAL'}"
    )

    results["baseline_results"] = {
        "f1": baseline_results["metrics"]["f1"],
        "precision": baseline_results["metrics"]["precision"],
        "recall": baseline_results["metrics"]["recall"],
        "cost_usd": baseline_results["cost_usd"],
        "processing_time": baseline_time,
        "predictions": baseline_results.get("predictions", {}),  # Include predictions for duplicate-aware eval
    }

    results["enhanced_results"] = {
        "f1": enhanced_results["f1"],
        "precision": enhanced_results["precision"],
        "recall": enhanced_results["recall"],
        "cost_usd": enhanced_results["cost"],
        "processing_time": enhanced_time,
        "early_decisions": enhanced_results.get("early_decisions", 0),
        "llm_calls": enhanced_results.get("llm_calls", 0),
        "llm_call_reduction": enhanced_results.get("llm_call_reduction", 0),
        "predictions": enhanced_results.get("predictions", {}),  # Include predictions for failure analysis
    }

    results["ab_comparison"] = {
        "f1_improvement": f1_improvement,
        "cost_change": cost_change,
        "rules_helped": f1_improvement > 0.01,
    }

    # FINAL SUMMARY
    rule_gen_cost = rule_generation_cost.get("total_cost_usd", 0.0)
    total_cost = dev_results["cost_usd"] + baseline_results["cost_usd"] + enhanced_results["cost"] + rule_gen_cost
    total_time = dev_time + baseline_time + enhanced_time

    print(f"\n🏆 FINAL RESULTS FOR {dataset.upper()}")
    print(f"Dev F1:        {dev_results['metrics']['f1']:.4f} (${dev_results['cost_usd']:.3f})")
    print(f"Rule Generation: {rule_generation_cost.get('method', 'unknown')} (${rule_gen_cost:.4f})")
    print(
        f"Test Baseline: {baseline_results['metrics']['f1']:.4f} (${baseline_results['cost_usd']:.3f}) - optimal params only"
    )
    print(f"Test Enhanced: {enhanced_results['f1']:.4f} (${enhanced_results['cost']:.3f}) - optimal params + rules")
    print(f"Improvement:   {f1_improvement:+.4f} F1 points")
    print(f"Total Cost: ${total_cost:.3f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Rules: {heuristics_file}")
    print(f"LLM Call Reduction: {enhanced_results.get('llm_call_reduction', 0):.1f}%")

    # Check if we beat the leaderboard (check both baseline and enhanced)
    target_f1 = get_leaderboard_target_f1(dataset)
    baseline_beats_leaderboard = (
        baseline_results["metrics"]["f1"] > target_f1 / 100
        if target_f1 > 10
        else baseline_results["metrics"]["f1"] > target_f1
    )
    enhanced_beats_leaderboard = (
        enhanced_results["f1"] > target_f1 / 100 if target_f1 > 10 else enhanced_results["f1"] > target_f1
    )
    beat_leaderboard = enhanced_beats_leaderboard or baseline_beats_leaderboard

    if baseline_beats_leaderboard and enhanced_beats_leaderboard:
        leaderboard_msg = f"🎉 BOTH BEAT LEADERBOARD TARGET ({target_f1:.1f})!"
    elif baseline_beats_leaderboard and not enhanced_beats_leaderboard:
        leaderboard_msg = f"🎉 BASELINE BEATS LEADERBOARD ({target_f1:.1f}) - Rules hurt performance!"
    elif enhanced_beats_leaderboard and not baseline_beats_leaderboard:
        leaderboard_msg = f"🎉 ENHANCED BEATS LEADERBOARD ({target_f1:.1f}) - Rules helped!"
    else:
        leaderboard_msg = f"📈 Still working on it (target: {target_f1:.1f})"

    print(f"Leaderboard: {leaderboard_msg}")

    # ENHANCED: Save comprehensive optimization artifacts for reuse
    optimization_artifacts = {}

    # Save generated heuristics file content if it exists
    if heuristics_file and os.path.exists(heuristics_file):
        with open(heuristics_file) as f:
            optimization_artifacts["generated_heuristics"] = json.load(f)
        optimization_artifacts["heuristics_file_path"] = heuristics_file

    # Save sample data used for optimization
    sample_data_file = f"results/temp/sample_data_{dataset}.json"
    if os.path.exists(sample_data_file):
        with open(sample_data_file) as f:
            optimization_artifacts["sample_data"] = json.load(f)

    # Save optimal hyperparameters discovered
    optimization_artifacts["optimal_hyperparameters"] = {
        "baseline": {
            "max_candidates": baseline_results.get("max_candidates", optimal_candidates),
            "semantic_weight": baseline_results.get("semantic_weight", default_params.get("semantic_weight")),
            "trigram_weight": baseline_results.get("trigram_weight", default_params.get("trigram_weight")),
            "syntactic_weight": baseline_results.get("syntactic_weight", default_params.get("syntactic_weight")),
        }
    }

    # Extract enhanced hyperparameters if available
    if heuristics_file and os.path.exists(heuristics_file):
        try:
            with open(heuristics_file) as f:
                enhanced_config = json.load(f)
            if "hyperparameters" in enhanced_config:
                optimization_artifacts["optimal_hyperparameters"]["enhanced"] = enhanced_config["hyperparameters"]
        except:
            pass

    # Save EVERYTHING needed to reproduce results
    optimization_artifacts["reproduction_data"] = {}

    # 1. Save current prompt structure (if modified)
    try:
        current_prompt = get_prompt_data()
        optimization_artifacts["reproduction_data"]["final_prompt_structure"] = current_prompt
    except Exception as e:
        optimization_artifacts["reproduction_data"]["prompt_error"] = str(e)

    # 2. Save exact command to reproduce
    optimization_artifacts["reproduction_data"]["exact_command"] = {
        "script": "run_enhanced_matching.py",
        "args": [
            f"--dataset {dataset}",
            f"--max-candidates {baseline_results.get('max_candidates', optimal_candidates)}",
            f"--semantic-weight {baseline_results.get('semantic_weight', default_params.get('semantic_weight', 0.5))}",
            f"--trigram-weight {baseline_results.get('trigram_weight', default_params.get('trigram_weight'))}",
            f"--syntactic-weight {baseline_results.get('syntactic_weight', default_params.get('syntactic_weight'))}",
            f"--heuristic-file {heuristics_file}" if heuristics_file and os.path.exists(heuristics_file) else "",
            "--use-validation",
        ],
        "full_command": f"python run_enhanced_matching.py --dataset {dataset} --max-candidates {baseline_results.get('max_candidates', optimal_candidates)} --semantic-weight {baseline_results.get('semantic_weight', default_params.get('semantic_weight', 0.5))} --use-validation"
        + (f" --heuristic-file {heuristics_file}" if heuristics_file and os.path.exists(heuristics_file) else ""),
    }

    # 3. Save rule generation conversation logs with CONTENT
    log_files = [
        f"results/temp/claude_conversation_{dataset}.log",
        f"results/temp/optimization_log_{dataset}.json",
        "results/temp/mcp_server.log",
    ]
    optimization_artifacts["conversation_logs"] = []
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                # Read and include the actual log content for full reproduction
                with open(log_file, encoding="utf-8") as f:
                    content = f.read()
                    # Truncate if too long, but keep the important parts
                    if len(content) > 50000:  # 50KB limit
                        content = content[-50000:]  # Keep last 50KB (most recent)
                        content = "...[truncated]...\n" + content
                optimization_artifacts["conversation_logs"].append(
                    {
                        "file": log_file,
                        "content": content,
                        "size_bytes": len(content),
                        "note": "Full conversation content included for reproduction",
                    }
                )
            except Exception as e:
                optimization_artifacts["conversation_logs"].append(
                    {"file": log_file, "error": str(e), "note": "Failed to read log content"}
                )

    # 4. Save temp files that Claude generated
    temp_files = [
        f"results/temp/sample_data_{dataset}.json",
        "results/temp/generated_rules.json",
        "results/temp/prompt_data.json",
    ]
    optimization_artifacts["temp_artifacts"] = []
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                with open(temp_file) as f:
                    temp_content = json.load(f) if temp_file.endswith(".json") else f.read()
                optimization_artifacts["temp_artifacts"].append(
                    {
                        "file": temp_file,
                        "content": temp_content,
                        "note": "Temporary artifact generated during optimization",
                    }
                )
            except Exception as e:
                optimization_artifacts["temp_artifacts"].append({"file": temp_file, "error": str(e)})

    # Save method comparison and recommendation
    optimization_artifacts["method_recommendation"] = {
        "best_method": "enhanced" if enhanced_results["f1"] > baseline_results["metrics"]["f1"] else "baseline",
        "f1_improvement": f1_improvement,
        "should_use_rules": f1_improvement > 0.01,
        "reasoning": (
            "Rules provide significant improvement"
            if f1_improvement > 0.01
            else "Baseline sufficient, rules add complexity without benefit"
            if f1_improvement < -0.01
            else "Marginal difference, use simpler baseline approach"
        ),
    }

    results["optimization_artifacts"] = optimization_artifacts

    results["summary"] = {
        "dev_f1": dev_results["metrics"]["f1"],
        "baseline_f1": baseline_results["metrics"]["f1"],
        "enhanced_f1": enhanced_results["f1"],
        "f1_improvement": f1_improvement,
        "total_cost_usd": total_cost,
        "rule_generation_cost_usd": rule_gen_cost,
        "rule_generation_method": rule_generation_cost.get("method", "unknown"),
        "total_time_seconds": total_time,
        "beat_leaderboard": beat_leaderboard,
        "leaderboard_target": target_f1,
        "reusable_config_path": heuristics_file if heuristics_file and os.path.exists(heuristics_file) else None,
        # Experiment reproducibility and registry
        "pipeline_registry_id": registry.pipeline_run_id,
        "registry_path": str(registry.save_registry()),
        "experiment_genealogy": registry.get_experiment_genealogy(),
    }

    # Save results with comprehensive JSON serialization
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{dataset}_complete_pipeline.json"
    with open(results_file, "w") as f:
        json.dump(json_serialize(results), f, indent=2)

    print(f"📋 Results saved to: {results_file}")

    # Extract detailed failure analysis
    print("🔍 Extracting detailed failure analysis...")
    failure_analysis = extract_failure_records(dataset, results)
    results["failure_analysis"] = failure_analysis

    if failure_analysis.get("detailed_failures"):
        print(
            f"📊 Captured {failure_analysis['total_failures']} failures ({failure_analysis['false_positives']} FP, {failure_analysis['false_negatives']} FN)"
        )

    # Re-save results with failure analysis
    with open(results_file, "w") as f:
        json.dump(json_serialize(results), f, indent=2)

    # Generate updated internal leaderboard
    print("📊 Generating updated internal leaderboard...")
    try:
        result = subprocess.run(
            ["python", "generate_internal_leaderboard.py"], check=False, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            print("✅ Internal leaderboard updated successfully")
        else:
            print(f"⚠️ Leaderboard generation failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Could not generate leaderboard: {e}")

    return results


def extract_failure_records(dataset: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed failure records from results for analysis"""
    data_root = pathlib.Path("data") / "raw" / dataset

    # Load the original tables
    try:
        A_df = pd.read_csv(data_root / "tableA.csv")
        B_df = pd.read_csv(data_root / "tableB.csv")
        test_pairs = pd.read_csv(data_root / "test.csv")

        A_records = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
        B_records = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}

        predictions = results.get("enhanced_results", {}).get("predictions", {})
        if not predictions:
            return {"failure_analysis": "No predictions available for failure analysis"}

        failures = []

        # Use duplicate-aware evaluation for failure analysis
        from src.entity_matching.duplicate_aware_evaluation import duplicate_aware_evaluate

        # Prepare pairs data for duplicate-aware evaluation
        pairs_data = [(row.ltable_id, row.rtable_id, row.label) for _, row in test_pairs.iterrows()]
        
        # Get duplicate-aware predictions and labels
        preds, labels = duplicate_aware_evaluate(
            {int(k): int(v) for k, v in predictions.items()},  # Convert to int keys/values
            pairs_data, 
            B_records, 
            verbose=False
        )

        # Identify failures using duplicate-aware evaluation results
        for i, (_, row) in enumerate(test_pairs.iterrows()):
            left_id = row.ltable_id
            right_id = row.rtable_id
            true_label = row.label
            predicted_label = preds[i]

            # Check if this is a failure according to duplicate-aware evaluation
            if true_label != predicted_label:
                failure_type = "false_positive" if (true_label == 0 and predicted_label == 1) else "false_negative"
                predicted_right_id = predictions.get(str(left_id))  # Get original prediction

                failures.append(
                    {
                            "left_id": left_id,
                            "right_id": right_id,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "failure_type": failure_type,
                            "left_record": A_records.get(left_id, {}),
                            "right_record": B_records.get(right_id, {}),
                            "predicted_right_id": predicted_right_id,
                            "predicted_right_record": B_records.get(predicted_right_id, {})
                            if predicted_right_id
                            else None,
                        }
                    )

        return {
            "total_failures": len(failures),
            "false_positives": len([f for f in failures if f["failure_type"] == "false_positive"]),
            "false_negatives": len([f for f in failures if f["failure_type"] == "false_negative"]),
            "detailed_failures": failures,
        }

    except Exception as e:
        return {"failure_analysis_error": f"Could not extract failure records: {e}"}


async def main():
    parser = argparse.ArgumentParser(description="Complete entity matching pipeline")
    parser.add_argument("--dataset", help="Dataset name (e.g. beer, walmart_amazon)")
    parser.add_argument("--datasets", help="Run on comma-separated list of datasets or 'all' for all datasets")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent API requests (default: 20)")
    parser.add_argument(
        "--model", default="gpt-4.1-nano", help="Model to use for analysis-driven optimization (default: gpt-4.1-nano)"
    )
    parser.add_argument(
        "--mode",
        choices=["weights-only", "prompt-modification", "heuristics"],
        default="prompt-modification",
        help="Optimization mode: weights-only (fast weight tuning), prompt-modification (dynamic LLM guidance), heuristics (full rule generation)",
    )
    parser.add_argument(
        "--known-best-params",
        help='JSON string with known best hyperparameters (e.g. \'{"max_candidates": 50, "semantic_weight": 0.7}\')',
    )

    # Hyperparameter arguments
    parser.add_argument("--max-candidates", type=int, help="Maximum number of candidates to generate")
    parser.add_argument("--semantic-weight", type=float, help="Semantic similarity weight")
    parser.add_argument("--trigram-weight", type=float, help="Trigram similarity weight")
    parser.add_argument("--syntactic-weight", type=float, help="Syntactic similarity weight")
    parser.add_argument(
        "--use-train-for-rules",
        action="store_true",
        help="Use train set with optimal params to get more error examples for rule generation",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching and force regeneration of all analysis files",
    )

    parser.add_argument(
        "--embedding-base-url",
        help="Base URL for embedding API (e.g., http://localhost:8080 for TEI or https://api.openai.com/ for OpenAI). If not provided, uses local SentenceTransformer",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name (default: all-MiniLM-L6-v2 for local, or 'tei' for TEI endpoint)",
    )

    args = parser.parse_args()

    # Validate dataset arguments
    if not args.dataset and not args.datasets:
        parser.error("Either --dataset or --datasets must be specified")
    if args.dataset and args.datasets:
        parser.error("Cannot specify both --dataset and --datasets")

    # Get list of datasets to process
    if args.datasets:
        if args.datasets == "all":
            datasets = get_available_datasets()
            if not datasets:
                print("❌ No datasets found in data/raw directory")
                return None
            print(f"🗂️ Found {len(datasets)} datasets: {', '.join(datasets)}")
        else:
            # Parse comma-separated list
            datasets = [d.strip() for d in args.datasets.split(",")]
            print(f"🗂️ Processing {len(datasets)} datasets: {', '.join(datasets)}")

            # Validate dataset names
            available_datasets = get_available_datasets()
            invalid_datasets = [d for d in datasets if d not in available_datasets]
            if invalid_datasets:
                print(f"❌ Invalid datasets: {', '.join(invalid_datasets)}")
                print(f"Available datasets: {', '.join(available_datasets)}")
                return None
    else:
        datasets = [args.dataset]

    # Parse known best params if provided
    known_best_params = None
    if args.known_best_params:
        try:
            known_best_params = json.loads(args.known_best_params)
            print(f"🎯 Using known best parameters: {known_best_params}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON for known-best-params: {e}")
            return None

    # Build known_best_params from CLI arguments if provided
    if not known_best_params and any(
        [args.max_candidates, args.semantic_weight, args.trigram_weight, args.syntactic_weight]
    ):
        known_best_params = {}
        if args.max_candidates:
            known_best_params["max_candidates"] = args.max_candidates
        if args.semantic_weight is not None:
            known_best_params["semantic_weight"] = args.semantic_weight
        if args.trigram_weight is not None:
            known_best_params["trigram_weight"] = args.trigram_weight
        if args.syntactic_weight is not None:
            known_best_params["syntactic_weight"] = args.syntactic_weight
        print(f"🎯 Using parameters from CLI arguments: {known_best_params}")

    # Run pipeline on all datasets
    all_results = {}
    failed_datasets = []

    for i, dataset in enumerate(datasets):
        print(f"\n{'=' * 80}")
        print(f"📊 PROCESSING DATASET {i + 1}/{len(datasets)}: {dataset.upper()}")
        print(f"{'=' * 80}")

        try:
            result = await run_complete_pipeline(
                dataset,
                args.resume,
                args.concurrency,
                args.model,
                known_best_params,
                args.use_train_for_rules,
                args.mode,
                args.no_cache,
                args.embedding_base_url,
                args.embedding_model,
            )
            all_results[dataset] = result
            print(f"✅ {dataset}: F1={result.get('enhanced_results', {}).get('f1', 0):.4f}")

        except Exception as e:
            print(f"❌ {dataset}: FAILED - {e!s}")
            failed_datasets.append(dataset)
            all_results[dataset] = {"error": str(e)}

    # Print summary if multiple datasets
    if len(datasets) > 1:
        print(f"\n{'=' * 80}")
        print(f"📊 SUMMARY: {len(datasets)} DATASETS PROCESSED")
        print(f"{'=' * 80}")

        successful_datasets = [d for d in datasets if d not in failed_datasets]
        print(f"✅ Successful: {len(successful_datasets)}")
        print(f"❌ Failed: {len(failed_datasets)}")

        if successful_datasets:
            print("\n📈 RESULTS:")
            for dataset in successful_datasets:
                result = all_results[dataset]
                f1 = result.get("enhanced_results", {}).get("f1", 0)
                target = result.get("leaderboard_target", 0)
                beat_target = "🎯" if f1 >= target else "  "
                print(f"  {beat_target} {dataset:15} F1={f1:.4f} (target: {target:.1f})")

        if failed_datasets:
            print("\n❌ FAILED DATASETS:")
            for dataset in failed_datasets:
                print(f"  - {dataset}: {all_results[dataset].get('error', 'Unknown error')}")

    return all_results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        import sys

        sys.exit(1)
    except Exception as e:
        print(f"💥 Pipeline failed with unhandled exception: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)
    finally:
        # Clean up any remaining asyncio tasks
        try:
            # Cancel all pending tasks
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    pending_tasks = asyncio.all_tasks(loop)
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()
                            # Don't await here - just cancel
            except RuntimeError:
                # No event loop running, nothing to clean up
                pass
        except Exception:
            pass  # Ignore cleanup errors
