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

import asyncio
import json
import argparse
import pathlib
import time
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd

from src.entity_matching.hybrid_matcher import run_matching
from src.experiments.claude_sdk_heuristic_generator import ClaudeSDKHeuristicGenerator
from src.experiments.agentic_heuristic_generator import generate_agentic_heuristics
from run_enhanced_matching import run_enhanced_matching
from src.experiments.claude_sdk_optimizer import ClaudeSDKOptimizer
from src.experiments.improved_sweep import run_improved_sweep


async def run_basic_dev_sweep(dataset: str, early_exit: bool = False, model: str = 'gpt-4.1-nano', concurrency: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """DEPRECATED: Use run_improved_sweep instead - this function now redirects to the clean implementation"""
    print("⚠️ LEGACY SWEEP: Redirecting to improved sweep (no file swapping)")
    from src.experiments.improved_sweep import run_improved_sweep
    return await run_improved_sweep(dataset, early_exit, model, concurrency)


async def run_dev_only_analysis_with_params(dataset: str, params: Dict[str, Any], model: str = 'gpt-4.1-nano', concurrency: int = 3) -> Dict[str, Any]:
    """Run dev set analysis with specific hyperparameters - NO FILE SWAPPING"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    
    # Create temporary dataset - NO FILE SWAPPING
    os.makedirs("results/temp", exist_ok=True)
    temp_dataset_dir = pathlib.Path('results/temp') / f'{dataset}_dev_temp'
    temp_dataset_dir.mkdir(exist_ok=True)
    
    try:
        import shutil
        # Copy essential files
        shutil.copy(data_root / 'tableA.csv', temp_dataset_dir / 'tableA.csv')
        shutil.copy(data_root / 'tableB.csv', temp_dataset_dir / 'tableB.csv')
        
        # Decide what to use as dev set
        if (data_root / 'valid.csv').exists():
            print("✅ Using validation set for dev analysis (no test leakage)")
            shutil.copy(data_root / 'valid.csv', temp_dataset_dir / 'test.csv')
        elif (data_root / 'train.csv').exists():
            print("✅ Using slice of training set for dev analysis (no test leakage)")
            train_pairs = pd.read_csv(data_root / 'train.csv')
            dev_slice_size = min(100, len(train_pairs))
            train_slice = train_pairs.head(dev_slice_size)
            print(f"📊 Using {dev_slice_size} pairs from training set for dev analysis")
            train_slice.to_csv(temp_dataset_dir / 'test.csv', index=False)
        else:
            print("⚠️ No validation or training set - using test set for dev analysis (test won't be clean)")
            shutil.copy(data_root / 'test.csv', temp_dataset_dir / 'test.csv')
        
        # Run matching on temporary dataset
        # We need to use a different approach since run_matching expects data/raw/dataset structure
        # Let's create a symlink or copy to the expected location
        expected_path = pathlib.Path('data/raw') / f'temp_{dataset}_dev_temp'
        if expected_path.exists():
            shutil.rmtree(expected_path)
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_dataset_dir, expected_path)
        
        try:
            dev_results = await run_matching(
                dataset=f'temp_{dataset}_dev_temp',
                limit=None,
                max_candidates=params.get('max_candidates', 150),
                model=model,
                semantic_weight=params.get('semantic_weight', 0.5),
                use_semantic=params.get('use_semantic', True),
                embeddings_cache_dataset=dataset,  # Reuse cache from original dataset
                concurrency=concurrency
            )
        finally:
            # Clean up the expected path copy
            if expected_path.exists():
                shutil.rmtree(expected_path)
        
        return dev_results
        
    finally:
        # Clean up temporary dataset
        if temp_dataset_dir.exists():
            shutil.rmtree(temp_dataset_dir)


async def run_dev_only_analysis(dataset: str, model: str = 'gpt-4.1-nano', concurrency: int = 3) -> Dict[str, Any]:
    """Run dev set analysis without test set leakage - NO FILE SWAPPING"""
    return await run_dev_only_analysis_with_params(
        dataset=dataset,
        params={'max_candidates': 150, 'semantic_weight': 0.5, 'use_semantic': True},
        model=model,
        concurrency=concurrency
    )


async def run_train_for_rule_data(dataset: str, optimal_params: Dict[str, Any], model: str = 'gpt-4.1-nano', concurrency: int = 3) -> Dict[str, Any]:
    """Run on train set with optimal params to get more error examples for rule generation - NO FILE SWAPPING"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    
    if not (data_root / 'train.csv').exists():
        print("⚠️ No train.csv found - cannot use train set for rule data")
        return None
    
    print(f"🎯 Running on TRAIN SET with optimal params to get more error examples...")
    print(f"   This gives Claude much better signal for rule generation")
    print(f"   📁 NO FILE SWAPPING - using temporary dataset parameter")
    
    # Load train data directly - NO FILE MANIPULATION
    train_pairs = pd.read_csv(data_root / 'train.csv')
    
    # Use a reasonable subset for better error signal, but avoid timeouts
    # Adjust size based on dataset - larger datasets need smaller samples
    max_train_size = min(200 if len(train_pairs) > 1000 else 300, len(train_pairs))  # Smaller for large datasets
    train_subset = train_pairs.head(max_train_size)
    
    print(f"📊 Using {len(train_subset)} pairs from train set for error analysis")
    
    # Create a temporary dataset file in results/temp
    os.makedirs("results/temp", exist_ok=True)
    temp_train_file = f"results/temp/{dataset}_train_subset.csv"
    train_subset.to_csv(temp_train_file, index=False)
    
    try:
        # Run matching on the temporary train subset file by temporarily creating a mini-dataset
        temp_dataset_dir = pathlib.Path('results/temp') / f'{dataset}_train_temp'
        temp_dataset_dir.mkdir(exist_ok=True)
        
        # Copy the essential files
        import shutil
        shutil.copy(data_root / 'tableA.csv', temp_dataset_dir / 'tableA.csv')
        shutil.copy(data_root / 'tableB.csv', temp_dataset_dir / 'tableB.csv')
        shutil.copy(temp_train_file, temp_dataset_dir / 'test.csv')  # Use train subset as test for this run
        
        # Run matching on temporary dataset
        # Copy to expected data/raw location since run_matching expects that structure
        expected_path = pathlib.Path('data/raw') / f'temp_{dataset}_train_temp'
        if expected_path.exists():
            shutil.rmtree(expected_path)
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_dataset_dir, expected_path)
        
        try:
            train_results = await run_matching(
                dataset=f'temp_{dataset}_train_temp',
                limit=None,
                max_candidates=optimal_params['max_candidates'],
                model=model,
                semantic_weight=optimal_params['semantic_weight'],
                use_semantic=optimal_params.get('use_semantic', True),
                concurrency=concurrency
            )
        finally:
            # Clean up the expected path copy
            if expected_path.exists():
                shutil.rmtree(expected_path)
        
        print(f"✅ Train analysis: F1={train_results['metrics']['f1']:.4f}, {len(train_results.get('predictions', {}))} predictions")
        return train_results
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_train_file):
            os.unlink(temp_train_file)
        if temp_dataset_dir.exists():
            shutil.rmtree(temp_dataset_dir)


async def generate_actual_rules(dataset: str, dev_results: Dict[str, Any], model: str = 'gpt-4.1-nano', use_agentic: bool = True, use_train_for_rules: bool = False, optimal_params: Optional[Dict[str, Any]] = None, concurrency: int = 3) -> Tuple[str, Dict[str, Any]]:
    """Generate actual executable rules using Claude SDK heuristic generator"""
    print(f"🧠 STEP 2: Generating ACTUAL EXECUTABLE RULES using Claude SDK")
    
    os.makedirs("results/generated_rules", exist_ok=True)
    heuristics_file = f"results/generated_rules/{dataset}_generated_heuristics.json"
    
    try:
        # Decide which data to use for rule generation
        rule_data = dev_results
        if use_train_for_rules and optimal_params:
            print(f"🎯 Using TRAIN SET for better error signal in rule generation")
            try:
                train_results = await run_train_for_rule_data(dataset, optimal_params, model, concurrency)
                if train_results:
                    rule_data = train_results
                    print(f"🔄 Using train results (F1={train_results['metrics']['f1']:.4f}) for rule generation")
                else:
                    print(f"⚠️ Train analysis failed, falling back to dev results")
                    print(f"🔄 Using dev results (F1={dev_results['metrics']['f1']:.4f}) for rule generation")
            except Exception as e:
                print(f"⚠️ Train analysis failed with error: {e}")
                print(f"🔄 Falling back to dev results (F1={dev_results['metrics']['f1']:.4f}) for rule generation")
        else:
            print(f"🔄 Using dev results (F1={dev_results['metrics']['f1']:.4f}) for rule generation")
        
        rule_cost_info = {'total_cost_usd': 0.0, 'method': 'unknown'}
        
        if use_agentic:
            print(f"🤖 Using AGENTIC rule generation (Claude can test and iterate)")
            heuristics_file, rule_cost_info = await generate_agentic_heuristics(dataset, rule_data, heuristics_file)
            print(f"💰 Agentic rule generation cost: ${rule_cost_info.get('total_cost_usd', 0):.4f}")
            rule_cost_info['method'] = 'agentic'
        else:
            print(f"📋 Using LEGACY rule generation (simple prompt-response)")
            generator = ClaudeSDKHeuristicGenerator(dataset)
            
            # Generate failure patterns from rule_data (could be dev or train)
            patterns = generator.analyze_comprehensive_failure_patterns(rule_data)
            
            # Generate heuristics from the patterns
            rules = await generator.generate_heuristics(patterns)
            
            # Save the generated rules
            if rules:
                generator.save_heuristics(rules, heuristics_file)
            
            rule_cost_info = {'total_cost_usd': 0.0, 'method': 'legacy'}  # Legacy doesn't track costs yet
        
        print(f"✅ Generated executable rules saved to: {heuristics_file}")
        return heuristics_file, rule_cost_info
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Rule generation failed: {e}")
        print(f"❌ Claude SDK is required for this pipeline to work properly")
        print(f"❌ Install Claude SDK or fix the error above")
        raise RuntimeError(f"Rule generation failed and is required for pipeline: {e}")


async def validate_and_optimize_rules(dataset: str, heuristic_file: str, optimal_params: Dict[str, Any], concurrency: int) -> str:
    """Validate rules on dev set and optimize them using Claude SDK - NO FILE SWAPPING"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    claude_optimizer = ClaudeSDKOptimizer()
    
    if not claude_optimizer.claude_executable:
        print("⚠️ Claude SDK not available - skipping rule optimization")
        return heuristic_file
    
    print(f"🔍 Running rule validation on dev set...")
    
    # Create temporary dataset for validation - NO FILE SWAPPING
    os.makedirs("results/temp", exist_ok=True)
    temp_dataset_dir = pathlib.Path('results/temp') / f'{dataset}_validation_temp'
    temp_dataset_dir.mkdir(exist_ok=True)
    
    try:
        import shutil
        # Copy essential files
        shutil.copy(data_root / 'tableA.csv', temp_dataset_dir / 'tableA.csv')
        shutil.copy(data_root / 'tableB.csv', temp_dataset_dir / 'tableB.csv')
        
        # Choose validation data
        if (data_root / 'valid.csv').exists():
            print("✅ Using validation set for rule validation")
            shutil.copy(data_root / 'valid.csv', temp_dataset_dir / 'test.csv')
        elif (data_root / 'train.csv').exists():
            print("✅ Using slice of training set for rule validation")
            train_pairs = pd.read_csv(data_root / 'train.csv')
            dev_slice_size = min(200, len(train_pairs))
            train_slice = train_pairs.head(dev_slice_size)
            print(f"📊 Using {dev_slice_size} pairs from training set for rule validation")
            train_slice.to_csv(temp_dataset_dir / 'test.csv', index=False)
        else:
            print("⚠️ No validation or training set - skipping rule optimization to avoid test leakage")
            return heuristic_file
        
        # Run enhanced matching on temporary dataset
        print(f"🔄 RULE VALIDATION: Testing rules on validation data...")
        print(f"   📊 This is NOT the final test - just validating rules")
        print(f"   🎯 Purpose: Check if rules help/hurt performance before final test")
        
        # Copy to expected data/raw location
        expected_path = pathlib.Path('data/raw') / f'temp_{dataset}_validation_temp'
        if expected_path.exists():
            shutil.rmtree(expected_path)
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_dataset_dir, expected_path)
        
        try:
            dev_results = await run_enhanced_matching(
                dataset=f'temp_{dataset}_validation_temp',
                limit=None,
                max_candidates=optimal_params['max_candidates'],
                model=optimal_params['model'],
                semantic_weight=optimal_params['semantic_weight'],
                heuristic_file=heuristic_file,
                concurrency=concurrency
            )
        finally:
            # Clean up the expected path copy
            if expected_path.exists():
                shutil.rmtree(expected_path)
        
        print(f"✅ RULE VALIDATION completed: F1={dev_results['f1']:.4f}, Early decisions={dev_results.get('early_decisions', 0)}")
        
    finally:
        # Clean up temporary dataset
        if temp_dataset_dir.exists():
            shutil.rmtree(temp_dataset_dir)
    
    # Analyze performance and optimize rules
    print(f"📊 Dev Results: F1={dev_results['f1']:.4f}, P={dev_results['precision']:.4f}, R={dev_results['recall']:.4f}")
    
    # Load current heuristics
    with open(heuristic_file, 'r') as f:
        heuristics = json.load(f)
    
    # Get leaderboard target
    from src.experiments.claude_sdk_heuristic_generator import get_leaderboard_target_f1
    target_f1 = get_leaderboard_target_f1(dataset)
    
    # Create optimization prompt
    prompt = f"""You are an expert at entity matching rule optimization. Analyze these rule performance results and decide which rules to disable to improve F1 score.

DATASET: {dataset}
TARGET: F1 > {target_f1:.1f} (leaderboard target)

CURRENT DEV PERFORMANCE:
- F1 Score: {dev_results['f1']:.4f}
- Precision: {dev_results['precision']:.4f}  
- Recall: {dev_results['recall']:.4f}
- Early Decisions: {dev_results.get('early_decisions', 0)}
- LLM Call Reduction: {dev_results.get('llm_call_reduction', 0):.1f}%

ASSESSMENT: {'ABOVE TARGET' if dev_results['f1'] > target_f1/100 else 'BELOW TARGET - NEEDS OPTIMIZATION'}

If F1 is below target and precision > 0.9, disable overly conservative rules to improve recall.
If F1 is above target, make minimal changes.

Generate optimized heuristics with problematic rules disabled:

{{
  "analysis": "Performance assessment and optimization strategy",
  "rules_to_disable": ["rule_name1", "rule_name2"],
  "optimized_heuristics": {json.dumps(heuristics, indent=2)}
}}

Only disable rules if F1 < target. If F1 >= target, return empty rules_to_disable array."""
    
    try:
        # Call Claude SDK
        result = subprocess.run(
            [claude_optimizer.claude_executable, "--print", prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"⚠️ Claude SDK optimization failed: {result.stderr}")
            return heuristic_file
        
        response = result.stdout
        
        # Extract JSON from response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]
        
        optimization_result = json.loads(json_str)
        
        rules_to_disable = optimization_result.get('rules_to_disable', [])
        
        if not rules_to_disable:
            print("✅ No rule optimization needed - performance is acceptable")
            return heuristic_file
        
        print(f"🔧 Optimizing rules: disabling {len(rules_to_disable)} rules")
        for rule_name in rules_to_disable:
            print(f"   - Disabling: {rule_name}")
        
        # Save optimized heuristics
        optimized_heuristics = optimization_result['optimized_heuristics']
        optimized_heuristics['timestamp'] = datetime.now().isoformat()
        optimized_heuristics['optimization_notes'] = f"Disabled {len(rules_to_disable)} rules: {', '.join(rules_to_disable)}"
        
        optimized_file = heuristic_file.replace('.json', '_optimized.json')
        with open(optimized_file, 'w') as f:
            json.dump(optimized_heuristics, f, indent=2)
        
        print(f"✅ Optimized heuristics saved to: {optimized_file}")
        return optimized_file
        
    except Exception as e:
        print(f"⚠️ Rule optimization failed: {e}")
        return heuristic_file


async def run_complete_pipeline(dataset: str, early_exit: bool = False, resume: bool = False, concurrency: int = 3, validate_rules: bool = False, model: str = 'gpt-4.1-nano', use_agentic_rules: bool = True, known_best_params: Optional[Dict[str, Any]] = None, use_train_for_rules: bool = False) -> Dict[str, Any]:
    """Complete pipeline: dev analysis -> ACTUAL rule generation -> test with enhanced matching"""
    
    print(f"🚀 COMPLETE ENTITY MATCHING PIPELINE", flush=True)
    print(f"Dataset: {dataset}", flush=True)
    if resume:
        print("🔄 RESUME MODE: Will skip completed steps", flush=True)
    print("=" * 60, flush=True)
    
    # Check for existing checkpoint
    checkpoint_file = f"results/{dataset}_pipeline_checkpoint.json"
    checkpoint = {}
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        print(f"📁 Loaded checkpoint: {list(checkpoint.keys())}", flush=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "pipeline_version": "complete_v3_working_rules"
    }
    
    # STEP 1: Basic hyperparameter optimization on dev set (NO TEST LEAKAGE)
    if known_best_params:
        print(f"✅ STEP 1: Using provided hyperparameters: {known_best_params}")
        print(f"⏳ Running single dev evaluation to get predictions for rule generation...")
        
        start_time = time.time()
        dev_results = await run_dev_only_analysis_with_params(dataset, known_best_params, model, concurrency)
        dev_time = time.time() - start_time
        
        # Ensure optimal_params has all required fields
        optimal_params = {
            'max_candidates': known_best_params.get('max_candidates', 150),
            'semantic_weight': known_best_params.get('semantic_weight', 0.5),
            'model': known_best_params.get('model', model),  # Use dev model if not specified
            'use_semantic': known_best_params.get('use_semantic', True)
        }
        
        print(f"✅ Dev Results with known params: F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}")
        
    elif 'dev_results' in checkpoint and 'optimal_params' in checkpoint:
        print(f"✅ STEP 1: Using cached dev results from checkpoint")
        dev_results = checkpoint['dev_results']
        optimal_params = checkpoint['optimal_params']
        dev_time = checkpoint.get('dev_time', 0)
        
        # Validate consistency if we have unified results
        if 'unified_sweep_result' in dev_results:
            unified = dev_results['unified_sweep_result']
            expected_f1 = unified['best_f1']
            actual_f1 = dev_results['metrics']['f1']
            expected_config = unified['config_that_achieved_best_f1']
            
            if abs(expected_f1 - actual_f1) > 0.001:
                print(f"⚠️ WARNING: F1 mismatch in checkpoint! Expected {expected_f1:.4f}, got {actual_f1:.4f}")
            if expected_config != optimal_params:
                print(f"⚠️ WARNING: Config mismatch in checkpoint!")
                print(f"   Expected: {expected_config}")
                print(f"   Got: {optimal_params}")
        
        print(f"📊 Loaded: F1={dev_results['metrics']['f1']:.4f}, Config={optimal_params}")
    else:
        print(f"🎯 STEP 1: Hyperparameter optimization on dev set")
        print(f"⏳ This will run a basic sweep to find good parameters quickly...")
        
        start_time = time.time()
        print("🔧 Using improved sweep implementation (no file swapping)")
        dev_results, optimal_params = await run_improved_sweep(dataset, early_exit, model, concurrency)
        dev_time = time.time() - start_time
        
        # Save checkpoint with robust JSON handling - convert numpy types
        def clean_for_json(obj):
            """Convert numpy/pandas types to JSON-serializable types"""
            if hasattr(obj, 'item'):  # numpy/pandas scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        checkpoint.update({
            'dev_results': clean_for_json(dev_results),
            'optimal_params': clean_for_json(optimal_params),
            'dev_time': dev_time
        })
        os.makedirs("results", exist_ok=True)
        
        # Write checkpoint atomically to avoid corruption
        temp_checkpoint_file = checkpoint_file + '.tmp'
        try:
            with open(temp_checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            # Only replace the real file if write succeeded
            import shutil
            shutil.move(temp_checkpoint_file, checkpoint_file)
        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint: {e}")
            if os.path.exists(temp_checkpoint_file):
                os.unlink(temp_checkpoint_file)
    
    print(f"✅ Best Dev Results: F1={dev_results['metrics']['f1']:.4f}, Cost=${dev_results['cost_usd']:.3f}")
    print(f"🎯 Optimal Parameters: {optimal_params['max_candidates']} candidates, {optimal_params['semantic_weight']:.2f} semantic weight, {optimal_params['model']}")
    
    results['dev_results'] = {
        'f1': dev_results['metrics']['f1'],
        'precision': dev_results['metrics']['precision'],
        'recall': dev_results['metrics']['recall'],
        'cost_usd': dev_results['cost_usd'],
        'processing_time': dev_time
    }
    results['optimal_params'] = optimal_params
    
    # STEP 2: Generate ACTUAL EXECUTABLE RULES using Claude SDK heuristic generator
    heuristics_file = f"results/generated_rules/{dataset}_generated_heuristics.json"
    rule_generation_cost = {'total_cost_usd': 0.0, 'method': 'cached'}
    
    if 'heuristics_file' in checkpoint and os.path.exists(checkpoint['heuristics_file']):
        print(f"✅ STEP 2: Using cached heuristics from checkpoint")
        heuristics_file = checkpoint['heuristics_file']
        # Check if cached cost info exists
        if 'rule_generation_cost' in checkpoint:
            rule_generation_cost = checkpoint['rule_generation_cost']
    else:
        print(f"\n🧠 STEP 2: Rule generation (analyzing dev results...)")
        heuristics_file, rule_generation_cost = await generate_actual_rules(dataset, dev_results, model, use_agentic=use_agentic_rules, use_train_for_rules=use_train_for_rules, optimal_params=optimal_params, concurrency=concurrency)
        
        # Save checkpoint with robust JSON handling
        checkpoint['heuristics_file'] = heuristics_file
        checkpoint['rule_generation_cost'] = rule_generation_cost
        temp_checkpoint_file = checkpoint_file + '.tmp'
        try:
            with open(temp_checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            import shutil
            shutil.move(temp_checkpoint_file, checkpoint_file)
        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint: {e}")
            if os.path.exists(temp_checkpoint_file):
                os.unlink(temp_checkpoint_file)
    
    if heuristics_file:
        results['heuristics_file'] = heuristics_file
        results['rule_generation'] = "claude_sdk_success"
        print(f"✅ Rules generated: {heuristics_file}")
    else:
        results['rule_generation'] = "failed"
        print(f"❌ Rule generation failed")
        return results
    
    # STEP 2.5: Rule validation and optimization (optional)
    if validate_rules:
        if 'optimized_heuristics_file' in checkpoint and os.path.exists(checkpoint['optimized_heuristics_file']):
            print(f"✅ STEP 2.5: Using cached optimized rules from checkpoint")
            heuristics_file = checkpoint['optimized_heuristics_file']
        else:
            print(f"\n🔍 STEP 2.5: Rule validation and optimization on dev set")
            optimized_file = await validate_and_optimize_rules(dataset, heuristics_file, optimal_params, concurrency)
            if optimized_file != heuristics_file:
                heuristics_file = optimized_file
                results['rule_optimization'] = "claude_sdk_success"
                results['optimized_heuristics_file'] = heuristics_file
                
                # Save checkpoint
                checkpoint['optimized_heuristics_file'] = heuristics_file
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
            else:
                results['rule_optimization'] = "no_changes_needed"
                
            print(f"✅ Using heuristics: {heuristics_file}")
    
    # STEP 3A: Test set evaluation WITHOUT rules (baseline with optimal params)
    print(f"\n🎯 STEP 3A: FINAL TEST EVALUATION WITHOUT rules (optimal params baseline)")
    print(f"⏳ Running baseline matching on FULL TEST SET (this is the real evaluation)...")
    
    start_time = time.time()
    baseline_results = await run_matching(
        dataset=dataset,
        limit=None,
        max_candidates=optimal_params['max_candidates'],  # Use optimized candidates
        model='gpt-4.1-mini',  # Use better model for test (upgrade from dev model)
        semantic_weight=optimal_params['semantic_weight'],  # Use optimized semantic weight
        use_semantic=optimal_params['use_semantic'],
        concurrency=concurrency
    )
    baseline_time = time.time() - start_time
    
    print(f"✅ Baseline Results (no rules): F1={baseline_results['metrics']['f1']:.4f}, Cost=${baseline_results['cost_usd']:.3f}")
    
    # STEP 3B: Test set evaluation WITH generated rules
    print(f"\n🎯 STEP 3B: FINAL TEST EVALUATION WITH rules (enhanced approach)") 
    print(f"⏳ Running enhanced matching with rules on FULL TEST SET (this is the real evaluation)...")
    
    start_time = time.time()
    enhanced_results = await run_enhanced_matching(
        dataset=dataset,
        limit=None,
        max_candidates=optimal_params['max_candidates'],  # Use optimized candidates
        model='gpt-4.1-mini',  # Use better model for test (upgrade from dev model)
        semantic_weight=optimal_params['semantic_weight'],  # Use optimized semantic weight
        heuristic_file=heuristics_file
    )
    enhanced_time = time.time() - start_time
    
    print(f"✅ Enhanced Results (with rules): F1={enhanced_results['f1']:.4f}, Cost=${enhanced_results['cost']:.3f}")
    
    # Calculate improvement
    f1_improvement = enhanced_results['f1'] - baseline_results['metrics']['f1']
    cost_change = enhanced_results['cost'] - baseline_results['cost_usd']
    
    print(f"\n📊 A/B COMPARISON:")
    print(f"F1 Change: {f1_improvement:+.4f} ({'✅ IMPROVED' if f1_improvement > 0 else '❌ WORSE' if f1_improvement < 0 else '➡️ NO CHANGE'})")
    print(f"Cost Change: ${cost_change:+.3f}")
    print(f"Rules {'✅ HELPED' if f1_improvement > 0.01 else '❌ DID NOT HELP' if f1_improvement < -0.01 else '➡️ NEUTRAL'}")
    
    results['baseline_results'] = {
        'f1': baseline_results['metrics']['f1'],
        'precision': baseline_results['metrics']['precision'],
        'recall': baseline_results['metrics']['recall'],
        'cost_usd': baseline_results['cost_usd'],
        'processing_time': baseline_time
    }
    
    results['enhanced_results'] = {
        'f1': enhanced_results['f1'],
        'precision': enhanced_results['precision'],
        'recall': enhanced_results['recall'],
        'cost_usd': enhanced_results['cost'],
        'processing_time': enhanced_time,
        'early_decisions': enhanced_results.get('early_decisions', 0),
        'llm_calls': enhanced_results.get('llm_calls', 0),
        'llm_call_reduction': enhanced_results.get('llm_call_reduction', 0),
        'predictions': enhanced_results.get('predictions', {})  # Include predictions for failure analysis
    }
    
    results['ab_comparison'] = {
        'f1_improvement': f1_improvement,
        'cost_change': cost_change,
        'rules_helped': f1_improvement > 0.01
    }
    
    # FINAL SUMMARY
    rule_gen_cost = rule_generation_cost.get('total_cost_usd', 0.0)
    total_cost = dev_results['cost_usd'] + baseline_results['cost_usd'] + enhanced_results['cost'] + rule_gen_cost
    total_time = dev_time + baseline_time + enhanced_time
    
    print(f"\\n🏆 FINAL RESULTS FOR {dataset.upper()}")
    print(f"Dev F1:        {dev_results['metrics']['f1']:.4f} (${dev_results['cost_usd']:.3f})")
    print(f"Rule Generation: {rule_generation_cost.get('method', 'unknown')} (${rule_gen_cost:.4f})")
    print(f"Test Baseline: {baseline_results['metrics']['f1']:.4f} (${baseline_results['cost_usd']:.3f}) - optimal params only")
    print(f"Test Enhanced: {enhanced_results['f1']:.4f} (${enhanced_results['cost']:.3f}) - optimal params + rules")
    print(f"Improvement:   {f1_improvement:+.4f} F1 points")
    print(f"Total Cost: ${total_cost:.3f}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"Rules: {heuristics_file}")
    print(f"LLM Call Reduction: {enhanced_results.get('llm_call_reduction', 0):.1f}%")
    
    # Check if we beat the leaderboard (check both baseline and enhanced)
    from src.experiments.claude_sdk_heuristic_generator import get_leaderboard_target_f1
    target_f1 = get_leaderboard_target_f1(dataset)
    baseline_beats_leaderboard = baseline_results['metrics']['f1'] > target_f1/100 if target_f1 > 10 else baseline_results['metrics']['f1'] > target_f1
    enhanced_beats_leaderboard = enhanced_results['f1'] > target_f1/100 if target_f1 > 10 else enhanced_results['f1'] > target_f1
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
    
    # Add recommendation
    if baseline_beats_leaderboard and not enhanced_beats_leaderboard:
        print(f"💡 RECOMMENDATION: Use baseline approach (no rules) for best performance!")
    elif enhanced_beats_leaderboard and f1_improvement > 0.01:
        print(f"💡 RECOMMENDATION: Use enhanced approach with rules for best performance!")
    elif baseline_beats_leaderboard and enhanced_beats_leaderboard:
        if f1_improvement > 0:
            print(f"💡 RECOMMENDATION: Use enhanced approach with rules (slightly better)!")
        else:
            print(f"💡 RECOMMENDATION: Use baseline approach (simpler, similar performance)!")
    
    results['summary'] = {
        'dev_f1': dev_results['metrics']['f1'],
        'baseline_f1': baseline_results['metrics']['f1'],
        'enhanced_f1': enhanced_results['f1'],
        'f1_improvement': f1_improvement,
        'total_cost_usd': total_cost,
        'rule_generation_cost_usd': rule_gen_cost,
        'rule_generation_method': rule_generation_cost.get('method', 'unknown'),
        'total_time_seconds': total_time,
        'beat_leaderboard': beat_leaderboard,
        'leaderboard_target': target_f1
    }
    
    # Save results  
    os.makedirs("results", exist_ok=True)
    results_file = f"results/{dataset}_complete_pipeline.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📋 Results saved to: {results_file}")
    
    # Extract detailed failure analysis
    print(f"🔍 Extracting detailed failure analysis...")
    failure_analysis = extract_failure_records(dataset, results)
    results['failure_analysis'] = failure_analysis
    
    if failure_analysis.get('detailed_failures'):
        print(f"📊 Captured {failure_analysis['total_failures']} failures ({failure_analysis['false_positives']} FP, {failure_analysis['false_negatives']} FN)")
    
    # Re-save results with failure analysis
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate updated internal leaderboard
    print(f"📊 Generating updated internal leaderboard...")
    try:
        result = subprocess.run(['python', 'generate_internal_leaderboard.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ Internal leaderboard updated successfully")
        else:
            print(f"⚠️ Leaderboard generation failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Could not generate leaderboard: {e}")
    
    return results


def extract_failure_records(dataset: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed failure records from results for analysis"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    
    # Load the original tables
    try:
        A_df = pd.read_csv(data_root / 'tableA.csv')
        B_df = pd.read_csv(data_root / 'tableB.csv')
        test_pairs = pd.read_csv(data_root / 'test.csv')
        
        A_records = {row['id']: row.to_dict() for _, row in A_df.iterrows()}
        B_records = {row['id']: row.to_dict() for _, row in B_df.iterrows()}
        
        predictions = results.get('enhanced_results', {}).get('predictions', {})
        if not predictions:
            return {'failure_analysis': 'No predictions available for failure analysis'}
        
        failures = []
        
        for _, row in test_pairs.iterrows():
            left_id = row.ltable_id
            right_id = row.rtable_id
            true_label = row.label
            
            if left_id in predictions:
                predicted_right_id = predictions[left_id]
                predicted_match = (predicted_right_id == right_id)
                predicted_label = 1 if predicted_match else 0
                
                # Check if this is a failure
                if true_label != predicted_label:
                    failure_type = 'false_positive' if (true_label == 0 and predicted_label == 1) else 'false_negative'
                    
                    failures.append({
                        'left_id': left_id,
                        'right_id': right_id,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'failure_type': failure_type,
                        'left_record': A_records.get(left_id, {}),
                        'right_record': B_records.get(right_id, {}),
                        'predicted_right_id': predicted_right_id,
                        'predicted_right_record': B_records.get(predicted_right_id, {}) if predicted_right_id else None
                    })
        
        return {
            'total_failures': len(failures),
            'false_positives': len([f for f in failures if f['failure_type'] == 'false_positive']),
            'false_negatives': len([f for f in failures if f['failure_type'] == 'false_negative']),
            'detailed_failures': failures
        }
        
    except Exception as e:
        return {'failure_analysis_error': f"Could not extract failure records: {e}"}


async def main():
    parser = argparse.ArgumentParser(description="Complete entity matching pipeline")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. beer, walmart_amazon)')
    parser.add_argument('--early-exit', action='store_true', help='Stop sweep early if F1 beats leaderboard target')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--concurrency', type=int, default=3, help='Number of concurrent API requests')
    parser.add_argument('--validate-rules', action='store_true', help='Validate and optimize rules on dev set before test')
    parser.add_argument('--model', default='gpt-4.1-nano', help='Model to use for dev sweep (default: gpt-4.1-nano)')

    parser.add_argument('--use-agentic-rules', action='store_true', default=True, help='Use agentic rule generation (default: True)')
    parser.add_argument('--use-legacy-rules', dest='use_agentic_rules', action='store_false', help='Use legacy rule generation instead of agentic')
    parser.add_argument('--known-best-params', help='JSON string with known best hyperparameters (e.g. \'{"max_candidates": 50, "semantic_weight": 0.7}\')')
    parser.add_argument('--use-train-for-rules', action='store_true', help='Use train set with optimal params to get more error examples for rule generation')
    
    args = parser.parse_args()
    
    # Parse known best params if provided
    known_best_params = None
    if args.known_best_params:
        try:
            known_best_params = json.loads(args.known_best_params)
            print(f"🎯 Using known best parameters: {known_best_params}")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON for known-best-params: {e}")
            return
    
    # Update concurrency settings based on user input
    if args.concurrency != 3:
        print(f"🔧 Setting concurrency to {args.concurrency}")
        # Update concurrency in source files would require more complex logic
        # For now, just show the setting
    
    results = await run_complete_pipeline(args.dataset, args.early_exit, args.resume, args.concurrency, args.validate_rules, args.model, args.use_improved_sweep, args.use_agentic_rules, known_best_params, args.use_train_for_rules)
    return results


if __name__ == "__main__":
    results = asyncio.run(main())