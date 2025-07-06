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
from typing import Dict, Any, Tuple
import pandas as pd

from src.entity_matching.hybrid_matcher import run_matching
from src.experiments.claude_sdk_heuristic_generator import ClaudeSDKHeuristicGenerator
from src.experiments.intelligent_sweep import IntelligentSweeper
from run_enhanced_matching import run_enhanced_matching
from src.experiments.claude_sdk_optimizer import ClaudeSDKOptimizer


async def run_basic_dev_sweep(dataset: str, early_exit: bool = False, model: str = 'gpt-4.1-nano') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run basic hyperparameter sweep on dev set to find good parameters quickly"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    
    # Set up dev set (no test leakage)
    limit = None  # Use full dev set for reliable optimization
    
    if (data_root / 'valid.csv').exists():
        print("✅ Using validation set for hyperparameter sweep (no test leakage)")
        # Temporarily swap test.csv with valid.csv for dev analysis
        test_backup = data_root / 'test.csv.backup'
        valid_file = data_root / 'valid.csv'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use validation
        test_file.rename(test_backup)
        valid_file.rename(test_file)
        
        try:
            # Run strategic sweep - extreme/low/middle candidates × low/medium/high semantic weights
            sweeper = IntelligentSweeper(dataset, limit, early_exit, model)
            await sweeper.run_strategic_sweep()  # Strategic 3×3 grid sweep
            
            # Get best result
            best_result = max(sweeper.results, key=lambda r: r.f1_score)
            
            # Convert to format expected by rest of pipeline
            dev_results = {
                'metrics': {
                    'f1': best_result.f1_score,
                    'precision': best_result.precision,
                    'recall': best_result.recall,
                },
                'cost_usd': best_result.cost_usd,
                'processed_pairs': best_result.processed_pairs,
                'predictions_made': best_result.predictions_made
            }
            
            # Extract optimal parameters
            optimal_params = {
                'max_candidates': best_result.config.max_candidates,
                'semantic_weight': best_result.config.semantic_weight,
                'model': best_result.config.model,
                'use_semantic': best_result.config.use_semantic
            }
            
        finally:
            # Restore original files
            test_file.rename(valid_file)
            test_backup.rename(test_file)
            
    elif (data_root / 'train.csv').exists():
        print("✅ Using slice of training set for hyperparameter sweep (no test leakage)")
        # Use a reasonable slice of training set
        train_pairs = pd.read_csv(data_root / 'train.csv')
        dev_slice_size = min(100, len(train_pairs))  # Take up to 100 pairs for sweep
        train_slice = train_pairs.head(dev_slice_size)
        
        print(f"📊 Using {dev_slice_size} pairs from training set for hyperparameter optimization")
        
        # Create temporary dev file
        dev_file = data_root / 'dev_temp.csv'
        train_slice.to_csv(dev_file, index=False)
        
        # Temporarily swap test.csv with dev slice
        test_backup = data_root / 'test.csv.backup'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use dev slice
        test_file.rename(test_backup)
        dev_file.rename(test_file)
        
        try:
            # Run strategic sweep - extreme/low/middle candidates × low/medium/high semantic weights
            sweeper = IntelligentSweeper(dataset, None, early_exit, model)  # Use full dev slice
            await sweeper.run_strategic_sweep()  # Strategic 3×3 grid sweep
            
            # Get best result
            best_result = max(sweeper.results, key=lambda r: r.f1_score)
            
            # Convert to format expected by rest of pipeline
            dev_results = {
                'metrics': {
                    'f1': best_result.f1_score,
                    'precision': best_result.precision,
                    'recall': best_result.recall,
                },
                'cost_usd': best_result.cost_usd,
                'processed_pairs': best_result.processed_pairs,
                'predictions_made': best_result.predictions_made
            }
            
            # Extract optimal parameters
            optimal_params = {
                'max_candidates': best_result.config.max_candidates,
                'semantic_weight': best_result.config.semantic_weight,
                'model': best_result.config.model,
                'use_semantic': best_result.config.use_semantic
            }
            
        finally:
            # Restore original files
            test_file.rename(dev_file)
            test_backup.rename(test_file)
            # Clean up temp file
            dev_file.unlink()
    else:
        print("⚠️ No validation or training set - using fallback parameters")
        # Fallback to reasonable defaults
        dev_results = {
            'metrics': {'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
            'cost_usd': 0.0,
            'processed_pairs': 0,
            'predictions_made': 0
        }
        optimal_params = {
            'max_candidates': 50,
            'semantic_weight': 0.5,
            'model': model,
            'use_semantic': True
        }
    
    return dev_results, optimal_params


async def run_dev_only_analysis(dataset: str) -> Dict[str, Any]:
    """Run dev set analysis without test set leakage"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    
    # Check if validation set exists
    if (data_root / 'valid.csv').exists():
        print("✅ Using validation set for dev analysis (no test leakage)")
        # Temporarily swap test.csv with valid.csv for dev analysis
        test_backup = data_root / 'test.csv.backup'
        valid_file = data_root / 'valid.csv'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use validation
        test_file.rename(test_backup)
        valid_file.rename(test_file)
        
        try:
            dev_results = await run_matching(
                dataset=dataset,
                limit=None,
                max_candidates=150,
                model=model,
                semantic_weight=0.5,
                use_semantic=True,
                concurrency=concurrency
            )
        finally:
            # Restore original files
            test_file.rename(valid_file)
            test_backup.rename(test_file)
    elif (data_root / 'train.csv').exists():
        print("✅ Using slice of training set for dev analysis (no test leakage)")
        
        # Load train.csv and take a reasonable slice (e.g., first 100 pairs)
        train_pairs = pd.read_csv(data_root / 'train.csv')
        dev_slice_size = min(100, len(train_pairs))  # Take up to 100 pairs
        train_slice = train_pairs.head(dev_slice_size)
        
        print(f"📊 Using {dev_slice_size} pairs from training set for dev analysis")
        
        # Create temporary dev file
        dev_file = data_root / 'dev_temp.csv'
        train_slice.to_csv(dev_file, index=False)
        
        # Temporarily swap test.csv with dev slice
        test_backup = data_root / 'test.csv.backup'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use dev slice
        test_file.rename(test_backup)
        dev_file.rename(test_file)
        
        try:
            dev_results = await run_matching(
                dataset=dataset,
                limit=None,
                max_candidates=150,
                model=model,
                semantic_weight=0.5,
                use_semantic=True,
                concurrency=concurrency
            )
        finally:
            # Restore original files
            test_file.rename(dev_file)
            test_backup.rename(test_file)
            # Clean up temp file
            dev_file.unlink()
    else:
        print("⚠️ No validation or training set - using test set for dev analysis (test won't be clean)")
        dev_results = await run_matching(
            dataset=dataset,
            limit=None,
            max_candidates=150,
            model='gpt-4.1-nano',
            semantic_weight=0.5,
            use_semantic=True,
            concurrency=concurrency
        )
    
    return dev_results


async def generate_actual_rules(dataset: str, dev_results: Dict[str, Any]) -> str:
    """Generate actual executable rules using Claude SDK heuristic generator"""
    print(f"🧠 STEP 2: Generating ACTUAL EXECUTABLE RULES using Claude SDK")
    
    generator = ClaudeSDKHeuristicGenerator(dataset)
    
    # Use the proper comprehensive analysis method that generates real rules
    config = {
        'model': model,
        'max_candidates': 150,
        'semantic_weight': 0.5,
        'use_semantic': True,
        'limit': None  # Full dev set for comprehensive analysis
    }
    
    try:
        # This generates actual executable rules using the existing dev_results
        os.makedirs("results/generated_rules", exist_ok=True)
        heuristics_file = f"results/generated_rules/{dataset}_generated_heuristics.json"
        
        # Skip running dev analysis again - use existing results!
        print(f"🔄 Using existing dev results (F1={dev_results['metrics']['f1']:.4f}) for rule generation")
        
        # Create mock comprehensive results using the dev_results we already have
        comprehensive_results = {'validation': dev_results}
        
        # Generate failure patterns from existing results
        patterns = generator.analyze_comprehensive_failure_patterns(dev_results)
        
        # Generate heuristics from the patterns
        rules = await generator.generate_heuristics(patterns)
        
        # Save the generated rules
        if rules:
            generator.save_heuristics(rules, heuristics_file)
        
        print(f"✅ Generated executable rules saved to: {heuristics_file}")
        return heuristics_file
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Rule generation failed: {e}")
        print(f"❌ Claude SDK is required for this pipeline to work properly")
        print(f"❌ Install Claude SDK or fix the error above")
        raise RuntimeError(f"Rule generation failed and is required for pipeline: {e}")


async def validate_and_optimize_rules(dataset: str, heuristic_file: str, optimal_params: Dict[str, Any], concurrency: int) -> str:
    """Validate rules on dev set and optimize them using Claude SDK"""
    data_root = pathlib.Path('data') / 'raw' / dataset
    claude_optimizer = ClaudeSDKOptimizer()
    
    if not claude_optimizer.claude_executable:
        print("⚠️ Claude SDK not available - skipping rule optimization")
        return heuristic_file
    
    print(f"🔍 Running rule validation on dev set...")
    
    # Run enhanced matching on dev/validation set
    if (data_root / 'valid.csv').exists():
        print("✅ Using validation set for rule validation")
        # Temporarily swap test.csv with valid.csv
        test_backup = data_root / 'test.csv.backup'
        valid_file = data_root / 'valid.csv'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use validation
        test_file.rename(test_backup)
        valid_file.rename(test_file)
        
        try:
            dev_results = await run_enhanced_matching(
                dataset=dataset,
                limit=None,  # Use full validation set
                max_candidates=optimal_params['max_candidates'],
                model=optimal_params['model'],
                semantic_weight=optimal_params['semantic_weight'],
                heuristic_file=heuristic_file,
                concurrency=concurrency
            )
        finally:
            # Restore original files
            test_file.rename(valid_file)
            test_backup.rename(test_file)
            
    elif (data_root / 'train.csv').exists():
        print("✅ Using slice of training set for rule validation")
        # Use a reasonable slice of training set
        train_pairs = pd.read_csv(data_root / 'train.csv')
        dev_slice_size = min(200, len(train_pairs))  # Use up to 200 pairs for validation
        train_slice = train_pairs.head(dev_slice_size)
        
        print(f"📊 Using {dev_slice_size} pairs from training set for rule validation")
        
        # Create temporary dev file
        dev_file = data_root / 'dev_temp.csv'
        train_slice.to_csv(dev_file, index=False)
        
        # Temporarily swap test.csv with dev slice
        test_backup = data_root / 'test.csv.backup'
        test_file = data_root / 'test.csv'
        
        # Backup original test and use dev slice
        test_file.rename(test_backup)
        dev_file.rename(test_file)
        
        try:
            print(f"🔄 RULE VALIDATION: Running enhanced matching on {dev_slice_size} training pairs...")
            print(f"   📊 This is NOT the final test - just validating rules on training data")
            print(f"   🎯 Purpose: Check if rules help/hurt performance before final test")
            dev_results = await run_enhanced_matching(
                dataset=dataset,
                limit=None,  # Use full dev slice
                max_candidates=optimal_params['max_candidates'],
                model=optimal_params['model'],
                semantic_weight=optimal_params['semantic_weight'],
                heuristic_file=heuristic_file,
                concurrency=concurrency
            )
            print(f"✅ RULE VALIDATION completed: F1={dev_results['f1']:.4f}, Early decisions={dev_results.get('early_decisions', 0)}")
        finally:
            # Restore original files
            test_file.rename(dev_file)
            test_backup.rename(test_file)
            # Clean up temp file
            dev_file.unlink()
    else:
        print("⚠️ No validation or training set - skipping rule optimization to avoid test leakage")
        return heuristic_file
    
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


async def run_complete_pipeline(dataset: str, early_exit: bool = False, resume: bool = False, concurrency: int = 3, validate_rules: bool = False, model: str = 'gpt-4.1-nano') -> Dict[str, Any]:
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
    if 'dev_results' in checkpoint and 'optimal_params' in checkpoint:
        print(f"✅ STEP 1: Using cached dev results from checkpoint")
        dev_results = checkpoint['dev_results']
        optimal_params = checkpoint['optimal_params']
        dev_time = checkpoint.get('dev_time', 0)
    else:
        print(f"🎯 STEP 1: Hyperparameter optimization on dev set")
        print(f"⏳ This will run a basic sweep to find good parameters quickly...")
        
        start_time = time.time()
        dev_results, optimal_params = await run_basic_dev_sweep(dataset, early_exit, model)
        dev_time = time.time() - start_time
        
        # Save checkpoint
        checkpoint.update({
            'dev_results': dev_results,
            'optimal_params': optimal_params,
            'dev_time': dev_time
        })
        os.makedirs("results", exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
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
    if 'heuristics_file' in checkpoint and os.path.exists(checkpoint['heuristics_file']):
        print(f"✅ STEP 2: Using cached heuristics from checkpoint")
        heuristics_file = checkpoint['heuristics_file']
    else:
        print(f"\n🧠 STEP 2: Rule generation (analyzing dev results...)")
        heuristics_file = await generate_actual_rules(dataset, dev_results)
        
        # Save checkpoint
        checkpoint['heuristics_file'] = heuristics_file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
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
        'llm_call_reduction': enhanced_results.get('llm_call_reduction', 0)
    }
    
    results['ab_comparison'] = {
        'f1_improvement': f1_improvement,
        'cost_change': cost_change,
        'rules_helped': f1_improvement > 0.01
    }
    
    # FINAL SUMMARY
    total_cost = dev_results['cost_usd'] + baseline_results['cost_usd'] + enhanced_results['cost']
    total_time = dev_time + baseline_time + enhanced_time
    
    print(f"\\n🏆 FINAL RESULTS FOR {dataset.upper()}")
    print(f"Dev F1:        {dev_results['metrics']['f1']:.4f} (${dev_results['cost_usd']:.3f})")
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


async def main():
    parser = argparse.ArgumentParser(description="Complete entity matching pipeline")
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. beer, walmart_amazon)')
    parser.add_argument('--early-exit', action='store_true', help='Stop sweep early if F1 beats leaderboard target')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint if available')
    parser.add_argument('--concurrency', type=int, default=3, help='Number of concurrent API requests')
    parser.add_argument('--validate-rules', action='store_true', help='Validate and optimize rules on dev set before test')
    parser.add_argument('--model', default='gpt-4.1-nano', help='Model to use for dev sweep (default: gpt-4.1-nano)')
    
    args = parser.parse_args()
    
    # Update concurrency settings based on user input
    if args.concurrency != 3:
        print(f"🔧 Setting concurrency to {args.concurrency}")
        # Update concurrency in source files would require more complex logic
        # For now, just show the setting
    
    results = await run_complete_pipeline(args.dataset, args.early_exit, args.resume, args.concurrency, args.validate_rules, args.model)
    return results


if __name__ == "__main__":
    results = asyncio.run(main())