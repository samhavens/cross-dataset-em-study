#!/usr/bin/env python3
"""
Improved hyperparameter sweep implementation.

Clean, maintainable replacement for the existing sweep logic with:
- No file swapping madness
- Proper predictions saving 
- Clean separation of concerns
- Robust error handling
- Detailed logging
"""

import asyncio
import json
import pathlib
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from ..entity_matching.hybrid_matcher import run_matching


@dataclass
class SweepConfig:
    """Clean configuration for hyperparameter sweep"""
    dataset: str
    model: str = 'gpt-4.1-nano'
    early_exit: bool = False
    max_configs: int = 9  # 3x3 grid by default
    concurrency: int = 3


@dataclass 
class SweepResult:
    """Individual sweep result with all data preserved"""
    config: Dict[str, Any]
    metrics: Dict[str, float]  # f1, precision, recall
    cost_usd: float
    processed_pairs: int
    predictions_made: int
    predictions: List[Dict[str, Any]]  # The missing piece!
    processing_time: float
    error: Optional[str] = None


class ImprovedSweeper:
    """Clean hyperparameter sweeper without file manipulation"""
    
    def __init__(self, config: SweepConfig):
        self.config = config
        self.results: List[SweepResult] = []
        self.data_root = pathlib.Path('data') / 'raw' / config.dataset
        
    def _get_data_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Load data splits without file manipulation"""
        # Use validation set for dev if available, otherwise use train/test split
        if (self.data_root / 'valid.csv').exists():
            print("✅ Using validation set for hyperparameter sweep (clean dev set)")
            dev_data = pd.read_csv(self.data_root / 'valid.csv')
            test_data = pd.read_csv(self.data_root / 'test.csv') if (self.data_root / 'test.csv').exists() else None
            return dev_data, dev_data, test_data  # Use dev for both train and dev
            
        elif (self.data_root / 'train.csv').exists():
            print("✅ Using train set for hyperparameter sweep (dev slice)")
            train_data = pd.read_csv(self.data_root / 'train.csv')
            test_data = pd.read_csv(self.data_root / 'test.csv') if (self.data_root / 'test.csv').exists() else None
            
            # Split train into dev slice (no test leakage)
            dev_size = min(200, len(train_data) // 2)  # Reasonable dev size
            dev_data = train_data.sample(n=dev_size, random_state=42)
            return train_data, dev_data, test_data
            
        else:
            print("⚠️ No train/valid sets found - using test set (will contaminate results)")
            test_data = pd.read_csv(self.data_root / 'test.csv')
            return test_data, test_data, None
    
    def _generate_strategic_configs(self) -> List[Dict[str, Any]]:
        """Generate strategic 3x3 grid of configurations based on dataset size"""
        import random
        
        # Load tableB to get dataset size
        B_df = pd.read_csv(self.data_root / 'tableB.csv')
        total_candidates = len(B_df)
        
        # Calculate candidate ranges based on dataset size
        min_candidates = max(10, int(0.001 * total_candidates))  # max(10, 0.1% of candidates)
        
        # Estimate context window limit (assuming ~100 tokens per candidate, 128k context = ~1280 candidates max)
        # Use 85% of theoretical capacity for safety margin
        theoretical_max_candidates = 1280  # 128k tokens / ~100 tokens per candidate
        context_limit_candidates = min(int(0.85 * theoretical_max_candidates), total_candidates)  # min(85% of context capacity, 100% candidates)
        max_candidates = context_limit_candidates
        
        # Geometric mean for middle option
        mid_candidates = int(math.sqrt(min_candidates * max_candidates))
        
        candidates_options = [min_candidates, mid_candidates, max_candidates]
        semantic_weights = [0.15, 0.5, 0.85]  # low, medium, high
        
        print(f"📊 Dataset size: {total_candidates} candidates")
        print(f"🎯 Candidate sweep: {candidates_options} ({min_candidates/total_candidates*100:.1f}%, {mid_candidates/total_candidates*100:.1f}%, {max_candidates/total_candidates*100:.1f}%)")
        print(f"⚖️ Semantic weights: {semantic_weights}")
        
        # Generate all combinations systematically first
        configs = []
        for candidates in candidates_options:
            for weight in semantic_weights:
                configs.append({
                    'max_candidates': candidates,
                    'semantic_weight': weight,
                    'model': self.config.model,
                    'use_semantic': True
                })
        
        # 🎲 SHUFFLE to bounce around parameter space instead of linear traversal!
        random.shuffle(configs)
        
        print(f"🎯 Generated {len(configs)} strategic configurations")
        print(f"🎲 Shuffled order to bounce around parameter space!")
        
        # Show the randomized order
        print(f"🔀 Randomized sweep order:")
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config['max_candidates']} candidates, weight={config['semantic_weight']:.2f}")
        
        return configs
    
    async def _run_single_config(self, config: Dict[str, Any], dev_data: pd.DataFrame) -> SweepResult:
        """Run a single configuration and return complete results - NO FILE SWAPPING"""
        print(f"  Testing: {config['max_candidates']} candidates, weight={config['semantic_weight']:.1f}")
        
        start_time = time.time()
        try:
            # Create temporary dataset directory - NO FILE SWAPPING
            import os
            import shutil
            os.makedirs("results/temp", exist_ok=True)
            temp_dataset_dir = pathlib.Path('results/temp') / f'{self.config.dataset}_sweep_temp_{int(time.time()*1000)}'
            temp_dataset_dir.mkdir(exist_ok=True)
            
            try:
                # Copy essential files
                shutil.copy(self.data_root / 'tableA.csv', temp_dataset_dir / 'tableA.csv')
                shutil.copy(self.data_root / 'tableB.csv', temp_dataset_dir / 'tableB.csv')
                
                # Write dev data as test file for this run
                dev_data.to_csv(temp_dataset_dir / 'test.csv', index=False)
                
                # Copy to expected data/raw location since run_matching expects that structure
                expected_path = pathlib.Path('data/raw') / f'temp_{temp_dataset_dir.name}'
                if expected_path.exists():
                    shutil.rmtree(expected_path)
                expected_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(temp_dataset_dir, expected_path)
                
                try:
                    # Run matching with the configuration on temporary dataset
                    # Use original dataset name for embeddings cache efficiency
                    results = await run_matching(
                        dataset=f'temp_{temp_dataset_dir.name}',
                        limit=None,  # Use full dev set
                        embeddings_cache_dataset=self.config.dataset,  # Reuse cache from original dataset
                        **config,
                        concurrency=self.config.concurrency
                    )
                finally:
                    # Clean up the expected path copy
                    if expected_path.exists():
                        shutil.rmtree(expected_path)
                
                processing_time = time.time() - start_time
                
                # Extract predictions if available
                predictions = results.get('predictions', {})
                if not predictions:
                    print(f"    ⚠️ No predictions saved for config {config}")
                
                return SweepResult(
                    config=config,
                    metrics={
                        'f1': results['metrics']['f1'],
                        'precision': results['metrics']['precision'], 
                        'recall': results['metrics']['recall']
                    },
                    cost_usd=results['cost_usd'],
                    processed_pairs=results['processed_pairs'],
                    predictions_made=results['predictions_made'],
                    predictions=predictions,
                    processing_time=processing_time
                )
                
            finally:
                # Clean up temporary dataset
                if temp_dataset_dir.exists():
                    shutil.rmtree(temp_dataset_dir)
                    
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"    ❌ Config failed: {e}")
            return SweepResult(
                config=config,
                metrics={'f1': 0.0, 'precision': 0.0, 'recall': 0.0},
                cost_usd=0.0,
                processed_pairs=0,
                predictions_made=0, 
                predictions=[],
                processing_time=processing_time,
                error=str(e)
            )
    
    async def run_sweep(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the complete hyperparameter sweep"""
        print(f"🚀 Starting improved hyperparameter sweep for {self.config.dataset}")
        
        # Load data splits cleanly
        train_data, dev_data, test_data = self._get_data_splits()
        print(f"📊 Dev set: {len(dev_data)} pairs")
        
        # Generate configurations
        configs = self._generate_strategic_configs()
        
        # Run sweep
        print(f"⏳ Testing {len(configs)} configurations...")
        for config in configs:
            result = await self._run_single_config(config, dev_data)
            self.results.append(result)
            
            if not result.error:
                print(f"    → F1: {result.metrics['f1']:.4f}, Cost: ${result.cost_usd:.4f}")
            
            # Check early exit
            if (self.config.early_exit and 
                result.metrics['f1'] > 0.85 and  # Reasonable threshold
                not result.error):
                print(f"🎉 Early exit: F1 {result.metrics['f1']:.4f} > threshold")
                break
        
        # Find best result
        valid_results = [r for r in self.results if not r.error]
        if not valid_results:
            raise RuntimeError("All sweep configurations failed!")
            
        best_result = max(valid_results, key=lambda r: r.metrics['f1'])
        
        print(f"✅ Best configuration: F1={best_result.metrics['f1']:.4f}")
        print(f"   {best_result.config}")
        
        # Create a unified sweep result that keeps config and metrics together atomically
        unified_result = {
            'best_config': best_result.config.copy(),
            'best_metrics': best_result.metrics.copy(),
            'best_cost_usd': best_result.cost_usd,
            'best_processed_pairs': best_result.processed_pairs,
            'best_predictions_made': best_result.predictions_made,
            'best_predictions': best_result.predictions,  # Real predictions for agentic analysis
            'best_processing_time': best_result.processing_time,
            'all_tested_configs': [
                {
                    'config': r.config,
                    'f1': r.metrics['f1'],
                    'precision': r.metrics['precision'],
                    'recall': r.metrics['recall'],
                    'cost_usd': r.cost_usd,
                    'error': r.error
                } for r in self.results
            ],
            'sweep_summary': {
                'total_configs_tested': len(valid_results),
                'total_configs_failed': len([r for r in self.results if r.error]),
                'best_f1': best_result.metrics['f1'],
                'config_that_achieved_best_f1': best_result.config
            }
        }
        
        # For backwards compatibility with existing pipeline
        dev_results = {
            'metrics': unified_result['best_metrics'],
            'cost_usd': unified_result['best_cost_usd'],
            'processed_pairs': unified_result['best_processed_pairs'],
            'predictions_made': unified_result['best_predictions_made'],
            'predictions': unified_result['best_predictions'],
            'unified_sweep_result': unified_result  # Include the complete result
        }
        optimal_params = unified_result['best_config']
        
        return dev_results, optimal_params


async def run_improved_sweep(dataset: str, early_exit: bool = False, model: str = 'gpt-4.1-nano', concurrency: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Drop-in replacement for run_basic_dev_sweep() with clean implementation.
    
    Compatible interface for use in run_complete_pipeline.py
    """
    config = SweepConfig(
        dataset=dataset,
        model=model,
        early_exit=early_exit,
        concurrency=concurrency
    )
    
    sweeper = ImprovedSweeper(config)
    return await sweeper.run_sweep()


if __name__ == "__main__":
    # Test the improved sweeper
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--early-exit', action='store_true')
    parser.add_argument('--model', default='gpt-4.1-nano')
    
    args = parser.parse_args()
    
    async def test():
        dev_results, optimal_params = await run_improved_sweep(
            args.dataset, args.early_exit, args.model
        )
        print(f"\n🎯 Final Results:")
        print(f"F1: {dev_results['metrics']['f1']:.4f}")
        print(f"Optimal: {optimal_params}")
        print(f"Predictions saved: {len(dev_results['predictions'])}")
    
    asyncio.run(test())