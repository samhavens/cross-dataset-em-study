#!/usr/bin/env python
"""
Intelligent hyperparameter sweeping using Claude Code SDK for analysis-driven optimization.
"""

import asyncio
import json
import os
import tempfile
import pathlib
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from ..entity_matching.hybrid_matcher import run_matching
from .claude_sdk_optimizer import ClaudeSDKOptimizer, OptimizationSuggestion


@dataclass
class HyperparamConfig:
    """Configuration for a hyperparameter setting"""
    max_candidates: int
    semantic_weight: float
    model: str
    use_semantic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "semantic_weight": self.semantic_weight,
            "model": self.model,
            "use_semantic": self.use_semantic
        }


@dataclass
class SweepResult:
    """Result from a single hyperparameter configuration"""
    config: HyperparamConfig
    f1_score: float
    precision: float
    recall: float
    cost_usd: float
    elapsed_seconds: float
    predictions_made: int
    processed_pairs: int
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "config": self.config.to_dict(),
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "cost_usd": self.cost_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "predictions_made": self.predictions_made,
            "processed_pairs": self.processed_pairs
        }
        if self.error:
            result["error"] = self.error
        return result


class IntelligentSweeper:
    """Intelligent hyperparameter sweeper with Claude Code SDK integration"""
    
    def __init__(self, dataset: str, limit: Optional[int] = None, early_exit: bool = False, model: str = "gpt-4.1-nano"):
        self.dataset = dataset
        self.limit = limit
        self.early_exit = early_exit
        self.results: List[SweepResult] = []
        self.claude_optimizer = ClaudeSDKOptimizer()
        
        # Load dataset info for strategic planning
        self.data_root = pathlib.Path('data') / 'raw' / dataset
        B_df = pd.read_csv(self.data_root / 'tableB.csv')
        self.table_b_size = len(B_df)
        
        # Get target F1 for early exit
        if self.early_exit:
            from .claude_sdk_heuristic_generator import get_leaderboard_target_f1
            self.target_f1 = get_leaderboard_target_f1(dataset) / 100.0  # Convert to decimal
            print(f"🎯 Early exit enabled: will stop if F1 > {self.target_f1:.3f} (leaderboard target)")
        
        # Hyperparameter search spaces
        self.candidate_options = [5, 10, 25, 50, 100]
        self.semantic_weight_options = [0.0, 0.3, 0.5, 0.7, 1.0]
        self.model_options = [model]  # Use the specified model
        
    def get_config_cache_path(self, config: HyperparamConfig) -> pathlib.Path:
        """Get cache file path for a specific config"""
        # Create a unique hash for this config + dataset + limit
        config_str = f"{self.dataset}_{self.limit}_{config.to_dict()}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        cache_dir = pathlib.Path('.sweep_cache')
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self.dataset}_{config_hash}.json"
    
    async def run_single_config(self, config: HyperparamConfig) -> SweepResult:
        """Run a single hyperparameter configuration"""
        # Check cache first
        cache_path = self.get_config_cache_path(config)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                print(f"📁 Using cached result: {config.model}, {config.max_candidates} candidates, semantic_weight={config.semantic_weight}")
                return SweepResult(
                    config=config,
                    f1_score=cached_data['f1_score'],
                    precision=cached_data['precision'],
                    recall=cached_data['recall'],
                    cost_usd=cached_data['cost_usd'],
                    elapsed_seconds=cached_data['elapsed_seconds'],
                    predictions_made=cached_data['predictions_made'],
                    processed_pairs=cached_data['processed_pairs']
                )
            except Exception as e:
                print(f"⚠️ Cache read failed: {e}, running fresh")
        
        try:
            semantic_desc = f"semantic_weight={config.semantic_weight}" if config.use_semantic else "trigram_only"
            print(f"Testing config: {config.model}, {config.max_candidates} candidates, {semantic_desc}")
            
            # Run the matching with this configuration
            result = await run_matching(
                dataset=self.dataset,
                limit=self.limit,
                max_candidates=config.max_candidates,
                model=config.model,
                use_semantic=config.use_semantic,
                semantic_weight=config.semantic_weight,
                concurrency=3  # Even lower concurrency to avoid timeouts
            )
            
            # Extract metrics
            metrics = result["metrics"]
            sweep_result = SweepResult(
                config=config,
                f1_score=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                cost_usd=result["cost_usd"],
                elapsed_seconds=result["elapsed_seconds"],
                predictions_made=result["predictions_made"],
                processed_pairs=result["processed_pairs"]
            )
            
            # Cache the result
            try:
                with open(cache_path, 'w') as f:
                    json.dump({
                        'f1_score': sweep_result.f1_score,
                        'precision': sweep_result.precision,
                        'recall': sweep_result.recall,
                        'cost_usd': sweep_result.cost_usd,
                        'elapsed_seconds': sweep_result.elapsed_seconds,
                        'predictions_made': sweep_result.predictions_made,
                        'processed_pairs': sweep_result.processed_pairs
                    }, f)
            except Exception as e:
                print(f"⚠️ Cache write failed: {e}")
            
            return sweep_result
            
        except Exception as e:
            print(f"Error with config {config.to_dict()}: {e}")
            return SweepResult(
                config=config,
                f1_score=0.0,
                precision=0.0,
                recall=0.0,
                cost_usd=0.0,
                elapsed_seconds=0.0,
                predictions_made=0,
                processed_pairs=0,
                error=str(e)
            )
    
    def generate_strategic_configs(self) -> List[HyperparamConfig]:
        """Generate strategic configurations: extreme/low/middle candidates × low/medium/high semantic weights"""
        # Calculate strategic candidate values based on token limits
        table_b_size = self.table_b_size
        
        # Token limits for models (80% of max context)
        model_token_limits = {
            'gpt-4.1-nano': int(1_000_000 * 0.8),   # 800k tokens
            'gpt-4.1-mini': int(1_000_000 * 0.8),   # 800k tokens  
            'o3-mini': int(128_000 * 0.8),          # 102k tokens
            'o4-mini': int(128_000 * 0.8),          # 102k tokens
        }
        
        # Use gpt-4.1-nano for sweep (1M token model)
        max_tokens = model_token_limits.get('gpt-4.1-nano', 800_000)
        
        # Estimate tokens per candidate (rough approximation)
        # Each candidate is ~200-500 tokens depending on record size
        # Be more conservative to avoid timeouts
        tokens_per_candidate = 500  # More conservative estimate
        
        # Calculate max candidates that fit in token budget
        token_based_max = max_tokens // tokens_per_candidate
        
        # Use 10% of table B, but cap at token limit
        percentage_based_max = int(table_b_size * 0.1)
        extreme_candidates = min(percentage_based_max, token_based_max)  # NO artificial 200 cap!
        
        low_candidates = 25
        middle_candidates = int((extreme_candidates * low_candidates) ** 0.5)  # Geometric mean
        
        print(f"📊 Token budget: {max_tokens:,} tokens (~{token_based_max} candidates max)")
        print(f"📊 Table B size: {table_b_size} records (10% = {percentage_based_max})")
        
        print(f"📊 Strategic sweep: {low_candidates} (low) / {middle_candidates} (mid) / {extreme_candidates} (extreme) candidates")
        
        # Semantic weights: low/medium/high
        semantic_weights = [0.2, 0.5, 0.8]  # Low, medium, high
        
        configs = []
        
        # Test all combinations of candidate counts × semantic weights
        for candidates in [low_candidates, middle_candidates, extreme_candidates]:
            for semantic_weight in semantic_weights:
                configs.append(HyperparamConfig(
                    max_candidates=candidates,
                    semantic_weight=semantic_weight,
                    model="gpt-4.1-nano",  # Use cheap model for sweep
                    use_semantic=True
                ))
        
        print(f"📋 Generated {len(configs)} strategic configurations")
        return configs
    
    async def run_strategic_sweep(self) -> List[SweepResult]:
        """Run strategic sweep with extreme/low/middle candidates × low/medium/high semantic weights"""
        print(f"🎯 Running strategic hyperparameter sweep...")
        
        configs = self.generate_strategic_configs()
        
        # Run configurations sequentially to avoid API rate limits
        for i, config in enumerate(configs):
            result = await self.run_single_config(config)
            self.results.append(result)
            print(f"  → F1: {result.f1_score:.4f}, Cost: ${result.cost_usd:.4f}")
            
            # Check for early exit
            if self.early_exit and result.f1_score > self.target_f1:
                print(f"\n🎉 EARLY EXIT: Found F1 {result.f1_score:.4f} > target {self.target_f1:.3f}")
                print(f"✅ Stopping sweep after {i+1}/{len(configs)} configs")
                print(f"🏆 Best config: {config.to_dict()}")
                break
        
        return self.results
    
    def analyze_results_for_claude(self) -> str:
        """Prepare results analysis for Claude Code SDK"""
        if not self.results:
            return "No results available for analysis."
        
        # Sort by F1 score
        sorted_results = sorted(self.results, key=lambda r: r.f1_score, reverse=True)
        
        analysis = "# Hyperparameter Sweep Results Analysis\n\n"
        analysis += f"Dataset: {self.dataset}\n"
        analysis += f"Total configurations tested: {len(self.results)}\n\n"
        
        analysis += "## Top 5 Performing Configurations:\n"
        for i, result in enumerate(sorted_results[:5], 1):
            config = result.config
            analysis += f"{i}. F1={result.f1_score:.4f}, Model={config.model}, "
            analysis += f"Candidates={config.max_candidates}, Semantic_weight={config.semantic_weight}, "
            analysis += f"Cost=${result.cost_usd:.4f}\n"
        
        analysis += "\n## Performance Patterns:\n"
        
        # Analyze by model
        model_performance = {}
        for result in self.results:
            model = result.config.model
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(result.f1_score)
        
        analysis += "### By Model:\n"
        for model, scores in model_performance.items():
            avg_f1 = sum(scores) / len(scores)
            analysis += f"- {model}: avg F1 = {avg_f1:.4f} ({len(scores)} configs)\n"
        
        # Analyze by semantic usage
        semantic_true = [r.f1_score for r in self.results if r.config.use_semantic]
        semantic_false = [r.f1_score for r in self.results if not r.config.use_semantic]
        
        analysis += "\n### By Semantic Similarity:\n"
        if semantic_true:
            analysis += f"- With semantic: avg F1 = {sum(semantic_true)/len(semantic_true):.4f} ({len(semantic_true)} configs)\n"
        if semantic_false:
            analysis += f"- Without semantic: avg F1 = {sum(semantic_false)/len(semantic_false):.4f} ({len(semantic_false)} configs)\n"
        
        # Cost analysis
        analysis += "\n## Cost Efficiency:\n"
        for result in sorted_results[:3]:
            f1_per_dollar = result.f1_score / max(result.cost_usd, 0.001)  # Avoid division by zero
            analysis += f"- Config with F1={result.f1_score:.4f}: ${result.cost_usd:.4f} (F1 per $: {f1_per_dollar:.1f})\n"
        
        # Raw data for Claude analysis
        analysis += "\n## Raw Results Data:\n"
        analysis += "```json\n"
        analysis += json.dumps([r.to_dict() for r in sorted_results], indent=2)
        analysis += "\n```\n"
        
        return analysis
    
    def get_claude_suggestions(self) -> List[HyperparamConfig]:
        """Get Claude SDK suggestions for next configurations to test"""
        analysis = self.analyze_results_for_claude()
        suggestions = self.claude_optimizer.analyze_and_suggest(analysis)
        
        # Convert OptimizationSuggestion to HyperparamConfig
        configs = []
        for suggestion in suggestions:
            config = HyperparamConfig(
                max_candidates=suggestion.max_candidates,
                semantic_weight=suggestion.semantic_weight,
                model=suggestion.model,
                use_semantic=suggestion.use_semantic
            )
            configs.append(config)
            
        return configs
    
    async def run_claude_iteration(self, max_new_configs: int = 5) -> List[SweepResult]:
        """Run one iteration of Claude-suggested configurations"""
        print("\n" + "="*80)
        print("🤖 GETTING CLAUDE SDK SUGGESTIONS FOR NEXT ITERATION")
        print("="*80)
        
        suggested_configs = self.get_claude_suggestions()
        
        print(f"Claude suggested {len(suggested_configs)} new configurations:")
        for i, config in enumerate(suggested_configs[:max_new_configs], 1):
            print(f"  {i}. {config.model}, {config.max_candidates} candidates, semantic_weight={config.semantic_weight}")
        
        print(f"\nTesting top {min(max_new_configs, len(suggested_configs))} suggestions...")
        
        new_results = []
        for config in suggested_configs[:max_new_configs]:
            result = await self.run_single_config(config)
            self.results.append(result)
            new_results.append(result)
            print(f"  → F1: {result.f1_score:.4f}, Cost: ${result.cost_usd:.4f}")
        
        return new_results
    
    async def run_intelligent_sweep(self, 
                                    initial_configs: int = 8,
                                    iterations: int = 2,
                                    configs_per_iteration: int = 5) -> List[SweepResult]:
        """Run complete intelligent sweep: initial + Claude-guided iterations"""
        
        print("🚀 STARTING INTELLIGENT HYPERPARAMETER SWEEP")
        print(f"  Initial configs: {initial_configs}")
        print(f"  Claude iterations: {iterations}")
        print(f"  Configs per iteration: {configs_per_iteration}")
        print("="*80)
        
        # Initial sweep
        await self.run_initial_sweep(initial_configs)
        
        # Claude-guided iterations
        for iteration in range(iterations):
            print(f"\n🔄 CLAUDE ITERATION {iteration + 1}/{iterations}")
            await self.run_claude_iteration(configs_per_iteration)
            
            # Show current best
            best_result = max(self.results, key=lambda r: r.f1_score)
            print(f"\nCurrent best F1: {best_result.f1_score:.4f} with config: {best_result.config.to_dict()}")
            
            # Check if we've hit the target
            if best_result.f1_score >= 0.912:
                print(f"🎉 TARGET F1 > 91.2% ACHIEVED! (F1 = {best_result.f1_score:.4f})")
                break
        
        return self.results
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"Results saved to {filepath}")


async def main():
    """CLI entry point for intelligent sweeping"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intelligent hyperparameter sweeping with Claude SDK")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--limit', type=int, help='Limit number of test pairs (default: use validation set)')
    
    # Sweep mode options
    parser.add_argument('--mode', choices=['basic', 'intelligent'], default='intelligent',
                       help='Sweep mode: basic (initial only) or intelligent (Claude-guided)')
    parser.add_argument('--initial-configs', type=int, default=8, help='Number of initial configurations')
    parser.add_argument('--iterations', type=int, default=2, help='Number of Claude iterations')
    parser.add_argument('--configs-per-iteration', type=int, default=5, help='Configs per Claude iteration')
    
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize sweeper
    sweeper = IntelligentSweeper(args.dataset, args.limit)
    
    if args.mode == 'basic':
        # Run basic initial sweep only
        results = await sweeper.run_initial_sweep(args.initial_configs)
    else:
        # Run full intelligent sweep with Claude iterations
        results = await sweeper.run_intelligent_sweep(
            initial_configs=args.initial_configs,
            iterations=args.iterations,
            configs_per_iteration=args.configs_per_iteration
        )
    
    # Print final analysis
    analysis = sweeper.analyze_results_for_claude()
    print("\n" + "="*80)
    print("📊 FINAL ANALYSIS")
    print("="*80)
    print(analysis)
    
    # Show best result
    best_result = max(sweeper.results, key=lambda r: r.f1_score)
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"F1 Score: {best_result.f1_score:.4f}")
    print(f"Config: {best_result.config.to_dict()}")
    print(f"Cost: ${best_result.cost_usd:.4f}")
    
    if best_result.f1_score >= 0.912:
        print(f"🎉 SUCCESS! Achieved target F1 > 91.2%")
    else:
        gap = 0.912 - best_result.f1_score
        print(f"📈 Still {gap:.3f} F1 points away from 91.2% target")
    
    # Save results if requested
    if args.output:
        sweeper.save_results(args.output)
    
    return sweeper


if __name__ == "__main__":
    sweeper = asyncio.run(main())