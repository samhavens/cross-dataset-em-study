#!/usr/bin/env python
"""
Demonstration of the enhanced experimental workflow.
Shows structured output, hyperparameter sweeping, and auto-testing.
"""

import subprocess
import pathlib
import json
import pandas as pd
import os

def demo_basic_structured_output():
    """Demo basic structured output functionality"""
    print("🎯 DEMO 1: Structured Output")
    print("-" * 40)
    
    # Ensure mock mode
    env = os.environ.copy()
    if 'OPENAI_API_KEY' in env:
        del env['OPENAI_API_KEY']
    
    cmd = [
        "python", "llm_em_hybrid.py",
        "--dataset", "beer",
        "--limit", "10",
        "--candidate-ratio", "0.05",
        "--output-json", "demo_results.json",
        "--output-csv", "demo_log.csv"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        # Show JSON structure
        with open("demo_results.json") as f:
            data = json.load(f)
        
        print(f"\n✅ JSON Results Structure:")
        print(f"   Dataset: {data['dataset']}")
        print(f"   Model: {data['model']}")
        print(f"   Candidate Ratio: {data['candidate_ratio']:.1%}")
        print(f"   F1 Score: {data['metrics']['f1']:.1f}")
        print(f"   Cost: ${data['cost_usd']:.3f}")
        print(f"   Processing Time: {data['elapsed_seconds']:.1f}s")
        
        # Show CSV logging
        df = pd.read_csv("demo_log.csv")
        print(f"\n✅ CSV Log:")
        print(f"   Rows: {len(df)}")
        print(f"   Key columns: {df[['dataset', 'candidate_ratio', 'f1', 'cost_usd']].to_string(index=False)}")
    else:
        print("❌ Demo 1 failed")

def demo_hyperparameter_sweep():
    """Demo hyperparameter sweeping with auto-test"""
    print(f"\n🎯 DEMO 2: Hyperparameter Sweep + Auto-Test")
    print("-" * 50)
    
    env = os.environ.copy()
    if 'OPENAI_API_KEY' in env:
        del env['OPENAI_API_KEY']
    
    cmd = [
        "python", "tools/sweep_candidates.py",
        "--dataset", "beer",
        "--limit", "10", 
        "--num-points", "3",
        "--auto-test",
        "--output", "demo_sweep.json",
        "--output-csv", "demo_log.csv"  # Append to same CSV
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    
    if result.returncode == 0:
        # Show sweep results
        with open("demo_sweep.json") as f:
            data = json.load(f)
        
        print(f"\n✅ Sweep Results:")
        sweep_results = data["sweep_results"]
        print(f"   Trials run: {len(sweep_results)}")
        
        best_val = data["best_validation"]
        print(f"   Best validation F1: {best_val['f1']:.1f} at {best_val['candidate_ratio']:.1%}")
        
        test_result = data["test_results"]
        print(f"   Test set F1: {test_result['f1']:.1f}")
        print(f"   Validation→Test difference: {test_result['f1'] - best_val['f1']:+.1f}")
        
        # Show updated CSV log
        df = pd.read_csv("demo_log.csv")
        print(f"\n✅ Updated CSV Log:")
        print(f"   Total experiments: {len(df)}")
        print(f"   Datasets: {df['dataset'].unique()}")
        print(f"   F1 range: {df['f1'].min():.1f} - {df['f1'].max():.1f}")
    else:
        print("❌ Demo 2 failed")

def demo_analysis():
    """Demo results analysis"""
    print(f"\n🎯 DEMO 3: Results Analysis")
    print("-" * 35)
    
    if pathlib.Path("demo_log.csv").exists():
        df = pd.read_csv("demo_log.csv")
        
        print(f"📊 Experiment Log Analysis:")
        print(f"   Total runs: {len(df)}")
        print(f"   Unique datasets: {df['dataset'].nunique()}")
        print(f"   Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")
        
        # Group by dataset
        summary = df.groupby('dataset').agg({
            'f1': ['count', 'max', 'mean'],
            'cost_usd': 'sum',
            'candidate_ratio': 'nunique'
        }).round(3)
        
        print(f"\n📈 Per-Dataset Summary:")
        print(summary)
        
        # Best configuration per dataset
        best_configs = df.loc[df.groupby('dataset')['f1'].idxmax()]
        print(f"\n🏆 Best Configurations:")
        for _, row in best_configs.iterrows():
            print(f"   {row['dataset']:15}: F1={row['f1']:5.1f}, Ratio={row['candidate_ratio']:6.1%}, Cost=${row['cost_usd']:6.3f}")
    else:
        print("❌ No log file found for analysis")

def cleanup():
    """Clean up demo files"""
    demo_files = [
        "demo_results.json", "demo_sweep.json", "demo_log.csv"
    ]
    
    for file in demo_files:
        if pathlib.Path(file).exists():
            pathlib.Path(file).unlink()

if __name__ == "__main__":
    print("🚀 ENHANCED EXPERIMENTAL WORKFLOW DEMO")
    print("=" * 60)
    print("This demo shows the new structured output, hyperparameter")
    print("sweeping, and automatic test evaluation capabilities.")
    print("=" * 60)
    
    try:
        demo_basic_structured_output()
        demo_hyperparameter_sweep()
        demo_analysis()
        
        print(f"\n🎉 DEMO COMPLETE!")
        print(f"The enhanced system provides:")
        print(f"  ✅ Structured JSON/CSV output for reproducibility")
        print(f"  ✅ Automated hyperparameter optimization")
        print(f"  ✅ Validation→Test evaluation pipeline")
        print(f"  ✅ Comprehensive experiment tracking")
        print(f"  ✅ Results analysis and competitive benchmarking")
        
    finally:
        print(f"\n🧹 Cleaning up demo files...")
        cleanup()