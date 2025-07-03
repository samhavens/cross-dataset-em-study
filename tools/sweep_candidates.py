#!/usr/bin/env python
"""
Hyperparameter sweep for --candidate-ratio in llm_em_hybrid.py.
Finds the optimal candidate ratio that maximizes F1 on the validation set.
"""

import argparse
import json
import subprocess
import pathlib
import time
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd

from dataset_info import parse_sizes_md, estimate_max_safe_candidates, get_competitive_f1_threshold

def run_llm_em_hybrid(dataset: str, candidate_ratio: float = None, max_candidates: int = None,
                      model: str = "gpt-4.1-nano", use_valid: bool = True, limit: int = None, 
                      csv_file: str = None) -> Dict:
    """
    Run llm_em_hybrid.py with specific parameters and parse results.
    
    Args:
        dataset: Dataset name
        candidate_ratio: Ratio of table B to use as candidates (alternative to max_candidates)
        max_candidates: Absolute number of candidates (alternative to candidate_ratio)
        model: Model to use
        use_valid: Whether to use validation set
        limit: Limit on test pairs
        csv_file: CSV file to log results
    
    Returns:
        Dict with metrics: precision, recall, f1, predictions_made, etc.
    """
    # Find the llm_em_hybrid.py script relative to this file
    script_dir = pathlib.Path(__file__).parent
    llm_script = script_dir.parent / "llm_em_hybrid.py"
    
    cmd = [
        "python", str(llm_script),
        "--dataset", dataset,
        "--model", model
    ]
    
    # Add candidate selection parameter
    if candidate_ratio is not None:
        cmd.extend(["--candidate-ratio", str(candidate_ratio)])
    elif max_candidates is not None:
        cmd.extend(["--max-candidates", str(max_candidates)])
    else:
        raise ValueError("Either candidate_ratio or max_candidates must be specified")
    
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    if csv_file:
        cmd.extend(["--output-csv", csv_file])
    
    # Use validation set for hyperparameter tuning
    if use_valid:
        # Temporarily copy valid.csv to test.csv for evaluation
        root = pathlib.Path("data/raw") / dataset
        test_backup = root / "test_backup.csv"
        test_file = root / "test.csv"
        valid_file = root / "valid.csv"
        
        if valid_file.exists():
            # Backup original test file
            if test_file.exists():
                test_file.rename(test_backup)
            
            # Copy valid to test
            import shutil
            shutil.copy2(valid_file, test_file)
    
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout for sweeps
        
        if result.returncode != 0:
            print(f"Error running command: {result.stderr}")
            return {"error": result.stderr}
        
        # Parse output for metrics
        output = result.stdout
        print(f"Output preview: {output[-500:]}")  # Show last 500 chars
        
        metrics = {}
        
        # Extract metrics from output
        for line in output.split('\n'):
            if 'Precision:' in line:
                metrics['precision'] = float(line.split('Precision:')[1].strip())
            elif 'Recall:' in line:
                metrics['recall'] = float(line.split('Recall:')[1].strip())
            elif 'F1-Score:' in line:
                metrics['f1'] = float(line.split('F1-Score:')[1].strip())
            elif 'Predictions made:' in line:
                metrics['predictions_made'] = int(line.split('Predictions made:')[1].strip())
            elif 'Processed:' in line and 'pairs' in line:
                metrics['total_pairs'] = int(line.split('Processed:')[1].split('pairs')[0].strip())
        
        # Extract cost information if available
        if '→ $' in output:
            cost_line = [line for line in output.split('\n') if '→ $' in line][-1]
            cost_match = cost_line.split('→ $')[1].split()[0]
            try:
                metrics['cost'] = float(cost_match.replace(',', ''))
            except:
                pass
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Restore original test file
        if use_valid:
            root = pathlib.Path("data/raw") / dataset
            test_backup = root / "test_backup.csv"
            test_file = root / "test.csv"
            
            if test_backup.exists():
                if test_file.exists():
                    test_file.unlink()
                test_backup.rename(test_file)

def generate_candidate_counts(table_b_size: int, min_candidates: int = 10, num_points: int = 10) -> List[int]:
    """
    Generate candidate counts for sweeping from min_candidates to table_b_size.
    Uses logarithmic spacing to cover the full range efficiently.
    """
    if min_candidates >= table_b_size:
        return [table_b_size]
    
    # Logarithmic spacing from min_candidates to table_b_size
    log_min = np.log10(max(1, min_candidates))
    log_max = np.log10(table_b_size)
    
    log_counts = np.linspace(log_min, log_max, num_points)
    counts = [int(10 ** lc) for lc in log_counts]
    
    # Ensure we include min_candidates and table_b_size
    counts = [min_candidates] + counts + [table_b_size]
    
    # Remove duplicates and sort
    counts = sorted(list(set(counts)))
    
    # Limit to requested number of points
    if len(counts) > num_points:
        # Keep min, max, and evenly spaced points in between
        indices = np.linspace(0, len(counts) - 1, num_points).astype(int)
        counts = [counts[i] for i in indices]
    
    return counts

def generate_candidate_ratios(max_ratio: float, num_points: int = 8) -> List[float]:
    """
    Generate candidate ratios for sweeping.
    Uses logarithmic spacing to cover more of the interesting range.
    """
    # Start from 0.5% up to max_ratio
    min_ratio = 0.005  # 0.5%
    
    if max_ratio <= min_ratio:
        return [max_ratio]
    
    # Logarithmic spacing
    log_min = np.log10(min_ratio)
    log_max = np.log10(max_ratio)
    
    log_ratios = np.linspace(log_min, log_max, num_points)
    ratios = [10 ** lr for lr in log_ratios]
    
    # Add a few linear points at the low end
    linear_ratios = [0.001, 0.002, 0.005] 
    all_ratios = linear_ratios + ratios
    
    # Remove duplicates and sort
    all_ratios = sorted(list(set(r for r in all_ratios if r <= max_ratio)))
    
    return all_ratios

def sweep_dataset(dataset: str, model: str = "gpt-4.1-nano", num_points: int = 10, 
                  limit: int = None, csv_file: str = None, use_max_candidates: bool = True) -> List[Dict]:
    """
    Perform hyperparameter sweep for a single dataset.
    
    Returns:
        List of results with candidate_ratio and metrics
    """
    print(f"\n=== SWEEPING {dataset.upper()} ===")
    
    # Get dataset info
    try:
        sizes = parse_sizes_md()
        dataset_info = sizes[dataset]
        table_b_size = dataset_info.table_b_records
        competitive_f1 = get_competitive_f1_threshold(dataset)
        print(f"Table B size: {table_b_size:,} records")
        print(f"Competitive F1 threshold: {competitive_f1:.1f}")
    except Exception as e:
        print(f"Warning: Could not get dataset info: {e}")
        table_b_size = 1000  # Default fallback
        competitive_f1 = 80.0
    
    # Generate candidate counts or ratios
    if use_max_candidates:
        candidate_counts = generate_candidate_counts(table_b_size, min_candidates=10, num_points=num_points)
        print(f"Testing {len(candidate_counts)} candidate counts: {candidate_counts}")
        test_params = [(None, count) for count in candidate_counts]  # (ratio, max_candidates)
    else:
        max_candidates, max_ratio = estimate_max_safe_candidates(dataset)
        ratios = generate_candidate_ratios(max_ratio, num_points)
        print(f"Testing {len(ratios)} candidate ratios: {[f'{r:.1%}' for r in ratios]}")
        test_params = [(ratio, None) for ratio in ratios]  # (ratio, max_candidates)
    
    results = []
    
    for i, (ratio, max_candidates) in enumerate(test_params):
        if max_candidates:
            param_desc = f"{max_candidates:,} candidates"
            print(f"\n--- Trial {i+1}/{len(test_params)}: {max_candidates:,} candidates ---")
        else:
            param_desc = f"{ratio:.1%} ratio"
            print(f"\n--- Trial {i+1}/{len(test_params)}: {ratio:.1%} ratio ---")
        
        start_time = time.time()
        metrics = run_llm_em_hybrid(dataset, candidate_ratio=ratio, max_candidates=max_candidates,
                                   model=model, use_valid=True, limit=limit, csv_file=csv_file)
        elapsed = time.time() - start_time
        
        if "error" in metrics:
            print(f"FAILED: {metrics['error']}")
            continue
        
        result = {
            "dataset": dataset,
            "candidate_ratio": ratio if ratio else max_candidates / table_b_size,
            "max_candidates": max_candidates if max_candidates else int(ratio * table_b_size),
            "model": model,
            "elapsed_seconds": elapsed,
            **metrics
        }
        
        results.append(result)
        
        f1 = metrics.get('f1', 0)
        predictions = metrics.get('predictions_made', 0)
        total = metrics.get('total_pairs', 0)
        prediction_rate = predictions / total * 100 if total > 0 else 0
        
        print(f"F1: {f1:.1f}, Predictions: {predictions}/{total} ({prediction_rate:.1f}%), Time: {elapsed:.1f}s")
        
        # Early stopping if we're doing really well
        if f1 > competitive_f1 * 1.05:  # 5% above competitive threshold
            print(f"🎉 Excellent result! F1 {f1:.1f} >> competitive threshold {competitive_f1:.1f}")
    
    return results

def run_best_on_test(dataset: str, best_ratio: float = None, best_max_candidates: int = None,
                     model: str = "gpt-4.1-nano", csv_file: str = None) -> Dict:
    """
    Run the best hyperparameters on the actual test set.
    
    Returns:
        Dict with test set results
    """
    print(f"\n=== RUNNING BEST CONFIG ON TEST SET ===")
    print(f"Dataset: {dataset}")
    if best_max_candidates:
        print(f"Best max candidates: {best_max_candidates:,}")
    else:
        print(f"Best candidate ratio: {best_ratio:.1%}")
    print(f"Model: {model}")
    
    # Run on test set (use_valid=False means use actual test.csv)
    start_time = time.time()
    test_results = run_llm_em_hybrid(dataset, candidate_ratio=best_ratio, max_candidates=best_max_candidates,
                                    model=model, use_valid=False, csv_file=csv_file)
    elapsed = time.time() - start_time
    
    if "error" in test_results:
        print(f"❌ Test run FAILED: {test_results['error']}")
        return test_results
    
    f1 = test_results.get('f1', 0)
    predictions = test_results.get('predictions_made', 0)
    total = test_results.get('total_pairs', 0)
    cost = test_results.get('cost', 0)
    
    print(f"✅ Test Results:")
    print(f"   F1: {f1:.1f}")
    print(f"   Predictions: {predictions}/{total}")
    print(f"   Cost: ${cost:.2f}")
    print(f"   Time: {elapsed:.1f}s")
    
    # Compare against competitive threshold
    try:
        competitive_f1 = get_competitive_f1_threshold(dataset)
        if f1 >= competitive_f1:
            print(f"🎉 COMPETITIVE: F1 {f1:.1f} >= threshold {competitive_f1:.1f}")
        else:
            print(f"⚠️  Below competitive: F1 {f1:.1f} < threshold {competitive_f1:.1f}")
    except:
        print(f"📊 F1 achieved: {f1:.1f}")
    
    return test_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset to sweep")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--num-points", type=int, default=8, help="Number of candidate ratios to test")
    parser.add_argument("--limit", type=int, help="Limit number of test pairs (for faster debugging)")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--output-csv", help="CSV file to log all results")
    parser.add_argument("--auto-test", action="store_true", help="Automatically run best config on test set")
    parser.add_argument("--use-max-candidates", action="store_true", default=True, 
                       help="Use absolute candidate counts instead of ratios (default: True)")
    args = parser.parse_args()
    
    # Set up CSV logging if requested
    csv_file = args.output_csv
    if csv_file:
        print(f"Logging results to: {csv_file}")
    
    # Run the sweep
    results = sweep_dataset(args.dataset, args.model, args.num_points, args.limit, csv_file, args.use_max_candidates)
    
    if not results:
        print("No successful results!")
        return
    
    # Find best result
    best_result = max(results, key=lambda x: x.get('f1', 0))
    
    print(f"\n=== SWEEP COMPLETE FOR {args.dataset.upper()} ===")
    print(f"Tested {len(results)} configurations")
    print(f"Best F1: {best_result['f1']:.1f} at {best_result['candidate_ratio']:.1%} ratio")
    print(f"Best config: {best_result['candidate_ratio']:.1%} ratio, {best_result.get('predictions_made', 0)} predictions")
    
    # Show all results sorted by F1
    print(f"\nAll results (sorted by F1):")
    sorted_results = sorted(results, key=lambda x: x.get('f1', 0), reverse=True)
    for i, r in enumerate(sorted_results[:5]):  # Top 5
        f1 = r.get('f1', 0)
        ratio = r['candidate_ratio']
        preds = r.get('predictions_made', 0)
        print(f"{i+1}. F1: {f1:.1f}, Ratio: {ratio:.1%}, Predictions: {preds}")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        output_file = f"sweep_results_{args.dataset}_{args.model.replace('/', '_')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Provide recommendation
    try:
        competitive_f1 = get_competitive_f1_threshold(args.dataset)
        if best_result['f1'] >= competitive_f1:
            print(f"🎉 SUCCESS: F1 {best_result['f1']:.1f} >= competitive threshold {competitive_f1:.1f}")
        else:
            print(f"⚠️  Below competitive: F1 {best_result['f1']:.1f} < threshold {competitive_f1:.1f}")
    except:
        pass
    
    print(f"\nRecommended command:")
    print(f"python llm_em_hybrid.py --dataset {args.dataset} --candidate-ratio {best_result['candidate_ratio']:.3f} --model {args.model}")
    
    # Auto-test on test set if requested
    if args.auto_test:
        best_ratio = best_result.get('candidate_ratio')
        best_max_candidates = best_result.get('max_candidates')
        test_results = run_best_on_test(args.dataset, best_ratio=best_ratio, 
                                       best_max_candidates=best_max_candidates, 
                                       model=args.model, csv_file=csv_file)
        
        # Add test results to saved data
        if args.output:
            output_data = {
                "sweep_results": results,
                "best_validation": best_result,
                "test_results": test_results
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nComplete results (sweep + test) saved to: {args.output}")
    
    print(f"\n🎯 Summary:")
    print(f"   Best validation F1: {best_result['f1']:.1f} at {best_result['candidate_ratio']:.1%} ratio")
    if args.auto_test and test_results and 'f1' in test_results:
        test_f1 = test_results['f1']
        print(f"   Test set F1: {test_f1:.1f}")
        print(f"   Validation → Test difference: {test_f1 - best_result['f1']:+.1f}")

if __name__ == "__main__":
    main()