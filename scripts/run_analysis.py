#!/usr/bin/env python3
"""
CLI script for running dataset analysis using the production analysis module.

Usage:
    python scripts/run_analysis.py --dataset itunes_amazon
    python scripts/run_analysis.py --dataset beer --max-pairs 200
"""

import argparse
import pathlib
import sys

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from entity_matching.analysis import analyze_dataset_for_claude


def main():
    parser = argparse.ArgumentParser(description="Generate rich similarity analysis for Claude optimization")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. itunes_amazon, beer)")
    parser.add_argument("--max-pairs", type=int, default=200, help="Maximum pairs to analyze")
    parser.add_argument("--max-candidates", type=int, default=100, help="Candidates threshold for analysis")
    parser.add_argument("--output", help="Output JSON file (default: results/{dataset}_claude_analysis.json)")

    args = parser.parse_args()

    # Set default output if not specified
    if not args.output:
        args.output = f"results/{args.dataset}_claude_analysis.json"

    # Run analysis
    try:
        analyze_dataset_for_claude(
            dataset=args.dataset,
            max_pairs=args.max_pairs,
            max_candidates=args.max_candidates,
            output_file=args.output,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()