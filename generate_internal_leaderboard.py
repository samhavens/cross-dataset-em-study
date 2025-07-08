#!/usr/bin/env python
"""
Generate internal leaderboard from pipeline results.

Scans results/ folder for *_complete_pipeline.json files and creates
an internal leaderboard showing our best results (baseline vs enhanced).
"""

import argparse
import json
import pathlib
import re

from typing import Dict, Tuple


class InternalLeaderboardGenerator:
    """Generate internal leaderboard from pipeline results"""

    def __init__(self):
        self.results_dir = pathlib.Path("results")
        self.dataset_mappings = {
            "abt_buy": "abt",
            "dblp_acm": "dblp\\_acm",
            "dblp_scholar": "dblp\\_scholar",
            "fodors_zagat": "fodors\\_zagat",
            "zomato_yelp": "zomato\\_yelp",
            "amazon_google": "amazon\\_google",
            "beer": "beer",
            "itunes_amazon": "itunes\\_amazon",
            "rotten_imdb": "rotten\\_imdb",
            "walmart_amazon": "walmart\\_amazon",
        }

    def load_pipeline_results(self) -> Dict[str, Dict]:
        """Load all pipeline results from results/ folder"""
        results = {}

        # Find all complete pipeline result files
        pipeline_files = list(self.results_dir.glob("*_complete_pipeline.json"))

        for file in pipeline_files:
            dataset = file.stem.replace("_complete_pipeline", "")

            try:
                with open(file) as f:
                    data = json.load(f)
                    results[dataset] = data
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
                continue

        return results

    def get_best_result(self, pipeline_data: Dict) -> Tuple[float, str, Dict]:
        """Get the best result (baseline or enhanced) from pipeline data"""
        baseline_f1 = pipeline_data.get("baseline_results", {}).get("f1", 0.0)
        enhanced_f1 = pipeline_data.get("enhanced_results", {}).get("f1", 0.0)

        if enhanced_f1 > baseline_f1:
            # Enhanced is better
            return enhanced_f1, "enhanced", pipeline_data.get("enhanced_results", {})
        # Baseline is better (or equal)
        return baseline_f1, "baseline", pipeline_data.get("baseline_results", {})

    def load_original_leaderboard(self) -> str:
        """Load the original leaderboard.md for reference"""
        leaderboard_file = pathlib.Path("leaderboard.md")
        if leaderboard_file.exists():
            with open(leaderboard_file) as f:
                return f.read()
        return ""

    def extract_leaderboard_target(self, dataset: str) -> float:
        """Extract the top F1 score from original leaderboard for this dataset"""
        original_content = self.load_original_leaderboard()

        # Get the dataset section name
        section_name = self.dataset_mappings.get(dataset, dataset)

        # Find the dataset section
        section_pattern = rf"### {re.escape(section_name)}(?:\s+\([^)]+\))?\s*\n"
        match = re.search(section_pattern, original_content, re.IGNORECASE)

        if not match:
            return 0.0

        # Extract the section content until the next ### or end of file
        start = match.end()
        next_section = re.search(r"\n### ", original_content[start:])
        if next_section:
            section_content = original_content[start : start + next_section.start()]
        else:
            section_content = original_content[start:]

        # Find the bold F1 score (which is the top non-jellyfish score)
        bold_f1_score = None

        for line in section_content.split("\n"):
            if "|" in line and any(char.isdigit() for char in line):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    f1_text = parts[-1]  # Last column is F1

                    # Look for bold score: **92.4** (check this first!)
                    if f1_text.startswith("**") and f1_text.endswith("**"):
                        f1_match = re.search(r"(\d+\.?\d*)", f1_text)
                        if f1_match:
                            bold_f1_score = float(f1_match.group(1))
                            break  # Found the bold score, we're done

                    # Skip italicized scores (jellyfish) - only single asterisks
                    if f1_text.startswith("*") and f1_text.endswith("*") and not f1_text.startswith("**"):
                        continue

        return bold_f1_score if bold_f1_score is not None else 0.0

    def generate_internal_leaderboard(self, results: Dict[str, Dict]) -> str:
        """Generate internal leaderboard markdown"""

        markdown = """# Internal Leaderboard

Our best results on entity matching datasets. Shows the better of baseline (optimal hyperparameters) vs enhanced (with heuristic rules).

**Legend:**
- 🎯 **Our Result**: Best F1 from our pipeline (baseline or enhanced)
- 📊 **Leaderboard Target**: Top published result from main leaderboard.md
- ✅ **Beat Target**: Our result exceeds the published leaderboard
- 📈 **Below Target**: Still working to beat the published result
- ❌ **Not Tested**: No pipeline results yet

| Dataset | Our F1 | Method | vs Target | Leaderboard Target | Notes |
|---------|--------|--------|-----------|-------------------|-------|
"""

        # Include all datasets from mapping, not just ones with results
        all_datasets = set(self.dataset_mappings.keys()) | set(results.keys())
        sorted_datasets = sorted(all_datasets)

        for dataset in sorted_datasets:
            # Check if we have pipeline results for this dataset
            if dataset in results:
                pipeline_data = results[dataset]
                # Get our best result
                our_f1, method, result_details = self.get_best_result(pipeline_data)
            else:
                # No results yet - create empty entry
                our_f1, method, result_details = 0.0, "Not tested", {}

            # Get leaderboard target
            target_f1 = self.extract_leaderboard_target(dataset)

            # Determine if we beat the target
            if dataset not in results:
                vs_target = "❌ Not tested"
                status_icon = "❌"
            elif target_f1 > 0:
                if our_f1 > target_f1 / 100 if target_f1 > 10 else our_f1 > target_f1:
                    vs_target = "✅ Beat"
                    status_icon = "✅"
                else:
                    gap = (target_f1 / 100 if target_f1 > 10 else target_f1) - our_f1
                    vs_target = f"📈 -{gap:.3f}"
                    status_icon = "📈"
            else:
                vs_target = "❓ Unknown"
                status_icon = "❓"

            # Format our F1 score
            our_f1_display = f"**{our_f1:.3f}**" if status_icon == "✅" else f"{our_f1:.3f}"

            # Get additional details
            if dataset in results:
                precision = result_details.get("precision", 0.0)
                recall = result_details.get("recall", 0.0)
                result_details.get("cost_usd", 0.0)

                # Create method description
                if method == "enhanced":
                    method_desc = "Enhanced (rules)"
                    early_decisions = pipeline_data.get("enhanced_results", {}).get("early_decisions", 0)
                    llm_reduction = pipeline_data.get("enhanced_results", {}).get("llm_call_reduction", 0)
                    notes = f"P:{precision:.3f}, R:{recall:.3f}, Early:{early_decisions}, -LLM:{llm_reduction:.1f}%"
                elif method == "baseline":
                    method_desc = "Baseline (optimal)"
                    optimal_params = pipeline_data.get("optimal_params", {})
                    candidates = optimal_params.get("max_candidates", "?")
                    semantic_weight = optimal_params.get("semantic_weight", "?")
                    notes = f"P:{precision:.3f}, R:{recall:.3f}, {candidates}c, sw:{semantic_weight}"
                else:
                    method_desc = method
                    notes = f"P:{precision:.3f}, R:{recall:.3f}"
            else:
                # No results - create placeholder
                method_desc = "❌ Not tested"
                notes = "Run: python run_complete_pipeline.py --dataset " + dataset

            # Format target
            target_display = f"{target_f1:.1f}" if target_f1 > 0 else "Unknown"

            markdown += f"| {dataset} | {our_f1_display} | {method_desc} | {vs_target} | {target_display} | {notes} |\n"

        # Add summary section
        total_datasets = len(all_datasets)
        tested_datasets = len(results)
        beat_targets = sum(1 for dataset in results if self._beats_target(dataset, results[dataset]))

        markdown += f"""
## Summary

- **Total Datasets**: {total_datasets}
- **Tested**: {tested_datasets}/{total_datasets} datasets
- **Beat Leaderboard**: {beat_targets}/{tested_datasets} tested datasets
- **Success Rate**: {beat_targets / tested_datasets * 100:.1f}% (of tested)
- **Remaining**: {total_datasets - tested_datasets} datasets to test

"""

        # Add methodology section
        markdown += """## Methodology

Our pipeline:
1. **Hyperparameter Optimization**: Strategic sweep on dev/validation set to find optimal parameters
2. **Rule Generation**: Claude SDK generates domain-specific heuristic rules based on failure analysis
3. **A/B Testing**: Compare baseline (optimal params only) vs enhanced (optimal params + rules)
4. **Best Result**: Report whichever approach (baseline or enhanced) achieves higher F1

**Baseline Approach**: Hybrid trigram + semantic similarity with optimized hyperparameters
**Enhanced Approach**: Baseline + heuristic rules for early decisions, score adjustments, and weight tuning

Results show that sometimes optimal hyperparameters alone beat complex rule systems!
"""

        return markdown

    def _beats_target(self, dataset: str, pipeline_data: Dict) -> bool:
        """Check if our result beats the leaderboard target"""
        our_f1, _, _ = self.get_best_result(pipeline_data)
        target_f1 = self.extract_leaderboard_target(dataset)

        if target_f1 > 0:
            return our_f1 > target_f1 / 100 if target_f1 > 10 else our_f1 > target_f1
        return False

    def save_leaderboard(self, markdown: str, output_file: str = "internal_leaderboard.md"):
        """Save the internal leaderboard to a file"""
        with open(output_file, "w") as f:
            f.write(markdown)
        print(f"📊 Internal leaderboard saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate internal leaderboard from pipeline results")
    parser.add_argument(
        "--output", default="internal_leaderboard.md", help="Output file (default: internal_leaderboard.md)"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    generator = InternalLeaderboardGenerator()

    # Load all pipeline results
    print("🔍 Scanning results/ folder for pipeline results...")
    results = generator.load_pipeline_results()

    if not results:
        print("❌ No pipeline results found in results/ folder")
        print("   Run some pipelines first: python run_complete_pipeline.py --dataset <name>")
        return

    print(f"📊 Found results for {len(results)} datasets: {', '.join(results.keys())}")

    if args.verbose:
        for dataset, data in results.items():
            baseline_f1 = data.get("baseline_results", {}).get("f1", 0.0)
            enhanced_f1 = data.get("enhanced_results", {}).get("f1", 0.0)
            best_f1, method, _ = generator.get_best_result(data)
            print(f"  {dataset}: baseline={baseline_f1:.3f}, enhanced={enhanced_f1:.3f}, best={best_f1:.3f} ({method})")

    # Generate leaderboard
    print("📝 Generating internal leaderboard...")
    markdown = generator.generate_internal_leaderboard(results)

    # Save leaderboard
    generator.save_leaderboard(markdown, args.output)

    # Show summary
    beat_count = sum(1 for dataset in results if generator._beats_target(dataset, results[dataset]))
    print(f"🎯 Summary: {beat_count}/{len(results)} datasets beat their leaderboard targets")


if __name__ == "__main__":
    main()
