#!/usr/bin/env python3
"""
Agentic Claude Code SDK heuristic generation system.

Clean, agentic approach where Claude can:
1. Generate rules
2. Test them on sample data
3. See results
4. Iterate and improve
5. Validate final performance

NO file swapping, NO subprocess calls to claude binary.
Uses only claude_code_sdk Python imports.
"""

import asyncio
import json
import os
import pathlib

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)


def get_leaderboard_target_f1(dataset: str) -> float:
    """Get the top F1 score from leaderboard.md for the given dataset"""
    try:
        # Dataset name mappings (dataset -> leaderboard section name)
        dataset_mappings = {
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

        # Read the leaderboard file
        leaderboard_path = pathlib.Path("leaderboard.md")
        if not leaderboard_path.exists():
            print("⚠️ leaderboard.md not found, using default target of 85.0")
            return 85.0

        with open(leaderboard_path) as f:
            content = f.read()

        # Get the dataset section name
        section_name = dataset_mappings.get(dataset, dataset)

        # Find the dataset section and extract highest F1 score
        import re

        section_pattern = rf"### {re.escape(section_name)}(?:\s+\([^)]+\))?\s*\n"
        match = re.search(section_pattern, content, re.IGNORECASE)

        if not match:
            print(f"⚠️ Dataset {dataset} not found in leaderboard, using default target of 85.0")
            return 85.0

        # Extract the section content until the next ### or end of file
        start = match.end()
        next_section = re.search(r"\n### ", content[start:])
        section_content = content[start : start + next_section.start()] if next_section else content[start:]

        # Find the highest F1 score in bold (non-italicized)
        f1_scores = []
        for line in section_content.split("\n"):
            if "|" in line and any(char.isdigit() for char in line):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:  # model | F1 at minimum
                    f1_text = parts[-1]  # Last column is F1

                    # Skip italicized scores (jellyfish)
                    if f1_text.startswith("*") and f1_text.endswith("*"):
                        continue

                    # Extract number from **92.4** or 92.4
                    f1_match = re.search(r"(\d+\.?\d*)", f1_text)
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))

        if f1_scores:
            target_f1 = max(f1_scores)
            print(f"🎯 Leaderboard target for {dataset}: F1 = {target_f1:.1f}")
            return target_f1
        print(f"⚠️ No F1 scores found for {dataset}, using default target of 85.0")
        return 85.0

    except Exception as e:
        print(f"⚠️ Error parsing leaderboard for {dataset}: {e}, using default target of 85.0")
        return 85.0


@dataclass
class SampleData:
    """Clean sample data for Claude to analyze and test rules on"""

    dataset: str
    dev_pairs: List[Dict[str, Any]]  # Sampled dev pairs with left/right records
    dev_predictions: Dict[int, int]  # Model predictions on dev set
    dev_metrics: Dict[str, float]  # F1, precision, recall on dev set
    target_f1: float  # Leaderboard target
    table_a: Dict[int, Dict]  # ID -> record mapping
    table_b: Dict[int, Dict]  # ID -> record mapping
    candidate_analysis: Dict[int, Dict]  # Candidate generation analysis


class AgenticHeuristicGenerator:
    """Clean agentic heuristic generator using only claude_code_sdk"""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.data_root = pathlib.Path("data") / "raw" / dataset
        self.target_f1 = get_leaderboard_target_f1(dataset)

    def _load_clean_data(self) -> Tuple[Dict[int, Dict], Dict[int, Dict], pd.DataFrame]:
        """Load dataset cleanly without file manipulation"""
        # Load tables
        A_df = pd.read_csv(self.data_root / "tableA.csv")
        B_df = pd.read_csv(self.data_root / "tableB.csv")

        # Create ID-to-record mappings
        A = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
        B = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}

        # Load dev set (validation if available, otherwise train slice)
        if (self.data_root / "valid.csv").exists():
            dev_pairs = pd.read_csv(self.data_root / "valid.csv")
            print(f"✅ Using validation set: {len(dev_pairs)} pairs")
        elif (self.data_root / "train.csv").exists():
            train_pairs = pd.read_csv(self.data_root / "train.csv")
            dev_slice_size = min(100, len(train_pairs))
            dev_pairs = train_pairs.head(dev_slice_size)
            print(f"✅ Using train slice: {dev_slice_size} pairs")
        else:
            raise ValueError("No validation or training set available for clean rule generation")

        return A, B, dev_pairs

    def _analyze_candidate_generation(
        self, A: Dict[int, Dict], B: Dict[int, Dict], dev_pairs: pd.DataFrame, predictions: Dict[int, int]
    ) -> Dict[int, Dict]:
        """Analyze candidate generation for each left record to understand false negatives"""
        from ..entity_matching.hybrid_matcher import Config, get_top_candidates

        # Create a basic config for candidate generation (same as what was used in dev run)
        cfg = Config()
        cfg.use_semantic = True
        cfg.semantic_weight = 0.5  # Default from dev run
        cfg.use_heuristics = False  # No heuristics in original run

        # Load embeddings cache if available for this dataset
        try:
            from ..entity_matching.hybrid_matcher import compute_dataset_embeddings

            cfg.embeddings = compute_dataset_embeddings(self.dataset, cfg)
            print("✅ Loaded embeddings cache for candidate analysis")
        except Exception as e:
            print(f"⚠️ Could not load embeddings for candidate analysis: {e}")
            cfg.embeddings = None
            cfg.use_semantic = False  # Fall back to trigram only

        candidate_analysis = {}

        print("🔍 Analyzing candidate generation for false negatives...")

        # Focus on pairs where we have predictions (the model actually evaluated these)
        analyzed_pairs = 0
        for _, row in dev_pairs.iterrows():
            left_id = row.ltable_id
            right_id = row.rtable_id
            true_label = row.label

            # Only analyze pairs the model actually processed
            if left_id not in predictions:
                continue

            left_record = A[left_id]

            # Get the same candidates the model would have seen
            # Use a reasonable candidate count (similar to what was used in dev)
            max_candidates = 150  # Default from hybrid_matcher
            candidates = get_top_candidates(left_record, B, max_candidates, cfg, self.dataset)

            # Check if the ground truth made it into the candidates
            ground_truth_in_candidates = False
            ground_truth_rank = None
            candidate_ids = [candidate_id for candidate_id, _ in candidates]

            if true_label == 1:  # Only check for positive pairs
                if right_id in candidate_ids:
                    ground_truth_in_candidates = True
                    ground_truth_rank = candidate_ids.index(right_id) + 1  # 1-based rank

            predicted_right_id = predictions[left_id]
            predicted_match = predicted_right_id == right_id

            # Categorize the result
            if true_label == 1 and not predicted_match:
                category = "false_negative"
            elif true_label == 0 and predicted_match:
                category = "false_positive"
            elif true_label == 1 and predicted_match:
                category = "true_positive"
            else:
                category = "true_negative"

            candidate_analysis[left_id] = {
                "left_record": left_record,
                "true_right_id": right_id if true_label == 1 else None,
                "predicted_right_id": predicted_right_id,
                "category": category,
                "ground_truth_in_candidates": ground_truth_in_candidates,
                "ground_truth_rank": ground_truth_rank,
                "total_candidates": len(candidates),
                "candidate_ids": candidate_ids[:10],  # Top 10 for analysis
                "all_candidates": candidates,  # Full list for detailed analysis
            }

            analyzed_pairs += 1
            if analyzed_pairs % 20 == 0:
                print(f"   Analyzed {analyzed_pairs} pairs...")

        return candidate_analysis

    def _create_sample_data(self, dev_results: Dict[str, Any], analysis_data: Optional[Dict] = None) -> SampleData:
        """Create clean sample data for Claude to work with"""
        A, B, dev_pairs = self._load_clean_data()

        # Extract predictions and metrics - NEVER use mock data
        predictions = dev_results.get("predictions", {})
        metrics = dev_results.get("metrics", {})

        if not predictions:
            raise ValueError(
                f"No predictions found in dev_results. Cannot generate rules without real model predictions. Got keys: {list(dev_results.keys())}"
            )

        print(f"📊 Real model predictions: {len(predictions)} predictions")

        # Run candidate generation analysis to understand what the model "saw"
        candidate_analysis = self._analyze_candidate_generation(A, B, dev_pairs, predictions)
        print(f"📊 Candidate analysis: {len(candidate_analysis)} pairs analyzed")

        # Create enriched sample pairs with actual records and candidate analysis
        # Prioritize errors for better rule generation signal
        sample_pairs = []
        error_pairs = []
        success_pairs = []
        candidate_issues = []  # Track when ground truth wasn't in candidates

        for left_id, analysis in candidate_analysis.items():
            category = analysis["category"]

            # Get ground truth info
            true_right_id = analysis["true_right_id"]
            predicted_right_id = analysis["predicted_right_id"]

            pair_data = {
                "left_id": left_id,
                "right_id": true_right_id,
                "left_record": analysis["left_record"],
                "right_record": B.get(true_right_id) if true_right_id else None,
                "predicted_right_id": predicted_right_id,
                "predicted_right_record": B.get(predicted_right_id) if predicted_right_id else None,
                "category": category,
                "ground_truth_in_candidates": analysis["ground_truth_in_candidates"],
                "ground_truth_rank": analysis["ground_truth_rank"],
                "candidate_ids": analysis["candidate_ids"],
                "total_candidates": analysis["total_candidates"],
            }

            # Track candidate generation issues
            if category == "false_negative" and not analysis["ground_truth_in_candidates"]:
                candidate_issues.append(pair_data)

            # Separate errors from successes for prioritization
            if category in ["false_positive", "false_negative"]:
                error_pairs.append(pair_data)
            else:
                success_pairs.append(pair_data)

        # Prioritize errors - include ALL errors, then fill with successes
        sample_pairs = error_pairs.copy()  # Include all errors

        # Add some successful examples for context, but prioritize errors
        remaining_slots = max(0, 50 - len(error_pairs))  # Target ~50 total examples
        sample_pairs.extend(success_pairs[:remaining_slots])

        print(
            f"📊 Sample composition: {len(error_pairs)} errors (high signal) + {len(success_pairs[:remaining_slots])} successes"
        )
        print(f"📊 Candidate issues: {len(candidate_issues)} false negatives where ground truth wasn't in candidates")

        sample_data = SampleData(
            dataset=self.dataset,
            dev_pairs=sample_pairs,
            dev_predictions=predictions,
            dev_metrics=metrics,
            target_f1=self.target_f1,
            table_a=A,
            table_b=B,
            candidate_analysis=candidate_analysis,
        )

        # Add rich analysis data if provided
        if analysis_data:
            sample_data.analysis_insights = analysis_data

        return sample_data

    def _create_agentic_prompt(self, sample_data: SampleData) -> str:
        """Create agentic prompt that lets Claude test and iterate on rules"""

        # Analyze sample data for patterns
        error_examples = [p for p in sample_data.dev_pairs if p["category"] in ["false_positive", "false_negative"]]
        [p for p in sample_data.dev_pairs if p["category"] in ["true_positive", "true_negative"]]

        # Calculate candidate generation statistics
        fn_examples = [p for p in error_examples if p["category"] == "false_negative"]
        fn_not_in_candidates = len([p for p in fn_examples if not p.get("ground_truth_in_candidates", True)])
        total_fn = len(fn_examples)

        # Add rich analysis insights if available
        analysis_section = ""
        if hasattr(sample_data, 'analysis_insights') and sample_data.analysis_insights:
            insights = sample_data.analysis_insights
            similarity_stats = insights.get("similarity_analysis", {})
            candidate_stats = insights.get("candidate_analysis", {})
            true_matches = similarity_stats.get("true_matches", {})
            false_positives = similarity_stats.get("false_positives", {})

            analysis_section = f"""

**🔬 RICH SIMILARITY ANALYSIS** (from {insights.get("metadata", {}).get("total_pairs_analyzed", 0)} validation pairs):

**True Match Patterns**:
- Syntactic similarity: {true_matches.get("syntactic", {}).get("mean", 0):.3f} ± {true_matches.get("syntactic", {}).get("std", 0):.3f} (range: {true_matches.get("syntactic", {}).get("min", 0):.3f}-{true_matches.get("syntactic", {}).get("max", 0):.3f})
- Trigram similarity: {true_matches.get("trigram", {}).get("mean", 0):.3f} ± {true_matches.get("trigram", {}).get("std", 0):.3f} (range: {true_matches.get("trigram", {}).get("min", 0):.3f}-{true_matches.get("trigram", {}).get("max", 0):.3f})
{"- Semantic similarity: " + f"{true_matches.get('semantic', {}).get('mean', 0):.3f} ± {true_matches.get('semantic', {}).get('std', 0):.3f} (range: {true_matches.get('semantic', {}).get('min', 0):.3f}-{true_matches.get('semantic', {}).get('max', 0):.3f})" if true_matches.get("semantic") else "- Semantic similarity: Not available"}

**False Positive Patterns** (what the model mistakes for matches):
- Syntactic similarity: {false_positives.get("syntactic", {}).get("mean", 0):.3f} ± {false_positives.get("syntactic", {}).get("std", 0):.3f} (range: {false_positives.get("syntactic", {}).get("min", 0):.3f}-{false_positives.get("syntactic", {}).get("max", 0):.3f})
- Trigram similarity: {false_positives.get("trigram", {}).get("mean", 0):.3f} ± {false_positives.get("trigram", {}).get("std", 0):.3f} (range: {false_positives.get("trigram", {}).get("min", 0):.3f}-{false_positives.get("trigram", {}).get("max", 0):.3f})
{"- Semantic similarity: " + f"{false_positives.get('semantic', {}).get('mean', 0):.3f} ± {false_positives.get('semantic', {}).get('std', 0):.3f} (range: {false_positives.get('semantic', {}).get('min', 0):.3f}-{false_positives.get('semantic', {}).get('max', 0):.3f})" if false_positives.get("semantic") else "- Semantic similarity: Not available"}

**🎯 OPTIMAL THRESHOLDS** (based on data analysis):
- Syntactic threshold: ~{max(0.6, (true_matches.get("syntactic", {}).get("mean", 0.7) + false_positives.get("syntactic", {}).get("mean", 0.4)) / 2):.3f} (separates true matches from false positives)
- Trigram threshold: ~{max(0.3, (true_matches.get("trigram", {}).get("mean", 0.6) + false_positives.get("trigram", {}).get("mean", 0.2)) / 2):.3f}
{"- Semantic threshold: ~" + f"{max(0.7, (true_matches.get('semantic', {}).get('mean', 0.8) + false_positives.get('semantic', {}).get('mean', 0.6)) / 2):.3f}" if true_matches.get("semantic") and false_positives.get("semantic") else ""}

**📊 CANDIDATE GENERATION RECALL**:
{chr(10).join([f"- Recall@{k.split('_')[-1]}: {v:.1%} ({int(v * insights.get('metadata', {}).get('positive_pairs', 27))}/{insights.get('metadata', {}).get('positive_pairs', 27)} matches found)" for k, v in candidate_stats.items() if k.startswith("recall_at")]) if candidate_stats else "- No candidate recall data available"}

**🚨 KEY INSIGHTS FOR RULE GENERATION**:
{"- ⚡ HIGH PRECISION OPPORTUNITY: True matches have much higher " + ("semantic" if true_matches.get("semantic", {}).get("mean", 0) - false_positives.get("semantic", {}).get("mean", 0) > 0.1 else "syntactic" if true_matches.get("syntactic", {}).get("mean", 0) - false_positives.get("syntactic", {}).get("mean", 0) > 0.2 else "trigram") + " similarity - use this for confident auto-accepts" if (true_matches.get("semantic", {}).get("mean", 0) - false_positives.get("semantic", {}).get("mean", 0) > 0.1) or (true_matches.get("syntactic", {}).get("mean", 0) - false_positives.get("syntactic", {}).get("mean", 0) > 0.2) else "- ⚠️ OVERLAPPING SIMILARITIES: True matches and false positives have similar scores - need nuanced rules"}
- 🔍 RECALL OPPORTUNITY: {f"Only {candidate_stats.get('recall_at_100', 0):.1%} of true matches are in top 100 candidates" if candidate_stats.get('recall_at_100', 1) < 0.98 else f"Good candidate recall ({candidate_stats.get('recall_at_100', 1):.1%}) - focus on LLM decision quality"}
{f"- 🎯 PERFECT MATCHES: {(true_matches.get('syntactic', {}).get('max', 0) > 0.95) + (true_matches.get('trigram', {}).get('max', 0) > 0.95) + (true_matches.get('semantic', {}).get('max', 0) > 0.95 if true_matches.get('semantic') else 0)} similarity types reach near-perfect scores - auto-accept these" if any([true_matches.get('syntactic', {}).get('max', 0) > 0.95, true_matches.get('trigram', {}).get('max', 0) > 0.95, true_matches.get('semantic', {}).get('max', 0) > 0.95 if true_matches.get('semantic') else False]) else ""}
{"- 📈 CONCRETE EXAMPLES: " + f"{len(insights.get('concrete_examples', {}).get('true_matches', []))} true match examples and {len(insights.get('concrete_examples', {}).get('confusing_non_matches', []))} confusing non-match examples available for pattern analysis" if insights.get('concrete_examples') else ""}"""

        # Convert SampleData to dictionary for JSON serialization
        from ..entity_matching.analysis import clean_record_for_json

        def clean_dict_for_json(obj):
            """Recursively clean dictionary for JSON serialization"""
            import numpy as np
            
            if isinstance(obj, dict):
                return {k: clean_dict_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_dict_for_json(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                # Use clean_record_for_json for complex objects
                return clean_record_for_json({"value": obj})["value"]

        sample_data_dict = clean_dict_for_json({
            "dataset": sample_data.dataset,
            "dev_pairs": sample_data.dev_pairs,
            "dev_predictions": sample_data.dev_predictions,
            "dev_metrics": sample_data.dev_metrics,
            "target_f1": sample_data.target_f1,
            "table_a": sample_data.table_a,
            "table_b": sample_data.table_b,
            "candidate_analysis": sample_data.candidate_analysis,
        })

        prompt = f"""You are an expert at entity matching optimization. You can iteratively develop and test BOTH hyperparameters AND rules for optimal performance.

**DATASET**: {sample_data.dataset}
**TARGET**: F1 > {sample_data.target_f1:.1f} (leaderboard target)
**CURRENT DEV PERFORMANCE**: F1={sample_data.dev_metrics.get("f1", 0):.4f}, P={sample_data.dev_metrics.get("precision", 0):.4f}, R={sample_data.dev_metrics.get("recall", 0):.4f}
**DETAILS**: {json.dumps(sample_data_dict, indent=2)}

**YOUR TOOLS**:
- `Read`: Read files (e.g., sample data, existing rules)
- `Write`: Write configuration files with both hyperparameters and rules
- `Bash`: Test configuration by running matching with your generated config

**TASK**: Generate optimal entity matching configuration including:
1. **HYPERPARAMETERS** (max_candidates, similarity weights, thresholds)
2. **CANDIDATE GENERATION RULES** (ensure correct matches get into the candidate list)
3. **DECISION RULES** (auto-accept/reject to reduce LLM costs)
4. **SCORE ADJUSTMENT RULES** (boost likely matches, penalize unlikely ones)
5. **DYNAMIC WEIGHT RULES** (adjust semantic vs trigram weights based on context)

**RULE TYPES**:
```python
# 1. CANDIDATE RULES - Boost similarity during candidate selection to surface missed matches
CandidateAction(similarity_boost=0.4, confidence=0.8, reason="Boost partial name matches in candidate generation")

# 2. DECISION RULES - Early termination to skip LLM
DecisionAction(terminate_early=True, final_result=1, confidence=0.95, reason="Exact match")
DecisionAction(terminate_early=True, final_result=0, skip_llm=True, reason="Very low similarity")

# 3. SCORE RULES - Adjust similarity scores
ScoreAction(score_adjustment=0.3, confidence=0.8, reason="Strong field match")

# 4. WEIGHT RULES - Dynamic weight adjustment
WeightAction(semantic_weight=0.9, confidence=0.7, reason="Text-heavy comparison")
```

**PIPELINE STAGES**:
- `candidate_generation`: During candidate selection (boost similarity to surface missed matches)
- `pre_llm`: Best for early decisions (auto-accept/reject)
- `post_semantic`: After semantic similarity, good for score adjustments
- `pre_semantic`: Before semantic calculation, good for weight adjustments

**PERFORMANCE ANALYSIS**:
Current metrics: F1={sample_data.dev_metrics.get("f1", 0):.4f}, P={sample_data.dev_metrics.get("precision", 0):.4f}, R={sample_data.dev_metrics.get("recall", 0):.4f}
Model made {len(error_examples)} errors ({len([p for p in error_examples if p["category"] == "false_positive"])} false positives, {len([p for p in error_examples if p["category"] == "false_negative"])} false negatives)

⚠️ **CANDIDATE GENERATION ANALYSIS**:
- False negatives: {total_fn} total
- FN where ground truth NOT in candidates: {fn_not_in_candidates}/{total_fn} ({fn_not_in_candidates / total_fn * 100 if total_fn > 0 else 0:.1f}%)
{f"- 🚨 MAJOR ISSUE: {fn_not_in_candidates / total_fn * 100:.1f}% of false negatives are due to candidate generation failures!" if fn_not_in_candidates > 0 and total_fn > 0 and fn_not_in_candidates / total_fn >= 0.3 else "- ✅ Most false negatives have ground truth in candidates (LLM decision issue)" if fn_not_in_candidates == 0 else "- ⚠️ Some candidate generation issues detected"}

⚠️ **STRATEGY ANALYSIS**:
- Precision = {sample_data.dev_metrics.get("precision", 0):.4f}: {"Perfect! Model never wrong when it says match." if sample_data.dev_metrics.get("precision", 0) >= 0.99 else "Has false positives - model too aggressive"}
- Recall = {sample_data.dev_metrics.get("recall", 0):.4f}: {"Low - model missing true matches (too conservative)" if sample_data.dev_metrics.get("recall", 0) < 0.9 else "Good recall"}

**RECOMMENDED OPTIMIZATION STRATEGY**:
{"🎯 **BOOST RECALL**: Increase max_candidates and semantic_weight. Generate rules to catch missed matches (score boosts, relaxed thresholds)" if sample_data.dev_metrics.get("precision", 0) >= 0.99 and sample_data.dev_metrics.get("recall", 0) < 0.9 else "🎯 **IMPROVE PRECISION**: Lower decision threshold, add rejection rules to reduce false positives" if sample_data.dev_metrics.get("precision", 0) < 0.9 else "🎯 **BALANCED APPROACH**: Fine-tune hyperparameters and rules for optimal F1"}

⚠️ **HIGH SIGNAL DATA**: The sample file contains ALL false positives and false negatives (not just a subset). This gives you comprehensive error patterns to analyze.

{analysis_section}

**FALSE POSITIVE EXAMPLES** (incorrectly predicted as matches):
"""

        # Add false positive examples
        fp_examples = [p for p in error_examples if p["category"] == "false_positive"][:3]
        for i, example in enumerate(fp_examples):
            prompt += f"""
Example {i + 1}:
Left:  {json.dumps(example["left_record"], indent=2)}
Right: {json.dumps(example["right_record"], indent=2)}
"""

        prompt += """
**FALSE NEGATIVE EXAMPLES** (missed real matches):
"""

        # Add false negative examples with candidate analysis
        fn_examples = [p for p in error_examples if p["category"] == "false_negative"][:3]
        for i, example in enumerate(fn_examples):
            in_candidates = example.get("ground_truth_in_candidates", "Unknown")
            rank = example.get("ground_truth_rank", "N/A")
            total_candidates = example.get("total_candidates", "Unknown")

            prompt += f"""
Example {i + 1}:
Left:  {json.dumps(example["left_record"], indent=2)}
Right: {json.dumps(example["right_record"], indent=2)}
⚠️ Ground truth in top {total_candidates} candidates: {in_candidates}
⚠️ Ground truth rank: {rank if rank else "Not in candidates"}
🔍 Top candidate IDs: {example.get("candidate_ids", [])}
"""

            # If ground truth wasn't in candidates, this is a candidate generation issue
            if not in_candidates:
                prompt += f"""
❌ CANDIDATE GENERATION ISSUE: The correct match wasn't even in the top {total_candidates} candidates!
   This suggests we need CANDIDATE GENERATION rules to surface the correct matches.
"""

        prompt += f"""

**SPECIFIC GUIDANCE FOR THIS DATASET**:
{f"Since precision is perfect ({sample_data.dev_metrics.get('precision', 0):.4f}) but recall is low ({sample_data.dev_metrics.get('recall', 0):.4f}), focus on FALSE NEGATIVES. Generate rules to BOOST scores for missed matches. Avoid rejection rules since precision is already perfect." if sample_data.dev_metrics.get("precision", 0) >= 0.99 and sample_data.dev_metrics.get("recall", 0) < 0.9 else "Focus on both precision and recall optimization."}

**ITERATIVE WORKFLOW**:
1. **Write initial config** to `results/temp/test_config.json` with both hyperparameters and rules
2. **Test them** with: `python run_enhanced_matching.py --dataset {sample_data.dataset} --heuristic-file results/temp/test_config.json --limit 20`
3. **Analyze results** - did F1 improve? Which hyperparameters/rules helped/hurt?
4. **Iterate** - refine both hyperparameters and rules, test again
5. **Final config** - when satisfied, write final configuration to `results/temp/generated_rules.json`

**COMPLEX CANDIDATE GENERATION HEURISTICS**:
You can write sophisticated candidate generation rules that:
- Boost similarity calculations during candidate search to surface missed matches
- Use field-specific logic (e.g., prioritize name matches, handle abbreviations)
- Apply domain knowledge (e.g., year ranges, category mappings)
- Handle data quality issues (missing fields, typos, formatting differences)

**EXAMPLE COMPLEX CANDIDATE RULES**:
Examples of sophisticated candidate generation rules:

1. **Partial Name Matching**: Boost similarity for partial name matches
2. **Abbreviation Handling**: Detect and boost acronym matches
3. **Year Range Matching**: Handle temporal data with year ranges
4. **Field-Specific Logic**: Custom logic for different data types

These rules execute during candidate generation to surface missed matches.

**IMPORTANT**: All rule files MUST be saved in the `results/temp/` directory. Do NOT save files in the root directory.

**CONFIGURATION FORMAT** ({"Focus on CANDIDATE GENERATION and SCORE BOOSTS since precision is perfect. INCREASE max_candidates and semantic_weight to improve recall." if sample_data.dev_metrics.get("precision", 0) >= 0.99 and sample_data.dev_metrics.get("recall", 0) < 0.9 else "Balance hyperparameters and rules for optimal F1"}):
```json
{{
  "hyperparameters": {{
    "max_candidates": 150,
    "semantic_weight": 0.7,
    "syntactic_weight": 0.2,
    "trigram_weight": 0.1,
    "decision_threshold": 0.5,
    "auto_accept_threshold": 0.95,
    "auto_reject_threshold": 0.1
  }},
  "candidate_rules": [
    {{
      "rule_name": "boost_partial_name_candidates",
      "description": "Boost similarity for partial name matches during candidate generation",
      "implementation": "def boost_partial_name_candidates(left_record, right_record):\\n    left_name = normalize(left_record.get('name', ''))\\n    right_name = normalize(right_record.get('name', ''))\\n    if left_name and right_name and (left_name in right_name or right_name in left_name):\\n        return CandidateAction(similarity_boost=0.4, confidence=0.8, reason='Partial name match - surface in candidates')\\n    return None",
      "confidence": 0.8,
      "stage": "candidate_generation"
    }}
  ],
  "score_rules": [
    {{
      "rule_name": "partial_name_boost",
      "description": "Boost score for partial name matches to catch missed pairs",
      "implementation": "def partial_name_boost(left_record, right_record):\\n    left_name = normalize(left_record.get('name', ''))\\n    right_name = normalize(right_record.get('name', ''))\\n    if left_name and right_name and (left_name in right_name or right_name in left_name):\\n        return ScoreAction(score_adjustment=0.3, confidence=0.8, reason='Partial name match - boost to catch missed pairs')\\n    return None",
      "confidence": 0.8,
      "stage": "post_semantic"
    }}
  ],
  "decision_rules": [
    {{
      "rule_name": "high_similarity_auto_accept",
      "description": "Auto-accept very high similarity to catch conservative misses",
      "implementation": "def high_similarity_auto_accept(left_record, right_record, current_score):\\n    if current_score > 0.9:\\n        return DecisionAction(terminate_early=True, final_result=1, confidence=0.95, reason='Very high similarity - likely missed match')\\n    return None",
      "confidence": 0.95,
      "stage": "pre_llm"
    }}
  ],
  "weight_rules": [
    {{
      "rule_name": "boost_semantic_weight",
      "description": "Increase semantic weight to be more aggressive in matching",
      "implementation": "def boost_semantic_weight(left_record, right_record):\\n    return WeightAction(semantic_weight=0.8, confidence=0.7, reason='Boost semantic matching to catch more pairs')",
      "confidence": 0.7,
      "stage": "pre_semantic"
    }}
  ],
  "pipeline_rules": []
}}
```

**START HERE**: Analyze the error patterns and similarity statistics above, then write your first configuration with optimal hyperparameters and rules to `results/temp/test_config.json` and test them!
"""

        return prompt

    def _write_sample_data_file(self, sample_data: SampleData) -> str:
        """Write sample data to a file Claude can read"""
        os.makedirs("results/temp", exist_ok=True)
        sample_file = f"results/temp/sample_data_{sample_data.dataset}.json"

        def clean_for_json(obj):
            """Convert pandas types to JSON-serializable types"""
            if hasattr(obj, "item"):  # numpy/pandas scalar
                return obj.item()
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            return obj

        # Separate by category for better analysis
        false_positives = [p for p in sample_data.dev_pairs if p["category"] == "false_positive"]
        false_negatives = [p for p in sample_data.dev_pairs if p["category"] == "false_negative"]
        true_positives = [p for p in sample_data.dev_pairs if p["category"] == "true_positive"]
        true_negatives = [p for p in sample_data.dev_pairs if p["category"] == "true_negative"]

        # Calculate candidate generation statistics
        fn_not_in_candidates = [p for p in false_negatives if not p.get("ground_truth_in_candidates", True)]

        # Prepare clean sample data with error prioritization
        clean_data = {
            "dataset": sample_data.dataset,
            "target_f1": sample_data.target_f1,
            "current_metrics": sample_data.dev_metrics,
            "error_summary": {
                "total_false_positives": len(false_positives),
                "total_false_negatives": len(false_negatives),
                "total_true_positives": len(true_positives),
                "total_true_negatives": len(true_negatives),
                "error_rate": len(false_positives + false_negatives) / len(sample_data.dev_pairs)
                if sample_data.dev_pairs
                else 0,
            },
            "candidate_generation_analysis": {
                "false_negatives_not_in_candidates": len(fn_not_in_candidates),
                "total_false_negatives": len(false_negatives),
                "candidate_failure_rate": len(fn_not_in_candidates) / len(false_negatives) if false_negatives else 0,
                "needs_candidate_rules": len(fn_not_in_candidates) > 0,
            },
            "detailed_errors": {
                "false_positives": false_positives,  # ALL false positives
                "false_negatives": false_negatives,  # ALL false negatives
            },
            "sample_successes": {
                "true_positives": true_positives[:5],  # Sample of successes for context
                "true_negatives": true_negatives[:5],
            },
        }

        # Clean all pandas types for JSON serialization
        clean_data = clean_for_json(clean_data)

        with open(sample_file, "w") as f:
            json.dump(clean_data, f, indent=2)

        print(f"📊 Sample data written to {sample_file}")
        return sample_file

    async def _call_claude_agentic(self, prompt: str, sample_file: str) -> Tuple[str, Dict[str, Any]]:
        """Call Claude Code SDK in agentic mode with file access"""
        try:
            print("🤖 Starting agentic Claude session...")
            print(f"📊 Sample data: {sample_file}")
            print("🔄 Claude will now read data, generate rules, test them, and iterate...")

            # Add sample file info to prompt
            enhanced_prompt = f"""{prompt}

**SAMPLE DATA FILE**: `{sample_file}` contains detailed error analysis and examples.
You can read this file to see all the false positive/negative examples.

**TESTING COMMANDS**:
```bash
# Quick test (20 pairs for fast feedback)
python run_enhanced_matching.py --dataset {self.dataset} --heuristic-file results/temp/test_rules.json --limit 20 --max-candidates 150 --semantic-weight 0.75 --concurrency 10

# FULL DEV EVALUATION (use this regularly to track progress - FAST with concurrency!)
python run_enhanced_matching.py --dataset {self.dataset} --heuristic-file results/temp/test_rules.json --max-candidates 150 --semantic-weight 0.75 --concurrency 20
```

**CRITICAL**: After each rule iteration, run the FULL dev evaluation! It should take < 30 seconds with proper concurrency.
Your target is F1 > 82.0. Always show: "Current F1: X.XXX, Target: 82.0" after testing.
Use `run_enhanced_matching.py` - this is the SAME function used in the main pipeline!

**CRITICAL FILE PATHS**:
- Save initial rules to: `results/temp/test_rules.json`
- Save final rules to: `results/temp/generated_rules.json`
- Do NOT create files in root directory!

Start by reading the sample data file to understand the patterns, then iteratively develop and test rules!
"""

            options = ClaudeCodeOptions(
                allowed_tools=["Read", "Write", "Bash"],
                permission_mode="acceptEdits",
                cwd=os.getcwd(),
            )

            response_parts = []
            turn_count = 0
            total_cost_usd = 0.0
            session_id = None
            duration_ms = 0

            print("🎭 Claude is working on rule generation...")

            # Log the prompt to a file for debugging
            os.makedirs("results/temp", exist_ok=True)
            prompt_log_file = f"results/temp/{self.dataset}_claude_prompt.txt"
            with open(prompt_log_file, "w") as f:
                f.write(enhanced_prompt)
            print(f"📝 Claude prompt logged to: {prompt_log_file}")

            try:
                async for message in query(prompt=enhanced_prompt, options=options):
                    if isinstance(message, AssistantMessage):
                        turn_count += 1
                        print(f"   💭 Turn {turn_count}: Claude is thinking and taking actions...")

                        for block in message.content:
                            if isinstance(block, TextBlock):
                                response_parts.append(block.text)
                                # Show much more of what Claude is thinking for better visibility
                                if len(block.text.strip()) > 50:
                                    lines = block.text.strip().split('\n')
                                    # Show first few lines or up to 1000+ characters
                                    preview_lines = []
                                    char_count = 0
                                    for line in lines:
                                        if char_count + len(line) > 1500:  # Show up to 1500 chars
                                            break
                                        preview_lines.append(line)
                                        char_count += len(line)

                                    preview = '\n'.join(preview_lines).strip()  # Keep line breaks for readability
                                    if len(preview) > 1500:
                                        preview = preview[:1500] + "..."
                                    print(f"      🔤 {preview}")

                    elif isinstance(message, ResultMessage):
                        # Capture cost and session info
                        total_cost_usd = message.total_cost_usd or 0.0
                        session_id = message.session_id
                        duration_ms = message.duration_ms or 0

                        print("✅ Claude session completed!")
                        print(f"   📊 {turn_count} turns, {duration_ms / 1000:.1f}s, ${total_cost_usd:.4f}")

                        if message.is_error:
                            print(f"   ⚠️ Session ended with error: {message.result}")

                        # end of run
                        break

            except Exception as claude_error:
                print(f"❌ Claude Code SDK error: {claude_error}")
                print("   This might be due to async conflicts or Claude SDK issues")
                print("   Trying to continue with best effort...")

                # Return empty response but don't crash the whole pipeline
                return "", {"total_cost_usd": 0.0, "method": "failed", "error": str(claude_error)}

            full_response = "".join(response_parts)

            cost_info = {
                "total_cost_usd": total_cost_usd,
                "session_id": session_id,
                "duration_ms": duration_ms,
                "turn_count": turn_count,
                "response_length": len(full_response),
            }

            return full_response, cost_info

        except Exception as e:
            raise RuntimeError(f"Claude SDK agentic call failed: {e}")

    async def generate_agentic_rules(
        self, dev_results: Dict[str, Any], output_file: Optional[str] = None, analysis_data: Optional[Dict] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate rules using agentic Claude approach"""
        print(f"🚀 AGENTIC HEURISTIC GENERATION FOR {self.dataset}")
        print("=" * 60)

        # Prepare clean sample data with rich analysis insights
        sample_data = self._create_sample_data(dev_results, analysis_data)
        sample_file = self._write_sample_data_file(sample_data)

        # Create agentic prompt
        prompt = self._create_agentic_prompt(sample_data)

        try:
            # Run agentic Claude session
            response, cost_info = await self._call_claude_agentic(prompt, sample_file)

            # Clean up any files created in wrong location and move them
            wrong_location_files = ["generated_rules.json", "test_rules.json", "final_rules.json"]
            for wrong_file in wrong_location_files:
                if os.path.exists(wrong_file):
                    correct_file = f"results/temp/{wrong_file}"
                    print(f"🔧 Moving {wrong_file} to correct location: {correct_file}")
                    os.makedirs("results/temp", exist_ok=True)
                    import shutil

                    shutil.move(wrong_file, correct_file)

            # Look for generated rules file (check temp directory first, then root)
            possible_files = [
                "results/temp/generated_rules.json",
                "results/temp/test_rules.json",
                "results/temp/final_rules.json",
                "generated_rules.json",  # Legacy locations (shouldn't exist after cleanup)
                "test_rules.json",
                f"{self.dataset}_generated_heuristics.json",
                "final_rules.json",
            ]

            generated_file = None
            for filename in possible_files:
                if os.path.exists(filename):
                    generated_file = filename
                    print(f"✅ Found generated rules: {filename}")
                    break

            if not generated_file:
                print("⚠️ No rules file found, Claude may have encountered issues")
                print(f"Response preview: {response[:500]}...")
                return None, cost_info

            # Copy to final output location if specified
            if output_file and generated_file != output_file:
                with open(generated_file) as src:
                    rules_data = json.load(src)

                # Add metadata including cost information
                rules_data["timestamp"] = datetime.now().isoformat()
                rules_data["dataset"] = self.dataset
                rules_data["generation_method"] = "agentic_claude_sdk"
                rules_data["target_f1"] = self.target_f1
                rules_data["generation_cost_info"] = cost_info

                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "w") as dst:
                    json.dump(rules_data, dst, indent=2)

                print(f"✅ Final rules saved to: {output_file}")
                return output_file, cost_info
            return generated_file, cost_info

        except Exception as e:
            print(f"❌ Agentic rule generation failed: {e}")
            raise
        finally:
            # Clean up sample file
            if os.path.exists(sample_file):
                os.unlink(sample_file)

            # Final cleanup: remove any stray files in root directory
            stray_files = ["generated_rules.json", "test_rules.json", "final_rules.json"]
            for stray_file in stray_files:
                if os.path.exists(stray_file):
                    print(f"🧹 Cleaning up stray file: {stray_file}")
                    os.unlink(stray_file)


async def generate_agentic_heuristics(
    dataset: str, dev_results: Dict[str, Any], output_file: Optional[str] = None, analysis_data: Optional[Dict] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Clean interface for agentic heuristic generation.

    Args:
        dataset: Dataset name
        dev_results: Development set results with predictions and metrics
        output_file: Optional output file path

    Returns:
        Tuple of (path to generated heuristics file, cost information)
    """
    generator = AgenticHeuristicGenerator(dataset)
    return await generator.generate_agentic_rules(dev_results, output_file, analysis_data)


if __name__ == "__main__":
    # Test CLI
    import argparse

    parser = argparse.ArgumentParser(description="Generate agentic heuristics using Claude Code SDK")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--dev-results", required=True, help="Path to dev results JSON file")
    parser.add_argument("--output", help="Output file for generated heuristics")

    args = parser.parse_args()

    # Load dev results
    with open(args.dev_results) as f:
        dev_results = json.load(f)

    async def main():
        output_file = await generate_agentic_heuristics(args.dataset, dev_results, args.output)
        print(f"🎉 Agentic heuristics generated: {output_file}")

    asyncio.run(main())
