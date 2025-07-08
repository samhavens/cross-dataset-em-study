#!/usr/bin/env python
"""
General-purpose Claude Code SDK heuristic generation system.

This script analyzes failure patterns between validation and test sets,
then calls Claude Code SDK to generate domain-specific matching rules.
"""

import asyncio
import json
import os
import pathlib
import re

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

from ..entity_matching.hybrid_matcher import run_matching
from .claude_sdk_optimizer import ClaudeSDKOptimizer


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
            "walmart_amazon": "walmart\\_amazon",  # All underscores escaped in markdown
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

        # Find the dataset section
        # Look for pattern like "### walmart_amazon (waam)" or "### beer"
        section_pattern = rf"### {re.escape(section_name)}(?:\s+\([^)]+\))?\s*\n"
        match = re.search(section_pattern, content, re.IGNORECASE)

        if not match:
            print(f"⚠️ Dataset {dataset} not found in leaderboard, using default target of 85.0")
            return 85.0

        # Extract the section content until the next ### or end of file
        start = match.end()
        next_section = re.search(r"\n### ", content[start:])
        section_content = content[start : start + next_section.start()] if next_section else content[start:]

        # Find the highest F1 score in bold (non-italicized, i.e., not *jellyfish*)
        # Look for patterns like "| **92.4** |" but not "*97.7*"
        f1_scores = []

        # Extract all F1 scores from the table
        for line in section_content.split("\n"):
            if "|" in line and any(char.isdigit() for char in line):
                # Extract F1 scores (last column typically)
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
class FailurePattern:
    """Represents a failure pattern with examples"""

    pattern_type: str  # e.g., "validation_success_test_failure"
    count: int
    examples: List[Dict[str, Any]]
    description: str


@dataclass
class HeuristicRule:
    """Generated heuristic rule"""

    rule_name: str
    description: str
    implementation: str  # Python code
    confidence: float  # 0.0-1.0
    test_cases: List[Dict[str, Any]]


class ClaudeSDKHeuristicGenerator:
    """General-purpose heuristic generation using Claude Code SDK"""

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.claude_optimizer = ClaudeSDKOptimizer()
        self.data_root = pathlib.Path("data") / "raw" / dataset

    def load_dataset(self) -> Tuple[Dict[int, Dict], Dict[int, Dict], pd.DataFrame, pd.DataFrame]:
        """Load dataset tables and test pairs with proper ID mapping"""
        A_df = pd.read_csv(self.data_root / "tableA.csv")
        B_df = pd.read_csv(self.data_root / "tableB.csv")

        # Create ID-to-record mappings (not array indices)
        A = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
        B = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}

        # Load dev set (validation if available, otherwise train slice)
        if (self.data_root / "valid.csv").exists():
            validation_pairs = pd.read_csv(self.data_root / "valid.csv")
        elif (self.data_root / "train.csv").exists():
            # Use the same slice logic as in run_comprehensive_analysis
            train_pairs = pd.read_csv(self.data_root / "train.csv")
            dev_slice_size = min(100, len(train_pairs))
            validation_pairs = train_pairs.head(dev_slice_size)
        else:
            validation_pairs = None

        test_pairs = pd.read_csv(self.data_root / "test.csv")

        print(f"📊 Dataset loaded: {len(A)} records in tableA, {len(B)} records in tableB")
        print(f"📊 ID ranges: tableA {min(A.keys())}-{max(A.keys())}, tableB {min(B.keys())}-{max(B.keys())}")

        return A, B, validation_pairs, test_pairs

    async def run_comprehensive_analysis(self, config: Dict[str, Any], concurrency: int = 3) -> Dict[str, Any]:
        """Run comprehensive analysis on ONLY validation/dev set for heuristic generation (no test leakage)"""
        print("🔍 RUNNING COMPREHENSIVE ANALYSIS FOR HEURISTIC GENERATION")
        print(f"Config: {config}")
        print("✅ Running ONLY on dev set (no test leakage)")

        results = {}

        # Run on validation set (if available) - FULL SET
        if (self.data_root / "valid.csv").exists():
            print("Running on FULL validation set...")

            # Temporarily swap test.csv with valid.csv
            test_backup = self.data_root / "test.csv.backup"
            valid_file = self.data_root / "valid.csv"
            test_file = self.data_root / "test.csv"

            # Backup original test and use validation
            test_file.rename(test_backup)
            valid_file.rename(test_file)

            try:
                val_results = await run_matching(
                    dataset=self.dataset,
                    limit=None,  # NO LIMIT - run full validation set
                    max_candidates=config["max_candidates"],
                    model=config["model"],
                    use_semantic=config.get("use_semantic", True),
                    semantic_weight=config.get("semantic_weight", 0.5),
                    concurrency=concurrency,
                )
                results["validation"] = val_results
            finally:
                # Restore original files
                test_file.rename(valid_file)
                test_backup.rename(test_file)
        elif (self.data_root / "train.csv").exists():
            print("Running on training set slice...")

            # Load train.csv and take a reasonable slice (e.g., first 100 pairs)
            train_pairs = pd.read_csv(self.data_root / "train.csv")
            dev_slice_size = min(100, len(train_pairs))  # Take up to 100 pairs
            train_slice = train_pairs.head(dev_slice_size)

            print(f"📊 Using {dev_slice_size} pairs from training set for dev analysis")

            # Create temporary dev file
            dev_file = self.data_root / "dev_temp.csv"
            train_slice.to_csv(dev_file, index=False)

            # Temporarily swap test.csv with dev slice
            test_backup = self.data_root / "test.csv.backup"
            test_file = self.data_root / "test.csv"

            # Backup original test and use dev slice
            test_file.rename(test_backup)
            dev_file.rename(test_file)

            try:
                val_results = await run_matching(
                    dataset=self.dataset,
                    limit=None,  # NO LIMIT - run full dev slice
                    max_candidates=config["max_candidates"],
                    model=config["model"],
                    use_semantic=config.get("use_semantic", True),
                    semantic_weight=config.get("semantic_weight", 0.5),
                    concurrency=concurrency,
                )
                results["validation"] = val_results  # Store as validation for consistency
            finally:
                # Restore original files
                test_file.rename(dev_file)
                test_backup.rename(test_file)
                # Clean up temp file
                dev_file.unlink()
        else:
            print("⚠️ No validation or training set available - cannot generate rules without test leakage")
            raise ValueError("Dataset must have validation or training set for clean rule generation")

        # DO NOT run on test set - that would be data leakage
        print("✅ Skipping test set analysis to avoid data leakage")

        return results

    def analyze_comprehensive_failure_patterns(
        self, validation_results: Dict, test_results: Dict = None
    ) -> List[FailurePattern]:
        """Analyze actual prediction errors from validation results for heuristic generation"""
        print("📊 COMPREHENSIVE FAILURE PATTERN ANALYSIS (MODEL ERRORS ONLY)")

        A, B, validation_pairs, test_pairs = self.load_dataset()

        patterns = []

        # Performance info
        if validation_results:
            val_f1 = validation_results["metrics"]["f1"]
            print(f"Validation F1: {val_f1:.4f}")

        # Get actual predictions from validation results
        predictions = validation_results.get("predictions", {})
        if not predictions:
            print("⚠️ No predictions found in validation results - falling back to ground truth sampling")
            return self._fallback_ground_truth_analysis(validation_results, A, B, validation_pairs)

        # Analyze actual model errors
        tp_examples = []  # True positives (correctly predicted matches)
        fp_examples = []  # False positives (incorrectly predicted matches)
        fn_examples = []  # False negatives (missed real matches)
        tn_examples = []  # True negatives (correctly predicted non-matches) - sample a few

        # Process each validation pair and categorize the model's prediction
        for _, row in validation_pairs.iterrows():
            left_id = row.ltable_id
            right_id = row.rtable_id
            true_label = row.label

            # Check if model made a prediction for this pair
            if left_id in predictions:
                pred_right_id = predictions[left_id]
                predicted_match = pred_right_id == right_id
                predicted_label = 1 if predicted_match else 0
            else:
                predicted_label = 0  # No prediction = negative

            left_record = A[left_id]
            right_record = B[right_id]

            example = {
                "left_record": left_record,
                "right_record": right_record,
                "left_id": left_id,
                "right_id": right_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "source": "validation",
            }

            # Categorize based on true vs predicted labels
            if true_label == 1 and predicted_label == 1:
                example["record_type"] = "true_positive"
                tp_examples.append(example)
            elif true_label == 0 and predicted_label == 1:
                example["record_type"] = "false_positive"  # Model error - learn why this was wrong
                fp_examples.append(example)
            elif true_label == 1 and predicted_label == 0:
                example["record_type"] = "false_negative"  # Model error - learn why this was missed
                fn_examples.append(example)
            else:  # true_label == 0 and predicted_label == 0
                example["record_type"] = "true_negative"
                if len(tn_examples) < 5:  # Sample a few TNs for contrast
                    tn_examples.append(example)

        print("📋 Model Error Analysis:")
        print(f"  TP (correct matches): {len(tp_examples)}")
        print(f"  FP (incorrect matches): {len(fp_examples)} ← LEARN FROM THESE")
        print(f"  FN (missed matches): {len(fn_examples)} ← LEARN FROM THESE")
        print(f"  TN (correct non-matches): {len(tn_examples)} (sampled)")

        # Focus on errors for rule generation
        error_examples = fp_examples + fn_examples
        context_examples = tp_examples[:10] + tn_examples  # Add some context

        all_examples = error_examples + context_examples

        print(f"📋 Using {len(error_examples)} error examples + {len(context_examples)} context examples")
        print("✅ Focused on actual model failures for rule generation")

        # Create error-focused pattern
        if validation_results:
            pattern = FailurePattern(
                pattern_type="model_error_analysis",
                count=len(all_examples),
                examples=all_examples,
                description=f"Model error analysis: {len(fp_examples)} false positives, {len(fn_examples)} false negatives, {len(context_examples)} context examples. Validation F1: {validation_results['metrics']['f1']:.4f}",
            )
        else:
            pattern = FailurePattern(
                pattern_type="model_error_analysis",
                count=len(all_examples),
                examples=all_examples,
                description=f"Model error analysis: {len(error_examples)} errors, {len(context_examples)} context examples.",
            )

        patterns.append(pattern)
        return patterns

    def _fallback_ground_truth_analysis(
        self, validation_results: Dict, A: Dict, B: Dict, validation_pairs
    ) -> List[FailurePattern]:
        """Fallback to ground truth sampling when predictions unavailable"""
        print("⚠️ Using fallback ground truth analysis")

        # Sample true matches and non-matches from ground truth
        true_matches = validation_pairs[validation_pairs.label == 1]
        non_matches = validation_pairs[validation_pairs.label == 0]

        examples = []

        # Add all true matches
        for _, row in true_matches.iterrows():
            left_record = A[row.ltable_id]
            right_record = B[row.rtable_id]
            examples.append(
                {
                    "left_record": left_record,
                    "right_record": right_record,
                    "left_id": row.ltable_id,
                    "right_id": row.rtable_id,
                    "true_label": 1,
                    "record_type": "ground_truth_match",
                    "source": "validation",
                }
            )

        # Sample non-matches
        non_match_sample = non_matches.head(min(20, len(non_matches)))
        for _, row in non_match_sample.iterrows():
            left_record = A[row.ltable_id]
            right_record = B[row.rtable_id]
            examples.append(
                {
                    "left_record": left_record,
                    "right_record": right_record,
                    "left_id": row.ltable_id,
                    "right_id": row.rtable_id,
                    "true_label": 0,
                    "record_type": "ground_truth_non_match",
                    "source": "validation",
                }
            )

        pattern = FailurePattern(
            pattern_type="ground_truth_fallback",
            count=len(examples),
            examples=examples,
            description=f"Ground truth fallback: {len(true_matches)} matches, {len(non_match_sample)} non-matches",
        )

        return [pattern]

    def create_heuristic_generation_prompt(self, patterns: List[FailurePattern]) -> str:
        """Create a simplified heuristic generation prompt that works reliably"""
        # Get leaderboard target for context
        target_f1 = get_leaderboard_target_f1(self.dataset)
        print(f"🎯 Leaderboard target for {self.dataset}: F1 = {target_f1:.1f}")

        # Create a focused prompt that works reliably with Claude
        return f"""Generate entity matching rules for {self.dataset} dataset. Target F1: {target_f1:.1f}

Create rules that:
- Auto-reject very low similarity (< 0.1) to save LLM costs
- Auto-accept exact matches with high confidence
- Boost scores for likely matches

IMPORTANT: Return ONLY the JSON below with actual rules, no explanation or description:
{{
  "score_rules": [
    {{
      "rule_name": "example_boost",
      "description": "Example rule description",
      "implementation": "def example_boost(left_record, right_record):\\n    return ScoreAction(score_adjustment=0.2, confidence=0.8, reason='example')",
      "confidence": 0.8,
      "stage": "post_semantic",
      "test_cases": []
    }}
  ],
  "decision_rules": [
    {{
      "rule_name": "low_similarity_reject",
      "description": "Reject very low similarity",
      "implementation": "def low_similarity_reject(left_record, right_record, current_score):\\n    if current_score < 0.1:\\n        return DecisionAction(terminate_early=True, final_result=0, skip_llm=True, confidence=0.95, reason='Low similarity')\\n    return None",
      "confidence": 0.95,
      "stage": "pre_llm",
      "test_cases": []
    }}
  ],
  "weight_rules": [],
  "pipeline_rules": [],
  "implementation_notes": "Rules for {self.dataset}",
  "cost_optimization_strategy": "Reduce LLM calls via early rejection"
}}"""

    def create_heuristic_generation_prompt_old(self, patterns: List[FailurePattern]) -> str:
        """Create prompt for Claude Code SDK to generate enhanced heuristics"""
        # Get the competitive target from leaderboard
        target_f1 = get_leaderboard_target_f1(self.dataset)

        prompt = f"""You are an expert at entity matching and sophisticated rule-based system design.

I have an entity matching system with the following pipeline:
1. Candidate Selection: Trigram similarity filtering
2. Semantic Ranking: Sentence transformer reranking
3. LLM Decision: GPT makes final binary decisions

The system is performing well on validation data but needs to achieve competitive performance.

DATASET: {self.dataset}
DOMAIN: Entity matching in the {self.dataset} domain
🎯 **TARGET: BEAT THE LEADERBOARD - F1 > {target_f1:.1f}** (current #1 score to beat)

PERFORMANCE ANALYSIS:
"""

        for pattern in patterns:
            prompt += f"""
PATTERN: {pattern.pattern_type}
DESCRIPTION: {pattern.description}
COUNT: {pattern.count} examples analyzed

SAMPLE DATA EXAMPLES:
"""

            # Add examples from the pattern
            for i, example in enumerate(pattern.examples[:5]):  # Show first 5 examples
                # Safety check for true_label field
                match_status = "UNKNOWN"
                if "true_label" in example:
                    match_status = "MATCH" if example["true_label"] == 1 else "NON-MATCH"
                elif "record_type" in example:
                    if "match" in example["record_type"].lower():
                        match_status = "MATCH" if "true_positive" in example["record_type"] else "NON-MATCH"
                    else:
                        match_status = "NON-MATCH"

                prompt += f"""
Example {i + 1} ({match_status}):
Left Record: {json.dumps(example["left_record"], indent=2)}
Right Record: {json.dumps(example["right_record"], indent=2)}
Record Type: {example.get("record_type", "unknown")}
---
"""

        prompt += f"""
TASK: Generate sophisticated control logic rules for entity matching.

You can now create 4 types of rules:

1. **SCORE_RULES**: Adjust similarity scores (existing capability)
2. **DECISION_RULES**: Make early termination decisions to skip expensive LLM calls
3. **WEIGHT_RULES**: Dynamically adjust semantic vs trigram weights based on conditions
4. **PIPELINE_RULES**: Control pipeline flow and candidate selection

RULE CAPABILITIES:
- **Early Decisions**: "If exact brewery+beer match, auto-accept with confidence 0.95"
- **Smart LLM Skipping**: "If similarity < 0.1, auto-reject and skip LLM call"
- **Dynamic Weights**: "If styles incompatible, increase semantic_weight to 0.9"
- **Conditional Logic**: "When X is true, do Y, otherwise do Z"

PIPELINE STAGES:
- candidate_selection: During initial candidate filtering
- pre_semantic: Before semantic similarity calculation
- post_semantic: After semantic similarity, before LLM
- pre_llm: Just before LLM call (best for early decisions)
- post_llm: After LLM response

AVAILABLE ACTIONS:
```python
# Score adjustment
ScoreAction(score_adjustment=0.3, confidence=0.9, reason="Exact brewery match")

# Early termination decision
DecisionAction(terminate_early=True, final_result=1, confidence=0.95, reason="Exact match detected")
DecisionAction(terminate_early=True, final_result=0, skip_llm=True, reason="Very low similarity")

# Dynamic weight adjustment
WeightAction(semantic_weight=0.9, confidence=0.8, reason="Compensating for style mismatch")

# Pipeline control
PipelineAction(skip_stage=True, reason="High confidence exact match")
```

EXAMPLE RULES:
```python
def exact_brewery_beer_match(left_record, right_record):
    # Check for exact matches
    left_brewery = normalize(left_record.get('Brew_Factory_Name', ''))
    right_brewery = normalize(right_record.get('Brew_Factory_Name', ''))
    left_beer = normalize(left_record.get('Beer_Name', ''))
    right_beer = normalize(right_record.get('Beer_Name', ''))

    if (left_brewery == right_brewery and left_beer == right_beer
        and left_brewery and left_beer):
        return DecisionAction(
            terminate_early=True,
            final_result=1,
            confidence=0.95,
            reason="Exact brewery and beer name match"
        )
    return None

def low_similarity_early_reject(left_record, right_record, current_score):
    if current_score < 0.1:
        return DecisionAction(
            terminate_early=True,
            final_result=0,
            skip_llm=True,
            confidence=0.85,
            reason="Very low similarity - skip expensive LLM call"
        )
    return None
```

Generate 5-8 sophisticated rules across different types that will improve performance and reduce costs.

Format your response as JSON:
{{
  "score_rules": [
    {{
      "rule_name": "rule_name",
      "description": "What this rule does",
      "implementation": "def rule_name(left_record, right_record):\\n    # Python code\\n    return ScoreAction(...)",
      "confidence": 0.8,
      "stage": "candidate_selection",
      "test_cases": [...]
    }}
  ],
  "decision_rules": [
    {{
      "rule_name": "rule_name",
      "description": "Early termination logic",
      "implementation": "def rule_name(left_record, right_record, current_score):\\n    # Python code\\n    return DecisionAction(...) or None",
      "confidence": 0.9,
      "stage": "pre_llm",
      "test_cases": [...]
    }}
  ],
  "weight_rules": [
    {{
      "rule_name": "rule_name",
      "description": "Dynamic weight adjustment",
      "implementation": "def rule_name(left_record, right_record, current_weights):\\n    # Python code\\n    return WeightAction(...) or None",
      "confidence": 0.7,
      "stage": "pre_semantic",
      "test_cases": [...]
    }}
  ],
  "pipeline_rules": [],
  "implementation_notes": "Notes about the sophisticated control logic",
  "cost_optimization_strategy": "How these rules will reduce LLM API costs"
}}

CRITICAL REQUIREMENTS:
- **BALANCE PRECISION AND RECALL**: Avoid rules that achieve perfect precision by killing recall
- **TARGET: F1 > {target_f1:.1f}**: Rules should BEAT THE LEADERBOARD and achieve state-of-the-art performance
- **Focus on actual model errors**: Generate rules based on the FP/FN examples above
- **Conservative early decisions**: Only auto-reject when 99% confident, not 85%
- **Boost recall**: Create rules that help catch missed matches (reduce FN)
- **Smart LLM reduction**: Skip LLM calls for obvious cases, but don't be too aggressive

SPECIFIC GUIDANCE:
- Auto-reject rules should use very high confidence thresholds (>0.95)
- Include rules that BOOST scores for likely matches to improve recall
- Focus on {self.dataset} domain patterns from the examples above
- Design weight adjustments that help catch subtle matches
- Avoid being overly conservative - better to let LLM decide than miss matches

USING CLAUDE CODE TOOLS:
- You have the Read, Write and Bash tools available.
- Write candidate rules to a file named `heuristics.json`.
- Test them by running `python run_enhanced_matching.py --dataset {self.dataset} --limit 100 --heuristics heuristics.json`.
- Inspect the F1 score and iterate on the rules until the score exceeds {target_f1:.1f}.
- When satisfied, output ONLY the final `heuristics.json` content as JSON.
"""

        return prompt

    async def generate_heuristics(self, patterns: List[FailurePattern]) -> List[HeuristicRule]:
        """Use Claude Code SDK to generate heuristic rules"""
        if not self.claude_optimizer.claude_executable:
            raise RuntimeError(
                "Claude Code SDK not available. This pipeline requires Claude SDK to generate heuristics. Please install it with: pip install claude-code-sdk and npm install -g @anthropic-ai/claude-code"
            )

        print("🤖 GENERATING HEURISTICS VIA CLAUDE CODE SDK")

        prompt = self.create_heuristic_generation_prompt(patterns)

        try:
            # Call Claude Code SDK
            response = await self._call_claude_sdk(prompt)
            return self._parse_heuristic_response(response)

        except Exception as e:
            print(f"❌ CRITICAL ERROR: Claude SDK heuristic generation failed: {e}")
            print("❌ This pipeline REQUIRES Claude SDK to generate real heuristics")
            print("❌ Pipeline will crash - you can resume later with working Claude SDK")
            raise RuntimeError(
                f"Claude SDK heuristic generation failed: {e}. This is required for the pipeline to work properly."
            )

    async def _call_claude_sdk(self, prompt: str) -> str:
        """Call Claude Code SDK with the heuristic generation prompt"""
        try:
            print(f"🔍 Calling Claude SDK with prompt length: {len(prompt)} chars")

            options = ClaudeCodeOptions(
                allowed_tools=["Read", "Write", "Bash"],
                permission_mode="acceptEdits",
                cwd=os.getcwd(),
            )

            response_parts = []
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    # end of run
                    break

            return "".join(response_parts)

        except Exception as e:
            raise RuntimeError(f"Claude SDK call failed: {e}")

    def _parse_heuristic_response(self, response: str) -> List[HeuristicRule]:
        """Parse Claude's enhanced heuristic generation response"""
        try:
            print(f"🔍 Parsing response length: {len(response)} chars")
            print(f"🔍 Response preview: {response[:300]!r}")

            # Handle markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end == -1:
                    json_end = len(response)
                json_str = response[json_start:json_end].strip()
            else:
                # Extract JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]

            print(f"🔍 Extracted JSON length: {len(json_str)} chars")
            print(f"🔍 JSON preview: {json_str[:200]!r}")

            data = json.loads(json_str)

            # Handle both legacy and enhanced formats
            if "heuristic_rules" in data:
                # Legacy format
                rules = []
                for rule_data in data.get("heuristic_rules", []):
                    rule = HeuristicRule(
                        rule_name=rule_data["rule_name"],
                        description=rule_data["description"],
                        implementation=rule_data["implementation"],
                        confidence=rule_data["confidence"],
                        test_cases=rule_data.get("test_cases", []),
                    )
                    rules.append(rule)
                return rules
            # Enhanced format - return the full data structure for enhanced engine
            return data

        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ CRITICAL ERROR: Failed to parse Claude SDK response: {e}")
            print(f"Raw response: {response}")
            print("❌ Claude SDK returned invalid JSON - pipeline will crash")
            print("❌ You can resume later when Claude SDK is working properly")
            raise RuntimeError(f"Claude SDK returned invalid response: {e}. Raw response: {response}")

    def save_heuristics(self, rules_data, output_file: str):
        """Save generated heuristics to file"""
        if isinstance(rules_data, list):
            # Legacy format
            heuristic_data = {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset,
                "rules": [
                    {
                        "rule_name": rule.rule_name,
                        "description": rule.description,
                        "implementation": rule.implementation,
                        "confidence": rule.confidence,
                        "test_cases": rule.test_cases,
                    }
                    for rule in rules_data
                ],
            }
        else:
            # Enhanced format - add metadata
            heuristic_data = {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset,
                **rules_data,  # Include all the enhanced rule types
            }

        with open(output_file, "w") as f:
            json.dump(heuristic_data, f, indent=2)

        print(f"💾 Enhanced heuristics saved to {output_file}")

    async def run_comprehensive_heuristic_generation(self, config: Dict[str, Any], output_file: Optional[str] = None):
        """Run comprehensive heuristic generation from full dataset analysis"""
        print("🚀 STARTING COMPREHENSIVE CLAUDE SDK HEURISTIC GENERATION")
        print(f"Dataset: {self.dataset}")
        print(f"Config: {config}")
        print("=" * 80)

        # Step 1: Run comprehensive analysis on FULL datasets
        results = await self.run_comprehensive_analysis(config)

        # Step 2: Comprehensive failure pattern analysis (validation only)
        validation_results = results.get("validation")
        patterns = self.analyze_comprehensive_failure_patterns(validation_results)

        # Step 3: Generate heuristics from comprehensive patterns
        rules = await self.generate_heuristics(patterns)

        # Step 4: Save results
        if output_file:
            self.save_heuristics(rules, output_file)

        # Handle different return types
        if isinstance(rules, list):
            print(f"✅ Generated {len(rules)} heuristic rules from comprehensive analysis")
            for rule in rules:
                print(f"  - {rule.rule_name}: {rule.description} (confidence: {rule.confidence:.2f})")
        else:
            # Enhanced format
            total_rules = 0
            for rule_type, rule_list in rules.items():
                if rule_type.endswith("_rules") and isinstance(rule_list, list):
                    total_rules += len(rule_list)
                    print(f"  {rule_type}: {len(rule_list)} rules")
                    for rule in rule_list:
                        if isinstance(rule, dict):
                            print(
                                f"    - {rule.get('rule_name', 'unknown')}: {rule.get('description', 'no description')}"
                            )
            print(f"✅ Generated {total_rules} enhanced rules from comprehensive analysis")

        return rules


async def main():
    """CLI entry point for heuristic generation"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate domain-specific heuristics using Claude Code SDK")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use for analysis")
    parser.add_argument("--max-candidates", type=int, default=100, help="Max candidates for analysis")
    parser.add_argument("--semantic-weight", type=float, default=0.8, help="Semantic weight for analysis")
    parser.add_argument("--limit", type=int, default=100, help="Limit pairs for analysis")
    parser.add_argument("--output", help="Output file for generated heuristics")
    parser.add_argument(
        "--enhanced", action="store_true", help="Generate enhanced heuristics with sophisticated control logic"
    )

    args = parser.parse_args()

    # Configuration for analysis
    config = {
        "model": args.model,
        "max_candidates": args.max_candidates,
        "semantic_weight": args.semantic_weight,
        "limit": args.limit,
        "use_semantic": True,
    }

    # Generate heuristics from comprehensive analysis
    generator = ClaudeSDKHeuristicGenerator(args.dataset)
    return await generator.run_comprehensive_heuristic_generation(config, args.output)


if __name__ == "__main__":
    rules = asyncio.run(main())
