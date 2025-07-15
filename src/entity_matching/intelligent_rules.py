"""
Intelligent rules engine that uses analysis insights to improve candidate generation and matching.
This addresses the recall issues by adding specialized rules based on data patterns.
"""

import json
import re

from typing import Dict, List, Tuple

from .hybrid_matcher import semantic_similarity, syntactic_similarity, trigram_similarity


class IntelligentRulesEngine:
    """
    Rules engine that uses analysis insights to improve matching performance.
    Focuses on fixing recall issues by identifying missed candidates.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.rules = self.config.get("rules", [])
        self.hyperparams = self.config.get("hyperparameters", {})
        self.insights = self.config.get("analysis_insights", {})

    def should_force_match(self, left_record: Dict, right_record: Dict, similarities: Dict) -> bool:
        """
        Check if this pair should be forced as a match based on intelligent rules.
        """
        for rule in self.rules:
            if rule["type"] == "force_match" and self._evaluate_condition(rule, left_record, right_record, similarities):
                return True
        return False

    def should_expand_candidates(self, left_record: Dict, current_candidates: List) -> bool:
        """
        Check if candidate generation should be expanded for this record.
        """
        for rule in self.rules:
            if rule["type"] == "candidate_expansion" and self._should_expand_for_rule(rule, left_record, current_candidates):
                return True
        return False

    def get_additional_candidates(self, left_record: Dict, B_records: List[Dict], current_candidates: List, cfg) -> List[Tuple[int, float]]:
        """
        Generate additional candidates using intelligent rules.
        """
        additional_candidates = []

        # Rule: Perfect syntactic matches that might be missed
        if self._has_rule_type("perfect_syntactic_match"):
            additional_candidates.extend(
                self._find_perfect_syntactic_matches(left_record, B_records, current_candidates, cfg)
            )

        # Rule: High semantic similarity matches
        if self._has_rule_type("high_semantic_match") and cfg.use_semantic:
            additional_candidates.extend(
                self._find_high_semantic_matches(left_record, B_records, current_candidates, cfg)
            )

        # Rule: Fuzzy name matches (for partial name variations)
        if self._has_rule_type("fuzzy_name_match"):
            additional_candidates.extend(
                self._find_fuzzy_name_matches(left_record, B_records, current_candidates, cfg)
            )

        return additional_candidates

    def _evaluate_condition(self, rule: Dict, left_record: Dict, right_record: Dict, similarities: Dict) -> bool:
        """
        Evaluate if a rule condition is met.
        """
        condition = rule["condition"]

        # Parse condition string
        if "syntactic_similarity" in condition:
            threshold = self._extract_threshold(condition)
            return similarities.get("syntactic", 0) >= threshold

        if "semantic_similarity" in condition:
            threshold = self._extract_threshold(condition)
            return similarities.get("semantic", 0) >= threshold

        if "trigram_similarity" in condition:
            threshold = self._extract_threshold(condition)
            return similarities.get("trigram", 0) >= threshold

        if "perfect_match" in condition:
            # Check for exact field matches
            return self._is_perfect_field_match(left_record, right_record)

        return False

    def _should_expand_for_rule(self, rule: Dict, left_record: Dict, current_candidates: List) -> bool:
        """
        Check if candidates should be expanded based on rule conditions.
        """
        condition = rule["condition"]

        if "recall_at_10 < 0.95" in condition:
            # Always expand if recall is poor
            return len(current_candidates) < 150

        if "missing_high_similarity" in condition:
            # Check if we're likely missing high similarity candidates
            return self._likely_missing_candidates(left_record, current_candidates)

        return False

    def _find_perfect_syntactic_matches(self, left_record: Dict, B_records: List[Dict], current_candidates: List, cfg) -> List[Tuple[int, float]]:
        """
        Find candidates with perfect or near-perfect syntactic similarity.
        """
        candidates = []
        current_ids = {c[0] for c in current_candidates} if current_candidates else set()

        # Get primary field for comparison
        left_key = self._get_primary_field(left_record)
        if not left_key:
            return candidates

        for right_record in B_records:
            right_id = right_record.get("id")
            if right_id in current_ids:
                continue

            right_key = self._get_primary_field(right_record)
            if not right_key:
                continue

            syntactic_sim = syntactic_similarity(left_key, right_key)
            if syntactic_sim >= 0.95:  # Near perfect match
                candidates.append((right_id, syntactic_sim))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:10]

    def _find_high_semantic_matches(self, left_record: Dict, B_records: List[Dict], current_candidates: List, cfg) -> List[Tuple[int, float]]:
        """
        Find candidates with high semantic similarity that might be missed.
        """
        candidates = []
        current_ids = {c[0] for c in current_candidates} if current_candidates else set()

        semantic_threshold = self.hyperparams.get("semantic_threshold", 0.8) - 0.1

        for right_record in B_records[:1000]:  # Limit search for performance
            right_id = right_record.get("id")
            if right_id in current_ids:
                continue

            try:
                semantic_sim = semantic_similarity(
                    json.dumps(left_record, ensure_ascii=False).lower(),
                    json.dumps(right_record, ensure_ascii=False).lower(),
                    cfg
                )

                if semantic_sim >= semantic_threshold:
                    candidates.append((right_id, semantic_sim))
            except:
                continue

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    def _find_fuzzy_name_matches(self, left_record: Dict, B_records: List[Dict], current_candidates: List, cfg) -> List[Tuple[int, float]]:
        """
        Find candidates using fuzzy name matching for partial variations.
        """
        candidates = []
        current_ids = {c[0] for c in current_candidates} if current_candidates else set()

        left_key = self._get_primary_field(left_record)
        if not left_key:
            return candidates

        # Extract key terms from left record
        left_terms = self._extract_key_terms(left_key)
        if not left_terms:
            return candidates

        for right_record in B_records[:2000]:  # Limit search for performance
            right_id = right_record.get("id")
            if right_id in current_ids:
                continue

            right_key = self._get_primary_field(right_record)
            if not right_key:
                continue

            # Check for partial term matches
            if self._has_partial_term_match(left_terms, right_key):
                trigram_sim = trigram_similarity(left_key, right_key)
                if trigram_sim >= 0.3:  # Reasonable similarity threshold
                    candidates.append((right_id, trigram_sim))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    def _get_primary_field(self, record: Dict) -> str:
        """
        Get the primary field for comparison (usually the first non-id field).
        """
        for key, value in record.items():
            if key != "id" and value:
                return str(value)
        return ""

    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text for fuzzy matching.
        """
        # Remove common stop words and extract meaningful terms
        terms = re.findall(r'\b\w{3,}\b', text.lower())
        stop_words = {'the', 'and', 'feat', 'featuring', 'version', 'remix', 'edit', 'clean', 'explicit'}
        return [term for term in terms if term not in stop_words]

    def _has_partial_term_match(self, left_terms: List[str], right_text: str) -> bool:
        """
        Check if any left terms appear in right text.
        """
        right_lower = right_text.lower()
        return any(term in right_lower for term in left_terms if len(term) >= 4)

    def _is_perfect_field_match(self, left_record: Dict, right_record: Dict) -> bool:
        """
        Check if records have perfect matches in key fields.
        """
        key_fields = ["Song_Name", "Artist_Name", "name", "title"]

        for field in key_fields:
            if field in left_record and field in right_record:
                left_val = str(left_record[field]).strip().lower()
                right_val = str(right_record[field]).strip().lower()
                if left_val == right_val and len(left_val) > 3:
                    return True

        return False

    def _extract_threshold(self, condition: str) -> float:
        """
        Extract threshold value from condition string.
        """
        match = re.search(r'[><=]+\s*([\d.]+)', condition)
        return float(match.group(1)) if match else 0.5

    def _has_rule_type(self, rule_type: str) -> bool:
        """
        Check if any rules of the given type exist.
        """
        return any(rule.get("type") == rule_type for rule in self.rules)

    def _likely_missing_candidates(self, left_record: Dict, current_candidates: List) -> bool:
        """
        Heuristic to determine if we're likely missing good candidates.
        """
        # If we have very few candidates, we might be missing some
        return len(current_candidates) < 10


def create_intelligent_rules_from_analysis(analysis: Dict) -> Dict:
    """
    Create intelligent rules based on analysis insights.
    """
    rules = []
    hyperparams = {}

    # Extract key statistics
    true_match_stats = analysis["similarity_analysis"]["true_matches"]
    false_positive_stats = analysis["similarity_analysis"]["false_positives"]
    recall_analysis = analysis["candidate_analysis"]

    # Rule 1: Perfect syntactic matches
    if true_match_stats["syntactic"]["max"] > 0.95:
        rules.append({
            "type": "perfect_syntactic_match",
            "condition": "syntactic_similarity >= 0.95",
            "action": "force_match",
            "params": {"threshold": 0.95}
        })

    # Rule 2: High semantic similarity (if available)
    if analysis["metadata"]["semantic_available"]:
        semantic_threshold = max(0.75, true_match_stats["semantic"]["mean"] - 0.1)
        rules.append({
            "type": "high_semantic_match",
            "condition": f"semantic_similarity >= {semantic_threshold}",
            "action": "force_match",
            "params": {"threshold": semantic_threshold}
        })

    # Rule 3: Candidate expansion for poor recall
    if recall_analysis.get("recall_at_10", 0) < 0.95:
        rules.append({
            "type": "candidate_expansion",
            "condition": "recall_at_10 < 0.95",
            "action": "expand_candidates",
            "params": {"max_candidates": 200, "use_fuzzy_matching": True}
        })

    # Rule 4: Fuzzy name matching for partial matches
    rules.append({
        "type": "fuzzy_name_match",
        "condition": "missing_high_similarity",
        "action": "expand_candidates",
        "params": {"min_term_length": 4, "max_additional": 10}
    })

    # Set hyperparameters based on analysis
    syntactic_threshold = max(0.6, (true_match_stats["syntactic"]["mean"] + false_positive_stats["syntactic"]["mean"]) / 2)
    trigram_threshold = max(0.3, (true_match_stats["trigram"]["mean"] + false_positive_stats["trigram"]["mean"]) / 2)

    hyperparams = {
        "syntactic_threshold": round(syntactic_threshold, 3),
        "trigram_threshold": round(trigram_threshold, 3),
        "max_candidates": 150,  # Increase for better recall
        "use_heuristics": True
    }

    if analysis["metadata"]["semantic_available"]:
        semantic_threshold = max(0.7, (true_match_stats["semantic"]["mean"] + false_positive_stats["semantic"]["mean"]) / 2)
        hyperparams["semantic_threshold"] = round(semantic_threshold, 3)
        hyperparams["semantic_weight"] = 0.4

    return {
        "rules": rules,
        "hyperparameters": hyperparams,
        "analysis_insights": {
            "true_match_syntactic_mean": true_match_stats["syntactic"]["mean"],
            "false_positive_syntactic_mean": false_positive_stats["syntactic"]["mean"],
            "recall_at_10": recall_analysis.get("recall_at_10", 0),
            "recall_at_100": recall_analysis.get("recall_at_100", 0),
            "semantic_available": analysis["metadata"]["semantic_available"]
        }
    }
