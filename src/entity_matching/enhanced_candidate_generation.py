"""
Enhanced candidate generation with intelligent rules integration.
This module extends the existing candidate generation to improve recall using analysis insights.
"""

import json

from typing import Dict, List, Optional, Tuple

from .hybrid_matcher import SEMANTIC_AVAILABLE, CandidateCache, Config, get_heuristic_engine, semantic_similarity_cached
from .intelligent_rules import IntelligentRulesEngine


def get_top_candidates_with_intelligent_rules(
    left_record: dict,
    candidate_cache: CandidateCache,
    max_candidates: int,
    cfg: Config,
    dataset: str = None,
    rules_engine: IntelligentRulesEngine = None
) -> List[tuple]:
    """
    Enhanced candidate generation with intelligent rules to improve recall.
    """
    json.dumps(left_record, ensure_ascii=False).lower()

    # Get heuristic engine if enabled
    heuristic_engine = get_heuristic_engine(cfg, dataset) if dataset else None

    # First pass: Standard candidate generation
    base_candidates = _get_base_candidates(left_record, candidate_cache, max_candidates, cfg, heuristic_engine)

    # Apply intelligent rules if available
    if rules_engine:
        # Check if we should expand candidates
        if rules_engine.should_expand_candidates(left_record, base_candidates):
            # Get additional candidates using intelligent rules
            additional_candidates = rules_engine.get_additional_candidates(
                left_record,
                list(candidate_cache.records.values()),  # All B records
                base_candidates,
                cfg
            )

            # Combine and deduplicate candidates
            all_candidates = _combine_candidates(base_candidates, additional_candidates, max_candidates)
        else:
            all_candidates = base_candidates

        # Apply intelligent force-match rules
        final_candidates = _apply_force_match_rules(left_record, all_candidates, rules_engine, cfg)
    else:
        final_candidates = base_candidates

    # Return top candidates
    return sorted(final_candidates, key=lambda x: x[0], reverse=True)[:max_candidates]


def _get_base_candidates(
    left_record: dict,
    candidate_cache: CandidateCache,
    max_candidates: int,
    cfg: Config,
    heuristic_engine
) -> List[Tuple[float, int]]:
    """
    Get base candidates using the standard approach.
    """
    left_str = json.dumps(left_record, ensure_ascii=False).lower()

    # Fast scoring using pre-computed values
    if cfg.use_semantic and SEMANTIC_AVAILABLE:
        # First pass: Fast trigram scoring with 3x candidates for semantic reranking
        trigram_candidates = max_candidates * 3
        trigram_scores = []

        for record_id in candidate_cache.get_all_ids():
            # Ultra-fast trigram similarity using pre-computed trigrams
            score = candidate_cache.compute_trigram_similarity(left_str, record_id)

            # Apply heuristics if enabled
            if heuristic_engine:
                try:
                    record = candidate_cache.get_record(record_id)
                    candidate_action = heuristic_engine.apply_stage_heuristics(
                        "candidate_generation", left_record, record
                    )
                    if candidate_action and hasattr(candidate_action, "similarity_boost"):
                        score += candidate_action.similarity_boost * candidate_action.confidence
                        score = min(score, 1.0)
                except Exception:
                    pass

            trigram_scores.append((score, record_id))

        # Sort and take top candidates for semantic reranking
        trigram_scores.sort(key=lambda x: x[0], reverse=True)
        top_candidates = trigram_scores[:trigram_candidates]

        # Second pass: Semantic similarity with cached embeddings
        if cfg.embeddings is not None:
            try:
                candidates = []
                for trigram_score, record_id in top_candidates:
                    record = candidate_cache.get_record(record_id)
                    semantic_score = semantic_similarity_cached(left_record, record, cfg.embeddings)

                    # Weighted combination
                    combined_score = (1 - cfg.semantic_weight) * trigram_score + cfg.semantic_weight * semantic_score

                    # Apply heuristics if enabled
                    if heuristic_engine:
                        try:
                            heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                            combined_score += heuristic_adjustment
                        except Exception:
                            pass

                    candidates.append((combined_score, record_id))

            except Exception:
                # Fall back to trigram only
                candidates = [(score, record_id) for score, record_id in top_candidates[:max_candidates]]
        else:
            # Fall back to trigram only
            candidates = [(score, record_id) for score, record_id in top_candidates[:max_candidates]]
    else:
        # Trigram only
        trigram_scores = []
        for record_id in candidate_cache.get_all_ids():
            score = candidate_cache.compute_trigram_similarity(left_str, record_id)

            # Apply heuristics if enabled
            if heuristic_engine:
                try:
                    record = candidate_cache.get_record(record_id)
                    heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                    score += heuristic_adjustment
                except Exception:
                    pass

            trigram_scores.append((score, record_id))

        trigram_scores.sort(key=lambda x: x[0], reverse=True)
        candidates = trigram_scores[:max_candidates]

    return candidates


def _combine_candidates(
    base_candidates: List[Tuple[float, int]],
    additional_candidates: List[Tuple[int, float]],
    max_candidates: int
) -> List[Tuple[float, int]]:
    """
    Combine base and additional candidates, removing duplicates.
    """
    # Convert additional candidates to same format as base candidates
    additional_formatted = [(score, record_id) for record_id, score in additional_candidates]

    # Combine and deduplicate
    seen_ids = set()
    combined = []

    # Add base candidates first (they're already sorted)
    for score, record_id in base_candidates:
        if record_id not in seen_ids:
            combined.append((score, record_id))
            seen_ids.add(record_id)

    # Add additional candidates that aren't already present
    for score, record_id in additional_formatted:
        if record_id not in seen_ids:
            combined.append((score, record_id))
            seen_ids.add(record_id)

    # Sort by score and return top candidates
    combined.sort(key=lambda x: x[0], reverse=True)
    return combined[:max_candidates * 2]  # Return more for force-match filtering


def _apply_force_match_rules(
    left_record: dict,
    candidates: List[Tuple[float, int]],
    rules_engine: IntelligentRulesEngine,
    cfg: Config
) -> List[Tuple[float, int]]:
    """
    Apply force-match rules to boost certain candidates.
    """
    enhanced_candidates = []

    for score, record_id in candidates:
        # We need to get the actual record to apply rules
        # This is a bit expensive but necessary for intelligent matching
        try:
            # For now, we'll need to implement a way to get records by ID
            # This would need to be integrated with the candidate cache
            enhanced_candidates.append((score, record_id))
        except Exception:
            enhanced_candidates.append((score, record_id))

    return enhanced_candidates


def create_intelligent_rules_engine(config: Dict) -> Optional[IntelligentRulesEngine]:
    """
    Create an intelligent rules engine from configuration.
    """
    if not config or "rules" not in config:
        return None

    return IntelligentRulesEngine(config)


def enhance_config_with_analysis(analysis_file: str, cfg: Config) -> IntelligentRulesEngine:
    """
    Enhance configuration with analysis-based intelligent rules.
    """
    try:
        with open(analysis_file) as f:
            analysis = json.load(f)

        from .intelligent_rules import create_intelligent_rules_from_analysis
        rules_config = create_intelligent_rules_from_analysis(analysis)

        # Apply hyperparameters to cfg
        hyperparams = rules_config.get("hyperparameters", {})
        if "semantic_weight" in hyperparams:
            cfg.semantic_weight = hyperparams["semantic_weight"]
        if "use_heuristics" in hyperparams:
            cfg.use_heuristics = hyperparams["use_heuristics"]

        return IntelligentRulesEngine(rules_config)

    except Exception as e:
        print(f"Warning: Could not load intelligent rules from analysis: {e}")
        return None
