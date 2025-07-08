# Implementation Plan: Claude-Driven Hyperparameter + Rules Generation

## Overview
Replace the current blind hyperparameter sweep with intelligent analysis where Claude sees rich similarity data and chooses BOTH hyperparameters AND rules in one shot.

## Current State Analysis

### What Works Well
- `dump_matches.py` shows syntactic, trigram, and semantic similarities 
- Uses dev/validation sets (no test leakage)
- Embeddings are cached for speed
- Rules engine exists and can generate heuristics

### Key Problems to Solve
1. **Syntactic similarity not available to rules engine** - hybrid_matcher.py only exposes trigram + semantic
2. **Blind hyperparameter search** - sweeps configs without understanding WHY they work
3. **Separated optimization** - hyperparameters chosen first, then rules based on failures
4. **Limited signal** - Claude only sees pass/fail, not rich similarity patterns

## Implementation Steps

### Step 1: Add Syntactic Similarity to hybrid_matcher.py
**Files to modify**: `src/entity_matching/hybrid_matcher.py`

**Changes needed**:
```python
# Add syntactic_similarity function near trigram_similarity (line ~122)
def syntactic_similarity(s1: str, s2: str) -> float:
    """Calculate syntactic similarity using difflib.SequenceMatcher"""
    from difflib import SequenceMatcher
    if not s1 or s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

# Modify combined_similarity function (line ~275) to include all three
def triple_similarity(s1: str, s2: str, cfg: Config) -> float:
    """Calculate combined trigram + syntactic + semantic similarity"""
    trigram_score = trigram_similarity(s1, s2)
    syntactic_score = syntactic_similarity(s1, s2)
    
    if not cfg.use_semantic or not SEMANTIC_AVAILABLE:
        # Equal weight between trigram and syntactic
        return 0.5 * trigram_score + 0.5 * syntactic_score
    
    semantic_score = semantic_similarity(s1, s2, cfg)
    
    # Three-way weighted combination
    return (cfg.trigram_weight * trigram_score + 
            cfg.syntactic_weight * syntactic_score +
            cfg.semantic_weight * semantic_score)

# Add weights to Config class (line ~36)
class Config:
    def __init__(self):
        # ... existing fields ...
        self.trigram_weight = 0.2      # Lower weight for trigram
        self.syntactic_weight = 0.3    # Medium weight for syntactic
        self.semantic_weight = 0.5     # Higher weight for semantic (sums to 1.0)
```

**Context**: Rules need access to syntactic similarity to make decisions like "if syntactic > 0.8 and trigram < 0.5, boost score because it's likely formatting difference"

### Step 2: Create Rich Analysis Output Script
**New file**: `scripts/analyze_for_claude.py`

**Purpose**: Generate structured JSON output that Claude can use to make intelligent decisions

**Key features**:
- Use dev/validation sets (no test leakage)  
- Show similarity distributions for matches vs non-matches
- Identify key patterns (e.g., "90% of true matches have semantic > 0.8")
- Include dataset characteristics (size, typical record structure)
- Export candidate generation effectiveness at different thresholds

**Output format**:
```json
{
  "dataset": "itunes_amazon",
  "analysis_type": "validation",
  "similarity_analysis": {
    "true_matches": {
      "syntactic": {"mean": 0.85, "std": 0.12, "median": 0.87},
      "trigram": {"mean": 0.65, "std": 0.18, "median": 0.68}, 
      "semantic": {"mean": 0.92, "std": 0.08, "median": 0.94}
    },
    "false_positives": {
      "syntactic": {"mean": 0.45, "std": 0.22, "median": 0.42},
      "trigram": {"mean": 0.71, "std": 0.15, "median": 0.73},
      "semantic": {"mean": 0.33, "std": 0.19, "median": 0.31}
    }
  },
  "candidate_analysis": {
    "recall_at_50": 0.78,
    "recall_at_100": 0.85,
    "recall_at_150": 0.89
  },
  "dataset_characteristics": {
    "table_a_size": 700,
    "table_b_size": 55000,
    "typical_fields": ["Song_Name", "Artist_Name", "Album_Name"],
    "common_patterns": ["[Explicit] tags", "featuring artists", "remastered versions"]
  }
}
```

### Step 3: Create Claude Config + Rules Generator
**New file**: `scripts/claude_config_generator.py`

**Purpose**: Send rich analysis to Claude and get back BOTH optimal hyperparameters AND rules

**Claude prompt structure**:
```
You are an expert at entity matching optimization. Based on this similarity analysis, 
determine the optimal hyperparameters AND generate heuristic rules.

DATASET ANALYSIS:
{json_analysis}

Please provide:
1. Optimal hyperparameters (max_candidates, trigram_weight, syntactic_weight, semantic_weight)
2. Heuristic rules that leverage all three similarity measures
3. Reasoning for your choices

Output format:
{
  "hyperparameters": {
    "max_candidates": 100,
    "trigram_weight": 0.2,
    "syntactic_weight": 0.3, 
    "semantic_weight": 0.5
  },
  "rules": [...],
  "reasoning": "..."
}
```

### Step 4: Modified Pipeline Entry Point
**New file**: `scripts/intelligent_pipeline.py` or modify `run_complete_pipeline.py`

**New flow**:
1. Run `analyze_for_claude.py` on dev set → rich_analysis.json
2. Send to Claude → get hyperparameters + rules  
3. Run final test with both optimized config AND rules
4. Compare against baseline (no rules, default config)

**Skip entirely**: The current dev hyperparameter sweep

### Step 5: Testing Scripts

**File**: `scripts/test_syntactic_integration.py`
- Test that syntactic similarity works in hybrid_matcher
- Verify rules can access all three similarity measures

**File**: `scripts/test_analysis_output.py`  
- Test that `analyze_for_claude.py` generates proper JSON
- Verify distributions make sense for known datasets

**File**: `scripts/test_claude_integration.py`
- Test full pipeline: analysis → Claude → execution
- Verify hyperparameters are reasonable

## Key Context for Next Session

### Current File Structure
- Main pipeline: `run_complete_pipeline.py` (has dev sweep we want to replace)
- Entity matching: `src/entity_matching/hybrid_matcher.py` (needs syntactic similarity)
- Analysis script: `scripts/dump_matches.py` (good foundation, needs JSON output)
- Rules engine: `src/entity_matching/enhanced_heuristic_engine.py` (already works)

### Critical Implementation Details
1. **Config class weights must sum to 1.0** for proper combination
2. **Use cached embeddings** from `compute_dataset_embeddings()` for speed
3. **Follow existing dev set logic** from `get_dev_dataset()` to avoid test leakage
4. **Claude SDK integration** already exists in `ClaudeSDKHeuristicGenerator`

### Expected Benefits
- **Better hyperparameters**: Chosen based on actual data patterns, not blind search
- **Better rules**: Informed by similarity analysis, not just failure cases  
- **Faster pipeline**: No expensive hyperparameter sweep
- **More interpretable**: Clear reasoning for why certain configs work

### Success Metrics
- F1 improvement over current pipeline
- Faster execution (no dev sweep)
- More consistent performance across datasets
- Better interpretability of generated rules

## Next Session Action Plan
1. Start with Step 1 (add syntactic similarity to hybrid_matcher.py)
2. Test integration with a simple script
3. Create the analysis script (Step 2)
4. Test on one dataset (itunes_amazon) to validate approach
5. Build Claude integration if initial results look promising