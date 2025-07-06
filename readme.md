# Cross-Dataset Entity Matching with Claude Code SDK

## Current Working System

**The pipeline has all the pieces but needs final connection:**

### ✅ What Works
- **Enhanced Matching**: `run_enhanced_matching.py` executes sophisticated rules
- **Dev Analysis**: Can analyze dev/validation sets without test leakage  
- **Rule Engine**: `enhanced_heuristic_engine.py` applies executable rules

### ❌ What's Broken
- **Rule Generation**: Has data leakage (analyzes both dev AND test sets)
- **Pipeline Integration**: Can't connect dev analysis → rule generation → test evaluation cleanly

## Quick Start

```bash
# Set up environment
./setup.sh
source .venv/bin/activate
export OPENAI_API_KEY="your-key-here"

# Test enhanced matching with existing rules
python run_enhanced_matching.py --dataset beer --heuristic-file zomato_yelp_restaurant_heuristics.json --max-candidates 50 --limit 10

# Test complete pipeline (currently has data leakage issue)
python run_complete_pipeline.py --dataset beer
```

## Core Files

- **`run_enhanced_matching.py`** - Main working enhanced matching system
- **`run_complete_pipeline.py`** - Attempts full pipeline but has data leakage in step 2
- **`src/entity_matching/enhanced_heuristic_engine.py`** - Rule execution engine  
- **`src/experiments/claude_sdk_heuristic_generator.py`** - Rule generation (has data leakage)
- **`zomato_yelp_restaurant_heuristics.json`** - Working example heuristics

## The Issue

The ClaudeSDKHeuristicGenerator runs comprehensive analysis on BOTH validation AND test sets (data leakage). Need to modify it to only analyze dev/validation for rule generation.

## Architecture

```
src/entity_matching/
├── hybrid_matcher.py              # Basic matching
├── enhanced_heuristic_engine.py   # Rule execution (works)
└── heuristic_engine.py            # Legacy

src/experiments/
├── claude_sdk_heuristic_generator.py  # Rule generation (data leakage)
└── claude_sdk_optimizer.py            # Hyperparameter optimization

Root:
├── run_enhanced_matching.py       # Enhanced matching (works)
└── run_complete_pipeline.py       # Full pipeline (broken rule generation)
```