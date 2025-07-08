# Cross-Dataset Entity Matching Pipeline

An agentic entity matching system where Claude analyzes matching failures and writes executable code to fix them. The system provides comprehensive candidate generation analysis and multi-stage heuristic optimization.

## 🚀 What This System Does

1. **Analyzes candidate generation failures** - identifies when correct matches aren't even in the candidate list
2. **Generates multi-stage executable rules** using agentic Claude SDK analysis 
3. **Automatically optimizes hyperparameters** with dataset-aware ranges (no test leakage)
4. **Tests both baseline and enhanced approaches** on clean test sets
5. **Provides detailed failure analysis** with actual record examples and candidate diagnostics

## ⚡ Quick Start

```bash
# Set up environment
./setup.sh
source .venv/bin/activate
export OPENAI_API_KEY="your-openai-key"

# Run complete automated pipeline
python run_complete_pipeline.py --dataset beer

# Run with resume capability for long experiments
python run_complete_pipeline.py --dataset walmart_amazon --resume

# Run with rule validation/optimization
python run_complete_pipeline.py --dataset beer --validate-rules
```

## 📊 System Architecture

The system runs a **4-stage agentic pipeline**:

### Stage 1: Improved Hyperparameter Sweep
- **Dataset-aware ranges**: Calculates candidate limits based on context window and dataset size
- **Geometric optimization**: Uses geometric mean between min (0.1% of candidates) and max (85% of context capacity)
- **Embeddings cache reuse**: Eliminates duplicate embedding computation across sweep configurations
- **Zero test leakage**: Uses only validation/training data

### Stage 2: Candidate Generation Analysis
- **Failure diagnosis**: Identifies when ground truth matches aren't in candidate lists  
- **Comprehensive coverage**: Analyzes ALL false negatives, not just samples
- **Ranking analysis**: Reports where ground truth ranked in candidates (if present)
- **Root cause identification**: Distinguishes candidate generation failures from LLM decision errors

### Stage 3: Agentic Multi-Stage Rule Generation
Claude iteratively writes and tests executable rules across **4 pipeline stages**:

- **`candidate_generation`**: Boost similarity to surface missed matches in candidate lists
- **`pre_llm`**: Make early decisions to skip expensive LLM calls  
- **`post_semantic`**: Adjust scores after semantic similarity calculation
- **`pre_semantic`**: Modify semantic/trigram weights dynamically

**Rule Types Generated**:
```python
CandidateAction(similarity_boost=0.4, confidence=0.8, reason="Partial name match")
DecisionAction(terminate_early=True, final_result=1, confidence=0.95, reason="Exact match")
ScoreAction(score_adjustment=0.3, confidence=0.8, reason="Strong field match")
WeightAction(semantic_weight=0.9, confidence=0.7, reason="Text-heavy comparison")
```

### Stage 4: A/B Test Evaluation
- **Baseline**: Optimal hyperparameters only
- **Enhanced**: Optimal hyperparameters + multi-stage generated rules
- **Detailed comparison**: F1, cost, performance, and LLM call reduction analysis

## 🎯 Example Output

```
🚀 COMPLETE ENTITY MATCHING PIPELINE: beer
================================================================

🎯 STAGE 1: Improved hyperparameter sweep
📊 Dataset size: 3000 candidates
🎯 Candidate sweep: [10, 39, 150] (0.3%, 1.3%, 5.0%)
⚖️ Semantic weights: [0.15, 0.5, 0.85]
✅ Best config: F1=0.8571, 150 candidates, 0.5 semantic weight

🔍 STAGE 2: Candidate generation analysis
📊 Analyzing 77 pairs for candidate generation failures...
📊 False negatives: 3 total
📊 FN where ground truth NOT in candidates: 2/3 (66.7%)
🚨 MAJOR ISSUE: 66.7% of false negatives are candidate generation failures!

🧠 STAGE 3: Agentic multi-stage rule generation
🤖 Claude is analyzing error patterns and writing executable rules...
✅ Generated candidate_rules: 2 rules for candidate generation stage
✅ Generated score_rules: 3 rules for post_semantic stage  
✅ Generated decision_rules: 1 rule for pre_llm stage
📄 Rules saved to: results/beer_generated_heuristics.json

🎯 STAGE 4A: Baseline evaluation (optimal params only)
✅ Baseline: F1=0.8571, Cost=$0.052, 150 candidates/0.5 weight

🎯 STAGE 4B: Enhanced evaluation (optimal params + multi-stage rules)
✅ Enhanced: F1=0.9200, Cost=$0.031, with candidate generation fixes

📊 A/B COMPARISON:
F1 Change: +0.0629 (✅ SIGNIFICANT IMPROVEMENT)
Cost Change: $-0.021 (40% reduction via early decisions)
Candidate generation issues: ✅ RESOLVED

🏆 FINAL RESULTS FOR BEER
Hyperparameter Optimization: F1=0.8571 ($0.045)
Baseline (params only):       F1=0.8571 ($0.052)
Enhanced (params + rules):    F1=0.9200 ($0.031)
Total Improvement:           +0.0629 F1 points
LLM Call Reduction:          40.2%
```

## 🛠️ Individual Components

### Basic Hybrid Matching
```bash
# Simple entity matching with trigram filtering
python llm_em_hybrid.py --dataset beer --max-candidates 50
python llm_em_hybrid.py --dataset beer --candidate-ratio 0.02
```

### Enhanced Matching with Rules
```bash
# Use pre-generated rules for better performance
python run_enhanced_matching.py --dataset beer \
  --heuristic-file results/generated_rules/beer_heuristics.json \
  --max-candidates 50
```

### Rule Generation Only
```bash
# Generate rules from dev set analysis
python -c "
from src.experiments.claude_sdk_heuristic_generator import ClaudeSDKHeuristicGenerator
generator = ClaudeSDKHeuristicGenerator('beer')
# ... rule generation code
"
```

## 📁 Project Structure

```
├── run_complete_pipeline.py           # 🚀 Main agentic pipeline
├── run_enhanced_matching.py           # Enhanced matching with multi-stage rules
├── generate_internal_leaderboard.py   # Performance tracking
│
├── src/
│   ├── entity_matching/
│   │   ├── hybrid_matcher.py          # Core matching with candidate generation heuristics
│   │   └── heuristic_engine.py        # Multi-stage rule execution engine
│   │
│   ├── experiments/
│   │   ├── agentic_heuristic_generator.py    # 🧠 Agentic rule generation with candidate analysis
│   │   ├── improved_sweep.py                 # Dataset-aware hyperparameter optimization
│   │   └── claude_sdk_heuristic_generator.py # Legacy rule generation (deprecated)
│   │
│   └── scripts/                       # Utility scripts
│
├── results/                           # Generated rules and detailed results
├── data/raw/                         # Entity matching datasets
├── .embeddings_cache/                # Cached semantic embeddings (reused across runs)
└── leaderboard.md                    # Performance comparison
```

## 🏆 Supported Datasets

| Dataset | Records | Test Pairs | Difficulty | Best F1 |
|---------|---------|------------|------------|---------|
| **beer** | 7,345 | 91 | Medium | 0.920 |
| **walmart_amazon** | 24,628 | 2,049 | Hard | 0.856 |
| **amazon_google** | 4,589 | 2,293 | Medium | 0.743 |
| **abt_buy** | 2,173 | 1,916 | Medium | 0.821 |
| **zomato_yelp** | 10,857 | 89 | Easy | 0.950 |

See `sizes.md` for complete dataset statistics.

## 🔧 Advanced Options

### Resume Long Experiments
```bash
# Pipeline saves checkpoints automatically
python run_complete_pipeline.py --dataset walmart_amazon --resume
```

### Control Concurrency
```bash
# Adjust API request parallelism
python run_complete_pipeline.py --dataset beer --concurrency 5
```

### Early Exit on Success
```bash
# Stop hyperparameter sweep if target F1 is reached
python run_complete_pipeline.py --dataset beer --early-exit
```

### Export Results
```bash
# Save detailed results to JSON/CSV
python llm_em_hybrid.py --dataset beer --output-json results.json
python llm_em_hybrid.py --dataset beer --output-csv experiments.csv
```

## 💰 Cost Management

The system provides detailed cost tracking:
- **Dev optimization**: ~$0.05-0.20 per dataset
- **Rule generation**: ~$0.01-0.05 per dataset  
- **Test evaluation**: ~$0.05-0.15 per dataset
- **Total typical cost**: ~$0.15-0.40 per complete pipeline run

## 🎯 Performance

Current leaderboard performance (F1 scores):
- **beer**: 0.920 (beats 0.889 previous best)
- **walmart_amazon**: 0.856 
- **amazon_google**: 0.743
- **abt_buy**: 0.821

The pipeline achieves **state-of-the-art performance** on most datasets while maintaining cost efficiency through rule-based early decisions.

## 🧠 How the Agentic System Works

### Candidate Generation Analysis
The system first **reproduces the exact candidate selection** that the original model used, then analyzes:
- Whether ground truth matches made it into the candidate list
- Where ground truth ranked in the candidates (if present)  
- Patterns in missed candidates vs. successful ones

This identifies the **root cause** of false negatives:
- **Candidate generation failure**: Correct match wasn't even considered (needs candidate rules)
- **LLM decision failure**: Correct match was in candidates but LLM chose wrong (needs score/decision rules)

### Multi-Stage Rule Architecture
Claude writes rules that execute at different pipeline stages:

```
Record Pair → [candidate_generation] → [pre_semantic] → [post_semantic] → [pre_llm] → LLM → Decision
                     ↑                     ↑               ↑              ↑
            CandidateAction         WeightAction    ScoreAction    DecisionAction
         (boost similarity)    (adjust weights)  (adjust scores) (skip LLM)
```

### Agentic Rule Generation Process
1. **Claude reads actual error data** - false positives, false negatives with candidate analysis
2. **Claude writes Python functions** that return Action objects based on record patterns
3. **Claude tests rules** by running the matching pipeline with `--heuristic-file`
4. **Claude iterates** - analyzes results, refines rules, tests again
5. **Claude outputs final rules** in multi-stage JSON format

### Example Generated Rule
```python
def boost_partial_name_candidates(left_record, right_record):
    left_name = normalize(left_record.get('Beer_Name', ''))
    right_name = normalize(right_record.get('Beer_Name', ''))
    if left_name and right_name and (left_name in right_name or right_name in left_name):
        return CandidateAction(similarity_boost=0.4, confidence=0.8, 
                              reason='Partial name match - surface in candidates')
    return None
```

This rule runs during `candidate_generation` stage and boosts trigram similarity by 0.4×0.8=0.32 for partial name matches, helping them surface in the candidate list.

## 🔬 Research Features

- **Zero test leakage**: Strict dev/test separation with temporary datasets
- **Agentic rule discovery**: Claude analyzes real failures and writes executable fixes
- **Multi-stage optimization**: Rules execute at 4 different pipeline stages
- **Candidate generation diagnosis**: Identifies root causes of matching failures
- **Embeddings cache efficiency**: Reuses expensive embeddings across experiments
- **Reproducible experiments**: Full checkpoint and resume support

## 🤝 Contributing

1. **Add new datasets** to `data/raw/` following the standard format (tableA.csv, tableB.csv, train/valid/test.csv)
2. **Extend agentic rule generation** in `src/experiments/agentic_heuristic_generator.py`
3. **Add new action types** to `src/entity_matching/heuristic_engine.py` and integrate into `hybrid_matcher.py`
4. **Enhance candidate analysis** by improving the failure diagnosis in candidate generation analysis
5. **Update leaderboard** with `python generate_internal_leaderboard.py`

### Key Files to Understand:
- `src/experiments/agentic_heuristic_generator.py` - Where Claude analyzes failures and generates rules
- `src/entity_matching/heuristic_engine.py` - Multi-stage rule execution engine
- `src/entity_matching/hybrid_matcher.py` - Core matching logic with candidate generation heuristics integration

## 📄 License

MIT License - see LICENSE file for details.