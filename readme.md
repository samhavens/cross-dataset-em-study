# Cross-Dataset Entity Matching Pipeline

A sophisticated automated entity matching system that combines traditional ML techniques with LLM-powered reasoning and rule generation.

## 🚀 What This System Does

1. **Automatically optimizes hyperparameters** on dev/validation sets (no test leakage)
2. **Generates executable matching rules** using Claude SDK analysis  
3. **Tests both baseline and enhanced approaches** on clean test sets
4. **Provides A/B comparison** and recommendations
5. **Tracks costs and performance** across the entire pipeline

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

## 📊 Pipeline Overview

The system runs a **3-step automated pipeline**:

### Step 1: Hyperparameter Optimization (Dev Set)
- Runs intelligent parameter sweep on validation/training data
- Finds optimal: candidate count, semantic weights, model choice
- **Zero test leakage** - uses only dev/validation sets

### Step 2: Rule Generation (Claude SDK)
- Analyzes dev set performance patterns
- Generates executable matching rules using Claude SDK
- Creates heuristics for early matching decisions
- Optional: Validates and optimizes rules on dev set

### Step 3: A/B Test Evaluation (Test Set)  
- Tests **baseline** approach (optimal params only)
- Tests **enhanced** approach (optimal params + generated rules)
- Compares F1, cost, and performance
- Provides clear recommendation

## 🎯 Example Output

```
🚀 COMPLETE ENTITY MATCHING PIPELINE
Dataset: beer
================================================================

🎯 STEP 1: Hyperparameter optimization on dev set
✅ Best Dev Results: F1=0.8500, Cost=$0.045
🎯 Optimal Parameters: 75 candidates, 0.60 semantic weight, gpt-4o-mini

🧠 STEP 2: Rule generation (analyzing dev results...)
✅ Generated executable rules saved to: results/generated_rules/beer_generated_heuristics.json

🎯 STEP 3A: FINAL TEST EVALUATION WITHOUT rules (optimal params baseline)
✅ Baseline Results (no rules): F1=0.8889, Cost=$0.052

🎯 STEP 3B: FINAL TEST EVALUATION WITH rules (enhanced approach)
✅ Enhanced Results (with rules): F1=0.9200, Cost=$0.031

📊 A/B COMPARISON:
F1 Change: +0.0311 (✅ IMPROVED)
Cost Change: $-0.021
Rules ✅ HELPED

🏆 FINAL RESULTS FOR BEER
Dev F1:        0.8500 ($0.045)
Test Baseline: 0.8889 ($0.052) - optimal params only  
Test Enhanced: 0.9200 ($0.031) - optimal params + rules
Improvement:   +0.0311 F1 points
Total Cost: $0.128
LLM Call Reduction: 40.2%
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
├── run_complete_pipeline.py           # 🚀 Main automated pipeline
├── run_enhanced_matching.py           # Enhanced matching with rules  
├── llm_em_hybrid.py                   # Basic hybrid matching
├── generate_internal_leaderboard.py   # Performance tracking
│
├── src/
│   ├── entity_matching/
│   │   ├── hybrid_matcher.py          # Core matching logic
│   │   └── enhanced_heuristic_engine.py # Rule execution engine
│   │
│   ├── experiments/
│   │   ├── claude_sdk_heuristic_generator.py # Rule generation
│   │   ├── claude_sdk_optimizer.py           # Parameter optimization  
│   │   └── intelligent_sweep.py              # Smart hyperparameter search
│   │
│   └── scripts/                       # Utility scripts
│
├── results/                           # Generated results and rules
├── data/raw/                         # Entity matching datasets
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

## 🔬 Research Features

- **Zero test leakage**: Strict dev/test separation
- **Automated rule discovery**: Claude SDK generates domain-specific rules
- **Adaptive optimization**: Smart hyperparameter search
- **Cost-performance tradeoffs**: Balances accuracy with efficiency
- **Reproducible experiments**: Full checkpoint and resume support

## 🤝 Contributing

1. Add new datasets to `data/raw/` following the standard format
2. Extend rule generation in `src/experiments/claude_sdk_heuristic_generator.py`
3. Add new matching strategies in `src/entity_matching/`
4. Update leaderboard with `python generate_internal_leaderboard.py`

## 📄 License

MIT License - see LICENSE file for details.