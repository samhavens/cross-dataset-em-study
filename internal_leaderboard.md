# Internal Leaderboard

Our best results on entity matching datasets. Shows the better of baseline (optimal hyperparameters) vs enhanced (with heuristic rules).

**Legend:**
- 🎯 **Our Result**: Best F1 from our pipeline (baseline or enhanced)
- 📊 **Leaderboard Target**: Top published result from main leaderboard.md
- ✅ **Beat Target**: Our result exceeds the published leaderboard
- 📈 **Below Target**: Still working to beat the published result
- ❌ **Not Tested**: No pipeline results yet

| Dataset | Our F1 | Method | vs Target | Leaderboard Target | Notes |
|---------|--------|--------|-----------|-------------------|-------|
| abt_buy | 0.893 | Claude-optimized hyperparams + prompt | 📈 -0.031 | 92.4 | P:0.966, R:0.830, Early:0, -LLM:0.0%, [exp_244fcf2b](results/experiments/exp_244fcf2b/) |
| amazon_google | 0.000 | ❌ Not tested | ❌ Not tested | 75.0 | File: results/amazon_google_complete_pipeline.json |
| beer | **0.966** | Default (or user-provided) hyperparams + default prompt | ✅ Beat | 95.3 | P:0.933, R:1.000, 50c, sw:0.9, tw:0.05, syn:0.05, [exp_8136d688](results/experiments/exp_8136d688/) |
| dblp_acm | **0.986** | Default (or user-provided) hyperparams + default prompt | ✅ Beat | 96.5 | P:0.976, R:0.995, 5c, sw:0.6, tw:0.2, syn:0.2,  |
| dblp_scholar | 0.571 | Default (or user-provided) hyperparams + default prompt | 📈 -0.327 | 89.8 | P:0.967, R:0.406, 100c, sw:0.7, tw:0.15, syn:0.15,  |
| fodors_zagat | **1.000** | Claude-optimized hyperparams + prompt | ✅ Beat | 99.6 | P:1.000, R:1.000, Early:0, -LLM:0.0%, [exp_aa86a90f](results/experiments/exp_aa86a90f/) |
| itunes_amazon | **0.851** | Default (or user-provided) hyperparams + default prompt | ✅ Beat | 85.0 | P:1.000, R:0.741, 50c, sw:0.9, tw:0.05, syn:0.05, [exp_fa9ab8c9](results/experiments/exp_fa9ab8c9/) |
| rotten_imdb | **0.987** | Default (or user-provided) hyperparams + default prompt | ✅ Beat | 97.2 | P:1.000, R:0.974, 100c, sw:0.6, tw:0.2, syn:0.2,  |
| walmart_amazon | 0.000 | ❌ Not tested | ❌ Not tested | 85.1 | File: results/walmart_amazon_complete_pipeline.json |
| zomato_yelp | **1.000** | Claude-optimized hyperparams + prompt | ✅ Beat | 98.2 | P:1.000, R:1.000, Early:0, -LLM:0.0%,  |

## Summary

- **Total Datasets**: 10
- **Tested**: 8/10 datasets
- **Beat Leaderboard**: 6/8 tested datasets
- **Success Rate**: 75.0% (of tested)
- **Remaining**: 2 datasets to test

## Methodology

Our pipeline:
1. **Hyperparameter Optimization**: Strategic sweep on dev/validation set to find optimal parameters
2. **Rule Generation**: Claude SDK generates domain-specific heuristic rules based on failure analysis
3. **A/B Testing**: Compare baseline (optimal params only) vs enhanced (optimal params + rules)
4. **Best Result**: Report whichever approach (baseline or enhanced) achieves higher F1

**Baseline Approach**: Hybrid trigram + semantic similarity with optimized hyperparameters
**Enhanced Approach**: Baseline + heuristic rules for early decisions, score adjustments, and weight tuning

Results show that sometimes optimal hyperparameters alone beat complex rule systems!
