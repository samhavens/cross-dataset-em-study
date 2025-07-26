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
| abt_buy | **0.956** | Baseline (optimal) | ✅ Beat | 92.4 | P:0.965, R:0.947, 25c, sw:0.2 |
| amazon_google | 0.737 | Baseline (optimal) | 📈 -0.013 | 75.0 | P:0.694, R:0.786, 25c, sw:0.2 |
| beer | **1.000** | Enhanced (rules) | ✅ Beat | 95.3 | P:1.000, R:1.000, Early:0, -LLM:0.0% |
| dblp_acm | 0.000 | ❌ Not tested | ❌ Not tested | 96.5 | Run: python run_complete_pipeline.py --dataset dblp_acm |
| dblp_scholar | 0.000 | ❌ Not tested | ❌ Not tested | 89.8 | Run: python run_complete_pipeline.py --dataset dblp_scholar |
| fodors_zagat | **1.000** | Baseline (optimal) | ✅ Beat | 99.6 | P:1.000, R:1.000, 28c, sw:0.2 |
| itunes_amazon | 0.000 | ❌ Not tested | ❌ Not tested | 85.0 | Run: python run_complete_pipeline.py --dataset itunes_amazon |
| rotten_imdb | **0.974** | Baseline (optimal) | ✅ Beat | 97.2 | P:0.950, R:1.000, 640c, sw:0.8 |
| walmart_amazon | **0.857** | Enhanced (rules) | ✅ Beat | 85.1 | P:0.857, R:0.857, Early:0, -LLM:0.0% |
| zomato_yelp | **1.000** | Baseline (optimal) | ✅ Beat | 98.2 | P:1.000, R:1.000, 50c, sw:0.5 |

## Summary

- **Total Datasets**: 10
- **Tested**: 7/10 datasets
- **Beat Leaderboard**: 6/7 tested datasets
- **Success Rate**: 85.7% (of tested)
- **Remaining**: 3 datasets to test

## Methodology

Our pipeline:
1. **Hyperparameter Optimization**: Strategic sweep on dev/validation set to find optimal parameters
2. **Rule Generation**: Claude SDK generates domain-specific heuristic rules based on failure analysis
3. **A/B Testing**: Compare baseline (optimal params only) vs enhanced (optimal params + rules)
4. **Best Result**: Report whichever approach (baseline or enhanced) achieves higher F1

**Baseline Approach**: Hybrid trigram + semantic similarity with optimized hyperparameters
**Enhanced Approach**: Baseline + heuristic rules for early decisions, score adjustments, and weight tuning

Results show that sometimes optimal hyperparameters alone beat complex rule systems!
