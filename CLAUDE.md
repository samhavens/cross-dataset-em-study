# Claude Code Session Notes

## 🎯 **CURRENT STATUS: ALMOST WORKING**

**All pieces exist but pipeline has data leakage in rule generation step.**

### ✅ **WORKING COMPONENTS:**
- **Enhanced matching system** → `run_enhanced_matching.py` executes sophisticated rules
- **Rule execution engine** → `enhanced_heuristic_engine.py` applies 4 rule types
- **Dev set analysis** → Can analyze validation sets without test leakage
- **Existing heuristics** → `zomato_yelp_restaurant_heuristics.json` works as example

### ❌ **THE ISSUE:**
- **Rule generation has data leakage** → `ClaudeSDKHeuristicGenerator.run_comprehensive_analysis()` analyzes BOTH validation AND test sets (lines 106-117)

## 🧪 **WHAT WORKS RIGHT NOW:**

```bash
# Enhanced matching with existing rules (WORKS)
python run_enhanced_matching.py --dataset beer --heuristic-file zomato_yelp_restaurant_heuristics.json --max-candidates 50 --limit 10

# Output: Loads 8 rules, processes pairs with sophisticated control logic
# ✅ exact_restaurant_name_boost (score, candidate_selection)
# ✅ phone_and_name_auto_accept (decision, pre_llm) 
# ✅ very_low_similarity_auto_reject (decision, pre_llm)
# etc.
```

## 🚧 **WHAT'S BROKEN:**

```bash
# Complete pipeline (BROKEN - data leakage in step 2)
python run_complete_pipeline.py --dataset beer

# Step 1: ✅ Dev analysis works
# Step 2: ❌ Rule generation analyzes BOTH dev AND test (data leakage)
# Step 3: ✅ Enhanced matching would work if step 2 generated clean rules
```

## 🔧 **THE FIX NEEDED:**

Modify `src/experiments/claude_sdk_heuristic_generator.py` line 106-117 to ONLY analyze dev/validation set:

```python
# CURRENT (BROKEN - data leakage):
# Line 106-117: Runs on BOTH validation AND test sets

# NEEDED (CLEAN):
# Only analyze validation set for rule generation
# Then apply generated rules to test set
```

## 📋 **PIPELINE SHOULD BE:**

1. **Dev Analysis** → `run_dev_only_analysis()` ✅ WORKS
2. **Rule Generation** → `generate_actual_rules()` ❌ HAS DATA LEAKAGE  
3. **Test with Rules** → `run_enhanced_matching()` ✅ WORKS

## 🏆 **SUCCESS CRITERIA:**

Once fixed, should achieve:
- Clean dev → rule generation → test pipeline
- No test set leakage in rule generation
- Sophisticated control logic reduces LLM calls
- Domain-specific rules improve F1 performance

## 📁 **KEY FILES:**

- `run_enhanced_matching.py` - Enhanced matching (works)
- `run_complete_pipeline.py` - Full pipeline (almost works, rule generation broken)  
- `src/experiments/claude_sdk_heuristic_generator.py` - Rule generation (needs data leakage fix)
- `zomato_yelp_restaurant_heuristics.json` - Working example heuristics