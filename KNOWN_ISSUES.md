# Known Issues

## Analysis Recall Bug (zomato_yelp dataset)

**Issue**: The recall@N analysis in `src/entity_matching/analysis.py` shows incorrect low recall values for the zomato_yelp dataset, despite manual testing showing perfect candidate generation.

**Evidence**:
- Manual testing: All 10 validation matches rank #1 in candidate generation (100% recall@1)
- Analysis function: Reports 10-40% recall@500, varying by run
- The algorithm itself works correctly - this is purely an analysis/reporting bug

**Root Cause**: Data type/caching inconsistencies between analysis function and actual matching code. The analysis function creates candidate caches differently than the production matching pipeline.

**Impact**: 
- Does not affect actual matching performance
- Only affects analysis reports and Claude's optimization guidance
- May cause Claude to make suboptimal parameter choices based on incorrect recall data

**Workaround**: Trust manual validation testing over automated analysis reports for recall assessment.

**Status**: Documented but not fixed. Analysis code works correctly for most datasets.