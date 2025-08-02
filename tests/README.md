# Test Suite

Quick tests to validate pipeline behavior without waiting for full experiments.

## Tests

### `test_variable_scoping.py`
Tests the baseline_results variable scoping fix that was causing crashes in prompt-modification mode.
- **Issue**: `baseline_results` was undefined in prompt-modification mode due to incorrect indentation
- **Fix**: Moved baseline evaluation (Step 3A) to top level so it runs for all modes
- **Usage**: `python tests/test_variable_scoping.py`

### `test_pipeline_imports.py` 
Tests that the pipeline can be imported and has correct function signatures.
- **Validates**: Import statements, async syntax, function parameters
- **Usage**: `python tests/test_pipeline_imports.py`

### `test_recall_analysis_duplicates.py`
Identifies the duplicate-awareness issue in recall@N candidate analysis.
- **Issue**: Current recall analysis doesn't use duplicate-aware evaluation
- **Impact**: Underestimates recall when candidates contain duplicates of ground truth
- **Usage**: `python tests/test_recall_analysis_duplicates.py`

### `test_f1_extraction_fix.py`
Tests the F1 score extraction fix in the simplified agentic generator.
- **Issue**: F1 scores appear at end of RunExperiment output but were truncated at 500 chars
- **Fix**: Extract F1 scores from full content before truncation, use last match (summary section)
- **Usage**: `python tests/test_f1_extraction_fix.py`

### `test_run_experiment_output.py`
Tests RunExperiment tool output format and F1 score extraction patterns.
- **Fast**: Pattern matching tests (always runs in test suite)
- **Slow**: Full RunExperiment tool test (run with `--full` flag)
- **Usage**: `python tests/test_run_experiment_output.py` or `python tests/test_run_experiment_output.py --full`

## Quick Testing Workflow

1. **Before making changes**: Run all tests to establish baseline
2. **After making changes**: Re-run tests to validate fixes
3. **Before long experiments**: Use tests to catch issues early

```bash
# Run all tests with summary (recommended)
python tests/run_all_tests.py

# Or individually
python tests/test_variable_scoping.py
python tests/test_pipeline_imports.py
python tests/test_recall_analysis_duplicates.py
```

### `run_all_tests.py`
Test runner that executes all tests and provides a summary.
- **Usage**: `python tests/run_all_tests.py`
- **Output**: Shows pass/fail status and detailed error info for failed tests

## Additional Quick Testing Options

- **Early Exit**: `python run_complete_pipeline.py beer --early-exit --mode prompt-modification`
- **Weights Only**: `python run_complete_pipeline.py beer --mode weights-only` (fastest)
- **Small Datasets**: Test on datasets with fewer records first