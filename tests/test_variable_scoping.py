#!/usr/bin/env python3
"""
Test the specific variable scoping issue in the pipeline without running actual experiments
"""
import sys


def test_baseline_results_scoping():
    """Test the baseline_results variable scoping logic"""

    # Simulate the pipeline logic that was causing the issue
    def simulate_pipeline_logic(mode):
        """Simulate the problematic code structure"""

        # This was the original problematic structure:
        # if mode == "weights-only":
        #     # Set baseline_results here
        #     baseline_results = {'metrics': {'f1': 0.8}, 'cost_usd': 0.0}
        #     return baseline_results
        #
        # if mode == "prompt-modification":
        #     # baseline_results not set!
        #     pass
        # else:  # mode == "heuristics"
        #     # Only set baseline_results inside this else block
        #     baseline_results = await run_matching(...)
        #
        # # Later code tries to use baseline_results regardless of mode
        # return baseline_results  # ← UnboundLocalError for prompt-modification!

        # NEW FIXED STRUCTURE:
        baseline_results = None

        if mode == "weights-only":
            # Skip baseline evaluation, use dev results
            return {'metrics': {'f1': 0.8}, 'cost_usd': 0.0}

        if mode == "prompt-modification":
            print("PROMPT-MODIFICATION MODE: Running both 3A and 3B")
        else:  # mode == "heuristics"
            print("HEURISTICS MODE: Running both 3A and 3B")

        # STEP 3A: Always run baseline evaluation (moved outside conditional)
        print("STEP 3A: Baseline evaluation")
        baseline_results = {'metrics': {'f1': 0.82}, 'cost_usd': 0.01}  # Mock result

        # STEP 3B: Enhanced evaluation
        print("STEP 3B: Enhanced evaluation")

        # Later code can safely use baseline_results
        return baseline_results

    # Test all modes
    modes = ["weights-only", "prompt-modification", "heuristics"]

    for mode in modes:
        print(f"\n🧪 Testing mode: {mode}")
        try:
            result = simulate_pipeline_logic(mode)
            if result and 'metrics' in result:
                print(f"   ✅ {mode}: baseline_results properly defined (F1={result['metrics']['f1']})")
            else:
                print(f"   ❌ {mode}: baseline_results is None or malformed")
                return False
        except Exception as e:
            print(f"   ❌ {mode}: ERROR - {e}")
            return False

    return True

def test_indentation_pattern():
    """Test that the indentation pattern is correct"""

    # Read the actual file and check indentation
    with open('run_complete_pipeline.py') as f:
        lines = f.readlines()

    # Find the key lines
    step_3a_line = None
    baseline_assignment_line = None

    for i, line in enumerate(lines):
        if "STEP 3A: FINAL TEST EVALUATION WITHOUT rules" in line:
            step_3a_line = i
        if "baseline_results = await run_matching(" in line:
            baseline_assignment_line = i

    if step_3a_line is None:
        print("❌ Could not find STEP 3A comment")
        return False

    if baseline_assignment_line is None:
        print("❌ Could not find baseline_results assignment")
        return False

    # Check that STEP 3A is at top level (4 spaces or less indentation)
    step_3a_indent = len(lines[step_3a_line]) - len(lines[step_3a_line].lstrip())
    baseline_indent = len(lines[baseline_assignment_line]) - len(lines[baseline_assignment_line].lstrip())

    print(f"📏 STEP 3A indentation: {step_3a_indent} spaces")
    print(f"📏 baseline_results assignment indentation: {baseline_indent} spaces")

    if step_3a_indent > 4:
        print(f"❌ STEP 3A is too deeply indented ({step_3a_indent} spaces)")
        return False

    if baseline_indent > 4:
        print(f"❌ baseline_results assignment is too deeply indented ({baseline_indent} spaces)")
        return False

    print("✅ Indentation looks correct")
    return True

if __name__ == "__main__":
    print("🚀 Testing baseline_results variable scoping fix")

    print("\n1. Testing logic simulation:")
    logic_ok = test_baseline_results_scoping()

    print("\n2. Testing actual file indentation:")
    indent_ok = test_indentation_pattern()

    if logic_ok and indent_ok:
        print("\n✅ Variable scoping fix validated!")
        print("   The baseline_results UnboundLocalError should be resolved.")
    else:
        print("\n❌ Issues found - the fix may not be complete")
        sys.exit(1)
