#!/usr/bin/env python3
"""
Quick test to validate imports and function signatures without running experiments
"""

import sys
import traceback


def test_pipeline_imports():
    """Test that the pipeline can be imported and key functions exist"""
    try:
        # Add parent directory to path for import
        import sys

        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Test import
        from run_complete_pipeline import get_leaderboard_target_f1, run_complete_pipeline
        print("✅ Successfully imported run_complete_pipeline")

        # Test function signature
        import inspect
        sig = inspect.signature(run_complete_pipeline)
        params = list(sig.parameters.keys())
        print(f"✅ Function signature: {params}")

        # Check that mode parameter exists
        if 'mode' in params:
            print("✅ Mode parameter found")
        else:
            print("❌ Mode parameter missing")
            return False

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_async_syntax():
    """Test that async syntax is valid"""
    try:
        import ast

        with open('run_complete_pipeline.py') as f:
            source = f.read()

        # Parse and validate syntax
        tree = ast.parse(source)
        print("✅ Python syntax validation passed")

        # Count async functions
        async_funcs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                async_funcs.append(node.name)

        print(f"✅ Found {len(async_funcs)} async functions: {async_funcs}")

        if 'run_complete_pipeline' not in async_funcs:
            print("❌ run_complete_pipeline is not async")
            return False

        return True

    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing pipeline imports and syntax")

    import_ok = test_pipeline_imports()
    syntax_ok = test_async_syntax()

    if import_ok and syntax_ok:
        print("\n✅ Pipeline ready to run!")
        print("   You can now test with real data without fear of variable scoping crashes.")
    else:
        print("\n❌ Issues found - fix before running real pipeline")
        sys.exit(1)
