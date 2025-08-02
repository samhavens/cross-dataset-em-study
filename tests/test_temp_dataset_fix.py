#!/usr/bin/env python3
"""
Test the temp dataset creation fix.

The issue was that temp datasets only had test.csv, but run_enhanced_matching
with use_validation=True expects valid.csv or train.csv to exist.
"""

import asyncio
import pathlib
import shutil
import sys


def test_temp_dataset_creation():
    """Test that temp datasets are created with proper file structure"""

    # Add parent directory to path
    parent_dir = pathlib.Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    # Import after adding to path
    from run_complete_pipeline import run_dev_only_analysis_with_params

    print("🧪 Testing temp dataset creation fix")

    # Test parameters
    params = {
        "max_candidates": 50,
        "semantic_weight": 0.6,
        "trigram_weight": 0.25,
        "syntactic_weight": 0.15,
        "use_semantic": True
    }

    try:
        # This should create a temp dataset without failing
        print("📁 Creating temp dataset for beer...")
        asyncio.run(run_dev_only_analysis_with_params("beer", params))

        # Check that temp dataset exists with proper files
        temp_path = pathlib.Path("data/raw/temp_beer_dev_temp")
        if temp_path.exists():
            print(f"✅ Temp dataset created: {temp_path}")

            files = list(temp_path.glob("*.csv"))
            file_names = [f.name for f in files]
            print(f"📄 Files in temp dataset: {file_names}")

            # Check for required files
            required_files = ["tableA.csv", "tableB.csv", "test.csv", "train.csv"]
            missing_files = [f for f in required_files if f not in file_names]

            if not missing_files:
                print("✅ All required files present")

                # Check that train.csv has data
                import pandas as pd
                train_df = pd.read_csv(temp_path / "train.csv")
                print(f"📊 train.csv has {len(train_df)} rows")

                if len(train_df) > 0:
                    print("✅ train.csv contains data")
                    return True
                print("❌ train.csv is empty")
                return False
            print(f"❌ Missing required files: {missing_files}")
            return False
        print("❌ Temp dataset was not created")
        return False

    except Exception as e:
        print(f"❌ Error creating temp dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temp dataset
        temp_path = pathlib.Path("data/raw/temp_beer_dev_temp")
        if temp_path.exists():
            shutil.rmtree(temp_path)
            print("🧹 Cleaned up temp dataset")

if __name__ == "__main__":
    success = test_temp_dataset_creation()

    if success:
        print("\n✅ Temp dataset creation fix is working!")
        print("   The pipeline should no longer fail with 'No validation or training data available'")
    else:
        print("\n❌ Temp dataset creation fix failed")
        sys.exit(1)
