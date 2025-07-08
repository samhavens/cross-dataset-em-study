#!/usr/bin/env python3
"""
Quick test to verify JSON serialization fixes work.
"""

import json
import pathlib
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from scripts.analyze_for_claude import clean_record_for_json


def test_nan_handling():
    """Test that NaN values are properly handled"""
    
    # Create test record with NaN
    test_record = {
        "id": 1,
        "Song_Name": "Test Song",
        "Artist_Name": "Test Artist",
        "Released": np.nan,  # This should be problematic
        "Price": "Album Only"
    }
    
    print("🧪 Testing NaN handling...")
    print(f"Original record: {test_record}")
    
    # Clean the record
    cleaned = clean_record_for_json(test_record)
    print(f"Cleaned record: {cleaned}")
    
    # Test JSON serialization
    try:
        json_str = json.dumps(cleaned)
        print(f"✅ JSON serialization successful: {json_str}")
        
        # Test deserialization
        parsed = json.loads(json_str)
        print(f"✅ JSON parsing successful: {parsed}")
        
    except Exception as e:
        print(f"❌ JSON error: {e}")
        return False
    
    # Verify NaN was replaced
    if cleaned["Released"] == "":
        print("✅ NaN successfully replaced with empty string")
        return True
    else:
        print(f"❌ NaN not properly replaced: {cleaned['Released']}")
        return False


def test_dataframe_nan():
    """Test NaN handling with DataFrame"""
    
    # Create test DataFrame with NaN
    df = pd.DataFrame({
        "id": [1, 2],
        "Song_Name": ["Song 1", "Song 2"],
        "Released": [2020, np.nan]  # Second record has NaN
    })
    
    print("\n🧪 Testing DataFrame NaN handling...")
    print("Original DataFrame:")
    print(df)
    
    # Convert to records and clean
    records = {}
    for _, row in df.iterrows():
        record_dict = row.to_dict()
        cleaned = clean_record_for_json(record_dict)
        records[row["id"]] = cleaned
    
    print(f"Cleaned records: {records}")
    
    # Test JSON serialization of all records
    try:
        json_str = json.dumps(records, indent=2)
        print(f"✅ Full records JSON serialization successful")
        return True
    except Exception as e:
        print(f"❌ Full records JSON error: {e}")
        return False


def main():
    print("🔬 TESTING JSON SERIALIZATION FIXES")
    print("=" * 50)
    
    test1 = test_nan_handling()
    test2 = test_dataframe_nan()
    
    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ NaN handling is working correctly")
        print("✅ JSON serialization is fixed")
    else:
        print("\n💥 SOME TESTS FAILED!")


if __name__ == "__main__":
    main()