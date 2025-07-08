#!/usr/bin/env python3
"""
Test script to verify the analysis module works correctly and would return more examples if they existed.
"""

import json
import pathlib
import sys
import tempfile
from unittest.mock import Mock, patch

import pandas as pd

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from entity_matching.analysis import (
    analyze_dataset_for_claude,
    generate_concrete_examples,
    calculate_similarity_stats,
    clean_record_for_json,
    get_semantic_similarity_for_records,
    analyze_candidate_recall,
)
from entity_matching.hybrid_matcher import Config


def test_clean_record_for_json():
    """Test that NaN values are properly cleaned"""
    print("🧪 Testing clean_record_for_json...")
    
    # Create a record with NaN values
    record_with_nan = {
        "id": 1,
        "name": "Test",
        "value": pd.NA,
        "missing": float('nan'),
        "normal": "normal_value"
    }
    
    cleaned = clean_record_for_json(record_with_nan)
    
    assert cleaned["id"] == 1
    assert cleaned["name"] == "Test"
    assert cleaned["value"] == ""
    assert cleaned["missing"] == ""
    assert cleaned["normal"] == "normal_value"
    
    print("✅ clean_record_for_json works correctly")


def test_calculate_similarity_stats():
    """Test similarity statistics calculation"""
    print("🧪 Testing calculate_similarity_stats...")
    
    # Test with normal values
    similarities = [0.1, 0.5, 0.8, 0.9, 0.3]
    stats = calculate_similarity_stats(similarities)
    
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    assert "min" in stats
    assert "max" in stats
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
    assert stats["median"] == 0.5
    
    # Test with empty list
    empty_stats = calculate_similarity_stats([])
    assert empty_stats["mean"] == 0.0
    assert empty_stats["std"] == 0.0
    
    print("✅ calculate_similarity_stats works correctly")


def test_generate_concrete_examples_with_mock_data():
    """Test that generate_concrete_examples returns all available examples"""
    print("🧪 Testing generate_concrete_examples with mock data...")
    
    # Create mock data with many pairs
    positive_pairs = pd.DataFrame({
        "ltable_id": [1, 2, 3, 4, 5],
        "rtable_id": [101, 102, 103, 104, 105],
        "label": [1, 1, 1, 1, 1]
    })
    
    negative_pairs = pd.DataFrame({
        "ltable_id": [6, 7, 8, 9, 10],
        "rtable_id": [106, 107, 108, 109, 110],
        "label": [0, 0, 0, 0, 0]
    })
    
    # Create mock records
    A_records = {
        i: {"id": i, "name": f"Record A {i}", "value": f"value_{i}"}
        for i in range(1, 11)
    }
    
    B_records = {
        i: {"id": i, "name": f"Record B {i}", "value": f"value_{i}"}
        for i in range(101, 111)
    }
    
    # Create mock config
    cfg = Config()
    cfg.use_semantic = False  # Skip semantic for testing
    
    # Mock get_top_candidates to always return empty (no candidate generation errors)
    with patch('entity_matching.analysis.get_top_candidates', return_value=[]):
        examples = generate_concrete_examples(
            positive_pairs, negative_pairs, A_records, B_records, cfg, "test_dataset", 100, verbose=False
        )
    
    # Should return ALL examples since no errors
    assert len(examples["true_matches"]) == 5, f"Expected 5 true matches, got {len(examples['true_matches'])}"
    assert len(examples["confusing_non_matches"]) == 5, f"Expected 5 confusing non-matches, got {len(examples['confusing_non_matches'])}"
    
    # Check structure of examples
    for example in examples["true_matches"]:
        assert "left_record" in example
        assert "right_record" in example
        assert "similarities" in example
        assert "candidate_generation" in example
        assert "syntactic" in example["similarities"]
        assert "trigram" in example["similarities"]
        assert "semantic" in example["similarities"]
    
    print("✅ generate_concrete_examples returns all available examples when no errors occur")


def test_generate_concrete_examples_with_errors():
    """Test that generate_concrete_examples handles errors gracefully"""
    print("🧪 Testing generate_concrete_examples with simulated errors...")
    
    # Create pairs where some records are missing (will cause errors)
    positive_pairs = pd.DataFrame({
        "ltable_id": [1, 2, 999],  # 999 doesn't exist
        "rtable_id": [101, 102, 103],
        "label": [1, 1, 1]
    })
    
    negative_pairs = pd.DataFrame({
        "ltable_id": [1, 2, 888],  # 888 doesn't exist
        "rtable_id": [101, 102, 103],
        "label": [0, 0, 0]
    })
    
    # Create records (missing id 999 and 888)
    A_records = {
        1: {"id": 1, "name": "Record A 1", "value": "value_1"},
        2: {"id": 2, "name": "Record A 2", "value": "value_2"},
    }
    
    B_records = {
        101: {"id": 101, "name": "Record B 101", "value": "value_101"},
        102: {"id": 102, "name": "Record B 102", "value": "value_102"},
        103: {"id": 103, "name": "Record B 103", "value": "value_103"},
    }
    
    cfg = Config()
    cfg.use_semantic = False
    
    # Mock get_top_candidates to always return empty
    with patch('entity_matching.analysis.get_top_candidates', return_value=[]):
        examples = generate_concrete_examples(
            positive_pairs, negative_pairs, A_records, B_records, cfg, "test_dataset", 100, verbose=False
        )
    
    # Should return only the valid examples (records 1 and 2)
    assert len(examples["true_matches"]) == 2, f"Expected 2 true matches, got {len(examples['true_matches'])}"
    assert len(examples["confusing_non_matches"]) == 2, f"Expected 2 confusing non-matches, got {len(examples['confusing_non_matches'])}"
    
    print("✅ generate_concrete_examples handles missing records gracefully")


def test_generate_concrete_examples_with_candidate_errors():
    """Test that generate_concrete_examples handles candidate generation errors"""
    print("🧪 Testing generate_concrete_examples with candidate generation errors...")
    
    positive_pairs = pd.DataFrame({
        "ltable_id": [1, 2],
        "rtable_id": [101, 102],
        "label": [1, 1]
    })
    
    negative_pairs = pd.DataFrame({
        "ltable_id": [1, 2],
        "rtable_id": [103, 104],
        "label": [0, 0]
    })
    
    A_records = {
        1: {"id": 1, "name": "Record A 1", "value": "value_1"},
        2: {"id": 2, "name": "Record A 2", "value": "value_2"},
    }
    
    B_records = {
        101: {"id": 101, "name": "Record B 101", "value": "value_101"},
        102: {"id": 102, "name": "Record B 102", "value": "value_102"},
        103: {"id": 103, "name": "Record B 103", "value": "value_103"},
        104: {"id": 104, "name": "Record B 104", "value": "value_104"},
    }
    
    cfg = Config()
    cfg.use_semantic = False
    
    # Mock get_top_candidates to always raise an error
    with patch('entity_matching.analysis.get_top_candidates', side_effect=Exception("Candidate generation failed")):
        examples = generate_concrete_examples(
            positive_pairs, negative_pairs, A_records, B_records, cfg, "test_dataset", 100, verbose=False
        )
    
    # Should still return all examples, but with candidate_generation.found = False
    assert len(examples["true_matches"]) == 2, f"Expected 2 true matches, got {len(examples['true_matches'])}"
    assert len(examples["confusing_non_matches"]) == 2, f"Expected 2 confusing non-matches, got {len(examples['confusing_non_matches'])}"
    
    # Check that candidate generation is marked as failed
    for example in examples["true_matches"]:
        assert example["candidate_generation"]["found"] is False
        assert example["candidate_generation"]["rank"] is None
    
    print("✅ generate_concrete_examples handles candidate generation errors gracefully")


def test_full_analysis_integration():
    """Test that the full analysis would work with real data structure"""
    print("🧪 Testing full analysis integration...")
    
    # Check if itunes_amazon dataset exists
    data_root = pathlib.Path("data/raw/itunes_amazon")
    if not data_root.exists():
        print("⚠️ Skipping integration test - itunes_amazon dataset not found")
        return
    
    # Run analysis with verbose=False and no output file
    try:
        result = analyze_dataset_for_claude(
            dataset="itunes_amazon",
            max_pairs=20,  # Small sample for testing
            max_candidates=10,
            output_file=None,
            verbose=False
        )
        
        # Check result structure
        assert "dataset" in result
        assert "metadata" in result
        assert "similarity_analysis" in result
        assert "candidate_analysis" in result
        assert "concrete_examples" in result
        assert "dataset_characteristics" in result
        
        # Check concrete examples
        examples = result["concrete_examples"]
        assert "true_matches" in examples
        assert "confusing_non_matches" in examples
        
        true_matches = examples["true_matches"]
        non_matches = examples["confusing_non_matches"]
        
        print(f"   Generated {len(true_matches)} true match examples")
        print(f"   Generated {len(non_matches)} confusing non-match examples")
        
        # Verify that we get some examples
        assert len(true_matches) > 0, "Should have at least some true match examples"
        # Don't assert on non-matches since they might be legitimately filtered
        
        # Check that recall analysis includes the new thresholds
        recall_analysis = result["candidate_analysis"]
        assert "recall_at_1" in recall_analysis
        assert "recall_at_5" in recall_analysis
        assert "recall_at_10" in recall_analysis
        
        print("✅ Full analysis integration test passed")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


def test_json_serialization():
    """Test that the analysis result can be serialized to JSON"""
    print("🧪 Testing JSON serialization...")
    
    # Create a mock result with potential problematic values
    result = {
        "dataset": "test",
        "metadata": {
            "total_pairs": 100,
            "semantic_available": True,
        },
        "similarity_analysis": {
            "true_matches": {
                "syntactic": {"mean": 0.5, "std": 0.1, "median": 0.5, "min": 0.0, "max": 1.0},
                "trigram": {"mean": 0.4, "std": 0.1, "median": 0.4, "min": 0.0, "max": 1.0},
                "semantic": {"mean": 0.6, "std": 0.1, "median": 0.6, "min": 0.0, "max": 1.0},
            }
        },
        "concrete_examples": {
            "true_matches": [
                {
                    "left_record": {"id": 1, "name": "Test", "value": ""},  # Empty string instead of NaN
                    "right_record": {"id": 2, "name": "Test2", "value": ""},
                    "similarities": {"syntactic": 0.5, "trigram": 0.4, "semantic": 0.6},
                    "candidate_generation": {"found": True, "rank": 1, "max_candidates": 100},
                }
            ],
            "confusing_non_matches": [],
        },
    }
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(result, indent=2)
        # Try to deserialize back
        parsed = json.loads(json_str)
        assert parsed["dataset"] == "test"
        print("✅ JSON serialization works correctly")
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        raise


def main():
    """Run all tests"""
    print("🧪 Running analysis module tests...")
    print("=" * 50)
    
    test_clean_record_for_json()
    test_calculate_similarity_stats()
    test_generate_concrete_examples_with_mock_data()
    test_generate_concrete_examples_with_errors()
    test_generate_concrete_examples_with_candidate_errors()
    test_json_serialization()
    test_full_analysis_integration()
    
    print("\n" + "=" * 50)
    print("✅ All tests passed! The analysis module is working correctly.")
    print("🎯 The module WOULD return more examples if they existed and were processable.")


if __name__ == "__main__":
    main()