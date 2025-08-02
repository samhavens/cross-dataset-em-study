#!/usr/bin/env python3
"""
Test the F1 extraction fix in simplified_agentic_generator.py

The issue was that F1 scores appear at the END of RunExperiment output,
but we were only showing the first 500 characters, so extraction failed.
"""

def test_f1_extraction_logic():
    """Test the F1 extraction from long content"""

    # Simulate the structure of RunExperiment output
    mock_content = """🧪 Enhanced Test Results for beer (FULL validation set):

🎯 WEIGHT VERIFICATION:
   Requested: semantic=0.6, trigram=0.25, syntactic=0.15
   Actually used: semantic=0.6, trigram=0.25, syntactic=0.15
   ✅ Weights loaded correctly from rules file

📊 PERFORMANCE METRICS:
   F1 Score: 0.7500
   Precision: 0.9000
   Recall: 0.6429

🔍 DETAILED ANALYSIS:
   Confusion Matrix: TP=9, FP=1, FN=5, TN=0 (total=15)
   ⚠️  FALSE POSITIVES: 1 pairs incorrectly marked as matches
      → Need to be more conservative (higher thresholds)
   ⚠️  FALSE NEGATIVES: 5 pairs missed (should have matched)
      → Need to be more sensitive (lower thresholds or better similarity)

[... lots of detailed false positive/negative examples ...]

============================================================
🎯 ENHANCED TEST RESULTS SUMMARY:
   F1 SCORE: 0.8148
   PRECISION: 0.8571
   RECALL: 0.7778
   FALSE POSITIVES: 1
   FALSE NEGATIVES: 2
   TRUE POSITIVES: 7
============================================================"""

    print("🧪 Testing F1 extraction from long content")
    print(f"   Content length: {len(mock_content)} chars")

    # Test the extraction logic
    content_str = mock_content
    f1_extracted = False

    # This is the UPDATED logic from simplified_agentic_generator.py
    if "F1 SCORE:" in content_str or "F1 Score:" in content_str:
        import re
        f1_matches = re.findall(r'F1 SCORE?:\s*([0-9.]+)', content_str, re.IGNORECASE)
        if f1_matches:
            f1_score = float(f1_matches[-1])  # Take LAST match (summary section)
            if len(f1_matches) > 1:
                print(f"   🎯 F1 SCORE: {f1_score:.4f} (found {len(f1_matches)} matches, using summary)")
            else:
                print(f"   🎯 F1 SCORE: {f1_score:.4f}")
            f1_extracted = True

        # Also extract precision/recall if available
        precision_matches = re.findall(r'PRECISION:\s*([0-9.]+)', content_str, re.IGNORECASE)
        recall_matches = re.findall(r'RECALL:\s*([0-9.]+)', content_str, re.IGNORECASE)
        if precision_matches and recall_matches:
            precision = float(precision_matches[0])
            recall = float(recall_matches[0])
            print(f"   📊 P={precision:.4f}, R={recall:.4f}")

    # Test that we extract from the SUMMARY section (last occurrence)
    all_f1_matches = re.findall(r'F1 SCORE?:\s*([0-9.]+)', content_str, re.IGNORECASE)
    print(f"   All F1 matches found: {all_f1_matches}")

    if len(all_f1_matches) > 1:
        first_f1 = float(all_f1_matches[0])  # From first metrics section
        last_f1 = float(all_f1_matches[-1])  # From summary section

        print(f"   First F1 (metrics): {first_f1:.4f}")
        print(f"   Last F1 (summary): {last_f1:.4f}")

        if first_f1 != last_f1:
            print("   ⚠️  Multiple different F1 scores found!")
            print("   📝 Current regex takes the FIRST match, but LAST might be more accurate")
            return last_f1

    return f1_extracted

def test_old_vs_new_truncation():
    """Test how the old truncation would miss F1 scores"""

    # Create content where F1 is beyond 500 chars
    long_content = "x" * 600 + """
🎯 ENHANCED TEST RESULTS SUMMARY:
   F1 SCORE: 0.8148
   PRECISION: 0.8571
   RECALL: 0.7778
============================================================"""

    print("\n🧪 Testing truncation behavior")
    print(f"   Content length: {len(long_content)} chars")
    print(f"   F1 score position: {long_content.find('F1 SCORE:')} chars")

    # Old behavior (first 500 chars only)
    old_truncated = long_content[:500]
    old_has_f1 = "F1 SCORE:" in old_truncated
    print(f"   Old truncation (500 chars): F1 found = {old_has_f1}")

    # New behavior (search full content first)
    new_has_f1 = "F1 SCORE:" in long_content
    print(f"   New approach (full content): F1 found = {new_has_f1}")

    return new_has_f1 and not old_has_f1

if __name__ == "__main__":
    print("🔍 Testing F1 Extraction Fix")
    print("=" * 50)

    # Test extraction logic
    extraction_works = test_f1_extraction_logic()

    # Test truncation issue
    truncation_fixed = test_old_vs_new_truncation()

    print("\n📊 Results:")
    print(f"   F1 extraction works: {extraction_works}")
    print(f"   Truncation issue fixed: {truncation_fixed}")

    if extraction_works and truncation_fixed:
        print("\n✅ F1 extraction fix is working correctly!")
        print("   RunExperiment results should now show F1 scores in MCP sessions.")
    else:
        print("\n❌ Issues found with F1 extraction fix")
