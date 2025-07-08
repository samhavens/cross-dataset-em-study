#!/usr/bin/env python3
"""
Test that syntactic similarity works correctly in hybrid_matcher.

Usage: python scripts/test_syntactic_integration.py
"""

import pathlib
import sys

# Add src to path
sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

from entity_matching.hybrid_matcher import Config, syntactic_similarity, trigram_similarity, triple_similarity


def test_syntactic_similarity():
    """Test syntactic similarity function"""
    print("🧪 Testing syntactic similarity function...")

    # Test cases
    test_cases = [
        ("hello world", "hello world", 1.0),
        ("", "", 0.0),
        ("hello", "", 0.0),
        ("", "hello", 0.0),
        ("The Beatles - Help!", "Beatles - Help!", 0.9),  # Should be high
        ("Song [Explicit]", "Song", 0.8),  # Should handle formatting differences well
    ]

    for s1, s2, expected_min in test_cases:
        similarity = syntactic_similarity(s1, s2)
        print(f"   '{s1}' vs '{s2}' = {similarity:.3f} (expected >= {expected_min})")
        assert similarity >= expected_min, f"Expected >= {expected_min}, got {similarity}"

    print("✅ Syntactic similarity tests passed")


def test_triple_similarity():
    """Test triple similarity function with all three weights"""
    print("\n🧪 Testing triple similarity function...")

    cfg = Config()
    # Verify weights sum to 1.0
    total_weight = cfg.trigram_weight + cfg.syntactic_weight + cfg.semantic_weight
    print(
        f"   Weight sum: {total_weight} (trigram={cfg.trigram_weight}, syntactic={cfg.syntactic_weight}, semantic={cfg.semantic_weight})"
    )
    assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight}"

    # Test with semantic disabled to check fallback
    cfg.use_semantic = False
    similarity = triple_similarity("hello", "hello", cfg)
    print(f"   Without semantic: 'hello' vs 'hello' = {similarity:.3f}")
    assert similarity == 1.0, f"Expected 1.0 for identical strings, got {similarity}"

    # Test with different strings
    similarity = triple_similarity("The Beatles - Help!", "Beatles - Help!", cfg)
    print(f"   Without semantic: 'The Beatles - Help!' vs 'Beatles - Help!' = {similarity:.3f}")

    print("✅ Triple similarity tests passed")


def test_weight_combination():
    """Test that weights are properly combined"""
    print("\n🧪 Testing weight combination...")

    cfg = Config()
    cfg.use_semantic = False  # Disable semantic for predictable results

    s1 = "The Beatles"
    s2 = "Beatles"

    trigram_score = trigram_similarity(s1, s2)
    syntactic_score = syntactic_similarity(s1, s2)

    # When semantic is disabled, should be 50/50 split
    expected_score = 0.5 * trigram_score + 0.5 * syntactic_score
    actual_score = triple_similarity(s1, s2, cfg)

    print(f"   Trigram: {trigram_score:.3f}")
    print(f"   Syntactic: {syntactic_score:.3f}")
    print(f"   Expected combined: {expected_score:.3f}")
    print(f"   Actual combined: {actual_score:.3f}")

    assert abs(actual_score - expected_score) < 0.001, "Weight combination mismatch"

    print("✅ Weight combination tests passed")


def main():
    print("🔬 TESTING SYNTACTIC SIMILARITY INTEGRATION")
    print("=" * 60)

    test_syntactic_similarity()
    test_triple_similarity()
    test_weight_combination()

    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Syntactic similarity integration is working correctly")
    print("✅ Config class supports three-way weights")
    print("✅ Triple similarity function works with weight fallback")


if __name__ == "__main__":
    main()
