#!/usr/bin/env python3
"""Test script to reproduce the slice indices error"""

def test_slice_error():
    """Reproduce the slice indices error"""
    
    # Simulate the problematic code
    max_candidates = 50
    
    # This is what happens in the code - produces float values
    trigram_candidates_fast = max_candidates * 1.5  # 75.0
    trigram_candidates_normal = max_candidates * 3  # 150.0
    
    print(f"max_candidates: {max_candidates} (type: {type(max_candidates)})")
    print(f"trigram_candidates_fast: {trigram_candidates_fast} (type: {type(trigram_candidates_fast)})")
    print(f"trigram_candidates_normal: {trigram_candidates_normal} (type: {type(trigram_candidates_normal)})")
    
    # Create a test list
    test_list = list(range(100))
    
    print("\nTesting slicing with float indices:")
    
    try:
        # This should fail with "slice indices must be integers or None or have an __index__ method"
        result = test_list[:trigram_candidates_fast]
        print(f"Fast analysis slice worked: {len(result)} items")
    except TypeError as e:
        print(f"Fast analysis slice failed: {e}")
    
    try:
        # This should also fail
        result = test_list[:trigram_candidates_normal]
        print(f"Normal analysis slice worked: {len(result)} items")
    except TypeError as e:
        print(f"Normal analysis slice failed: {e}")
    
    print("\nTesting slicing with integer indices:")
    
    try:
        # This should work
        result = test_list[:int(trigram_candidates_fast)]
        print(f"Fast analysis slice (int) worked: {len(result)} items")
    except TypeError as e:
        print(f"Fast analysis slice (int) failed: {e}")
    
    try:
        # This should work
        result = test_list[:int(trigram_candidates_normal)]
        print(f"Normal analysis slice (int) worked: {len(result)} items")
    except TypeError as e:
        print(f"Normal analysis slice (int) failed: {e}")

if __name__ == "__main__":
    test_slice_error()