#!/usr/bin/env python3
"""
Test script to demonstrate the batching performance improvement.
"""

def analyze_batching_performance():
    """Analyze the batching performance characteristics"""

    # Sample dataset sizes
    test_sizes = [100, 500, 1000, 5000, 10000]
    concurrency = 20

    print("🧪 Batching Performance Analysis")
    print("=" * 60)
    print(f"Concurrency: {concurrency}")
    print()

    for size in test_sizes:
        # Old approach (fixed small batches)
        old_batch_size = concurrency
        old_num_batches = (size + old_batch_size - 1) // old_batch_size  # Ceiling division

        # New approach (dynamic batching)
        new_batch_size = max(1, size // 20)
        new_num_batches = (size + new_batch_size - 1) // new_batch_size  # Ceiling division

        print(f"Dataset Size: {size:,} pairs")
        print(f"  Old approach: {old_num_batches:,} batches of {old_batch_size} pairs each")
        print(f"  New approach: {new_num_batches:,} batches of {new_batch_size} pairs each")
        print(f"  Improvement: {old_num_batches / new_num_batches:.1f}x fewer batches")
        print()

    print("💡 Key Benefits:")
    print("  • Fewer sequential batch operations")
    print("  • Better API utilization")
    print("  • Scales efficiently with dataset size")
    print("  • Same concurrency control as baseline")


if __name__ == "__main__":
    analyze_batching_performance()
