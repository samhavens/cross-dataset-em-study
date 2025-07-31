"""
Balanced sampling utility to prevent dangerous all-positive or all-negative slices.
"""

import pandas as pd


def balanced_train_sample(train_pairs: pd.DataFrame, target_size: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced sample from training pairs to avoid all-positive or all-negative slices.

    Args:
        train_pairs: DataFrame with 'label' column (0/1)
        target_size: Target total sample size (will be split equally between positive/negative)
        random_state: Random seed for reproducible sampling

    Returns:
        Balanced DataFrame with approximately target_size pairs

    Critical: Prevents Claude optimization from "freaking out" due to unreliable F1 signals
    caused by all-positive or all-negative evaluation sets.
    """
    positive_pairs = train_pairs[train_pairs['label'] == 1]
    negative_pairs = train_pairs[train_pairs['label'] == 0]

    # Split target size equally between positive and negative
    max_per_class = target_size // 2

    # Take up to max_per_class of each, limited by available data
    pos_sample = positive_pairs.head(min(max_per_class, len(positive_pairs)))
    neg_sample = negative_pairs.head(min(max_per_class, len(negative_pairs)))

    # Combine and shuffle for randomness
    return pd.concat([pos_sample, neg_sample]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)



def get_sample_info(balanced_pairs: pd.DataFrame, source_description: str = "sample") -> str:
    """Get a human-readable description of the balanced sample."""
    pos_count = sum(balanced_pairs['label'] == 1)
    neg_count = sum(balanced_pairs['label'] == 0)
    total_count = len(balanced_pairs)

    return f"✅ Using balanced {source_description}: {pos_count} positive + {neg_count} negative = {total_count} total pairs (no test data leakage)"
