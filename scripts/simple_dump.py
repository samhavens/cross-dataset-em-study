#!/usr/bin/env python3
"""
Simple dump of matches for iTunes Amazon to get a vibe.

Usage: python scripts/simple_dump.py
"""

import pathlib

from difflib import SequenceMatcher

import pandas as pd


def get_similarity(text1, text2):
    """Get similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()


def format_record(record, max_len=60):
    """Format a record for display"""
    formatted = []
    for key, value in record.items():
        if key == "id":
            continue
        value_str = str(value)
        if len(value_str) > max_len:
            value_str = value_str[: max_len - 3] + "..."
        formatted.append(f"{key}: {value_str}")
    return " | ".join(formatted)


def main():
    dataset = "itunes_amazon"

    print(f"🔍 SIMPLE ANALYSIS FOR {dataset.upper()}")
    print("=" * 80)

    # Load data
    data_root = pathlib.Path("data/raw") / dataset
    A_df = pd.read_csv(data_root / "tableA.csv")
    B_df = pd.read_csv(data_root / "tableB.csv")
    test_pairs = pd.read_csv(data_root / "test.csv")

    print(f"📊 Dataset: {len(A_df)} records in A, {len(B_df)} records in B")
    print(f"📊 Test pairs: {len(test_pairs)} pairs")
    print(f"📊 Positive pairs: {len(test_pairs[test_pairs.label == 1])} matches")
    print()

    # Convert to records dict
    A_records = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
    B_records = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}

    # Show some positive matches
    positive_pairs = test_pairs[test_pairs.label == 1].head(5)

    print("🎯 POSITIVE MATCHES (Ground Truth)")
    print("-" * 80)

    for i, (_, row) in enumerate(positive_pairs.iterrows(), 1):
        left_id = row.ltable_id
        right_id = row.rtable_id

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Compare song names
        left_song = left_record.get("Song_Name", "")
        right_song = right_record.get("Song_Name", "")
        song_sim = get_similarity(left_song, right_song)

        # Compare artists
        left_artist = left_record.get("Artist_Name", "")
        right_artist = right_record.get("Artist_Name", "")
        artist_sim = get_similarity(left_artist, right_artist)

        print(f"\n{i}. MATCH (IDs: {left_id} → {right_id})")
        print(f"   LEFT:  {format_record(left_record)}")
        print(f"   RIGHT: {format_record(right_record)}")
        print(f"   SIMILARITY: Song={song_sim:.3f}, Artist={artist_sim:.3f}")

    # Show some negative pairs
    negative_pairs = test_pairs[test_pairs.label == 0].head(5)

    print("\n\n🚫 NEGATIVE PAIRS (Non-matches)")
    print("-" * 80)

    for i, (_, row) in enumerate(negative_pairs.iterrows(), 1):
        left_id = row.ltable_id
        right_id = row.rtable_id

        left_record = A_records[left_id]
        right_record = B_records[right_id]

        # Compare song names
        left_song = left_record.get("Song_Name", "")
        right_song = right_record.get("Song_Name", "")
        song_sim = get_similarity(left_song, right_song)

        # Compare artists
        left_artist = left_record.get("Artist_Name", "")
        right_artist = right_record.get("Artist_Name", "")
        artist_sim = get_similarity(left_artist, right_artist)

        print(f"\n{i}. NON-MATCH (IDs: {left_id} → {right_id})")
        print(f"   LEFT:  {format_record(left_record)}")
        print(f"   RIGHT: {format_record(right_record)}")
        print(f"   SIMILARITY: Song={song_sim:.3f}, Artist={artist_sim:.3f}")

    # Show some sample records to understand the data
    print("\n\n🎲 SAMPLE RECORDS")
    print("-" * 80)

    print("SAMPLE FROM TABLE A:")
    sample_a = A_df.head(3)
    for _, row in sample_a.iterrows():
        print(f"   {format_record(row.to_dict())}")

    print("\nSAMPLE FROM TABLE B:")
    sample_b = B_df.head(3)
    for _, row in sample_b.iterrows():
        print(f"   {format_record(row.to_dict())}")

    print("\n🏁 INSIGHTS:")
    print("- Look at similarity scores between matches vs non-matches")
    print("- Notice [Explicit] tags and formatting differences")
    print("- See how dense the song space is")


if __name__ == "__main__":
    main()
