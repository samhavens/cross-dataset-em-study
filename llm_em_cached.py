#!/usr/bin/env python
import argparse
import json
import pathlib
import textwrap

import dspy
import pandas as pd

from llm_clustering import Predictor, cfg, report_cost, token_count

MAX = 1_000_000  # token limit for the matching prompt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--limit", type=int, default=None, help="number of test rows")
parser.add_argument("--batch-size", type=int, default=5, help="number of left records per batch")
parser.add_argument("--model", default=cfg.model)
args = parser.parse_args()

cfg.model = args.model
DATASET = args.dataset

# Configure DSPy - use real API if OPENAI_API_KEY is set, otherwise mock
import os

if os.getenv("OPENAI_API_KEY"):
    cfg.dry_run = False
    # Use real OpenAI API
    dspy.configure(lm=dspy.LM("openai/" + cfg.model))
    print("Using real OpenAI API with prompt caching")
else:
    cfg.dry_run = True  # Use mock predictor
    print("Using mock predictor")

root = pathlib.Path("data") / "raw" / DATASET
A = pd.read_csv(root / "tableA.csv").to_dict(orient="records")
B = pd.read_csv(root / "tableB.csv").to_dict(orient="records")

# load test pairs
pairs = pd.read_csv(root / "test.csv")
if args.limit:
    pairs = pairs.head(args.limit)

print(f"Dataset: {DATASET}")
print(f"TableA: {len(A)} records, TableB: {len(B)} records")
print(f"Test pairs: {len(pairs)}")
print(f"Batch size: {args.batch_size}")

# Prepare all tableB records (this will be cached!)
all_B_listing = [f"B{i}: {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(B)]
all_B_text = "\n".join(all_B_listing)
all_B_tokens = token_count(all_B_text)

print(f"All tableB records: {all_B_tokens:,} tokens (will be cached after first call)")

# CRITICAL: Create the static prefix that will be cached
# Put all the static content (instructions + tableB) at the BEGINNING
static_prefix = textwrap.dedent(f"""
You are performing entity matching between two tables.

Find matches between TABLE_A records and TABLE_B records.
A match means the records refer to the same real-world entity (same beer).

CRITICAL: For each TABLE_A record that has a match in TABLE_B, output EXACTLY this format:
A123 -> B456

Where 123 is the TABLE_A ID and 456 is the TABLE_B ID.
If a TABLE_A record has NO good match in TABLE_B, don't output anything for it.
Do NOT output B records pointing to A records.

TABLE_B (CANDIDATES):
{all_B_text}

""").strip()

static_tokens = token_count(static_prefix)
print(f"Static prefix (cached): {static_tokens:,} tokens")

# Get unique left record IDs from test pairs
test_left_ids = set(pairs["ltable_id"].unique())
print(f"Need to process {len(test_left_ids)} unique left records from test set")

# Process in batches
all_matches = {}  # ltable_id -> rtable_id (predicted matches)

left_ids_list = list(test_left_ids)
for batch_start in range(0, len(left_ids_list), args.batch_size):
    batch_end = min(batch_start + args.batch_size, len(left_ids_list))
    batch_left_ids = left_ids_list[batch_start:batch_end]

    print(f"Processing batch {batch_start // args.batch_size + 1}: left records {batch_start}-{batch_end - 1}")

    # Prepare batch of left records (this is the variable part)
    batch_A_records = [A[lid] for lid in batch_left_ids]
    batch_A_listing = [
        f"A{lid}: {json.dumps(r, ensure_ascii=False)}" for lid, r in zip(batch_left_ids, batch_A_records)
    ]
    batch_A_text = "\n".join(batch_A_listing)

    # Create the full prompt: STATIC PREFIX + VARIABLE BATCH
    # The static prefix should be cached after the first call
    full_prompt = (
        static_prefix
        + f"""

TABLE_A (TO MATCH):
{batch_A_text}

MATCHES (use format "A123 -> B456"):"""
    )

    total_tokens = token_count(full_prompt)
    print(f"  Total tokens: {total_tokens:,} (static: {static_tokens:,}, batch: {total_tokens - static_tokens:,})")

    if total_tokens > MAX * 0.9:
        print("  WARNING: Prompt too large, skipping...")
        continue

    # Send to LLM - the static prefix should be cached after first call
    result = Predictor("text")(full_prompt)
    if hasattr(result, "text"):
        out = result.text.strip()
    elif hasattr(result, "output"):
        out = str(result.output).strip()
    else:
        out = str(result).strip()

    print(f"  LLM response: {out[:200]}{'...' if len(out) > 200 else ''}")

    # Parse the response - handle both A->B and B->A formats
    matches_found = 0
    for line in out.split("\n"):
        line = line.strip()
        if "->" in line:
            try:
                left_part, right_part = line.split("->", 1)
                left_clean = left_part.strip()
                right_clean = right_part.strip()

                # Handle A123 -> B456 format (preferred)
                if left_clean.startswith("A") and right_clean.startswith("B"):
                    left_id = int(left_clean.replace("A", ""))
                    right_id = int(right_clean.replace("B", ""))
                    all_matches[left_id] = right_id
                    matches_found += 1
                    print(f"  Found match: A{left_id} -> B{right_id}")

                # Handle B456 -> A123 format (fix it)
                elif left_clean.startswith("B") and right_clean.startswith("A"):
                    left_id = int(right_clean.replace("A", ""))  # Swap
                    right_id = int(left_clean.replace("B", ""))  # Swap
                    all_matches[left_id] = right_id
                    matches_found += 1
                    print(f"  Found match (corrected): A{left_id} -> B{right_id}")

                # Handle plain numbers
                elif left_clean.isdigit() and right_clean.isdigit():
                    left_id = int(left_clean)
                    right_id = int(right_clean)
                    if left_id in batch_left_ids:  # Verify it's in our current batch
                        all_matches[left_id] = right_id
                        matches_found += 1
                        print(f"  Found match (inferred): A{left_id} -> B{right_id}")

            except (ValueError, IndexError) as e:
                print(f"  Failed to parse line: '{line}' - {e}")

    print(f"  Found {matches_found} matches in this batch")

print(f"\nFound {len(all_matches)} total matches")

# Evaluate against test pairs
preds = []
labels = []
for _, rec in pairs.iterrows():
    left_id = rec.ltable_id
    true_right_id = rec.rtable_id

    # Check if we predicted a match for this left record
    if left_id in all_matches:
        predicted_right_id = all_matches[left_id]
        pred = 1 if predicted_right_id == true_right_id else 0
    else:
        pred = 0  # No match predicted

    preds.append(pred)
    labels.append(rec.label)

report_cost()


# Calculate evaluation metrics
def calculate_metrics(predictions, ground_truth):
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if p == 0 and g == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


metrics = calculate_metrics(preds, labels)

print("\n=== RESULTS ===")
print(f"Dataset: {DATASET}")
print(f"Processed: {len(preds)} pairs")
print(f"Found matches: {len([p for p in preds if p == 1])}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")

print("\n=== CACHING INFO ===")
print(f"Static prefix: {static_tokens:,} tokens (cached after first call)")
print(f"Expected savings: ~{static_tokens * 0.8:,.0f} tokens per batch after first call")
print("Cache should last: 5-10 minutes (up to 1 hour during off-peak)")
