#!/usr/bin/env python
"""
build a 10-token 'mini-key' for every row in tableB using gpt-4.1-nano.
writes data/<dataset>_llmkeys.pkl
cost: ~35 tokens per row ⇒ $0.03 per 1 000 rows.
"""

import argparse, json, pathlib, pickle, textwrap, asyncio, concurrent.futures
import pandas as pd
import dspy
from llm_clustering import Predictor, cfg, token_count, report_cost
from tqdm import tqdm

BATCH = 40  # Back to larger batches
MAX_WORKERS = 5  # Parallel requests

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--model", default=cfg.model)
args = p.parse_args()
cfg.model = args.model

# Configure DSPy - use real API if OPENAI_API_KEY is set, otherwise mock
import os
if os.getenv("OPENAI_API_KEY"):
    cfg.dry_run = False
    # Use real OpenAI API
    dspy.configure(lm=dspy.LM("openai/" + cfg.model))
else:
    cfg.dry_run = True  # Use mock predictor
    dspy.configure(lm=dspy.LM("mock"))

root = pathlib.Path("data")/"raw"/args.dataset
tbl = pd.read_csv(root/"tableB.csv")
rows = tbl.to_dict(orient="records")


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def process_batch(offset, chunk):
    """Process a single batch of records."""
    listing = [f"{i}) {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(chunk)]
    records_text = '\n'.join(listing)
    prompt = textwrap.dedent(f"""
      create a *concise*, unique key (≤10 tokens, lowercase, hyphens ok)
      that would help match each record to itself later. respond as JSON
      mapping row number to key.
      records:
      {records_text}
    """)
    
    # Retry logic for API errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = Predictor("json")(prompt)
            mapping = response.output
            break
        except Exception as e:
            print(f"\nBatch {offset}, attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print(f"Skipping batch {offset} after {max_retries} attempts")
                return {}
            import time
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Handle case where output is a string that needs JSON parsing
    if isinstance(mapping, str):
        try:
            import json as _json
            mapping = _json.loads(mapping)
        except Exception as e:
            print(f"Failed to parse JSON, trying to fix: {e}")
            # Try to fix common JSON issues
            fixed_text = mapping
            # Fix unquoted keys like: 8: "value" -> "8": "value"
            import re
            fixed_text = re.sub(r'(\n\s*)(\d+):\s*', r'\1"\2": ', fixed_text)
            try:
                mapping = _json.loads(fixed_text)
                print("Successfully fixed JSON!")
            except Exception as e2:
                print(f"Could not fix JSON: {e2}")
                return {}
    
    # Convert to global indices
    batch_results = {}
    for local_idx, key in mapping.items():
        global_idx = offset * BATCH + int(local_idx)
        batch_results[global_idx] = {"key": key, "row": chunk[int(local_idx)]}
    
    return batch_results

# Check for existing progress
checkpoint_file = f"data/{args.dataset}_llmkeys_checkpoint.pkl"
if pathlib.Path(checkpoint_file).exists():
    print(f"Resuming from checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        out = pickle.load(f)
    completed_batches = len(out) // BATCH
else:
    out = {}
    completed_batches = 0

chunks_list = list(chunks(rows, BATCH))
total_batches = len(chunks_list)

print(f"Processing {len(rows)} records in {total_batches} batches (batch size: {BATCH})")
if completed_batches > 0:
    print(f"Resuming from batch {completed_batches}/{total_batches}")

# Process with progress bar
for offset, chunk in enumerate(tqdm(chunks_list[completed_batches:], 
                                   desc="Building keys", 
                                   initial=completed_batches,
                                   total=total_batches),
                                 start=completed_batches):
    
    batch_results = process_batch(offset, chunk)
    out.update(batch_results)
    
    # Checkpoint every 10 batches
    if (offset + 1) % 10 == 0:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(out, f)

# Final save
pickle.dump(out, open(f"data/{args.dataset}_llmkeys.pkl", "wb"))

# Clean up checkpoint file
if pathlib.Path(checkpoint_file).exists():
    pathlib.Path(checkpoint_file).unlink()

print("wrote", len(out), "keys, total prompt tokens:",
      sum(token_count(json.dumps(x["row"])) for x in out.values()))
report_cost()
