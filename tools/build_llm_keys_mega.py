#!/usr/bin/env python
"""
Build ALL keys for a dataset in 1-5 API calls using the full context window.
Much faster and simpler than batching!
"""

import argparse
import json
import pathlib
import pickle
import textwrap

import dspy
import pandas as pd

from tqdm import tqdm

from llm_clustering import Predictor, cfg, report_cost, token_count

# Use the full context window!
MAX_TOKENS = 800_000  # Conservative limit for gpt-4.1-nano
CHUNK_SIZE = 1000  # Records per chunk if we need multiple calls

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--model", default=cfg.model)
args = p.parse_args()
cfg.model = args.model

# Configure DSPy - use real API if OPENAI_API_KEY is set, otherwise mock
import os

if os.getenv("OPENAI_API_KEY"):
    cfg.dry_run = False
    # Configure with higher max_tokens for long responses
    lm = dspy.LM("openai/" + cfg.model, max_tokens=16000)
    dspy.configure(lm=lm)
else:
    cfg.dry_run = True
    dspy.configure(lm=dspy.LM("mock"))

root = pathlib.Path("data") / "raw" / args.dataset
tbl = pd.read_csv(root / "tableB.csv")
rows = tbl.to_dict(orient="records")

print(f"Processing {len(rows)} records for {args.dataset}")


def create_mega_prompt(records, start_idx=0):
    """Create one massive prompt for all records."""
    listing = [f"{start_idx + i}) {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(records)]
    records_text = "\n".join(listing)

    return textwrap.dedent(f"""
      Create a *concise*, unique key (≤10 tokens, lowercase, hyphens ok)
      for each record that would help match it to itself later.

      Respond ONLY with valid JSON mapping record number to key. No other text.

      Records:
      {records_text}
    """)


def chunk_by_tokens(records, max_tokens):
    """Split records into chunks that fit in token limit."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for record in records:
        record_tokens = token_count(json.dumps(record))

        if current_tokens + record_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [record]
            current_tokens = record_tokens
        else:
            current_chunk.append(record)
            current_tokens += record_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# Estimate total tokens
total_tokens = sum(token_count(json.dumps(row)) for row in rows)
print(f"Estimated tokens: {total_tokens:,}")

if total_tokens < MAX_TOKENS:
    # Single API call!
    print("✨ Doing it all in ONE API call!")
    prompt = create_mega_prompt(rows)
    response = Predictor("json")(prompt)
    mapping = response.output

    # Handle string response with robust JSON parsing
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Response length: {len(mapping)}")
            print(f"Response end: ...{mapping[-200:]}")

            # Try to fix truncated JSON
            if not mapping.strip().endswith("}"):
                print("Detected truncated response, trying to fix...")
                # Find the last complete entry
                lines = mapping.strip().split("\n")
                fixed_lines = []
                for line in lines:
                    if ":" in line and (line.strip().endswith(",") or line.strip().endswith('"')):
                        fixed_lines.append(line)
                    else:
                        break

                if fixed_lines:
                    # Remove trailing comma and close JSON
                    if fixed_lines[-1].strip().endswith(","):
                        fixed_lines[-1] = fixed_lines[-1].rstrip(",")
                    fixed_json = "{\n" + "\n".join(fixed_lines) + "\n}"
                    mapping = json.loads(fixed_json)
                    print(f"Fixed! Recovered {len(mapping)} entries")
                else:
                    raise e
            else:
                raise e

    out = {}
    for idx_str, key in mapping.items():
        idx = int(idx_str)
        out[idx] = {"key": key, "row": rows[idx]}

else:
    # Split into efficient chunks
    chunks = chunk_by_tokens(rows, MAX_TOKENS)
    print(f"Splitting into {len(chunks)} efficient chunks")

    out = {}
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        start_idx = sum(len(chunks[j]) for j in range(i))
        prompt = create_mega_prompt(chunk, start_idx)

        response = Predictor("json")(prompt)
        mapping = response.output

        if isinstance(mapping, str):
            mapping = json.loads(mapping)

        for idx_str, key in mapping.items():
            idx = int(idx_str)
            out[idx] = {"key": key, "row": rows[idx]}

# Save results
pickle.dump(out, open(f"data/{args.dataset}_llmkeys.pkl", "wb"))
print(f"✅ Wrote {len(out)} keys for {args.dataset}")
report_cost()
