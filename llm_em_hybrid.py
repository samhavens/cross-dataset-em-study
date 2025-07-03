#!/usr/bin/env python
import argparse, json, pathlib, textwrap, asyncio, os, datetime, time
from typing import List, Dict
import pandas as pd
from llm_clustering import MODEL_COSTS
import tiktoken
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Configuration
class Config:
    def __init__(self):
        self.model = "gpt-4o-mini"
        self.temperature = 0
        self.max_tokens = 100
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

cfg = Config()

# Token counting
def token_count(text: str) -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(cfg.model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text.split()) * 1.3

def report_cost():
    """Report total cost and token usage"""
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        print(f"WARNING: Model {cfg.model} not found in MODEL_COSTS. Using gpt-4.1-mini instead.")
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS['gpt-4.1-mini']

    input_cost = (cfg.total_input_tokens / 1000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print(f"≈{cfg.total_input_tokens/1000:.1f}K in, {cfg.total_output_tokens/1000:.1f}K out → ${total_cost:.2f}")

MAX = 1_000_000  # token limit for the matching prompt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--limit', type=int, default=None, help='number of test rows')

# Candidate selection - mutually exclusive options
candidate_group = parser.add_mutually_exclusive_group()
candidate_group.add_argument('--max-candidates', type=int, help='maximum candidates per left record (absolute number)')
candidate_group.add_argument('--candidate-ratio', type=float, help='candidates as ratio of table B size (e.g., 0.02 = 2%% of table B)')

parser.add_argument('--model', default=cfg.model)
parser.add_argument('--concurrency', type=int, default=20, help='number of concurrent API calls')
parser.add_argument('--output-json', type=str, help='save detailed results to JSON file')
parser.add_argument('--output-csv', type=str, help='append results to CSV file')
args = parser.parse_args()

# Set default if neither option provided
if args.max_candidates is None and args.candidate_ratio is None:
    DEFAULT_MAX_CANDIDATES = 50
    args.max_candidates = DEFAULT_MAX_CANDIDATES
    print(f"Using default max candidates: {DEFAULT_MAX_CANDIDATES}")

cfg.model = args.model
DATASET = args.dataset

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY environment variable not set")
    print("Set it with: export OPENAI_API_KEY='your-api-key'")
    exit(1)

# Initialize async OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print(f"Using OpenAI API with model: {cfg.model}")

async def call_openai_async(prompt: str) -> str:
    """Make async call to OpenAI API"""
    try:
        response = await client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens
        )

        # Track usage
        usage = response.usage
        cfg.total_input_tokens += usage.prompt_tokens
        cfg.total_output_tokens += usage.completion_tokens

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"OpenAI API error: {e}")


root = pathlib.Path('data')/'raw'/DATASET
A = pd.read_csv(root/'tableA.csv').to_dict(orient='records')
B = pd.read_csv(root/'tableB.csv').to_dict(orient='records')

# Calculate max_candidates based on parameter choice
if args.candidate_ratio is not None:
    if not 0.0 < args.candidate_ratio <= 1.0:
        raise ValueError(f"--candidate-ratio must be between 0.0 and 1.0, got {args.candidate_ratio}")
    max_candidates = max(1, int(len(B) * args.candidate_ratio))
    candidate_method = f"{args.candidate_ratio:.1%} of table B ({max_candidates} candidates)"
else:
    max_candidates = args.max_candidates
    candidate_method = f"{max_candidates} candidates ({max_candidates/len(B):.1%} of table B)"

# load test pairs
pairs = pd.read_csv(root/'test.csv')
if args.limit:
    pairs = pairs.head(args.limit)

def trigram_similarity(s1: str, s2: str) -> float:
    """Calculate trigram similarity between two strings"""
    def get_trigrams(s):
        s = s.lower()
        return set(s[i:i+3] for i in range(len(s)-2))

    t1, t2 = get_trigrams(s1), get_trigrams(s2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def get_top_candidates(left_record: dict, right_records: List[dict], max_candidates: int) -> List[tuple]:
    """Get top candidates for a left record using trigram similarity"""
    left_str = json.dumps(left_record, ensure_ascii=False).lower()

    # Calculate similarity scores
    candidates = []
    for i, right_record in enumerate(right_records):
        right_str = json.dumps(right_record, ensure_ascii=False).lower()
        score = trigram_similarity(left_str, right_str)
        candidates.append((score, i, right_record))

    # Sort by similarity and take top candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(idx, record) for _, idx, record in candidates[:max_candidates]]

async def match_single_record(left_record: dict, candidates: List[tuple]) -> int:
    """Match a single left record against its filtered candidates"""

    # Build prompt with candidates
    candidates_text = "\n".join(
        f"{idx}) {json.dumps(record, ensure_ascii=False)}"
        for idx, record in candidates
    )

    # Create prompt
    prompt = textwrap.dedent(f"""
      You are an expert at entity matching. Your task is to find the candidate that refers to the same real-world entity as the left record.

      Two records match if they refer to the same entity, even if:
      - They have different formatting or spellings
      - One has more/less information than the other
      - They use different abbreviations or representations

      LEFT RECORD:
      {json.dumps(left_record, ensure_ascii=False)}

      CANDIDATES:
      {candidates_text}

      Compare the left record against each candidate. Look for:
      1. Same entity name (allowing for variations in spelling/format)
      2. Matching key identifiers (IDs, codes, etc.)
      3. Consistent attribute values where they overlap
      4. No contradictory information

      Think step by step and identify the candidate that represents the same entity.

      Output format: If you find a match, output ONLY the candidate number (e.g., "1479").
      If no candidate represents the same entity, output "-1".

      ANSWER:
    """)

    # Check token count
    total_tokens = token_count(prompt)
    if total_tokens > MAX:
        print(f"  WARNING: Prompt too large ({total_tokens:,} tokens)")


    # Get LLM response
    response = await call_openai_async(prompt)

    # Parse response
    try:
        match_idx = int(response)
        return match_idx if match_idx != -1 else -1
    except ValueError:
        return -1

async def process_batch(batch_pairs: List[tuple], semaphore: asyncio.Semaphore) -> Dict[int, int]:
    """Process a batch of pairs with concurrency control"""
    async with semaphore:
        tasks = []
        for _, row in batch_pairs:
            left_id = row.ltable_id
            left_record = A[left_id]

            # Get top candidates for this left record
            candidates = get_top_candidates(left_record, B, max_candidates)

            # Create async task for matching
            task = match_single_record(left_record, candidates)
            tasks.append((left_id, task))

        # Execute all tasks in this batch
        batch_results = {}
        for left_id, task in tasks:
            match_idx = await task
            if match_idx != -1:
                batch_results[left_id] = match_idx

        return batch_results

async def main():
    """Main async function"""
    start_time = time.time()

    print(f"Processing {len(pairs)} pairs with {candidate_method} per record")
    print(f"Concurrency: {args.concurrency} parallel requests")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(args.concurrency)

    # Split pairs into batches for progress tracking
    batch_size = max(1, len(pairs) // 20)  # 20 progress updates
    batches = []
    for i in range(0, len(pairs), batch_size):
        batch = list(pairs.iloc[i:i+batch_size].iterrows())
        batches.append(batch)

    # Process batches with progress bar
    all_predictions = {}

    # Process batches with progress tracking
    tasks = [process_batch(batch, semaphore) for batch in batches]
    
    # Use tqdm to track progress
    for task in tqdm(asyncio.as_completed(tasks), total=len(batches), desc="Processing batches", unit="batch"):
        batch_results = await task
        all_predictions.update(batch_results)

    elapsed_time = time.time() - start_time
    matches_found = len(all_predictions)

    print(f"\nCompleted! Found {matches_found} matches out of {len(pairs)} pairs")
    print(f"Processing time: {elapsed_time:.1f} seconds")

    # Evaluate predictions
    preds = []
    labels = []

    for _, rec in pairs.iterrows():
        left_id = rec.ltable_id
        right_id = rec.rtable_id
        true_label = rec.label

        # Check if we predicted a match and if it's correct
        if left_id in all_predictions:
            pred_right_id = all_predictions[left_id]
            pred_label = 1 if pred_right_id == right_id else 0
        else:
            pred_label = 0  # No match predicted

        preds.append(pred_label)
        labels.append(true_label)

    # Calculate metrics
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(preds)

    # Calculate final cost
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        print(f"WARNING: Model {cfg.model} not found in MODEL_COSTS. Using gpt-4o-mini pricing.")
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get('gpt-4o-mini', (0.00015, 0.0006))

    input_cost = (cfg.total_input_tokens / 1000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Dataset: {DATASET}")
    print(f"Model: {cfg.model}")
    print(f"Processed: {len(pairs)} pairs")
    print(f"Candidate selection: {candidate_method}")
    print(f"Predictions made: {len(all_predictions)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    report_cost()

    # Create structured results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": DATASET,
        "model": cfg.model,
        "candidate_method": candidate_method,
        "max_candidates": max_candidates,
        "candidate_ratio": args.candidate_ratio if args.candidate_ratio else max_candidates / len(B),
        "processed_pairs": len(pairs),
        "predictions_made": len(all_predictions),
        "matches_found": matches_found,
        "elapsed_seconds": elapsed_time,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        },
        "cost_usd": total_cost,
        "tokens": {
            "input": cfg.total_input_tokens,
            "output": cfg.total_output_tokens,
            "total": cfg.total_input_tokens + cfg.total_output_tokens
        },
        "table_sizes": {
            "table_a": len(A),
            "table_b": len(B)
        },
        "args": vars(args)
    }

    # Save JSON results if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {args.output_json}")

    # Append to CSV if requested
    if args.output_csv:
        import csv
        import os

        # Flatten the results for CSV
        csv_row = {
            "timestamp": results["timestamp"],
            "dataset": results["dataset"],
            "model": results["model"],
            "candidate_ratio": results["candidate_ratio"],
            "max_candidates": results["max_candidates"],
            "processed_pairs": results["processed_pairs"],
            "predictions_made": results["predictions_made"],
            "matches_found": results["matches_found"],
            "elapsed_seconds": results["elapsed_seconds"],
            "precision": results["metrics"]["precision"],
            "recall": results["metrics"]["recall"],
            "f1": results["metrics"]["f1"],
            "accuracy": results["metrics"]["accuracy"],
            "tp": results["metrics"]["tp"],
            "fp": results["metrics"]["fp"],
            "fn": results["metrics"]["fn"],
            "tn": results["metrics"]["tn"],
            "cost_usd": results["cost_usd"],
            "input_tokens": results["tokens"]["input"],
            "output_tokens": results["tokens"]["output"],
            "total_tokens": results["tokens"]["total"],
            "table_a_size": results["table_sizes"]["table_a"],
            "table_b_size": results["table_sizes"]["table_b"]
        }

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(args.output_csv)

        with open(args.output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

        print(f"Results appended to: {args.output_csv}")

    return results

if __name__ == "__main__":
    results = asyncio.run(main())
    # Results are returned for potential programmatic use