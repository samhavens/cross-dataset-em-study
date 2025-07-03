#!/usr/bin/env python
import argparse, json, pathlib, pickle, difflib, textwrap
from typing import List, Dict
import pandas as pd
import dspy
from llm_clustering import Predictor, cfg, token_count, report_cost

MAX = 1_000_000  # token limit for the matching prompt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--limit', type=int, default=None, help='number of test rows')
parser.add_argument('--model', default=cfg.model)
args = parser.parse_args()

cfg.model = args.model
DATASET = args.dataset

# Configure DSPy - use real API if OPENAI_API_KEY is set, otherwise mock
import os
if os.getenv("OPENAI_API_KEY"):
    cfg.dry_run = False
    print("Using real OpenAI API")
    # Use real OpenAI API
    dspy.configure(lm=dspy.LM("openai/" + cfg.model))
else:
    cfg.dry_run = True  # Use mock predictor
    # For mock mode, we don't need to configure DSPy LM since we're using custom Predictor

root = pathlib.Path('data')/'raw'/DATASET
A = pd.read_csv(root/'tableA.csv').to_dict(orient='records')
B = pd.read_csv(root/'tableB.csv').to_dict(orient='records')

# load test pairs
pairs = pd.read_csv(root/'test.csv')
if args.limit:
    pairs = pairs.head(args.limit)

# load precomputed keys
KEYS = pickle.load(open(f"data/{DATASET}_llmkeys.pkl", 'rb'))
index: Dict[str, List[int]] = {}
for rid, obj in KEYS.items():
    index.setdefault(obj['key'], []).append(rid)

def llm_key(record: dict) -> str:
    """one nano call to turn a left record into its 10-token key"""
    prompt = f"give a 10-token matching key for: {json.dumps(record)}"
    result = Predictor('text')(prompt)
    # Handle both our custom response objects and DSPy Prediction objects
    if hasattr(result, 'text'):
        return result.text.strip()
    elif hasattr(result, 'output'):
        return str(result.output).strip()
    else:
        return str(result).strip()

def trigram_set(s: str, n: int = 3) -> set:
    return {s[i:i+n] for i in range(len(s)-n+1)} if len(s) >= n else {s}

def trigram_sim(a: str, b: str) -> float:
    na, nb = trigram_set(a.lower()), trigram_set(b.lower())
    if not na or not nb:
        return 0.0
    return len(na & nb)/len(na | nb)

def top50_trigram_fallback(left: dict) -> List[dict]:
    left_s = json.dumps(left, ensure_ascii=False)
    scores = [(trigram_sim(left_s, json.dumps(r, ensure_ascii=False)), r) for r in B]
    scores.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scores[:50]]

preds = []
labels = []
for _, rec in pairs.iterrows():
    left = A[rec.ltable_id]
    left_k = llm_key(left)

    bucket_ids = set()
    for k in (left_k,):
        bucket_ids.update(index.get(k, []))
    for k in index:
        if abs(len(k) - len(left_k)) <= 2 and difflib.SequenceMatcher(None, k, left_k).ratio() >= 0.85:
            bucket_ids.update(index[k])
    rights_block = [KEYS[i]['row'] for i in bucket_ids]

    if not rights_block or token_count('\n'.join(map(json.dumps, rights_block))) > MAX:
        rights_block = top50_trigram_fallback(left)

    listing = [f"{i}) {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(rights_block)]
    prompt = textwrap.dedent(f"""
      left record: {json.dumps(left, ensure_ascii=False)}
      choose the best match id from the list or -1 if none.
      rights:
      {'\n'.join(listing)}
    """)
    result = Predictor('text')(prompt)
    # Handle both our custom response objects and DSPy Prediction objects
    if hasattr(result, 'text'):
        out = result.text.strip()
    elif hasattr(result, 'output'):
        out = str(result.output).strip()
    else:
        out = str(result).strip()
    try:
        idx = int(out)
    except ValueError:
        idx = -1
    pred = 1 if idx >= 0 and rights_block[idx] == B[rec.rtable_id] else 0
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
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

metrics = calculate_metrics(preds, labels)

print(f'Dataset: {DATASET}')
print(f'Processed: {len(preds)} pairs')
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'F1-Score: {metrics["f1"]:.4f}')
print(f'Accuracy: {metrics["accuracy"]:.4f}')
print(f'TP: {metrics["tp"]}, FP: {metrics["fp"]}, FN: {metrics["fn"]}, TN: {metrics["tn"]}')
