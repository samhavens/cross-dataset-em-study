#!/usr/bin/env python
"""
build a 10-token 'mini-key' for every row in tableB using gpt-4.1-nano.
writes data/<dataset>_llmkeys.pkl
cost: ~35 tokens per row ⇒ $0.03 per 1 000 rows.
"""

import argparse, json, pathlib, pickle, textwrap
import pandas as pd
from llm_clustering import Predictor, cfg, token_count, report_cost

BATCH = 40

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--model", default=cfg.model)
args = p.parse_args()
cfg.model = args.model

root = pathlib.Path("data")/args.dataset/"raw"
tbl = pd.read_csv(root/"tableB.csv")
rows = tbl.to_dict(orient="records")


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


out = {}
for offset, chunk in enumerate(chunks(rows, BATCH)):
    listing = [f"{i}) {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(chunk)]
    prompt = textwrap.dedent(f"""
      create a *concise*, unique key (≤10 tokens, lowercase, hyphens ok)
      that would help match each record to itself later. respond as JSON
      mapping row number to key.
      records:
      {'\n'.join(listing)}
    """)
    mapping = Predictor("json")(prompt).output
    for local_idx, key in mapping.items():
        global_idx = offset * BATCH + int(local_idx)
        out[global_idx] = {"key": key, "row": chunk[int(local_idx)]}

pickle.dump(out, open(root.parent/f"{args.dataset}_llmkeys.pkl", "wb"))
print("wrote", len(out), "keys, total prompt tokens:",
      sum(token_count(json.dumps(x["row"])) for x in out.values()))
report_cost()
