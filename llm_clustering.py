# LLM clustering pipeline implemented with DSPy
# This is a sketch of the high-level pipeline described in the repository docs.
# It provides the building blocks to perform context-length-aware sampling,
# clustering with an LLM and a follow up classifier for remaining rows.

from __future__ import annotations

import random
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Tuple
import textwrap

__all__ = [
    "Config",
    "cfg",
    "token_count",
    "parse_id",
    "report_cost",
    "SampleForContext",
    "LLMCluster",
    "ClusterDescription",
    "ClusterNamer",
    "LLMClassifier",
    "VectorAssign",
    "handle_outliers",
    "ClusterPipeline",
    "Predictor",
    "MockPredict",
]

import dspy

MODEL_COSTS = {
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.2, 0.8),
    "o3": (0.4, 1.6),
    "o3-mini": (0.1, 0.4),
    "o4": (10.0, 30.0),
    "o4-mini": (0.15, 0.60),
    "claude-4-sonnet": (3.0, 15.0),
}

try:
    import tiktoken
except Exception:  # pragma: no cover - tiktoken may not be installed
    tiktoken = None  # type: ignore


def token_count(text: str, enc: Optional["tiktoken.Encoding"] = None) -> int:
    """Count tokens using ``tiktoken`` when available."""
    if tiktoken is not None:
        if enc is None:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))
    return len(text.split())


@dataclass
class Config:
    """Runtime configuration."""

    dry_run: bool = False
    mock_seed: int = 42
    in_cost: float = 0.50  # $ per 1M input tokens
    out_cost: float = 1.50  # $ per 1M output tokens
    model: str = "gpt-4.1-nano"



cfg = Config()

cost_log: List[Tuple[int, int]] = []  # (in_tokens, out_tokens)


def report_cost() -> None:
    """Print the total simulated token cost in tokens and dollars."""
    if not cost_log:
        print("no cost logged")
        return
    in_tot, out_tot = map(sum, zip(*cost_log))
    usd = in_tot / 1_000_000 * cfg.in_cost + out_tot / 1_000_000 * cfg.out_cost
    print(
        f"\u2248{in_tot/1_000:.1f}K in, {out_tot/1_000:.1f}K out → ${usd:,.2f}"
    )


class MockPredict:
    """Lightweight predictor used when ``cfg.dry_run`` is true."""

    def __init__(self, ret_type: str = "text"):
        self.ret_type = ret_type
        self.rng = random.Random(cfg.mock_seed)

    def __call__(self, prompt: str) -> dspy.Response:
        tokens_in = token_count(prompt)
        if self.ret_type == "json":
            ids = re.findall(r"(\d+)\)", prompt)
            tokens_out = max(1, 3 * len(ids))
            cost_log.append((tokens_in, tokens_out))
            mapping = {int(i): f"cluster_{int(i) % 3}" for i in ids}
            return dspy.Response(text="/*mock*/", output=mapping)
        if self.ret_type == "json-namer":
            tokens_out = 10
            cost_log.append((tokens_in, tokens_out))
            return dspy.Response(
                text="/*mock*/",
                output={
                    "name": f"cluster{self.rng.randint(0,9)}",
                    "features": "mock feats",
                },
            )
        tokens_out = 5
        cost_log.append((tokens_in, tokens_out))
        return dspy.Response(text=f"cluster_{self.rng.randint(0,2)}")


class LivePredict:
    """Wrapper around ``dspy.Predict`` that logs token usage."""

    def __init__(self, ret_type: str, model_name: str):
        self.inner = dspy.Predict(ret_type, model=model_name)

    def __call__(self, prompt: str) -> dspy.Response:
        resp = self.inner(prompt)
        cost_log.append((token_count(prompt), token_count(resp.text)))
        return resp


def Predictor(ret_type: str):
    """Return a real or mock predictor depending on ``cfg.dry_run``."""

    if cfg.dry_run:
        return MockPredict(ret_type)
    return LivePredict(ret_type, cfg.model)


_ID_RE = re.compile(r"^(\d+)\)")


def parse_id(row: str) -> Optional[int]:
    """Extract numeric ID from a row string."""
    m = _ID_RE.match(row.strip())
    if m:
        try:
            return int(m.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


class SampleForContext(dspy.Module):
    """Randomly sample rows so their total token count stays within a limit."""

    def __init__(
        self,
        limit_tokens: int,
        seed: int = 0,
        enc: Optional["tiktoken.Encoding"] = None,
    ):
        super().__init__()
        self.limit_tokens = limit_tokens
        self.seed = seed
        if enc is not None:
            self.enc = enc
        elif tiktoken is not None:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = tiktoken.get_encoding("gpt2")
        else:
            self.enc = None

    def forward(self, rows: List[str]) -> List[str]:
        rng = random.Random(self.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)

        sampled: List[str] = []
        total = 0
        for i in idx:
            row = rows[i]
            cost = token_count(row, self.enc)
            if total + cost > self.limit_tokens:
                break
            sampled.append(row)
            total += cost
        return sampled


class LLMCluster(dspy.Module):
    """Ask an LLM to cluster the given rows."""

    def __init__(self):
        super().__init__()
        self.reasoning: Optional[str] = None

    def forward(self, rows: List[str]) -> Dict[int, str]:
        prompt = textwrap.dedent(
            """
            system: you are an expert taxonomist.
            user: here are entries to cluster:
            {rows}
            assistant: think step-by-step and output a JSON mapping id to cluster name
            """
        ).format(rows="\n".join(rows))
        lm = Predictor("json")
        result = lm(prompt)
        self.reasoning = result.text
        return result.output  # type: ignore[return-value]


@dataclass
class ClusterDescription:
    name: str
    features: str


class ClusterNamer(dspy.Module):
    """Generate short cluster name and a brief feature list."""

    def forward(self, cluster_id: int, examples: Iterable[str]) -> ClusterDescription:
        prompt = textwrap.dedent(
            """
            Provide a short name (<=4 words) and key features for cluster {cid} given these examples:
            {rows}
            """
        ).format(cid=cluster_id, rows="\n".join(examples))
        lm = Predictor("json-namer")
        data = lm(prompt).output
        return ClusterDescription(
            name=data.get("name", "cluster"), features=data.get("features", "")
        )


class LLMClassifier(dspy.Module):
    """Classify an unseen row into an existing cluster."""

    def __init__(self, context: Dict[str, List[str]], tau: float = 0.15):
        super().__init__()
        self.context = context
        self.tau = tau

    def forward(self, row: str) -> str:
        ctx_lines = [f"{k}: {', '.join(v)}" for k, v in self.context.items()]
        prompt = textwrap.dedent(
            """
            {context}
            Choose the best cluster for the following row.
            Respond with the cluster name and a confidence score.
            {row}
            """
        ).format(context="\n".join(ctx_lines), row=row)
        lm = Predictor("text")
        res = lm(prompt).text.strip().split()
        label = res[0]
        conf = float(res[1]) if len(res) > 1 else 1.0
        return label if conf > self.tau else "OUTLIER"


class VectorAssign(dspy.Module):
    """Fallback vector based assignment using pre-computed embeddings.

    ``centroids`` should map cluster names to embedding vectors obtained
    externally (e.g. via an embeddings service).
    """

    def __init__(self, centroids: Dict[str, List[float]]):
        super().__init__()
        self.centroids = centroids

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-8)

    def forward(self, embedding: List[float]) -> str:
        if not self.centroids:
            return "OUTLIER"
        best = max(self.centroids.items(), key=lambda kv: self._cosine(kv[1], embedding))
        return best[0]


def handle_outliers(classifier: LLMClassifier, rows: List[str], depth: int = 0, max_depth: int = 1) -> Dict[int, str]:
    """Recursively cluster outlier rows."""
    if depth > max_depth or not rows:
        return {}
    mapping: Dict[int, str] = {}
    for row in rows:
        rid = parse_id(row)
        if rid is None:
            continue
        label = classifier(row)
        mapping[rid] = label
    leftover = [r for r, c in zip(rows, mapping.values()) if c == "OUTLIER"]
    if leftover:
        mapping.update(handle_outliers(classifier, leftover, depth + 1, max_depth))
    return mapping


class ClusterPipeline(dspy.Graph):
    """High level orchestration of the clustering workflow."""

    def __init__(self, limit_tokens: int = 1_000_000):
        super().__init__()
        self.sample = SampleForContext(limit_tokens)
        self.first_pass = LLMCluster()
        self.name_clusters = dspy.Map(ClusterNamer())
        self.assign_rest = dspy.Map(LLMClassifier({}))
        self.fallback_vec = VectorAssign({})
        self.sample_seed = 0

    @staticmethod
    def _quick_silhouette(by_cluster: Dict[str, List[str]]) -> float:
        """Placeholder silhouette heuristic using cluster sizes."""
        if not by_cluster:
            return 0.0
        sizes = [len(v) for v in by_cluster.values()]
        if len(sizes) < 2:
            return 0.0
        mean = sum(sizes) / len(sizes)
        var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        return 1 / (1 + var)

    def forward(self, rows: List[str]) -> Dict[int, str]:
        sampled = self.sample(rows)
        mapping = self.first_pass(sampled)

        by_cluster: Dict[str, List[str]] = {}
        for row in sampled:
            rid = parse_id(row)
            if rid is None:
                continue
            cid = mapping.get(rid)
            if cid is None:
                continue
            by_cluster.setdefault(cid, []).append(row)

        if self._quick_silhouette(by_cluster) < 0.05:
            self.sample_seed += 1
            self.sample.seed = self.sample_seed
            sampled = self.sample(rows)
            mapping = self.first_pass(sampled)
            by_cluster.clear()
            for row in sampled:
                rid = parse_id(row)
                if rid is None:
                    continue
                cid = mapping.get(rid)
                if cid is None:
                    continue
                by_cluster.setdefault(cid, []).append(row)

        # name clusters
        names: Dict[str, str] = {}
        for cid, examples in by_cluster.items():
            desc = self.name_clusters(cid, examples)
            names[cid] = desc.name

        # update classifier context with example lists
        self.assign_rest = dspy.Map(LLMClassifier(by_cluster))

        results: Dict[int, str] = {}
        outlier_rows: List[str] = []
        for row in rows:
            rid = parse_id(row)
            if rid in mapping:
                results[rid] = names.get(mapping[rid], mapping[rid])
            else:
                label = self.assign_rest(row)
                if label == "OUTLIER":
                    outlier_rows.append(row)
                else:
                    results[rid] = label

        if outlier_rows:
            extra_map = self.first_pass(outlier_rows)
            for row in outlier_rows:
                rid = parse_id(row)
                if rid is None:
                    continue
                cid = extra_map.get(rid)
                if cid is not None:
                    results[rid] = cid
                else:
                    results[rid] = "OUTLIER"
        return results


if __name__ == "__main__":
    import argparse
    import json
    import pathlib
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, help="path to txt/json file with one record per line")
    parser.add_argument("--model", default=cfg.model, help="model key (see MODEL_COSTS) or custom name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--in-cost", type=float)
    parser.add_argument("--out-cost", type=float)
    args = parser.parse_args()

    cfg.dry_run = args.dry_run
    cfg.model = args.model
    if args.in_cost and args.out_cost:
        cfg.in_cost, cfg.out_cost = args.in_cost, args.out_cost
    else:
        cfg.in_cost, cfg.out_cost = MODEL_COSTS.get(cfg.model, (cfg.in_cost, cfg.out_cost))

    p = pathlib.Path(args.rows)
    rows = json.loads(p.read_text()) if p.suffix == ".json" else p.read_text().splitlines()

    pipe = ClusterPipeline()
    labels = pipe(rows)
    report_cost()

    json.dump(labels, sys.stdout, indent=2)
