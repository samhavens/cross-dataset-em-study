# LLM clustering pipeline implemented with DSPy
# This is a sketch of the high-level pipeline described in the repository docs.
# It provides the building blocks to perform context-length-aware sampling,
# clustering with an LLM and a follow up classifier for remaining rows.

from __future__ import annotations

import random
import re
import textwrap
import warnings

from dataclasses import dataclass
from typing import Iterable

import tiktoken

__all__ = [
    "MODEL_COSTS",
    "ClusterDescription",
    "ClusterNamer",
    "ClusterPipeline",
    "Config",
    "LLMClassifier",
    "LLMCluster",
    "MockPredict",
    "Predictor",
    "SampleForContext",
    "VectorAssign",
    "cfg",
    "handle_outliers",
    "parse_id",
    "report_cost",
    "token_count",
]

import dspy

from .constants import MODEL_COSTS


def token_count(text: str, enc: tiktoken.Encoding | None = None) -> int:
    """Count tokens using ``tiktoken`` when available."""
    if enc is None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("gpt2")
    return len(enc.encode(text))


@dataclass
class Config:
    """Runtime configuration."""

    dry_run: bool = False
    mock_seed: int = 42
    in_cost: float = 0.50  # $ per 1M input tokens
    out_cost: float = 1.50  # $ per 1M output tokens
    model: str = "gpt-4.1-nano"


cfg = Config()

cost_log: list[tuple[int, int]] = []  # (in_tokens, out_tokens)


def report_cost() -> None:
    """Print the total simulated token cost in tokens and dollars."""
    if not cost_log:
        print("no cost logged")
        return
    in_tot, out_tot = map(sum, zip(*cost_log))
    usd = in_tot / 1_000_000 * cfg.in_cost + out_tot / 1_000_000 * cfg.out_cost
    print(f"\u2248{in_tot / 1_000:.1f}K in, {out_tot / 1_000:.1f}K out → ${usd:,.2f}")


class MockPredict:
    """Lightweight predictor used when ``cfg.dry_run`` is true."""

    def __init__(self, ret_type: str = "text"):
        self.ret_type = ret_type
        self.rng = random.Random(cfg.mock_seed)

    def __call__(self, prompt: str) -> object:
        tokens_in = token_count(prompt)
        if self.ret_type == "json":
            ids = re.findall(r"(\d+)\)", prompt)
            tokens_out = max(1, 3 * len(ids))
            cost_log.append((tokens_in, tokens_out))
            mapping = {int(i): f"cluster_{int(i) % 3}" for i in ids}

            # Create a simple object with text and output attributes
            class MockResponse:
                def __init__(self, text, output=None):
                    self.text = text
                    self.output = output

            return MockResponse(text="/*mock*/", output=mapping)
        if self.ret_type == "json-namer":
            tokens_out = 10
            cost_log.append((tokens_in, tokens_out))

            class MockResponse:
                def __init__(self, text, output=None):
                    self.text = text
                    self.output = output

            return MockResponse(
                text="/*mock*/",
                output={
                    "name": f"cluster{self.rng.randint(0, 9)}",
                    "features": "mock feats",
                },
            )
        tokens_out = 5
        cost_log.append((tokens_in, tokens_out))

        class MockResponse:
            def __init__(self, text, output=None):
                self.text = text
                self.output = output

        return MockResponse(text=f"cluster_{self.rng.randint(0, 2)}")


class LivePredict:
    """Wrapper around ``dspy.Predict`` that logs token usage."""

    def __init__(self, ret_type: str, model_name: str):
        # Use simple signature and handle JSON parsing manually
        signature = "prompt -> output"
        self.inner = dspy.Predict(signature)
        self.ret_type = ret_type

    def __call__(self, prompt: str) -> object:
        resp = self.inner(prompt=prompt)

        # Access the output field from the prediction
        out_text = resp.output if hasattr(resp, "output") else str(resp)

        # For JSON responses, try to parse the output
        if self.ret_type == "json" and out_text:
            try:
                import json as _json

                parsed_output = _json.loads(out_text)

                # Create a response object with parsed JSON
                class LiveResponse:
                    def __init__(self, text, output=None):
                        self.text = text
                        self.output = output

                result = LiveResponse(text=out_text, output=parsed_output)
            except Exception:
                # If parsing fails, create a simple response with just text
                class LiveResponse:
                    def __init__(self, text, output=None):
                        self.text = text
                        self.output = output

                result = LiveResponse(text=out_text, output=None)
        else:
            # For text responses, create a simple response object
            class LiveResponse:
                def __init__(self, text, output=None):
                    self.text = text
                    self.output = output

            result = LiveResponse(text=out_text, output=None)

        cost_log.append((token_count(prompt), token_count(str(out_text))))
        return result


def Predictor(ret_type: str):
    """Return a real or mock predictor depending on ``cfg.dry_run``."""

    if cfg.dry_run:
        return MockPredict(ret_type)
    real_type = "json" if ret_type == "json-namer" else ret_type
    return LivePredict(real_type, cfg.model)


_ID_RE = re.compile(r"^(\d+)\)")


def parse_id(row: str) -> int | None:
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
        enc: tiktoken.Encoding | None = None,
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

    def forward(self, rows: list[str]) -> list[str]:
        rng = random.Random(self.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)

        sampled: list[str] = []
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
        self.reasoning: str | None = None

    def forward(self, rows: list[str]) -> dict[int, str]:
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
        return ClusterDescription(name=data.get("name", "cluster"), features=data.get("features", ""))


class LLMClassifier(dspy.Module):
    """Classify an unseen row into an existing cluster."""

    def __init__(self, context: dict[str, list[str]], tau: float = 0.15):
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

    def __init__(self, centroids: dict[str, list[float]]):
        super().__init__()
        self.centroids = centroids

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb + 1e-8)

    def forward(self, embedding: list[float]) -> str:
        if not self.centroids:
            return "OUTLIER"
        best = max(self.centroids.items(), key=lambda kv: self._cosine(kv[1], embedding))
        return best[0]


def handle_outliers(classifier: LLMClassifier, rows: list[str], depth: int = 0, max_depth: int = 1) -> dict[int, str]:
    """Recursively cluster outlier rows."""
    if depth > max_depth or not rows:
        return {}
    mapping: dict[int, str] = {}
    processed: list[tuple[str, str]] = []
    for row in rows:
        rid = parse_id(row)
        if rid is None:
            continue
        label = classifier(row)
        mapping[rid] = label
        processed.append((row, label))
    leftover = [r for r, c in processed if c == "OUTLIER"]
    if leftover:
        mapping.update(handle_outliers(classifier, leftover, depth + 1, max_depth))
    return mapping


class ClusterPipeline(dspy.Module):
    """High level orchestration of the clustering workflow."""

    def __init__(self, limit_tokens: int = 1_000_000):
        super().__init__()
        self.sample = SampleForContext(limit_tokens)
        self.first_pass = LLMCluster()
        self.cluster_namer = ClusterNamer()
        self.fallback_vec = VectorAssign({})
        self.sample_seed = 0

    @staticmethod
    def _quick_silhouette(by_cluster: dict[str, list[str]]) -> float:
        """Placeholder silhouette heuristic using cluster sizes."""
        if not by_cluster:
            return 0.0
        sizes = [len(v) for v in by_cluster.values()]
        if len(sizes) < 2:
            return 0.0
        mean = sum(sizes) / len(sizes)
        var = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        return 1 / (1 + var)

    def forward(self, rows: list[str]) -> dict[int, str]:
        sampled = self.sample(rows)
        mapping = self.first_pass(sampled)

        by_cluster: dict[str, list[str]] = {}
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
        names: dict[str, str] = {}
        for cid, examples in by_cluster.items():
            desc = self.cluster_namer(cid, examples)
            names[cid] = desc.name

        # create classifier for remaining rows
        classifier = LLMClassifier(by_cluster)

        results: dict[int, str] = {}
        outlier_rows: list[str] = []
        for row in rows:
            rid = parse_id(row)
            if rid in mapping:
                results[rid] = names.get(mapping[rid], mapping[rid])
            else:
                label = classifier(row)
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
    if args.in_cost is not None and args.out_cost is not None:
        cfg.in_cost, cfg.out_cost = args.in_cost, args.out_cost
    elif cfg.model in MODEL_COSTS:
        cfg.in_cost, cfg.out_cost = MODEL_COSTS[cfg.model]
    else:
        warnings.warn(
            f"unknown model '{cfg.model}'; using default prices. Specify --in-cost and --out-cost to override.",
            stacklevel=2,
        )

    p = pathlib.Path(args.rows)
    rows = json.loads(p.read_text()) if p.suffix == ".json" else p.read_text().splitlines()

    pipe = ClusterPipeline()
    labels = pipe(rows)
    report_cost()

    json.dump(labels, sys.stdout, indent=2)
