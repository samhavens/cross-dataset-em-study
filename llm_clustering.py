# LLM clustering pipeline implemented with DSPy
# This is a sketch of the high-level pipeline described in the repository docs.
# It provides the building blocks to perform context-length-aware sampling,
# clustering with an LLM and a follow up classifier for remaining rows.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List

import dspy


def token_count(text: str) -> int:
    """Rough token estimator used to keep prompts under a limit."""
    return len(text.split())


class SampleForContext(dspy.Module):
    """Randomly sample rows so their total token count stays within a limit."""

    def __init__(self, limit_tokens: int, seed: int = 0):
        super().__init__()
        self.limit_tokens = limit_tokens
        self.seed = seed

    def forward(self, rows: List[str]) -> List[str]:
        rng = random.Random(self.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)

        sampled: List[str] = []
        total = 0
        for i in idx:
            row = rows[i]
            cost = token_count(row)
            if total + cost > self.limit_tokens:
                break
            sampled.append(row)
            total += cost
        return sampled


class LLMCluster(dspy.Module):
    """Ask an LLM to cluster the given rows."""

    def __init__(self):
        super().__init__()
        self.reasoning = None

    def forward(self, rows: List[str]) -> Dict[int, str]:
        prompt = (
            "You are an expert taxonomist. Cluster these entries:\n" + "\n".join(rows)
        )
        lm = dspy.Predict("json")
        result = lm(prompt)
        self.reasoning = result.text  # store full reasoning for later
        return result.output  # type: ignore[return-value]


@dataclass
class ClusterDescription:
    name: str
    features: str


class ClusterNamer(dspy.Module):
    """Generate short cluster name and a brief feature list."""

    def forward(self, cluster_id: int, examples: Iterable[str]) -> ClusterDescription:
        prompt = (
            f"Give a concise name and features for cluster {cluster_id}:\n"
            + "\n".join(examples)
        )
        lm = dspy.Predict("json")
        data = lm(prompt).output
        return ClusterDescription(name=data.get("name", "cluster"), features=data.get("features", ""))


class LLMClassifier(dspy.Module):
    """Classify an unseen row into an existing cluster."""

    def __init__(self, context: Dict[str, List[str]]):
        super().__init__()
        self.context = context

    def forward(self, row: str) -> str:
        ctx_lines = [f"{k}: {v}" for k, v in self.context.items()]
        prompt = "\n".join(ctx_lines + ["Row:", row])
        lm = dspy.Predict("text")
        return lm(prompt).output


class VectorAssign(dspy.Module):
    """Fallback vector based assignment using pre-computed embeddings."""

    def __init__(self, centroids: Dict[str, List[float]]):
        super().__init__()
        self.centroids = centroids

    def forward(self, embedding: List[float]) -> str:
        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(y * y for y in b) ** 0.5
            return dot / (na * nb + 1e-8)

        best = max(self.centroids.items(), key=lambda kv: cosine(kv[1], embedding))
        return best[0]


class ClusterPipeline(dspy.Graph):
    """High level orchestration of the clustering workflow."""

    def __init__(self, limit_tokens: int = 1_000_000):
        super().__init__()
        self.sample = SampleForContext(limit_tokens)
        self.first_pass = LLMCluster()
        self.name_clusters = dspy.Map(ClusterNamer())
        self.assign_rest = dspy.Map(LLMClassifier({}))
        self.fallback_vec = VectorAssign({})

    def forward(self, rows: List[str]) -> Dict[int, str]:
        sampled = self.sample(rows)
        mapping = self.first_pass(sampled)
        return mapping
