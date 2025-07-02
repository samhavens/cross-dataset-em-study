import argparse
import csv
import random
from typing import Dict, List, Tuple

DATASET_DIRS = {
    'abt': 'abt_buy',
    'amgo': 'amazon_google',
    'beer': 'beer',
    'dbac': 'dblp_acm',
    'dbgo': 'dblp_scholar',
    'foza': 'fodors_zagat',
    'itam': 'itunes_amazon',
    'roim': 'rotten_imdb',
    'waam': 'walmart_amazon',
    'wdc': 'wdc',
    'zoye': 'zomato_yelp',
}


def load_pairs(dataset: str) -> List[Tuple[str, str, int]]:
    dir_name = DATASET_DIRS.get(dataset, dataset)
    path = f"data/raw/{dir_name}/test.csv"
    pairs = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["ltable_id"], row["rtable_id"], int(row["label"])) )
    except FileNotFoundError:
        return []
    return pairs


def random_clustering(dataset: str, seed: int = 42, clusters: int = 5) -> float:
    pairs = load_pairs(dataset)
    left_ids = {p[0] for p in pairs}
    right_ids = {p[1] for p in pairs}

    random.seed(seed)
    left_clusters: Dict[str, int] = {lid: random.randrange(clusters) for lid in left_ids}
    right_clusters: Dict[str, int] = {rid: random.randrange(clusters) for rid in right_ids}

    preds: List[int] = []
    labels: List[int] = []
    for lid, rid, label in pairs:
        pred = 1 if left_clusters[lid] == right_clusters[rid] else 0
        preds.append(pred)
        labels.append(label)

    tp = sum(1 for p, l in zip(preds, labels) if p == l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_pairs = len(pairs)
    n_pos = sum(labels)
    expected_f1 = (2 * n_pos) / (n_pos * clusters + n_pairs) if n_pairs else 0.0

    print(f"{dataset} seed={seed} f1={f1:.4f} expected={expected_f1:.4f}")
    return f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Random clustering baseline")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()
    random_clustering(args.dataset, args.seed, args.clusters)


if __name__ == "__main__":
    main()
