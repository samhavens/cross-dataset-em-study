import random

from difflib import SequenceMatcher

import pandas as pd

from sklearn.metrics import f1_score


def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


datasets = ["abt", "wdc", "dbac", "dbgo", "foza", "zoye", "amgo", "beer", "itam", "roim", "waam"]
for seed in [42, 44, 46, 48, 50]:
    random.seed(seed)
    f1s = []
    for d in datasets:
        df = pd.read_csv(f"data/processed/{d}/test.csv")
        df = df.fillna("nan")
        cols = [c[:-2] for c in df.columns if c.endswith("_l")]
        random.shuffle(cols)
        l_cols = [c + "_l" for c in cols]
        r_cols = [c + "_r" for c in cols]

        df["textA"] = df.apply(lambda x: ", ".join([str(x[c]) for c in l_cols]), axis=1)
        df["textB"] = df.apply(lambda x: ", ".join([str(x[c]) for c in r_cols]), axis=1)
        df["prediction"] = df.apply(lambda x: 1 if string_similarity(x["textA"], x["textB"]) > 0.5 else 0, axis=1)
        f1 = f1_score(df["label"], df["prediction"])
        f1s.append(f1 * 100)
    print(f"Seed {seed}", f1s)
