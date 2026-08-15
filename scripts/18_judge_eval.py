"""Step 18: evaluate on the judge-labelled set.

If systems trained on our annotations transfer to items labelled by federal
judges, that is evidence about the annotations that does not depend on the
annotator. If they do not transfer, that is evidence too.
"""
import json, os, re, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from triggers import UNION

RNG = np.random.default_rng(3)
WINDOWS = {"SENT": None, "W256": 256, "W1024": 1024}


def ctx(df):
    ts, cols = TextStore(), {k: [] for k in WINDOWS}
    for r in df.itertuples(index=False):
        t = ts.get(r.opinion_id) or ""
        s, e = int(r.char_start), int(r.char_end)
        for k, w in WINDOWS.items():
            seg = (t[int(r.sent_start):int(r.sent_end)] if w is None
                   else t[max(0, s - w):min(len(t), e + w)])
            cols[k].append(re.sub(r"\s+", " ", seg).strip())
    ts.close()
    for k, v in cols.items():
        df[f"ctx_{k}"] = v
    return df


def boot(y, p, n=2000):
    i = np.arange(len(y))
    v = [f1_score(y[b], p[b], average="macro", zero_division=0)
         for b in (RNG.choice(i, len(i), replace=True) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    dev = pd.read_parquet(os.path.join(OUT, "09_items_with_ctx.parquet"))
    dev = dev[dev["split"] == "DEV"]
    j = ctx(pd.read_parquet(os.path.join(OUT, "17_judge_bench.parquet")))
    j["y"] = (j["label"] == "UNRESOLVED").astype(int)
    print(f"train {len(dev)} annotated DEV items; test {len(j):,} judge-labelled "
          f"({j['y'].mean():.3f} unresolved)")

    rows = []
    for k in WINDOWS:
        c = f"ctx_{k}"
        pred = j[c].map(lambda x: int(bool(UNION.search(x)))).values
        lo, hi = boot(j["y"].values, pred)
        rows.append({"model": "lexical-rule", "context": k, "n": len(j),
                     "acc": accuracy_score(j["y"], pred),
                     "macro_f1": f1_score(j["y"], pred, average="macro", zero_division=0),
                     "ci_lo": lo, "ci_hi": hi})
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        pipe.fit(dev[c], dev["y"])
        p = pipe.predict(j[c])
        lo, hi = boot(j["y"].values, p)
        rows.append({"model": "tfidf-lr", "context": k, "n": len(j),
                     "acc": accuracy_score(j["y"], p),
                     "macro_f1": f1_score(j["y"], p, average="macro", zero_division=0),
                     "ci_lo": lo, "ci_hi": hi})
    maj = int(dev["y"].mean() > 0.5)
    pm = np.full(len(j), maj)
    rows.append({"model": "majority", "context": "-", "n": len(j),
                 "acc": accuracy_score(j["y"], pm),
                 "macro_f1": f1_score(j["y"], pm, average="macro", zero_division=0),
                 "ci_lo": np.nan, "ci_hi": np.nan})
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, "18_judge_eval.csv"), index=False)
    print("\n" + r.round(3).to_string(index=False))
    json.dump(r.to_dict("records"), open(os.path.join(OUT, "18_judge_eval.json"), "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
