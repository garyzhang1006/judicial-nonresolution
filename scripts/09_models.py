"""Step 9: model evaluation across context conditions and held-out splits.

Four systems are compared. The lexical rule is the honest floor: it says how far
the frozen dictionary alone gets, and any system that fails to beat it is doing
nothing beyond phrase recognition. The issue-only classifier is a leakage
control: the issue statement is written to be neutral, so a classifier that sees
only the issue should be near chance.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from triggers import UNION

RNG = np.random.default_rng(7)
WINDOWS = {"SENT": None, "W256": 256, "W1024": 1024, "W4096": 4096}


def build(df):
    ts, cols = TextStore(), {k: [] for k in WINDOWS}
    for r in df.itertuples(index=False):
        t = ts.get(r.opinion_id) or ""
        s, e = int(r.char_start), int(r.char_end)
        for name, w in WINDOWS.items():
            if w is None:
                seg = t[int(r.sent_start):int(r.sent_end)]
            else:
                seg = t[max(0, s - w): min(len(t), e + w)]
            cols[name].append(re.sub(r"\s+", " ", seg).strip())
    ts.close()
    for k, v in cols.items():
        df[f"ctx_{k}"] = v
    return df


def boot_ci(y, p, fn, n=2000):
    idx = np.arange(len(y))
    vals = [fn(y[b], p[b]) for b in
            (RNG.choice(idx, len(idx), replace=True) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(name, ctx, y_true, y_pred, split, extra=None):
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    lo, hi = boot_ci(np.asarray(y_true), np.asarray(y_pred),
                     lambda a, b: f1_score(a, b, average="macro", zero_division=0))
    return {"model": name, "context": ctx, "split": split, "n": int(len(y_true)),
            "acc": float(accuracy_score(y_true, y_pred)), "macro_f1": float(mf1),
            "f1_unresolved": float(f1_score(y_true, y_pred, pos_label=1,
                                            zero_division=0)),
            "ci_lo": lo, "ci_hi": hi, **(extra or {})}


def main():
    df = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
    df = df[df["label"].isin(["DECIDED", "UNRESOLVED"])].copy()
    df["y"] = (df["label"] == "UNRESOLVED").astype(int)
    df = build(df)
    print(df.groupby("split")["y"].agg(["size", "mean"]).to_string())

    dev = df[df["split"] == "DEV"]
    tests = {s: df[df["split"] == s] for s in ("TEST", "TEMPORAL", "COURT")}
    results = []

    for ctx in WINDOWS:
        col = f"ctx_{ctx}"
        # 1. lexical rule
        for s, te in tests.items():
            pred = te[col].map(lambda x: int(bool(UNION.search(x)))).values
            results.append(evaluate("lexical-rule", ctx, te["y"].values, pred, s))
        # 2. tf-idf + logistic regression
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                            sublinear_tf=True, strip_accents="unicode"),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        pipe.fit(dev[col], dev["y"])
        for s, te in tests.items():
            results.append(evaluate("tfidf-lr", ctx, te["y"].values,
                                    pipe.predict(te[col]), s))

    # 3. leakage control: the neutral issue statement alone
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(dev["issue"].astype(str), dev["y"])
    for s, te in tests.items():
        results.append(evaluate("issue-only", "ISSUE", te["y"].values,
                                pipe.predict(te["issue"].astype(str)), s))

    # 4. majority class
    maj = int(dev["y"].mean() > 0.5)
    for s, te in tests.items():
        results.append(evaluate("majority", "-", te["y"].values,
                                np.full(len(te), maj), s))

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT, "09_results.csv"), index=False)
    print("\n" + res.pivot_table(index=["model", "context"], columns="split",
                                 values="macro_f1").round(3).to_string())

    # per-stratum accuracy for the strongest sparse model
    rows = []
    for ctx in WINDOWS:
        col = f"ctx_{ctx}"
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                            sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        pipe.fit(dev[col], dev["y"])
        te = pd.concat(tests.values())
        te = te.assign(pred=pipe.predict(te[col]),
                       lex=te[col].map(lambda x: int(bool(UNION.search(x)))))
        for st, g in te.groupby("stratum"):
            rows.append({"context": ctx, "stratum": st, "n": len(g),
                         "tfidf_acc": accuracy_score(g["y"], g["pred"]),
                         "lexical_acc": accuracy_score(g["y"], g["lex"])})
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(OUT, "09_by_stratum.csv"), index=False)
    print("\n" + st.pivot_table(index="stratum", columns="context",
                                values=["lexical_acc", "tfidf_acc"]).round(3).to_string())
    df.to_parquet(os.path.join(OUT, "09_items_with_ctx.parquet"), index=False)




def paired_context_test(out_csv="09_context_test.csv"):
    """Paired bootstrap on whether a wider window helps, model by model.

    The comparison is within model and within evaluation set, and it resamples
    the same items for both conditions, so it isolates the effect of the window
    rather than of anything that differs between splits.
    """
    df = pd.read_parquet(os.path.join(OUT, "09_items_with_ctx.parquet"))
    dev = df[df["split"] == "DEV"]
    rows = []
    for split in ("TEST", "TEMPORAL", "COURT"):
        te = df[df["split"] == split]
        if len(te) < 20:
            continue
        preds = {}
        for ctx in WINDOWS:
            col = f"ctx_{ctx}"
            pipe = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                sublinear_tf=True),
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
            pipe.fit(dev[col], dev["y"])
            preds[ctx] = pipe.predict(te[col])
        y = te["y"].values
        base = "SENT"
        for ctx in WINDOWS:
            if ctx == base:
                continue
            d = []
            idx = np.arange(len(y))
            for _ in range(2000):
                b = RNG.choice(idx, len(idx), replace=True)
                d.append(f1_score(y[b], preds[ctx][b], average="macro", zero_division=0)
                         - f1_score(y[b], preds[base][b], average="macro", zero_division=0))
            d = np.array(d)
            rows.append({"split": split, "context": ctx, "vs": base,
                         "delta_macro_f1": float(
                             f1_score(y, preds[ctx], average="macro", zero_division=0)
                             - f1_score(y, preds[base], average="macro", zero_division=0)),
                         "ci_lo": float(np.percentile(d, 2.5)),
                         "ci_hi": float(np.percentile(d, 97.5)),
                         "p_gt_zero": float((d > 0).mean())})
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, out_csv), index=False)
    print("\npaired context comparison (vs sentence-only):")
    print(r.round(3).to_string(index=False))
    return r


if __name__ == "__main__":
    main()
    paired_context_test()
