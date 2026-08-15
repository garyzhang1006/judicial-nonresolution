"""Step 21: escalation with issue-level matching rather than proximity.

The earlier proxy asked whether a non-resolution expression sat near the passage
a later court characterized. Hand-checking showed that usually catches an
expression governing a neighbouring question. This version instead compares the
proposition the origin declined to decide, taken from the issue statement
attached to each trigger, against the proposition the later court describes. Both
sides are propositions, so a match means the same question.
"""
import json, os, sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from issues import content_words

MIN_J = 0.34          # Jaccard between origin issue and characterization
RNG = np.random.default_rng(21)

fr = pd.read_parquet(os.path.join(OUT, "06_frame.parquet"),
                     columns=["opinion_id", "family", "issue"])
fr = fr[(fr.family == "TRIGGER")].dropna(subset=["issue"])
by_op = defaultdict(list)
for o, i in zip(fr.opinion_id, fr.issue):
    w = content_words(i)
    if len(w) >= 3:
        by_op[o].append((w, i))
print(f"opinions with usable trigger issues: {len(by_op):,}")

t = pd.read_parquet(os.path.join(OUT, "15_tight.parquet"))
rows = []
for r in t.itertuples(index=False):
    cw = content_words(r.characterization)
    if len(cw) < 4:
        continue
    best, bi = 0.0, None
    for w, txt in by_op.get(r.cited_opinion_id, []):
        j = len(cw & w) / len(cw | w)
        if j > best:
            best, bi = j, txt
    rows.append({"cited_opinion_id": r.cited_opinion_id, "citing_id": r.citing_id,
                 "verdict": r.verdict, "jac": best, "origin_issue": bi,
                 "characterization": r.characterization})

d = pd.DataFrame(rows)
d["matched"] = d["jac"] >= MIN_J
d["ah"] = (d.verdict == "L3").astype(int)
d.to_parquet(os.path.join(OUT, "21_issue_match.parquet"), index=False)
print(f"scored {len(d):,}; issue-matched {int(d.matched.sum()):,}")
print(d.groupby("matched")["ah"].agg(["size", "mean"]).round(4).to_string())

a, b = d[d.matched].ah.mean(), d[~d.matched].ah.mean()
ids = d.cited_opinion_id.values; uid = np.unique(ids)
idx = {u: np.flatnonzero(ids == u) for u in uid}
v = []
for _ in range(300):
    pick = RNG.choice(uid, len(uid), replace=True)
    x = d.iloc[np.concatenate([idx[p] for p in pick])]
    aa, bb = x[x.matched].ah, x[~x.matched].ah
    if len(aa) and len(bb):
        v.append(aa.mean() - bb.mean())
lo, hi = np.percentile(v, [2.5, 97.5])
print(f"contrast {a-b:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
json.dump({"n": int(len(d)), "n_matched": int(d.matched.sum()),
           "rate_matched": float(a), "rate_unmatched": float(b),
           "diff": float(a - b), "ci": [float(lo), float(hi)],
           "min_jaccard": MIN_J,
           "excludes_zero": bool(lo > 0 or hi < 0)},
          open(os.path.join(OUT, "21_issue_match.json"), "w"), indent=1)
