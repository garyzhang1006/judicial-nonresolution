"""Step 19: escalation at scale, using two frozen dictionaries and no annotator.

For each judicially authored characterization we locate, inside the cited
opinion, the passage the characterization is about, and ask whether that passage
carries a non-resolution expression. Where it does and the later court
nonetheless ascribes a holding, the later court has strengthened the decisional
status of the proposition. Both sides of that comparison come from dictionaries
fixed in advance, so the measurement scales past what annotation can reach. It
is coarser than annotation and its error rates are reported alongside it.
"""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from triggers import UNION
from issues import content_words

W = 400
RNG = np.random.default_rng(19)


def boot(df, stat, n=3000):
    g = [x for _, x in df.groupby("cited_opinion_id")]
    if len(g) < 5:
        return (np.nan, np.nan)
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(g), len(g))
        s = stat(pd.concat([g[k] for k in i], ignore_index=True))
        if s is not None and np.isfinite(s):
            v.append(s)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def main():
    t = pd.read_parquet(os.path.join(OUT, "15_tight.parquet"))
    elig = set(pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"),
                               columns=["opinion_id"])["opinion_id"])
    t = t[t["cited_opinion_id"].isin(elig)]
    print(f"characterizations of eligible cited opinions: {len(t):,}")

    ts, rows = TextStore(), []
    for n, r in enumerate(t.itertuples(index=False)):
        if n % 20000 == 0:
            print(f"  {n:,}/{len(t):,}", file=sys.stderr, flush=True)
        txt = ts.get(r.cited_opinion_id)
        if not txt:
            continue
        keys = content_words(r.characterization)
        if len(keys) < 4:
            continue
        best, pos = 0, None
        for m in re.finditer(r"[^.]{40,400}\.", txt):
            ov = len(content_words(m.group(0)) & keys)
            if ov > best:
                best, pos = ov, m.start()
        if best < 4 or pos is None:
            continue
        seg = txt[max(0, pos - W):min(len(txt), pos + W)]
        rows.append({"cited_opinion_id": r.cited_opinion_id, "citing_id": r.citing_id,
                     "verdict": r.verdict, "match": best,
                     "origin_marks_open": bool(UNION.search(seg))})
    ts.close()

    d = pd.DataFrame(rows)
    d.to_parquet(os.path.join(OUT, "19_escalation.parquet"), index=False)
    print(f"\nlocated: {len(d):,} across {d['cited_opinion_id'].nunique():,} opinions")

    d["ascribes_holding"] = (d["verdict"] == "L3").astype(int)
    out = {"n": int(len(d)), "n_opinions": int(d["cited_opinion_id"].nunique())}
    print("\n=== share of later characterizations ascribing a holding ===")
    for k, g in d.groupby("origin_marks_open"):
        lo, hi = boot(g, lambda x: x["ascribes_holding"].mean())
        lab = "origin marks the issue open" if k else "origin does not"
        out[str(k)] = {"n": int(len(g)), "rate": float(g["ascribes_holding"].mean()),
                       "ci": [lo, hi]}
        print(f"  {lab:30s} n={len(g):6,}  {g['ascribes_holding'].mean():.4f}  "
              f"[{lo:.4f}, {hi:.4f}]")

    def diff(x):
        a = x[x["origin_marks_open"]]["ascribes_holding"]
        b = x[~x["origin_marks_open"]]["ascribes_holding"]
        return a.mean() - b.mean() if len(a) and len(b) else None
    pt = diff(d)
    lo, hi = boot(d, diff)
    out["contrast"] = {"diff": float(pt), "ci": [lo, hi],
                       "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
    print(f"  contrast {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    json.dump(out, open(os.path.join(OUT, "19_escalation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
