"""Step 10c: layer-two statistics with the clustering the design requires.

Several citing passages can share an origin opinion, so passages are not
independent. Every interval here resamples origins, not passages. Nothing is
fitted: the contrast between unresolved and decided origins is descriptive, and
the matching is what makes it interpretable rather than an identification
strategy.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT

RNG = np.random.default_rng(20260811)
B = 4000


def cluster_boot(df, stat, n=B):
    """Percentile bootstrap resampling whole origins."""
    groups = [g for _, g in df.groupby("origin_item_id")]
    if len(groups) < 3:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(groups), len(groups))
        s = stat(pd.concat([groups[i] for i in idx], ignore_index=True))
        if s is not None and np.isfinite(s):
            vals.append(s)
    if not vals:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    l2 = pd.read_parquet(os.path.join(OUT, "10b_annotated.parquet"))
    u = l2[l2["usable"]].copy()
    print(f"usable layer-two annotations: {len(u):,} "
          f"across {u['origin_item_id'].nunique():,} origins")

    u["escalated"] = (u["status"] >= 2).astype(int)
    u["established"] = (u["status"] == 3).astype(int)
    out = {"n_usable": int(len(u)), "n_origins": int(u["origin_item_id"].nunique()),
           "n_off_issue": int(l2["off_issue"].sum()), "groups": {}}

    for lab, g in u.groupby("origin_label"):
        d = {"n": int(len(g)), "n_origins": int(g["origin_item_id"].nunique())}
        for name, col in (("escalated", "escalated"), ("established", "established"),
                          ("attributed", "A"), ("independent", "E")):
            d[name] = float(g[col].mean())
            lo, hi = cluster_boot(g, lambda x, c=col: x[c].mean())
            d[name + "_ci"] = [lo, hi]
        d["status_dist"] = {int(s): float((g["status"] == s).mean())
                            for s in range(4)}
        out["groups"][lab] = d
        print(f"\n{lab}: n={d['n']} origins={d['n_origins']}")
        for k in ("escalated", "established", "attributed", "independent"):
            lo, hi = d[k + "_ci"]
            print(f"  {k:12s} {d[k]:.3f}  [{lo:.3f}, {hi:.3f}]")

    # --- the contrast, with the same clustering
    if {"UNRESOLVED", "DECIDED"} <= set(u["origin_label"]):
        for col in ("escalated", "established", "A", "E"):
            def diff(x, c=col):
                a = x[x["origin_label"] == "UNRESOLVED"][c]
                b = x[x["origin_label"] == "DECIDED"][c]
                if not len(a) or not len(b):
                    return None
                return a.mean() - b.mean()
            point = diff(u)
            lo, hi = cluster_boot(u, diff)
            out.setdefault("contrast", {})[col] = {
                "diff": float(point), "ci": [lo, hi],
                "excludes_zero": bool(np.isfinite(lo) and np.isfinite(hi)
                                      and (lo > 0 or hi < 0))}
            print(f"contrast {col:12s} {point:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
                  f"{'  *' if (np.isfinite(lo) and (lo > 0 or hi < 0)) else ''}")

    # --- descriptive pathways through the network
    if len(u):
        path = u.assign(rel=np.where(u["is_scotus"], "Supreme Court",
                        np.where(u["same_court"], "same court", "other circuit")))
        tab = path.groupby(["origin_label", "rel"]).agg(
            n=("status", "size"), escalated=("escalated", "mean"),
            attributed=("A", "mean")).reset_index()
        tab.to_csv(os.path.join(OUT, "10c_pathways.csv"), index=False)
        print("\npathways:\n" + tab.to_string(index=False))
        out["pathways"] = tab.to_dict("records")

    json.dump(out, open(os.path.join(OUT, "10c_analysis.json"), "w"), indent=1)
    print("\nsaved 10c_analysis.json")


if __name__ == "__main__":
    main()
