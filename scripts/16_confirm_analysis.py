"""Step 16: use judicially authored characterizations two ways.

First as external validation. For a benchmark item whose originating opinion is
later characterized by another court on the same issue, that later court's
statement is a label produced by a federal judge. Agreement between it and the
annotation is a check no annotator was involved in.

Second as the layer-two measurement. Comparing how often later courts ascribe a
holding to origins we labelled UNRESOLVED against origins we labelled DECIDED is
exactly the escalation contrast, now at a sample size annotation cannot reach.

Both restrict to issue-specific characterizations, since an opinion is usually
cited for something other than the issue we annotated.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from issues import content_words

MIN_OVERLAP = 3
RNG = np.random.default_rng(11)


def cluster_boot(df, stat, n=4000):
    groups = [g for _, g in df.groupby("item_id")]
    if len(groups) < 3:
        return (np.nan, np.nan)
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(groups), len(groups))
        v = stat(pd.concat([groups[i] for i in idx], ignore_index=True))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main():
    # The tight set requires the characterizing verb to bind directly to the
    # citation, which raises precision from roughly a third to near ceiling on
    # inspection, at the cost of keeping 1.6% of loosely matched passages.
    jc = pd.read_parquet(os.path.join(OUT, "15_tight.parquet"))
    ann = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
    ann = ann[ann["label"].isin(["DECIDED", "UNRESOLVED"])]
    print(f"characterizations {len(jc):,}; annotated origins {len(ann):,}")

    m = jc.merge(ann[["item_id", "opinion_id", "label", "sublabel", "issue",
                      "court", "year", "stratum"]],
                 left_on="cited_opinion_id", right_on="opinion_id", how="inner")
    print(f"characterizations of annotated origins: {len(m):,} "
          f"across {m['item_id'].nunique():,} items")
    if not len(m):
        return

    keys = {i: content_words(s) for i, s in zip(ann["item_id"], ann["issue"])}
    m["overlap"] = [len(content_words(p) & keys.get(i, set()))
                    for p, i in zip(m["characterization"], m["item_id"])]
    iss = m[m["overlap"] >= MIN_OVERLAP].copy()
    print(f"issue-specific (overlap >= {MIN_OVERLAP}): {len(iss):,} "
          f"across {iss['item_id'].nunique():,} items")
    iss.to_parquet(os.path.join(OUT, "16_issue_specific.parquet"), index=False)

    out = {"n_char_total": int(len(jc)), "n_on_annotated": int(len(m)),
           "n_issue_specific": int(len(iss)),
           "n_items": int(iss["item_id"].nunique())}

    # ---------------- external validation
    # A judge writing "X did not decide" is asserting UNRESOLVED for X.
    v = iss.copy()
    v["judicial"] = np.where(v["verdict"] == "L0", "UNRESOLVED", "DECIDED")
    per_item = (v.groupby(["item_id", "label"])["judicial"]
                 .agg(lambda s: s.value_counts().idxmax()).reset_index())
    per_item["agree"] = per_item["label"] == per_item["judicial"]
    out["validation_n"] = int(len(per_item))
    out["validation_agreement"] = float(per_item["agree"].mean())
    tab = pd.crosstab(per_item["label"], per_item["judicial"])
    out["validation_table"] = tab.to_dict()
    print(f"\n=== EXTERNAL VALIDATION (majority judicial verdict per item) ===")
    print(f"items with an issue-specific characterization: {len(per_item)}")
    print(f"agreement with annotation: {per_item['agree'].mean():.3f}")
    print(tab.to_string())
    # Cohen's kappa between annotation and the judiciary
    n = len(per_item)
    if n:
        po = per_item["agree"].mean()
        pa = per_item["label"].value_counts(normalize=True)
        pb = per_item["judicial"].value_counts(normalize=True)
        pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in ("DECIDED", "UNRESOLVED"))
        out["validation_kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
        print(f"Cohen's kappa vs judicial statements: {out['validation_kappa']:.3f}")
    per_item.to_csv(os.path.join(OUT, "16_validation.csv"), index=False)

    # ---------------- escalation contrast
    iss["ascribes_holding"] = (iss["verdict"] == "L3").astype(int)
    print("\n=== ESCALATION (share of characterizations ascribing a holding) ===")
    for lab, g in iss.groupby("label"):
        lo, hi = cluster_boot(g, lambda x: x["ascribes_holding"].mean())
        out.setdefault("escalation", {})[lab] = {
            "n": int(len(g)), "n_items": int(g["item_id"].nunique()),
            "rate": float(g["ascribes_holding"].mean()), "ci": [lo, hi]}
        print(f"  {lab:11s} n={len(g):5,} items={g['item_id'].nunique():3}  "
              f"{g['ascribes_holding'].mean():.3f}  [{lo:.3f}, {hi:.3f}]")

    def diff(x):
        a = x[x["label"] == "UNRESOLVED"]["ascribes_holding"]
        b = x[x["label"] == "DECIDED"]["ascribes_holding"]
        return a.mean() - b.mean() if len(a) and len(b) else None

    if {"UNRESOLVED", "DECIDED"} <= set(iss["label"]):
        pt = diff(iss)
        lo, hi = cluster_boot(iss, diff)
        out["escalation_contrast"] = {"diff": float(pt), "ci": [lo, hi],
                                      "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))}
        print(f"  contrast   {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]"
              f"{'  *' if np.isfinite(lo) and (lo > 0 or hi < 0) else ''}")

    # ---------------- by sublabel and by court relation
    if len(iss):
        sub = iss[iss["sublabel"] != ""].groupby("sublabel")["ascribes_holding"].agg(
            ["size", "mean"]).reset_index()
        sub.to_csv(os.path.join(OUT, "16_by_sublabel.csv"), index=False)
        print("\nby sublabel:\n" + sub.round(3).to_string(index=False))

    json.dump(out, open(os.path.join(OUT, "16_analysis.json"), "w"), indent=1, default=str)
    print("\nsaved 16_analysis.json")


if __name__ == "__main__":
    main()
