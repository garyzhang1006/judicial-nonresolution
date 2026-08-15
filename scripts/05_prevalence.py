"""Step 5: eligibility filtering, deduplication, and population prevalence.

Eligibility criteria are applied in a fixed order and each one's cost is
reported, so that the denominator behind every prevalence figure is auditable.
"""
import hashlib, json, os, re, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, FED_APP, PERIODS
from triggers import SPEC, NAME, DICTIONARY_SHA1

MIN_CHARS = 1000

TYPE_GROUP = {
    "010combined": "majority/lead", "015unamimous": "majority/lead",
    "020lead": "majority/lead", "025plurality": "majority/lead",
    "030concurrence": "separate", "035concurrenceinpart": "separate",
    "040dissent": "separate", "050addendum": "other",
    "060remittitur": "other", "070rehearing": "other",
    "080onthemerits": "majority/lead", "090onmotiontostrike": "other",
}

m = pd.read_parquet(os.path.join(OUT, "03_opinions_meta.parquet"))
cl = pd.read_parquet(os.path.join(OUT, "02_clusters.parquet"))
df = m.merge(cl, on="cluster_id", how="left")
audit = [("opinions with extractable text in the 14 courts", len(df))]

# E2 usable filing date
df = df[df["year"].notna() & (df["year"] >= 1789) & (df["year"] <= 2026)]
audit.append(("E2 usable filing date", len(df)))

# E3 minimum length: one-line orders and judgment entries carry no reasoning
df = df[df["n_chars"] >= MIN_CHARS]
audit.append((f"E3 text at least {MIN_CHARS} characters", len(df)))

# E4 not blocked from public view
df = df[~df["blocked"].fillna(False)]
audit.append(("E4 not blocked", len(df)))

# E5 deduplicate: the same opinion is often stored more than once when it was
# ingested from several upstream sources.
fp = (df["court"].astype(str) + "|" + df["date_filed"].astype(str) + "|"
      + df["n_chars"].astype(str) + "|" + df["n_words"].astype(str) + "|"
      + df["type"].astype(str) + "|" + df["trigger_ids"].astype(str))
df = df.assign(_fp=[hashlib.sha1(x.encode()).hexdigest() for x in fp])
df = df.sort_values("opinion_id").drop_duplicates("_fp", keep="first")
audit.append(("E5 deduplicated", len(df)))

df["type_group"] = df["type"].map(TYPE_GROUP).fillna("other")
df["has_trigger"] = df["n_triggers"] > 0
df["court"] = pd.Categorical(df["court"], categories=FED_APP, ordered=True)
df["period"] = pd.Categorical(df["period"],
                              categories=[p[0] for p in PERIODS], ordered=True)
df.drop(columns=["_fp"]).to_parquet(os.path.join(OUT, "05_eligible.parquet"),
                                    index=False)

for k, v in audit:
    print(f"{v:>12,}  {k}")

# ------------------------------------------------------------------ prevalence
res = {"audit": audit, "dictionary_sha1": DICTIONARY_SHA1,
       "n_eligible": int(len(df)),
       "n_with_trigger": int(df["has_trigger"].sum()),
       "prevalence": float(df["has_trigger"].mean()),
       "n_occurrences": int(df["n_triggers"].sum()),
       "n_words": int(df["n_words"].sum())}
print(f"\npopulation prevalence: {res['prevalence']:.4f} "
      f"({res['n_with_trigger']:,} / {res['n_eligible']:,})")


def table(by):
    g = df.groupby(by, observed=True).agg(
        n=("opinion_id", "size"), k=("has_trigger", "sum"),
        occ=("n_triggers", "sum"), words=("n_words", "sum"))
    g["rate"] = g["k"] / g["n"]
    # Wilson interval; the population is a census, but the interval is reported
    # because the trigger measure is itself a noisy proxy.
    z = 1.96
    p, n = g["rate"], g["n"]
    den = 1 + z ** 2 / n
    g["lo"] = (p + z ** 2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / den
    g["hi"] = (p + z ** 2 / (2 * n) + z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / den
    g["per_100k_words"] = g["occ"] / g["words"] * 1e5
    return g.reset_index()


tabs = {}
for name, by in [("court", ["court"]), ("period", ["period"]),
                 ("court_period", ["court", "period"]),
                 ("type_group", ["type_group"]),
                 ("status", ["precedential_status"]),
                 ("year", ["year"]),
                 ("court_type", ["court", "type_group"])]:
    t = table(by)
    tabs[name] = t
    t.to_csv(os.path.join(OUT, f"05_prev_{name}.csv"), index=False)
    if name in ("court", "period", "type_group", "status"):
        print(f"\n--- {name} ---")
        print(t[by + ["n", "k", "rate", "per_100k_words"]].to_string(index=False))

# per-trigger counts
rows = []
for tid, tname, _ in SPEC:
    hit = df["trigger_ids"].str.contains(rf"(?:^|\|){tid}(?:\||$)", regex=True, na=False)
    rows.append((tid, tname, int(hit.sum()), float(hit.mean())))
tt = pd.DataFrame(rows, columns=["trigger", "expression", "n_opinions", "share_of_corpus"])
tt.to_csv(os.path.join(OUT, "05_prev_trigger.csv"), index=False)
print("\n--- per expression ---")
print(tt.to_string(index=False))

json.dump(res, open(os.path.join(OUT, "05_prevalence.json"), "w"), indent=1, default=str)
