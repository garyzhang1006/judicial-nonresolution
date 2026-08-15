"""Step 7: draw the stratified benchmark and define the evaluation splits.

Natural frequencies are preserved in the population tables produced by step 5.
The benchmark deliberately departs from them, oversampling the strata that carry
the most information about whether a system is reasoning or pattern matching.
Both are reported.

Splits are nested and disjoint, so no opinion appears in more than one of them:
  COURT     every item from two reserved courts
  TEMPORAL  every remaining item filed in 2018 or later
  TEST      a random half of what is left
  DEV       the other half, the only portion available for model development
"""
import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, FED_APP, PERIODS

SEED = 20260811
HELD_OUT_COURTS = ["ca5", "cadc"]   # one large regional circuit, one specialized
TEMPORAL_CUT = 2018
# Allocation follows the pilot, which found the H1 stratum to be close to evenly
# split between the two labels while PLAIN and CTRL_HOLD are nearly degenerate.
# Sample is therefore moved toward H1 for power. This changes how many items are
# drawn from each stratum; it changes no label definition and no stratum
# definition, both of which were fixed before the pilot ran.
QUOTA = {"PLAIN": 500, "H1": 450, "H2": 300, "CTRL_HOLD": 400, "CTRL_RAND": 150}

frame = pd.read_parquet(os.path.join(OUT, "06_frame.parquet"))
print("frame size:", f"{len(frame):,}")
print(frame["stratum"].value_counts().to_string())

# At most one item per opinion, so that context windows never overlap and the
# splits can be made disjoint at the opinion level.
frame = frame.sample(frac=1.0, random_state=SEED).drop_duplicates("opinion_id")
print(f"after one-item-per-opinion: {len(frame):,}")

parts = []
for stratum, n in QUOTA.items():
    sub = frame[frame["stratum"] == stratum]
    if len(sub) == 0:
        print(f"!! stratum {stratum} empty")
        continue
    # proportional-to-target allocation over court x period, then top up
    cell = sub.groupby(["court", "period"], observed=True)
    per_cell = max(1, n // max(1, cell.ngroups))
    took = cell.apply(lambda g: g.sample(min(len(g), per_cell), random_state=SEED),
                      include_groups=False).reset_index()
    took = took.merge(sub, on="item_id", suffixes=("", "_y"))
    if len(took) < n:
        rest = sub[~sub["item_id"].isin(took["item_id"])]
        took = pd.concat([took, rest.sample(min(len(rest), n - len(took)),
                                            random_state=SEED)])
    took = took.sample(min(n, len(took)), random_state=SEED)
    parts.append(took[frame.columns])
    print(f"{stratum}: requested {n}, drew {len(took)} from {len(sub):,} available")

bench = pd.concat(parts, ignore_index=True).drop_duplicates("item_id")

split = pd.Series("DEV", index=bench.index)
split[bench["court"].isin(HELD_OUT_COURTS)] = "COURT"
rest = split == "DEV"
split[rest & (bench["year"] >= TEMPORAL_CUT)] = "TEMPORAL"
pool = bench.index[split == "DEV"]
test_idx = bench.loc[pool].sample(frac=0.45, random_state=SEED).index
split[test_idx] = "TEST"
bench["split"] = split

print("\nsplit sizes:")
print(bench["split"].value_counts().to_string())
print("\nstratum x split:")
print(pd.crosstab(bench["stratum"], bench["split"]).to_string())
print("\ncourt x split:")
print(pd.crosstab(bench["court"], bench["split"]).to_string())
print("\nperiod x split:")
print(pd.crosstab(bench["period"], bench["split"]).to_string())

bench.to_parquet(os.path.join(OUT, "07_benchmark.parquet"), index=False)
json.dump({"seed": SEED, "quota": QUOTA, "held_out_courts": HELD_OUT_COURTS,
           "temporal_cut": TEMPORAL_CUT, "n": int(len(bench)),
           "splits": bench["split"].value_counts().to_dict()},
          open(os.path.join(OUT, "07_sample_config.json"), "w"), indent=1)
print(f"\nwrote {len(bench):,} benchmark items")
