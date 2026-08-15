"""Step 6: build the item frame from which the benchmark is sampled.

Three families of item are produced.

  TRIGGER   A passage containing a dictionary expression. Sub-stratified into
            PLAIN, H1 (the expression is plausibly about another institution or
            sits inside a quotation), and H2 (a later holding sentence in the
            same opinion is about the same proposition).
  CTRL_HOLD A matched passage from an opinion containing no dictionary
            expression at all, located by a holding cue.
  CTRL_RAND A matched passage from a zero-trigger opinion, located by a neutral
            marker that names an issue as an issue without any cue as to how it
            came out, so the benchmark does not rest on one control locator.

Structural flags stratify the sample. They never assign a label.
"""
import json, os, random, sys
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, read_jsonl
from textstore import TextStore
from issues import (issue_clause, sentence_span, h1_other_court, find_neutral_anchors,
                    h2_later_resolution, find_holding_anchors, HOLD, quoted_at)

RNG = random.Random(20260811)
elig = pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"))
elig_ids = set(elig["opinion_id"].tolist())
info = elig.set_index("opinion_id")[
    ["cluster_id", "court", "year", "period", "type_group",
     "precedential_status", "n_words", "citation_count", "case_name"]].to_dict("index")
print(f"eligible opinions: {len(elig):,}", flush=True)

by_op = defaultdict(list)
n_cand_raw = 0
for c in read_jsonl(os.path.join(OUT, "03_candidates.jsonl")):
    n_cand_raw += 1
    if c["opinion_id"] in elig_ids:
        by_op[c["opinion_id"]].append(c)
print(f"candidate passages: {n_cand_raw:,} raw, "
      f"{sum(len(v) for v in by_op.values()):,} in eligible opinions, "
      f"across {len(by_op):,} opinions", flush=True)

ts = TextStore()
rows = []
for i, (oid, cands) in enumerate(by_op.items()):
    if i % 50_000 == 0:
        print(f"  trigger items {i:,}/{len(by_op):,}", file=sys.stderr, flush=True)
    text = ts.get(oid)
    if not text:
        continue
    for c in cands:
        s, e = c["char_start"], c["char_end"]
        iss = issue_clause(text, s, e)
        if not iss:
            continue
        ss, se = sentence_span(text, s, e)
        h1 = h1_other_court(text, s, e)
        h2, h2pos = h2_later_resolution(text, s, e, iss)
        rows.append({
            "item_id": c["cand_id"], "family": "TRIGGER", "opinion_id": oid,
            "trigger_ids": c["trigger_ids"], "matched": c["matched"],
            "char_start": s, "char_end": e, "sent_start": ss, "sent_end": se,
            "issue": iss, "h1_other_court": bool(h1), "h2_later_holding": bool(h2),
            "h2_pos": h2pos, "stratum": "H1" if h1 else ("H2" if h2 else "PLAIN"),
        })

print(f"trigger items with an extractable issue: {len(rows):,}")

# ------------------------------------------------------------------ controls
zero = elig[(elig["n_triggers"] == 0) & (elig["n_words"] >= 400)]
print(f"zero-trigger eligible opinions: {len(zero):,}")
zero = zero.assign(wdec=pd.qcut(zero["n_words"], 10, labels=False, duplicates="drop"))
pool = defaultdict(list)
for oid, court, per, tg, wd in zip(zero["opinion_id"], zero["court"], zero["period"],
                                   zero["type_group"], zero["wdec"]):
    pool[(str(court), str(per), str(tg), int(wd))].append(int(oid))
for v in pool.values():
    RNG.shuffle(v)

trig = pd.DataFrame(rows)
trig["court"] = trig["opinion_id"].map(lambda o: str(info[o]["court"]))
trig["period"] = trig["opinion_id"].map(lambda o: str(info[o]["period"]))
trig["type_group"] = trig["opinion_id"].map(lambda o: str(info[o]["type_group"]))
trig["n_words"] = trig["opinion_id"].map(lambda o: info[o]["n_words"])
wq = elig["n_words"].quantile(np.linspace(0, 1, 11)).values
trig["wdec"] = np.clip(np.searchsorted(wq[1:-1], trig["n_words"]), 0, 9)

# One control opinion is drawn per distinct trigger-bearing opinion, matched on
# court, period, opinion type, and length decile, without replacement. Matching
# every trigger-bearing opinion in the corpus would take far longer than it is
# worth: the control quota is in the hundreds, so a large random subset of
# trigger opinions is matched instead, which leaves the stratification intact.
MAX_CONTROL_PAIRS = 80_000
need = trig.drop_duplicates("opinion_id")[
    ["opinion_id", "court", "period", "type_group", "wdec"]]
if len(need) > MAX_CONTROL_PAIRS:
    need = need.sample(MAX_CONTROL_PAIRS, random_state=20260811)
    print(f"matching controls for a random {MAX_CONTROL_PAIRS:,} of the "
          f"{trig['opinion_id'].nunique():,} trigger-bearing opinions")
used, pairs = set(), {}
for oid, court, per, tg, wd in need.itertuples(index=False):
    for key in [(court, per, tg, int(wd)),
                (court, per, tg, None), (court, per, None, None)]:
        cands = ([] if key[3] is None and key[2] is None
                 else pool.get(key, []) if None not in key else [])
        if None in key:
            cands = [o for k, v in pool.items()
                     if k[0] == key[0] and k[1] == key[1]
                     and (key[2] is None or k[2] == key[2]) for o in v[:40]]
        while cands:
            c = cands.pop()
            if c not in used:
                used.add(c)
                pairs[int(oid)] = c
                break
        if oid in pairs:
            break
print(f"matched control opinions found for {len(pairs):,} / {len(need):,} "
      f"trigger opinions")

crows = []
for k, (toid, coid) in enumerate(pairs.items()):
    if k % 50_000 == 0:
        print(f"  control items {k:,}/{len(pairs):,}", file=sys.stderr, flush=True)
    text = ts.get(coid)
    if not text:
        continue
    for j, (s, e, cue, iss) in enumerate(find_holding_anchors(text, limit=2)):
        ss, se = sentence_span(text, s, e)
        crows.append({"item_id": f"c{coid}-{j}", "family": "CTRL_HOLD",
                      "opinion_id": coid, "trigger_ids": "", "matched": cue,
                      "char_start": s, "char_end": e, "sent_start": ss,
                      "sent_end": se, "issue": iss, "h1_other_court": False,
                      "h2_later_holding": False, "h2_pos": None,
                      "stratum": "CTRL_HOLD", "match_for": toid})
    # An issue named without any disposition cue. This is the control that does
    # not depend on the holding locator, so it tests whether the benchmark is
    # solvable only because controls were found by looking for "we hold".
    for j, (s, e, cue, iss) in enumerate(find_neutral_anchors(text, limit=1)):
        ss, se = sentence_span(text, s, e)
        # A neutral anchor names the issue where it is raised, which is often
        # many pages before the court answers it. Locate the later holding
        # sentence so the annotator is not asked to label from the raising alone.
        later, lpos = h2_later_resolution(text, s, e, iss)
        crows.append({"item_id": f"r{coid}-{j}", "family": "CTRL_RAND",
                      "opinion_id": coid, "trigger_ids": "", "matched": cue,
                      "char_start": s, "char_end": e,
                      "sent_start": ss, "sent_end": se, "issue": iss,
                      "h1_other_court": False, "h2_later_holding": bool(later),
                      "h2_pos": lpos, "stratum": "CTRL_RAND", "match_for": toid})

ctrl = pd.DataFrame(crows)
for c in ("court", "period", "type_group"):
    ctrl[c] = ctrl["opinion_id"].map(lambda o: str(info[o][c]))
ctrl["n_words"] = ctrl["opinion_id"].map(lambda o: info[o]["n_words"])
ctrl["wdec"] = np.clip(np.searchsorted(wq[1:-1], ctrl["n_words"]), 0, 9)

frame = pd.concat([trig, ctrl], ignore_index=True)
frame["citation_count"] = frame["opinion_id"].map(lambda o: info[o]["citation_count"])
frame["case_name"] = frame["opinion_id"].map(lambda o: info[o]["case_name"])
frame["year"] = frame["opinion_id"].map(lambda o: info[o]["year"])
frame.to_parquet(os.path.join(OUT, "06_frame.parquet"), index=False)
print("\nframe by stratum:")
print(frame["stratum"].value_counts().to_string())
print("\nframe by family x period:")
print(pd.crosstab(frame["family"], frame["period"]).to_string())
ts.close()
