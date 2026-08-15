"""Step 10a: build issue-specific citation chains.

A case-level citation graph is not enough. An opinion that left issue q open is
usually cited for something else entirely, so the unit here is a citing passage:
the window of text in a later opinion that surrounds an actual reporter citation
to the origin, restricted to passages whose content words overlap the issue.
"""
import json, os, random, re, sys
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, stream_csv
from textstore import TextStore
from issues import content_words, sentence_span

MIN_OVERLAP = 3
WIN = 900
MAX_CITERS = 60
RNG = random.Random(31337)

ann = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
ann = ann[ann["label"].isin(["DECIDED", "UNRESOLVED"])]
print(f"annotated items usable as origins: {len(ann):,}")

elig = pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"))
op2cl = dict(zip(elig["opinion_id"], elig["cluster_id"]))
meta = elig.set_index("opinion_id")[["court", "year", "case_name",
                                     "citation_count", "n_words"]].to_dict("index")

# --- reporter citations, so that a citing passage can be located in raw text
cl2cite = defaultdict(list)
for r in stream_csv(os.path.join(DATA, "citations.csv.bz2"),
                    want=["volume", "reporter", "page", "cluster_id"],
                    progress_every=2_000_000, label="citations"):
    try:
        cl2cite[int(r["cluster_id"])].append(
            (r["volume"].strip(), r["reporter"].strip(), r["page"].strip()))
    except (ValueError, TypeError):
        continue
print(f"clusters with a reporter citation: {len(cl2cite):,}")

edges = pd.read_parquet(os.path.join(OUT, "04_citations.parquet"))
edges = edges[edges["citing_in_pop"]]
citers = defaultdict(list)
for a, b in zip(edges["citing_id"], edges["cited_id"]):
    citers[b].append(a)
print(f"cited opinions with in-population citers: {len(citers):,}")

ts = TextStore()
rows, no_cite, no_locate = [], 0, 0
for i, r in enumerate(ann.itertuples(index=False)):
    if i % 200 == 0:
        print(f"  origin {i}/{len(ann)}", file=sys.stderr, flush=True)
    oid = int(r.opinion_id)
    cl = op2cl.get(oid)
    cites = cl2cite.get(cl, [])
    if not cites:
        no_cite += 1
        continue
    pats = [re.compile(rf"\b{re.escape(v)}\s+{re.escape(rep)}\.?\s+{re.escape(p)}\b")
            for v, rep, p in cites if v and rep and p]
    if not pats:
        no_cite += 1
        continue
    keys = content_words(r.issue)
    all_citers = citers.get(oid, [])
    # A handful of origins are cited thousands of times. Scanning every citer
    # would let those few dominate both the runtime and the sample, so a random
    # subset is examined and the true in-degree is recorded alongside it.
    if len(all_citers) > MAX_CITERS:
        examined = RNG.sample(all_citers, MAX_CITERS)
    else:
        examined = all_citers
    for cit in examined:
        t = ts.get(cit)
        if not t:
            continue
        hit = None
        for pat in pats:
            m = pat.search(t)
            if m:
                hit = m
                break
        if not hit:
            no_locate += 1
            continue
        s, e = hit.start(), hit.end()
        ss, se = max(0, s - WIN), min(len(t), e + WIN)
        passage = t[ss:se]
        ov = len(content_words(passage) & keys)
        info = meta.get(cit, {})
        rows.append({
            "origin_item_id": r.item_id, "origin_opinion_id": oid,
            "origin_label": r.label, "origin_sublabel": r.sublabel,
            "origin_court": r.court, "origin_year": int(r.year),
            "origin_char_start": int(r.char_start), "origin_char_end": int(r.char_end),
            "origin_sent_start": int(r.sent_start), "origin_sent_end": int(r.sent_end),
            "origin_n_citers": len(all_citers), "n_examined": len(examined),
            "issue": r.issue, "citing_id": cit,
            "citing_court": str(info.get("court")), "citing_year": info.get("year"),
            "citing_case": info.get("case_name"), "cite_start": s, "cite_end": e,
            "win_start": ss, "win_end": se, "overlap": ov,
            "issue_specific": ov >= MIN_OVERLAP,
            "same_court": str(info.get("court")) == str(r.court),
            "is_scotus": str(info.get("court")) == "scotus",
        })
ts.close()

ch = pd.DataFrame(rows)
ch.to_parquet(os.path.join(OUT, "10a_chains.parquet"), index=False)
print(f"\ncitation passages located: {len(ch):,}")
print(f"origins with no reporter citation: {no_cite}; citers where the "
      f"citation string could not be located in text: {no_locate:,}")
if len(ch):
    print(f"issue-specific (overlap >= {MIN_OVERLAP}): "
          f"{ch['issue_specific'].mean():.3f}")
    print(pd.crosstab(ch["origin_label"], ch["issue_specific"]).to_string())
    print("\nciting court relation:")
    print(ch.groupby("origin_label")[["same_court", "is_scotus"]].mean().to_string())
