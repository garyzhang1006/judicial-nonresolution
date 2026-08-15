"""Step 13: bound what the frozen dictionary misses.

The dictionary was frozen before annotation, which protects the design but says
nothing about its coverage. Reading whole opinions at random is a weak audit: a
non-resolution can sit anywhere in forty pages and most opinions contain none.

Instead we run a second, deliberately wider probe list over the opinions the
dictionary called clean. The probe was written after the fact and is used only
here; it never touches candidate generation, the benchmark, or any prevalence
figure. It gives a lower bound on missed non-resolution, since it too is lexical,
and it identifies which expressions the frozen list should have contained.
"""
import json, os, random, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from issues import issue_clause, sentence_span

N_SAMPLE = 4000
SEED = 5150

PROBE = [
    ("P01", "need not address", r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?address\b"),
    ("P02", "need not consider", r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?consider\b"),
    ("P03", "need not resolve", r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?resolve\b"),
    ("P04", "no occasion to decide", r"\bno\s+occasion\s+to\s+(?:decide|reach|address|resolve)\b"),
    ("P05", "leave open the question", r"leav(?:e|es|ing)\s+(?:\w+\s+){0,3}?open\b"),
    ("P06", "decline to consider/resolve/reach",
     r"declin(?:e|es|ed|ing)\s+to\s+(?:\w+\s+){0,2}?(?:consider|resolve|reach|opine)\b"),
    ("P07", "assume arguendo", r"\bassum(?:e|es|ed|ing)\s+arguendo\b|\barguendo\b"),
    ("P08", "we may assume", r"\bwe\s+(?:may|will|shall)\s+assume\b"),
    ("P09", "pretermit", r"\bpretermit(?:s|ted|ting)?\b"),
    ("P10", "do not address/consider/resolve",
     r"\bdo(?:es)?\s+not\s+(?:\w+\s+){0,2}?(?:address|consider|resolve)\b"),
    ("P11", "save for another day", r"\bsav(?:e|es|ing)\s+(?:\w+\s+){0,3}?for\s+another\s+day\b"),
    ("P12", "for purposes of this appeal we assume",
     r"for\s+(?:present\s+)?purposes\s+(?:of\s+this\s+(?:appeal|case)\s+)?"
     r"(?:only\s+)?,?\s*we\s+(?:assume|accept)\b"),
    ("P13", "intimate no view", r"\bintimat(?:e|es|ing)\s+no\s+(?:view|opinion)\b"),
    ("P14", "put to one side / set aside the question",
     r"\bput\s+(?:that|this|the)\s+(?:question|issue)\s+to\s+one\s+side\b"),
    ("P15", "stop short of deciding", r"\bstop\s+short\s+of\s+(?:deciding|holding)\b"),
]
PATS = [(pid, name, re.compile(rx, re.I)) for pid, name, rx in PROBE]
PROBE_UNION = re.compile("|".join(f"(?:{rx})" for _, _, rx in PROBE), re.I)

elig = pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"))
clean = elig[(elig["n_triggers"] == 0) & (elig["n_words"] >= 400)]
print(f"opinions the dictionary called clean: {len(clean):,}")
samp = clean.sample(min(N_SAMPLE, len(clean)), random_state=SEED)

ts = TextStore()
hits, per_probe, n_any = [], {p[0]: 0 for p in PROBE}, 0
for r in samp.itertuples(index=False):
    t = ts.get(r.opinion_id)
    if not t or not PROBE_UNION.search(t):
        continue
    n_any += 1
    fired = []
    for pid, name, pat in PATS:
        m = pat.search(t)
        if m:
            per_probe[pid] += 1
            fired.append((pid, m))
    pid, m = fired[0]
    iss = issue_clause(t, m.start(), m.end())
    ss, se = sentence_span(t, m.start(), m.end())
    hits.append({"opinion_id": int(r.opinion_id), "court": str(r.court),
                 "year": int(r.year), "probe": "|".join(p for p, _ in fired),
                 "matched": m.group(0), "char_start": m.start(),
                 "char_end": m.end(), "sent_start": ss, "sent_end": se,
                 "issue": iss, "stratum": "RECALL", "h2_pos": None,
                 "item_id": f"p{r.opinion_id}"})
ts.close()

df = pd.DataFrame(hits)
df.to_parquet(os.path.join(OUT, "13_recall_frame.parquet"), index=False)
res = {"n_sampled": int(len(samp)), "n_with_probe": int(n_any),
       "share_with_probe": n_any / max(1, len(samp)),
       "n_with_extractable_issue": int(df["issue"].notna().sum()) if len(df) else 0,
       "per_probe": {p[0]: {"name": p[1], "n": per_probe[p[0]],
                            "share": per_probe[p[0]] / max(1, len(samp))}
                     for p in PROBE}}
json.dump(res, open(os.path.join(OUT, "13_recall.json"), "w"), indent=1)
print(f"of {len(samp):,} sampled clean opinions, {n_any:,} "
      f"({n_any/max(1,len(samp)):.3f}) contain probe language")
for p in PROBE:
    print(f"  {p[0]} {p[1]:38s} {per_probe[p[0]]:6,} "
          f"({per_probe[p[0]]/max(1,len(samp)):.4f})")
