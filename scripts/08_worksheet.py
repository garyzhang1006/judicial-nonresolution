"""Step 8: render annotation worksheets and ingest completed labels.

Items are emitted in a fixed random order that ignores stratum, split, court and
period. Annotation therefore proceeds down a shuffled list, and whatever prefix
is completed remains a valid random subsample of the drawn benchmark rather than
a convenience sample of whichever items were easiest or came first.

  python 08_worksheet.py render <batch> [n_per_batch]
  python 08_worksheet.py ingest
"""
import json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore

TAG = "" if os.environ.get("WS_SOURCE", "07") .startswith("07") else "_recall"
WS = os.path.join(OUT, "worksheets" + TAG)
LB = os.path.join(OUT, "labels" + TAG)
os.makedirs(WS, exist_ok=True)
os.makedirs(LB, exist_ok=True)
PRE, POST = 520, 560
CODES = {"D": ("DECIDED", ""), "A": ("UNRESOLVED", "AVOIDED"),
         "S": ("UNRESOLVED", "ASSUMED"), "R": ("UNRESOLVED", "RESERVED"),
         "X": ("UNCLEAR", "")}


SOURCE = os.environ.get("WS_SOURCE", "07_benchmark.parquet")
SEEDS = {"07_benchmark.parquet": 99991, "13_recall_frame.parquet": 77771}


def order():
    b = pd.read_parquet(os.path.join(OUT, SOURCE))
    for c in ("year", "type_group", "h2_pos", "split"):
        if c not in b.columns:
            b[c] = None
    return b.sample(frac=1.0, random_state=SEEDS.get(SOURCE, 12345)).reset_index(drop=True)


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def render_sentonly(batch, n, pool=160):
    """Re-render items showing only the anchor sentence.

    A second pass over the same items with the context removed measures how much
    of the annotator's own decision depended on material outside the sentence.
    It is not an inter-annotator statistic and is not reported as one. It gives a
    human reference point for the context ablation the models are put through.
    """
    b = order()
    sub = b.head(pool).sample(frac=1.0, random_state=4242).reset_index()
    lo, hi = batch * n, min((batch + 1) * n, len(sub))
    if lo >= len(sub):
        print("no items left")
        return
    ts, out = TextStore(), []
    for k in range(lo, hi):
        r = sub.iloc[k]
        t = ts.get(r["opinion_id"]) or ""
        seg = clean(t[int(r["sent_start"]):int(r["sent_end"])])
        out.append(f"### {k:04d} | {r['stratum']}\nISSUE: "
                   f"{clean(str(r['issue']))[:240]}\nSENT: {seg[:900]}")
    ts.close()
    path = os.path.join(WS, f"sent_{batch:03d}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# ANCHOR SENTENCE ONLY. Label from this alone.\n"
                 "# codes: D=decided  A=avoided  S=assumed  R=reserved  X=unclear\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"wrote {path} ({hi-lo} items)")


def ablation_stats(pool=160):
    """Compare sentence-only labels against the full-context labels."""
    b = order()
    sub = b.head(pool).sample(frac=1.0, random_state=4242).reset_index()
    full = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
    fmap = dict(zip(full["item_id"], full["code"]))
    smap = {}
    for fn in sorted(os.listdir(LB)):
        if not fn.startswith("sent_"):
            continue
        for line in open(os.path.join(LB, fn), encoding="utf-8"):
            m = re.match(r"\s*(\d{1,4})\s+([DASRX])\s*$", line)
            if m:
                smap[int(m.group(1))] = m.group(2)
    rows = []
    for k, code in smap.items():
        if k >= len(sub):
            continue
        iid = sub.iloc[k]["item_id"]
        if iid in fmap:
            rows.append({"item_id": iid, "stratum": sub.iloc[k]["stratum"],
                         "sent_code": code, "full_code": fmap[iid]})
    d = pd.DataFrame(rows)
    if not len(d):
        print("no paired items")
        return d
    bin_ = lambda c: "X" if c == "X" else ("U" if c in "ASR" else "D")
    d["sent_bin"] = d["sent_code"].map(bin_)
    d["full_bin"] = d["full_code"].map(bin_)
    d["agree"] = d["sent_bin"] == d["full_bin"]
    d.to_csv(os.path.join(OUT, "08_human_ablation.csv"), index=False)
    print(f"paired items: {len(d)}; sentence-only agrees with full context "
          f"on {d['agree'].mean():.3f}")
    print(d.groupby("stratum")["agree"].agg(["size", "mean"]).to_string())
    print(f"sentence-only UNCLEAR rate {(d['sent_bin']=='X').mean():.3f} "
          f"vs full-context {(d['full_bin']=='X').mean():.3f}")
    return d


def render_recheck(batch, n, pool=260):
    """Second pass over already-labelled items, blind to the first pass.

    Items are drawn in a different random order from the annotation order and
    carry fresh sequential ids, so the first-pass label cannot be read off the
    position. This measures whether the guideline is applied consistently. It is
    intra-annotator agreement and is reported as such; it does not establish
    that a second person would agree.
    """
    done = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
    done = done[done["label"].isin(["DECIDED", "UNRESOLVED", "UNCLEAR"])]
    sub = done.sample(min(pool, len(done)), random_state=8123).reset_index(drop=True)
    lo, hi = batch * n, min((batch + 1) * n, len(sub))
    if lo >= len(sub):
        print("no items left"); return
    ts, out = TextStore(), []
    for k in range(lo, hi):
        r = sub.iloc[k]
        t = ts.get(r["opinion_id"]) or ""
        s, e = int(r["char_start"]), int(r["char_end"])
        left, right = max(0, s - PRE), min(len(t), e + POST)
        c = (("..." if left > 0 else "") + clean(t[left:s]) + " <<" + clean(t[s:e])
             + ">> " + clean(t[e:right]) + ("..." if right < len(t) else ""))
        block = [f"### {k:04d}", f"ISSUE: {clean(str(r['issue']))[:260]}", f"CTX: {c}"]
        if pd.notna(r["h2_pos"]):
            pz = int(r["h2_pos"])
            if not (left <= pz <= right):
                block.append("LATER: ..." + clean(t[max(0, pz - 60):pz + 460]) + "...")
        out.append("\n".join(block))
    ts.close()
    sub[["item_id"]].to_csv(os.path.join(OUT, "recheck_order.csv"), index=False)
    path = os.path.join(WS, f"recheck_{batch:03d}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# SECOND PASS. Label from scratch.\n"
                 "# codes: D=decided  A=avoided  S=assumed  R=reserved  X=unclear\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"wrote {path} ({hi-lo} items, {os.path.getsize(path):,} bytes)")


def recheck_stats():
    order = pd.read_csv(os.path.join(OUT, "recheck_order.csv"))["item_id"].tolist()
    done = pd.read_parquet(os.path.join(OUT, "08_annotated.parquet"))
    first = dict(zip(done["item_id"], done["code"]))
    second = {}
    for fn in sorted(os.listdir(LB)):
        if not fn.startswith("recheck_"):
            continue
        for line in open(os.path.join(LB, fn), encoding="utf-8"):
            m = re.match(r"\s*(\d{1,4})\s+([DASRX])\s*$", line)
            if m:
                second[int(m.group(1))] = m.group(2)
    rows = []
    for k, c2 in second.items():
        if k >= len(order):
            continue
        iid = order[k]
        if iid in first:
            rows.append({"item_id": iid, "first": first[iid], "second": c2})
    d = pd.DataFrame(rows)
    if not len(d):
        print("no paired items"); return d
    b = lambda c: "X" if c == "X" else ("U" if c in "ASR" else "D")
    d["f_bin"], d["s_bin"] = d["first"].map(b), d["second"].map(b)
    d["agree"] = d["f_bin"] == d["s_bin"]
    d.to_csv(os.path.join(OUT, "08_recheck.csv"), index=False)
    po = d["agree"].mean()
    pa = d["f_bin"].value_counts(normalize=True); pb = d["s_bin"].value_counts(normalize=True)
    pe = sum(pa.get(k, 0) * pb.get(k, 0) for k in ("D", "U", "X"))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"paired {len(d)}; raw agreement {po:.3f}; Cohen's kappa {kappa:.3f}")
    print(pd.crosstab(d["f_bin"], d["s_bin"]).to_string())
    # binary-only, dropping items either pass called UNCLEAR
    bb = d[(d.f_bin != "X") & (d.s_bin != "X")]
    if len(bb):
        po2 = (bb.f_bin == bb.s_bin).mean()
        pa2 = bb.f_bin.value_counts(normalize=True); pb2 = bb.s_bin.value_counts(normalize=True)
        pe2 = sum(pa2.get(k, 0) * pb2.get(k, 0) for k in ("D", "U"))
        print(f"binary-only: n={len(bb)} agreement {po2:.3f} kappa "
              f"{(po2-pe2)/(1-pe2):.3f}")
    return d


def render(batch, n):
    b = order()
    lo, hi = batch * n, min((batch + 1) * n, len(b))
    if lo >= len(b):
        print("no items left")
        return
    ts, out = TextStore(), []
    for k in range(lo, hi):
        r = b.iloc[k]
        t = ts.get(r["opinion_id"]) or ""
        s, e = int(r["char_start"]), int(r["char_end"])
        left, right = max(0, s - PRE), min(len(t), e + POST)
        ws, we = left, right
        ctx = (("..." if left > 0 else "") + clean(t[left:s])
               + " <<" + clean(t[s:e]) + ">> " + clean(t[e:right])
               + ("..." if right < len(t) else ""))
        yr = int(r["year"]) if pd.notna(r["year"]) else 0
        head = (f"### {k:04d} | {r['court']} {yr} | "
                f"{r['type_group']} | {r['stratum']}")
        block = [head, f"ISSUE: {clean(str(r['issue']))[:260]}", f"CTX: {ctx}"]
        # Where the opinion later returns to the same proposition in a holding
        # sentence, show it. Rule L1-3 turns on exactly that passage, and
        # withholding it would make the item unanswerable rather than hard.
        if pd.notna(r["h2_pos"]):
            p = int(r["h2_pos"])
            if not (ws <= p <= we):
                block.append("LATER: ..." + clean(t[max(0, p - 60):p + 460]) + "...")
        out.append("\n".join(block))
    ts.close()
    path = os.path.join(WS, f"batch_{batch:03d}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# items {lo}-{hi-1} of {len(b)}\n"
                 f"# codes: D=decided  A=avoided  S=assumed  R=reserved  X=unclear\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"wrote {path}  ({hi-lo} items, {os.path.getsize(path):,} bytes)")


def ingest():
    b = order()
    labels = {}
    for fn in sorted(os.listdir(LB)):
        # sent_*.txt holds the context-ablation pass. Its indices run over a
        # different ordering, so reading it here silently overwrites main labels.
        if not fn.endswith(".txt") or fn.startswith("sent_"):
            continue
        for line in open(os.path.join(LB, fn), encoding="utf-8"):
            m = re.match(r"\s*(\d{1,4})\s+([DASRX])\s*$", line)
            if m:
                labels[int(m.group(1))] = m.group(2)
    print(f"parsed {len(labels)} labels from {len(os.listdir(LB))} files")
    rec = []
    for k, code in sorted(labels.items()):
        if k >= len(b):
            continue
        r = b.iloc[k]
        lab, sub = CODES[code]
        rec.append({**{c: r[c] for c in b.columns}, "order_idx": k,
                    "code": code, "label": lab, "sublabel": sub})
    df = pd.DataFrame(rec)
    df.to_parquet(os.path.join(OUT, f"08_annotated{TAG}.parquet"), index=False)
    print(f"annotated items: {len(df):,}")
    if len(df):
        print(df["label"].value_counts().to_string())
        print(pd.crosstab(df["stratum"], df["label"]).to_string())
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if cmd == "render":
        render(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 65)
    elif cmd == "sentonly":
        render_sentonly(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 80)
    elif cmd == "recheck":
        render_recheck(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 65)
    elif cmd == "recheckstats":
        recheck_stats()
    elif cmd == "ablation":
        ablation_stats()
    else:
        ingest()
