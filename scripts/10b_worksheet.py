"""Step 10b: render and ingest Layer-2 annotation of citing passages.

  python 10b_worksheet.py render <batch> [n]
  python 10b_worksheet.py ingest

Codes combine the status level with the two independent flags:
  0/1/2/3  status ascribed to the origin  (L0 unresolved ... L3 established)
  a        the later opinion attributes a resolution to the origin  (A=1)
  e        the later opinion resolves the issue itself              (E=1)
  o        off-issue          x  unusable
Example: "0142 2ae" is status L2, A=1, E=1.
"""
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore

MIN_OV = int(os.environ.get("L2_MIN_OVERLAP", 5))
TAG = "" if MIN_OV == 3 else "_r"
WS = os.path.join(OUT, "worksheets2" + TAG)
LB = os.path.join(OUT, "labels2" + TAG)
os.makedirs(WS, exist_ok=True)
os.makedirs(LB, exist_ok=True)
SEED = 4242
PER_ORIGIN = 3


def order():
    ch = pd.read_parquet(os.path.join(OUT, "10a_chains.parquet"))
    # Refined after a 23-passage pilot: raising the overlap threshold from 3 to 5
    # roughly doubled the share of passages that are genuinely about the issue.
    ch = ch[ch["overlap"] >= MIN_OV]
    # Cap per origin so that one heavily cited case cannot dominate the sample,
    # and so the clustered bootstrap has many small clusters rather than a few
    # large ones.
    ch = (ch.sample(frac=1.0, random_state=SEED)
            .groupby("origin_item_id", group_keys=False).head(PER_ORIGIN))
    return ch.sample(frac=1.0, random_state=SEED + 1).reset_index(drop=True)


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def render(batch, n):
    ch = order()
    lo, hi = batch * n, min((batch + 1) * n, len(ch))
    if lo >= len(ch):
        print("no items left")
        return
    ts, out = TextStore(), []
    for k in range(lo, hi):
        r = ch.iloc[k]
        t = ts.get(r["citing_id"]) or ""
        s, e = int(r["cite_start"]), int(r["cite_end"])
        ws, we = int(r["win_start"]), int(r["win_end"])
        passage = (clean(t[ws:s]) + " <<" + clean(t[s:e]) + ">> " + clean(t[e:we]))
        # The origin's own words are shown, never its label. Telling the
        # annotator that the origin left the issue open would make the
        # escalation contrast a comparison of primed judgements.
        ot = ts.get(r["origin_opinion_id"]) or ""
        oa, ob = int(r["origin_char_start"]), int(r["origin_char_end"])
        origin = (clean(ot[max(0, oa - 320):oa]) + " <<" + clean(ot[oa:ob])
                  + ">> " + clean(ot[ob:min(len(ot), ob + 360)]))
        out.append("\n".join([
            f"### {k:04d} | origin {r['origin_court']} {r['origin_year']}"
            f" -> citing {r['citing_court']} {r['citing_year']}",
            f"ISSUE: {clean(str(r['issue']))[:240]}",
            f"ORIGIN: ...{origin}...",
            f"CITING: {passage}"]))
    ts.close()
    path = os.path.join(WS, f"batch_{batch:03d}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# items {lo}-{hi-1} of {len(ch)}\n"
                 f"# code: <status 0-3>[a][e] | o=off-issue | x=unusable\n\n")
        fh.write("\n\n".join(out) + "\n")
    print(f"wrote {path} ({hi-lo} items, {os.path.getsize(path):,} bytes)")


def ingest():
    ch = order()
    labels = {}
    for fn in sorted(os.listdir(LB)):
        if not fn.endswith(".txt"):
            continue
        for line in open(os.path.join(LB, fn), encoding="utf-8"):
            m = re.match(r"\s*(\d{1,4})\s+([0-3][ae]{0,2}|o|x)\s*$", line.strip())
            if m:
                labels[int(m.group(1))] = m.group(2)
    rec = []
    for k, code in sorted(labels.items()):
        if k >= len(ch):
            continue
        r = ch.iloc[k]
        d = {c: r[c] for c in ch.columns}
        d["order_idx"], d["code"] = k, code
        if code in ("o", "x"):
            d.update(status=None, A=None, E=None, usable=False,
                     off_issue=(code == "o"))
        else:
            d.update(status=int(code[0]), A=int("a" in code), E=int("e" in code),
                     usable=True, off_issue=False)
        rec.append(d)
    df = pd.DataFrame(rec)
    df.to_parquet(os.path.join(OUT, f"10b_annotated{TAG}.parquet"), index=False)
    print(f"layer-2 annotations: {len(df):,}")
    if len(df):
        u = df[df["usable"]]
        print(f"usable: {len(u):,}  off-issue: {int(df['off_issue'].sum())}")
        if len(u):
            print(pd.crosstab(u["origin_label"], u["status"]).to_string())
            print(u.groupby("origin_label")[["A", "E"]].mean().to_string())
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if cmd == "render":
        render(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 55)
    else:
        ingest()
