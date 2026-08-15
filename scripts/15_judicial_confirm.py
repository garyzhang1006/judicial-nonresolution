"""Step 15: judicially confirmed labels.

The weakest point in this study is that its labels were produced by one
annotator. This step builds a validation set that no annotator touched.

When a later court writes "we left that question open in Smith" it is asserting,
on the record and under its own name, that Smith did not decide the question.
That is a label for Smith, authored by a federal judge. The same holds in the
other direction for "Smith held that." Harvesting both gives an external
standard against which the annotated labels can be checked.

The characterization dictionary below is frozen in the same sense as the trigger
dictionary: fixed before it was run against anything, and reported in full.
"""
import json, os, re, sys
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, stream_csv
from textstore import TextStore

# ---------------------------------------------------------------- dictionary
# L0: the later court says the cited court did not resolve the point.
NOT_DECIDED = re.compile(
    r"\b(?:left|leaving)\s+open\b"
    r"|\b(?:did|does)\s+not\s+(?:decide|reach|resolve|address)\b"
    r"|\b(?:declin\w+|refus\w+)\s+to\s+(?:decide|reach|resolve|address)\b"
    r"|\bexpress\w*\s+no\s+(?:opinion|view)\b"
    r"|\breserv\w+\s+(?:the\s+)?(?:question|issue)\b"
    r"|\bassum\w+\s+without\s+deciding\b"
    r"|\bwithout\s+(?:deciding|resolving)\b"
    r"|\bnot\s+decided\b|\bunresolved\b|\bleft\s+unanswered\b", re.I)

# L3: the later court says the cited court did resolve it.
DECIDED = re.compile(
    r"\bheld\b|\bholding\b|\bwe\s+decided\b|\bthe\s+court\s+decided\b"
    r"|\bconcluded\b|\bestablished\b|\bsquarely\s+(?:held|decided)\b"
    r"|\brejected\b|\badopted\b", re.I)

CITE = re.compile(
    r"\b(\d{1,4})\s+(F\.\s?\d?d|U\.\s?S\.|S\.\s?Ct\.|F\.\s?App'?x)\.?\s+(\d{1,4})\b")

WIN_BEFORE, WIN_AFTER = 260, 200


def norm_reporter(r):
    return re.sub(r"[\s.]", "", r).upper().replace("APPX", "FAPPX")


def main():
    elig = pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"),
                           columns=["opinion_id", "cluster_id", "court", "year"])
    op_of_cluster = defaultdict(list)
    for o, c in zip(elig["opinion_id"], elig["cluster_id"]):
        op_of_cluster[c].append(o)

    # reporter citation -> cluster
    cite2cluster = {}
    for r in stream_csv(os.path.join(DATA, "citations.csv.bz2"),
                        want=["volume", "reporter", "page", "cluster_id"],
                        progress_every=4_000_000, label="citations"):
        try:
            key = (r["volume"].strip(), norm_reporter(r["reporter"]), r["page"].strip())
            cite2cluster[key] = int(r["cluster_id"])
        except (ValueError, TypeError):
            continue
    print(f"reporter citations indexed: {len(cite2cluster):,}", flush=True)

    ts = TextStore()
    ids = elig["opinion_id"].tolist()
    rows, seen = [], 0
    for n, oid in enumerate(ids):
        if n % 100_000 == 0:
            print(f"  scanned {n:,}/{len(ids):,}, kept {len(rows):,}",
                  file=sys.stderr, flush=True)
        t = ts.get(oid)
        if not t:
            continue
        for m in CITE.finditer(t):
            seen += 1
            lo = max(0, m.start() - WIN_BEFORE)
            hi = min(len(t), m.end() + WIN_AFTER)
            win = t[lo:hi]
            nd = bool(NOT_DECIDED.search(win))
            dd = bool(DECIDED.search(win))
            if nd == dd:            # need exactly one signal, else ambiguous
                continue
            key = (m.group(1), norm_reporter(m.group(2)), m.group(3))
            cl = cite2cluster.get(key)
            if cl is None:
                continue
            for tgt in op_of_cluster.get(cl, []):
                if tgt == oid:
                    continue
                rows.append({"citing_id": oid, "cited_opinion_id": tgt,
                             "cited_cluster": cl, "verdict": "L0" if nd else "L3",
                             "cite_pos": m.start(), "passage": win})
    ts.close()

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "15_judicial_confirm.parquet"), index=False)
    print(f"\ncitations scanned: {seen:,}")
    print(f"characterizations extracted: {len(df):,}")
    if len(df):
        print(df["verdict"].value_counts().to_string())
        print(f"distinct cited opinions: {df['cited_opinion_id'].nunique():,}")
    json.dump({"n_citations_scanned": seen, "n_characterizations": int(len(df)),
               "n_cited_opinions": int(df["cited_opinion_id"].nunique()) if len(df) else 0},
              open(os.path.join(OUT, "15_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
