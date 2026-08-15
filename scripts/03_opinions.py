"""Step 3: single streaming pass over the 54 GB opinions table.

Everything that requires touching raw opinion text happens here: federal
appellate filtering, markup stripping, text-source precedence, trigger matching,
and sharded storage with a byte-offset index so later stages can seek to any
opinion without rereading the bulk file.

A cheap substring gate runs before the regex union, because only a minority of
opinions can possibly contain a trigger and regex scanning dominates the cost.
"""
import csv, io, json, os, subprocess, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, best_text
from triggers import find_all, merge_overlaps, UNION, DICTIONARY_SHA1

csv.field_size_limit(sys.maxsize)

SHARDS = 64
# A dry run over the leading portion of the bulk file validates every downstream
# stage before the full pass is committed to.
MAXROWS = int(os.environ.get("MAXROWS", "0")) or None
SUF = os.environ.get("SUF", "")
TEXTDIR = os.path.join(OUT, "text" + SUF)
os.makedirs(TEXTDIR, exist_ok=True)

# Necessary-condition substrings: every pattern in the dictionary contains at
# least one of these, so a document failing all of them cannot match.
GATE = ("decid", "reach", "declin", "reserv", "another day", "another case",
        "another occasion", "no opinion", "necessary to")

WANT = ["id", "cluster_id", "type", "per_curiam", "author_str", "joined_by_str",
        "page_count", "extracted_by_ocr", "sha1", "download_url",
        "html_columbia", "html_lawbox", "xml_harvard", "html_anon_2020",
        "html", "plain_text", "html_with_citations"]

clusters = pd.read_parquet(os.path.join(OUT, "02_clusters.parquet"),
                           columns=["cluster_id"])
keep = set(clusters["cluster_id"].tolist())
print(f"target clusters: {len(keep):,}", flush=True)

handles = [open(os.path.join(TEXTDIR, f"s{i:02d}.jsonl"), "w",
                encoding="utf-8", buffering=1 << 20) for i in range(SHARDS)]
cand_fh = open(os.path.join(OUT, f"03_candidates{SUF}.jsonl"), "w", encoding="utf-8")

meta, index = [], []
n_kept = n_seen = n_cand = n_bad = 0
t0 = time.time()

# Low-level reader: the cluster test runs against the raw row before any dict is
# built or any markup is stripped, so the ~90% of rows outside the population
# cost nothing beyond CSV tokenization.
proc = subprocess.Popen(["bzip2", "-dc", os.path.join(DATA, "opinions.csv.bz2")],
                        stdout=subprocess.PIPE, bufsize=1 << 22)
fh_in = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace", newline="")
reader = csv.reader(fh_in, escapechar="\\")
header = next(reader)
IX = {c: i for i, c in enumerate(header)}
CI, II = IX["cluster_id"], IX["id"]
WIDTH = len(header)

for row in reader:
    n_seen += 1
    if MAXROWS and n_seen > MAXROWS:
        break
    if n_seen % 1_000_000 == 0:
        print(f"  scanned {n_seen:,} | kept {n_kept:,} | cands {n_cand:,} | "
              f"{(time.time()-t0)/60:.1f} min", file=sys.stderr, flush=True)
    if len(row) != WIDTH:
        n_bad += 1
        continue
    try:
        cid = int(row[CI])
    except (ValueError, TypeError):
        continue
    if cid not in keep:
        continue
    r = {c: row[i] for c, i in IX.items()}
    oid = int(row[II])
    text, src = best_text(r)
    if not text:
        continue
    n_kept += 1

    hits = []
    low = text.lower()
    if any(g in low for g in GATE) and UNION.search(text):
        hits = merge_overlaps(find_all(text))

    sh = oid % SHARDS
    fh = handles[sh]
    off = fh.tell()
    payload = json.dumps({"opinion_id": oid, "cluster_id": cid, "text": text},
                         ensure_ascii=False)
    fh.write(payload + "\n")
    index.append((oid, sh, off, len(payload) + 1))

    tids = sorted({t for h in hits for t in h[0].split("+")})
    meta.append((oid, cid, r["type"], r["per_curiam"] == "t", r["author_str"],
                 r["joined_by_str"], src, r["extracted_by_ocr"] == "t",
                 len(text), text.count(" ") + 1, len(hits), "|".join(tids)))

    for k, (tid, s, e, m) in enumerate(hits):
        n_cand += 1
        cand_fh.write(json.dumps({
            "cand_id": f"{oid}-{k}", "opinion_id": oid, "cluster_id": cid,
            "trigger_ids": tid, "char_start": s, "char_end": e, "matched": m,
        }, ensure_ascii=False) + "\n")

for fh in handles:
    fh.close()
cand_fh.close()
fh_in.close()
rc = proc.wait()
# bzip2 verifies a CRC per block, so a non-zero exit here means the archive is
# damaged and the pass is incomplete. Fail loudly rather than writing a
# truncated corpus that would silently understate every count downstream.
if rc != 0:
    raise SystemExit(f"bzip2 exited {rc}: opinions archive is damaged or truncated")

cols = ["opinion_id", "cluster_id", "type", "per_curiam", "author_str",
        "joined_by_str", "text_source", "ocr", "n_chars", "n_words",
        "n_triggers", "trigger_ids"]
pd.DataFrame(meta, columns=cols).to_parquet(
    os.path.join(OUT, f"03_opinions_meta{SUF}.parquet"), index=False)
pd.DataFrame(index, columns=["opinion_id", "shard", "offset", "length"]).to_parquet(
    os.path.join(OUT, f"03_text_index{SUF}.parquet"), index=False)

json.dump({"n_opinion_rows_scanned": n_seen, "n_federal_appellate_kept": n_kept,
           "n_candidate_passages": n_cand, "n_malformed_rows": n_bad,
           "dictionary_sha1": DICTIONARY_SHA1,
           "minutes": (time.time() - t0) / 60},
          open(os.path.join(OUT, f"03_stats{SUF}.json"), "w"), indent=1)
print(f"scanned {n_seen:,}; kept {n_kept:,}; candidates {n_cand:,}; "
      f"malformed {n_bad:,}")
