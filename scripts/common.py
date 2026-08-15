"""Shared helpers for the judicial non-resolution pipeline."""
import bz2, csv, gzip, html, io, json, os, re, subprocess, sys

csv.field_size_limit(sys.maxsize)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.environ.get("OUTDIR") or os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- courts
# The population: U.S. federal appellate courts as defined by the study design.
FED_APP = ["scotus", "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8",
           "ca9", "ca10", "ca11", "cadc", "cafc"]
FED_APP_SET = set(FED_APP)
COURT_LABEL = {"scotus": "Supreme Court", "cadc": "D.C. Cir.", "cafc": "Fed. Cir."}
for _c in FED_APP:
    COURT_LABEL.setdefault(_c, _c.replace("ca", "") + " Cir.")

PERIODS = [("pre-1990", None, 1989), ("1990-2004", 1990, 2004),
           ("2005-2014", 2005, 2014), ("2015-2026", 2015, None)]


def period_of(year):
    if year is None:
        return None
    for name, lo, hi in PERIODS:
        if (lo is None or year >= lo) and (hi is None or year <= hi):
            return name
    return None


# ---------------------------------------------------------------- csv streaming
def stream_csv(path, want=None, progress_every=2_000_000, label=""):
    """Stream a bulk .csv.bz2 file, yielding dicts restricted to `want` columns.

    Decompression is delegated to the system bzip2 so it overlaps with parsing.
    """
    proc = subprocess.Popen(["bzip2", "-dc", path], stdout=subprocess.PIPE,
                            bufsize=1024 * 1024)
    fh = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace",
                          newline="")
    # The bulk export escapes embedded quotes with a backslash rather than by
    # doubling them. Parsing with the csv default silently desynchronizes the
    # reader on roughly two thirds of rows.
    reader = csv.reader(fh, escapechar="\\")
    header = next(reader)
    idx = {c: i for i, c in enumerate(header)}
    cols = want if want else header
    missing = [c for c in cols if c not in idx]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; have {header}")
    picks = [(c, idx[c]) for c in cols]
    width = len(header)
    n = bad = 0
    for row in reader:
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  [{label or os.path.basename(path)}] {n:,} rows "
                  f"({bad:,} malformed)", file=sys.stderr, flush=True)
        # A small number of rows carry an unescaped quote upstream, which shifts
        # every field after it. Width is the cheapest reliable detector.
        if len(row) != width:
            bad += 1
            continue
        yield {c: row[i] for c, i in picks}
    fh.close()
    proc.wait()
    STREAM_STATS[label or os.path.basename(path)] = {"rows": n, "malformed": bad}
    print(f"  [{label or os.path.basename(path)}] finished {n:,} rows, "
          f"{bad:,} malformed ({bad/max(n,1):.2e})", file=sys.stderr, flush=True)


STREAM_STATS = {}


# ---------------------------------------------------------------- text
_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_BLOCK = re.compile(r"</(p|div|blockquote|li|tr|h[1-6])\s*>|<br\s*/?>", re.I)
_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")

# CourtListener stores the same opinion in several markup flavours; this is the
# precedence its own `Opinion.text` property uses, minus the citation-annotated
# variant, which injects anchor text into the body.
TEXT_FIELDS = ["html_columbia", "html_lawbox", "xml_harvard", "html_anon_2020",
               "html", "plain_text", "html_with_citations"]


def to_text(raw):
    if not raw:
        return ""
    if "<" in raw:
        raw = _SCRIPT.sub(" ", raw)
        raw = _BLOCK.sub("\n", raw)
        raw = _TAG.sub(" ", raw)
    raw = html.unescape(raw)
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = _WS.sub(" ", raw)
    raw = _NL.sub("\n\n", raw)
    return raw.strip()


def best_text(row):
    for f in TEXT_FIELDS:
        v = row.get(f)
        if v and len(v) > 200:
            t = to_text(v)
            if len(t) > 200:
                return t, f
    for f in TEXT_FIELDS:
        v = row.get(f)
        if v:
            t = to_text(v)
            if t:
                return t, f
    return "", None


# ---------------------------------------------------------------- jsonl
def write_jsonl(path, rows):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
