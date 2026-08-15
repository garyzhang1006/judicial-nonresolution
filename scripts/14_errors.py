"""Step 14: error analysis material.

Selects the items that separate the systems, so the qualitative section can be
written from the record rather than from impression. Three groups are pulled:
where the dictionary is wrong and the model is right, where both are wrong, and
where widening the context flipped the model from wrong to right.
"""
import gzip, json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from triggers import UNION

GEN = os.path.join(os.path.dirname(OUT), "paper", "generated")
os.makedirs(GEN, exist_ok=True)
MAXQ = 320


class OpinionCache:
    """Small read-only replacement for TextStore over recovered eval opinions."""
    def __init__(self, path):
        self.text = {}
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                self.text[int(record["opinion_id"])] = record["text"]

    def get(self, opinion_id):
        return self.text.get(int(opinion_id))

    def close(self):
        pass


def snippet(ts, r, pre=260, post=300):
    if ts is None:
        t = str(r["ctx_W1024"])
        anchor = re.sub(r"\s+", " ", str(r.get("matched", ""))).strip()
        if not anchor:
            raise ValueError(f"{r['item_id']}: no anchor text for context-only snippet")
        starts = [m.start() for m in re.finditer(re.escape(anchor), t)]
        if not starts:
            raise ValueError(f"{r['item_id']}: anchor absent from ctx_W1024")
        s = min(starts, key=lambda p: abs(p - len(t) / 2))
        e = s + len(anchor)
    else:
        t = ts.get(r["opinion_id"]) or ""
        s, e = int(r["char_start"]), int(r["char_end"])
    seg = (t[max(0, s - pre):s] + " [[" + t[s:e] + "]] " + t[e:min(len(t), e + post)])
    return re.sub(r"\s+", " ", seg).strip()


def esc(x):
    for a, b in (("\u0097", "--"), ("’", "'"), ("‘", "`"), ("“", "``"), ("”", "''"),
                 ("—", "--"), ("–", "--")):
        x = x.replace(a, b)
    for a, b in (("\\", " "), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                 ("$", r"\$"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", " "), ("^", " ")):
        x = x.replace(a, b)
    return x


def main():
    items = pd.read_parquet(os.path.join(OUT, "09_items_with_ctx.parquet"))
    preds = {}
    p = os.path.join(OUT, "09c_llm_preds.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        for ctx, g in d.groupby("context"):
            preds[ctx] = dict(zip(g["item_id"], g["pred"]))

    text_index = os.path.join(OUT, "03_text_index.parquet")
    opinion_cache = os.path.join(OUT, "recovered_eval_opinions.jsonl.gz")
    ts = (TextStore() if os.path.exists(text_index) else
          OpinionCache(opinion_cache) if os.path.exists(opinion_cache) else None)
    te = items[items["split"].isin(["TEST", "TEMPORAL", "COURT"])].copy()
    te["lex"] = te["ctx_W1024"].map(lambda x: int(bool(UNION.search(str(x)))))
    te["lex_wrong"] = te["lex"] != te["y"]

    groups, rows = {}, []
    # 1. the dictionary is misled
    g1 = te[te["lex_wrong"]].head(60)
    groups["dictionary misled"] = g1
    # 2. widening the window changed the zero-shot model's answer
    if "SENT" in preds and "W4096" in preds:
        flip = te[te["item_id"].map(
            lambda i: i in preds["SENT"] and i in preds["W4096"]
            and preds["SENT"][i] != preds["W4096"][i])]
        fixed = flip[flip["item_id"].map(lambda i: preds["W4096"][i]) == flip["y"]]
        broke = flip[flip["item_id"].map(lambda i: preds["SENT"][i]) == flip["y"]]
        groups["context fixed it"] = fixed.head(40)
        groups["context broke it"] = broke.head(40)

    for name, g in groups.items():
        for _, r in g.iterrows():
            rows.append({"group": name, "item_id": r["item_id"],
                         "stratum": r["stratum"], "court": r["court"],
                         "year": r["year"], "gold": r["label"],
                         "lexical": "UNRESOLVED" if r["lex"] else "DECIDED",
                         "issue": str(r["issue"])[:200],
                         "passage": snippet(ts, r)})
    if ts is not None:
        ts.close()
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "14_error_cases.csv"), index=False)
    print(f"selected {len(df)} cases across {df['group'].nunique() if len(df) else 0} groups")
    if len(df):
        print(df.groupby(["group", "stratum"]).size().to_string())

    # A compact LaTeX exhibit: two cases per group, truncated.
    out = []
    for name, g in df.groupby("group"):
        for _, r in g.head(1).iterrows():
            out.append(
                "\\textbf{%s} \\emph{(%s, %s %d; gold %s, dictionary says %s)}\\\\\n"
                "\\textsc{issue}: %s\\\\\n\\textsc{passage}: \\dots %s \\dots\n\\smallskip\n"
                % (esc(name), esc(r["stratum"]), esc(str(r["court"])), int(r["year"]),
                   esc(r["gold"]), esc(r["lexical"]), esc(r["issue"][:MAXQ]),
                   esc(r["passage"][:420])))
    with open(os.path.join(GEN, "error_cases.tex"), "w") as fh:
        fh.write("\n".join(out) if out else "")
    print("wrote error_cases.tex")


if __name__ == "__main__":
    main()
