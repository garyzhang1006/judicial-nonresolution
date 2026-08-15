"""Step 17: an evaluation set labelled by federal judges rather than by us.

Every item here comes from one court telling us, in writing, what another court
did with an issue. A parenthetical reading "(holding that X)" asserts that the
cited opinion decided X. One reading "(declining to decide whether X)" asserts
that it did not. Attaching that assertion to the passage in the cited opinion
that the parenthetical is about gives a labelled item with no annotator in the
loop.

The set is small relative to the corpus and it is not a substitute for
annotation: it can only cover issues a later court thought worth characterizing,
and its labels inherit whatever the later court got wrong. It is reported as an
independent check on the annotated benchmark, not as a replacement for it.
"""
import json, os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT
from textstore import TextStore
from issues import content_words, sentence_span

MIN_MATCH = 4          # content words shared between parenthetical and anchor
MAX_PER_OPINION = 1


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def strip_lead(c):
    """Drop the characterizing verb so the issue text does not leak the label."""
    return re.sub(r"^[^.]{0,30}?\(?\s*(?:holding that|held that|concluding that|"
                  r"concluded that|rejecting|adopting|establishing that|left open|"
                  r"leaving open|declin\w+ to (?:decide|reach|resolve)|did not "
                  r"(?:decide|reach|resolve)|reserv\w+ the (?:question|issue)|"
                  r"express\w* no (?:opinion|view)|assum\w+ without deciding)\s*",
                  "", c, flags=re.I)


def main():
    t = pd.read_parquet(os.path.join(OUT, "15_tight.parquet"))
    elig = pd.read_parquet(os.path.join(OUT, "05_eligible.parquet"),
                           columns=["opinion_id", "court", "year", "period",
                                    "type_group"])
    info = elig.set_index("opinion_id").to_dict("index")
    t = t[t["cited_opinion_id"].isin(info)]
    print(f"tight characterizations of eligible opinions: {len(t):,}")

    # Balance: keep every L0, and a like-sized random sample of L3.
    l0 = t[t["verdict"] == "L0"]
    l3 = t[t["verdict"] == "L3"].sample(min(len(l0) * 3, (t["verdict"] == "L3").sum()),
                                        random_state=5)
    pool = pd.concat([l0, l3], ignore_index=True)
    print(f"pool: {len(l0):,} L0 + {len(l3):,} L3")

    ts, rows, used = TextStore(), [], {}
    for n, r in enumerate(pool.itertuples(index=False)):
        if n % 1000 == 0:
            print(f"  {n:,}/{len(pool):,} kept {len(rows):,}", file=sys.stderr, flush=True)
        if used.get(r.cited_opinion_id, 0) >= MAX_PER_OPINION:
            continue
        txt = ts.get(r.cited_opinion_id)
        if not txt:
            continue
        issue = strip_lead(r.characterization)
        keys = content_words(issue)
        if len(keys) < MIN_MATCH:
            continue
        # anchor: the sentence in the cited opinion sharing the most content words
        best, bpos = 0, None
        for m in re.finditer(r"[^.]{40,400}\.", txt):
            ov = len(content_words(m.group(0)) & keys)
            if ov > best:
                best, bpos = ov, (m.start(), m.end())
        if best < MIN_MATCH or bpos is None:
            continue
        s, e = bpos
        ss, se = sentence_span(txt, s, min(e, s + 60))
        used[r.cited_opinion_id] = used.get(r.cited_opinion_id, 0) + 1
        meta = info[r.cited_opinion_id]
        rows.append({
            "item_id": f"j{r.cited_opinion_id}-{n}", "opinion_id": r.cited_opinion_id,
            "citing_id": r.citing_id, "label": "UNRESOLVED" if r.verdict == "L0" else "DECIDED",
            "issue": clean(issue)[:300], "char_start": s, "char_end": min(e, s + 60),
            "sent_start": ss, "sent_end": se, "match": best,
            "court": str(meta["court"]), "year": meta["year"],
            "period": str(meta["period"]), "type_group": str(meta["type_group"]),
            "stratum": "JUDGE"})
    ts.close()

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, "17_judge_bench.parquet"), index=False)
    print(f"\njudge-labelled items: {len(df):,}")
    if len(df):
        print(df["label"].value_counts().to_string())
        print(f"distinct opinions: {df['opinion_id'].nunique():,}; "
              f"median match {df['match'].median():.0f}")
        print(df.groupby("court").size().sort_values(ascending=False).head(6).to_string())
    json.dump({"n": int(len(df)),
               "n_unresolved": int((df["label"] == "UNRESOLVED").sum()) if len(df) else 0,
               "n_decided": int((df["label"] == "DECIDED").sum()) if len(df) else 0},
              open(os.path.join(OUT, "17_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
