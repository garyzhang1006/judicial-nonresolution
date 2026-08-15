"""The frozen trigger dictionary.

Thirteen expressions, one per entry, matching the study design one-to-one. The
dictionary was fixed before any annotation was carried out and was not revised
afterwards; its SHA-1 is reported in the paper so that the frozen state is
verifiable. Matching is case-insensitive over whitespace-normalized text, and
each pattern tolerates a small number of intervening words so that "need not
now decide" and "we do not here decide" are captured alongside the bare forms.

One operational decision is recorded here rather than made silently. The design
lists the bare stem "reserve"/"reserved" (T11). In appellate prose that stem is
dominated by senses unrelated to non-resolution: reserved rights, reserved
easements, the Federal Reserve, objections reserved for the record. T11
therefore requires the stem to govern an object denoting a legal question or a
deferral. Every other expression is matched as written. T08 and T09 are proper
subsets of T07 by construction; they are retained as separate entries because
the design names them separately, and the prevalence tables count a passage once
while retaining every trigger identifier that fired on it.
"""
import hashlib, re

SPEC = [
    ("T01", "need not decide",
     r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?decide\b"),
    ("T02", "need not reach",
     r"need(?:s|ed)?\s+not\s+(?:\w+\s+){0,2}?reach\b"),
    ("T03", "do not decide",
     r"\bdo(?:es)?\s+not\s+(?:\w+\s+){0,2}?decide\b"),
    ("T04", "do not reach",
     r"\bdo(?:es)?\s+not\s+(?:\w+\s+){0,2}?reach\b"),
    ("T05", "decline to decide",
     r"declin(?:e|es|ed|ing)\s+to\s+(?:\w+\s+){0,2}?decide\b"),
    ("T06", "decline to address",
     r"declin(?:e|es|ed|ing)\s+to\s+(?:\w+\s+){0,2}?address\b"),
    ("T07", "without deciding",
     r"\bwithout\s+deciding\b"),
    ("T08", "assuming without deciding",
     r"\bassuming\s+(?:\w+\s+){0,4}?without\s+deciding\b"),
    ("T09", "assume without deciding",
     r"assum(?:e|es|ed)\s+(?:\w+\s+){0,4}?without\s+deciding\b"),
    ("T10", "express no opinion",
     r"express(?:es|ed|ing)?\s+no\s+(?:\w+\s+){0,2}?opinion\b"),
    ("T11", "reserve / reserved",
     r"reserv(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?"
     r"(?:question|questions|issue|issues|matter|judgment|decision)\b"
     r"|reserv(?:e|es|ed|ing)\s+(?:\w+\s+){0,3}?for\s+another\s+(?:day|case)\b"),
    ("T12", "leave for another day",
     r"leav(?:e|es|ing)\s+(?:\w+\s+){0,4}?"
     r"(?:for|to|until)\s+another\s+(?:day|case|occasion)\b"),
    ("T13", "not necessary to decide",
     r"\b(?:not|un)\s*necessary\s+to\s+(?:decide|reach|resolve|address)\b"),
]

NAME = {t: n for t, n, _ in SPEC}
PATTERNS = [(tid, name, re.compile(rx, re.I)) for tid, name, rx in SPEC]
UNION = re.compile("|".join(f"(?:{rx})" for _, _, rx in SPEC), re.I)

DICTIONARY_SHA1 = hashlib.sha1(
    "\n".join(f"{t}\t{n}\t{r}" for t, n, r in SPEC).encode()).hexdigest()


def find_all(text):
    """Return (trigger_id, start, end, matched_string) tuples, left-to-right."""
    hits = []
    for tid, _name, pat in PATTERNS:
        for m in pat.finditer(text):
            hits.append((tid, m.start(), m.end(), m.group(0)))
    hits.sort(key=lambda h: (h[1], -h[2]))
    return hits


def merge_overlaps(hits):
    """Collapse textually overlapping hits into one passage-level occurrence."""
    out = []
    for tid, s, e, txt in hits:
        if out and s < out[-1][2]:
            pid, ps, pe, ptxt = out[-1]
            ids = "+".join(sorted(set(pid.split("+") + [tid])))
            out[-1] = (ids, ps, max(pe, e), ptxt if len(ptxt) >= len(txt) else txt)
        else:
            out.append((tid, s, e, txt))
    return out


if __name__ == "__main__":
    print("dictionary sha1:", DICTIONARY_SHA1)
    for tid, name, rx in SPEC:
        print(f"{tid}  {name:26s}  {rx}")
