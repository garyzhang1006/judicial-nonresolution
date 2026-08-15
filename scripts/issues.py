"""Issue-clause extraction and hard-negative structural cues.

Nothing here assigns a label. These routines locate the proposition a passage is
about and flag passages whose surface form is likely to mislead a shallow
classifier, so that the sample can be stratified over them. Whether a flagged
passage really is a hard negative is settled by annotation, not by these rules.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triggers import UNION as _TRIG_IN_ISSUE

STOP = set("""a an the of to in on for by with that this those these which whether
and or but not is are was were be been being as at from it its his her their our
we they he she i you would could should may might must can will shall do does did
under over into than then when where who whom what there here such any all each
other another same more most less least also however therefore because since if""".split())

# ------------------------------------------------------------------ issue clause
_WHETHER = re.compile(r"\bwhether\b", re.I)
_SENT_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z\"(])|\n\n")


def sentence_span(text, start, end):
    """Character span of the sentence containing [start, end)."""
    lo = max(0, start - 1200)
    left = text.rfind(". ", lo, start)
    left2 = text.rfind("\n", lo, start)
    left = max(left + 2 if left != -1 else lo, left2 + 1 if left2 != -1 else lo)
    hi = min(len(text), end + 1200)
    m = _SENT_END.search(text, end, hi)
    right = m.start() + 1 if m else hi
    return left, right


_CITE = re.compile(
    r"\(?\b\d+\s+(?:F\.\s?\d?d|F\.\s?Supp\.?\s?\d?d?|U\.S\.|S\.\s?Ct\.|L\.\s?Ed\.\s?\d?d?"
    r"|F\.\s?App'?x|Fed\.\s?Appx)\.?\s+\d+(?:\s*[,(][^)]{0,60}\))?\)?"
    r"|\bid\.\s*(?:at\s+\d+)?|\bsupra\b|\bsee\s+(?:also\s+)?\b|\bcf\.\s*")
_LEADIN = re.compile(
    r"^\s*(?:whether|that|if|the\s+question\s+(?:of|whether)|"
    r"(?:up)?on|as\s+to|regarding|concerning|about|with\s+respect\s+to|"
    r"the\s+(?:issue|merits)\s+of)\b", re.I)
_JUNK_HEAD = re.compile(r"^[\s,;:.)\]\"'—–-]+")
# "we need not decide that question because ..." points back at an issue stated
# earlier, so the words after the anchor do not state it. Rejecting sends the
# extractor to the material before the anchor, and failing that, drops the item.
_ANAPHOR = re.compile(
    r"^(?:questions?|issues?|matters?|points?|ones?|things?)\b", re.I)


def _clean(s):
    return re.sub(r"\s+", " ", _CITE.sub(" ", s)).strip()


def _neutralize(seg, extra_pat):
    """Cut the clause at the first disposition word, so it cannot leak a label.

    Applied unconditionally, which is what lets the paper state that no issue
    statement contains a dictionary expression or a holding cue.
    """
    for pat in (_TRIG_IN_ISSUE, extra_pat):
        m = pat.search(seg) if pat else None
        if m:
            seg = seg[:m.start()]
    return seg.strip(" ,;:.()[]\"'")


# A clause opening with any of these is a sentence fragment rather than a
# statement of an issue, so it is rejected instead of being dressed up as one.
BAD_HEAD = set("""not does do did is are was were be been being and but or nor yet
so thus then because since although while however therefore we it they he she i
you there here at by from as if into upon about with such more most also again
had has have will would could should may might must can shall""".split())
_TRAIL = re.compile(r"(?:\s+(?:and|or|but|the|a|an|of|to|in|for|that|we|it|"
                    r"they|which|because|since|as|by|with|on))+$", re.I)


def _normalize(seg, max_words):
    seg = _JUNK_HEAD.sub("", seg)
    if not seg:
        return None, None
    m = _WHETHER.search(seg)
    if m and m.start() < 60:
        seg, form = seg[m.start():], "whether"
    else:
        lead = _LEADIN.match(seg)
        if lead:
            rest = seg[lead.end():].strip(" ,;:")
            if not rest or _ANAPHOR.match(rest):
                return None, None
            seg, form = "whether " + rest, "whether"
        else:
            form = "np"
    seg = re.sub(r"\s+", " ", seg).strip(" ,;:.()[]\"'")
    words = seg.split()
    if len(words) > max_words:
        seg = " ".join(words[:max_words])
    seg = _TRAIL.sub("", seg).strip(" ,;:.“”\"'()[]")
    head = re.sub(r"[^a-z]", "", (seg.split() or [""])[0].lower())
    # A one- or two-letter opening token is almost always a word the extractor
    # cut in half, as when a footnote marker splits "three" into "thre" + "e".
    if len(head) < 3 and head not in {"a", "an", "if", "in", "is", "it", "no",
                                      "of", "on", "or", "to", "we", "by", "as"}:
        return None, None
    # A whether-clause can be short and still be a complete issue ("whether the
    # statute is constitutional" has two content words). A bare noun phrase has
    # to carry more, because it has no complementizer to mark it as a question.
    if form == "np":
        if head in BAD_HEAD or len(content_words(seg)) < 4:
            return None, None
    elif len(content_words(seg)) < 2:
        return None, None
    alpha = sum(c.isalpha() for c in seg)
    if alpha < 0.6 * max(1, len(seg.replace(" ", ""))):
        return None, None
    return seg, form


def issue_clause(text, start, end, max_words=45, with_form=False):
    """Statement of the proposition the anchored passage is about.

    The clause is bounded by the sentence containing the anchor rather than by a
    fixed character budget, which keeps it from running into whatever the court
    said next. The complement following the anchor is preferred; where the
    trigger closes its sentence ("that question we do not reach"), the material
    before the anchor is used instead.
    """
    ss, se = sentence_span(text, start, end)
    before = _clean(text[ss:start])
    # The material before the anchor is only usable when it names the issue
    # itself; otherwise it is the court's reasoning, not the proposition.
    bm = re.search(r"\b(whether|the question of|the issue of)\b", before, re.I)
    before = before[bm.start():] if bm else ""
    for raw in (_clean(text[end:se]), before):
        if len(raw) < 8:
            continue
        seg, form = _normalize(raw, max_words)
        if not seg:
            continue
        seg = _neutralize(seg, HOLD)
        if len(content_words(seg)) < (2 if form == "whether" else 4):
            continue
        return (seg, form) if with_form else seg
    return (None, None) if with_form else None


def content_words(s):
    return {w for w in re.findall(r"[a-z]{4,}", (s or "").lower()) if w not in STOP}


# ------------------------------------------------------------------ hard negatives
# Case-insensitive: opinions capitalize "the District Court" as often as not, and
# a case-sensitive pattern silently misses half of them.
OTHER_COURT = re.compile(
    r"\b(district court|court below|trial court|magistrate|bankruptcy court|"
    r"tax court|state court|board of immigration|the (?:B\.?I\.?A\.?|Board)|"
    r"supreme court|(?:first|second|third|fourth|fifth|sixth|seventh|eighth|"
    r"ninth|tenth|eleventh|d\.c\.|federal) circuit|panel below|the agency|"
    r"the commission|arbitrator|the court (?:held|concluded|stated|noted|found|"
    r"reasoned|denied|rejected|observed))\b", re.I)
CITATION = re.compile(r"\d+\s+(?:F\.\s?\d?d|U\.S\.|S\.\s?Ct\.|F\.\s?App'x|Fed\.\s?Appx)\.?\s+\d+",
                      re.I)
FIRST_PERSON = re.compile(r"\b(we|this court|this panel|I)\b", re.I)

# "our holding" is deliberately absent: in the pilot it pointed at a named prior
# case far more often than at the present disposition, which made it a poor
# locator for a decided issue in the opinion at hand.
HOLD = re.compile(
    r"\bwe\s+(?:therefore\s+|now\s+|thus\s+|accordingly\s+)?"
    r"(?:hold|conclude|find|determine|decide|agree|reject|affirm|reverse|vacate)\b"
    r"|\bwe\s+are\s+persuaded\b|\bit\s+is\s+clear\s+that\b"
    r"|\bthe\s+district\s+court\s+(?:erred|correctly)\b"
    r"|\bwe\s+hold\s+that\b", re.I)


def quoted_at(text, pos, window=1500):
    """True when the character position sits inside a quotation."""
    lo = max(0, pos - window)
    seg = text[lo:pos]
    return (seg.count('"') % 2 == 1) or (seg.count("“") > seg.count("”"))


def h1_other_court(text, start, end):
    """Trigger whose grammatical subject is plausibly a different institution."""
    pre = text[max(0, start - 220):start]
    if quoted_at(text, start):
        return True
    fp = FIRST_PERSON.search(pre[-90:])
    oc = OTHER_COURT.search(pre) or CITATION.search(pre[-160:])
    return bool(oc) and not fp


def h2_later_resolution(text, start, end, issue, min_overlap=3):
    """A holding sentence later in the opinion that is about the same issue."""
    keys = content_words(issue)
    if len(keys) < min_overlap:
        return False, None
    rest = text[end:]
    for m in HOLD.finditer(rest):
        s, e = sentence_span(rest, m.start(), m.end())
        sent = rest[s:e]
        if len(content_words(sent) & keys) >= min_overlap:
            return True, end + s
    return False, None


def find_holding_anchors(text, limit=6):
    """Locate resolved issues in an opinion for use as matched controls."""
    out = []
    for m in HOLD.finditer(text):
        if quoted_at(text, m.start()):
            continue
        iss = issue_clause(text, m.start(), m.end())
        if iss:
            out.append((m.start(), m.end(), m.group(0), iss))
        if len(out) >= limit:
            break
    return out


# A neutral locator: an issue named as an issue, with no cue as to how it came
# out. Unlike the holding locator it is not correlated with the label, which is
# what makes it a check on whether the benchmark depends on the holding cue.
ISSUE_MARKER = re.compile(
    r"\b(?:the\s+(?:question|issue)\s+(?:of\s+)?whether|"
    r"the\s+(?:question|issue)\s+(?:of|is|was|presented|before\s+us)|"
    r"whether)\b", re.I)


def find_neutral_anchors(text, limit=4):
    """Locate issues named without any disposition cue, for neutral controls."""
    out = []
    for m in ISSUE_MARKER.finditer(text):
        if quoted_at(text, m.start()):
            continue
        s, e = sentence_span(text, m.start(), m.end())
        if HOLD.search(text[s:e]):
            continue
        iss = issue_clause(text, m.start(), m.end())
        if iss and len(content_words(iss)) >= 4:
            out.append((m.start(), m.end(), m.group(0), iss))
        if len(out) >= limit:
            break
    return out
