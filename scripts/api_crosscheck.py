"""External cross-check of corpus coverage against the CourtListener search API.

The bulk release and the public search index are built from the same database
but are refreshed on different schedules and tokenized differently. Querying the
index for literal phrases gives an independent estimate of how many federal
appellate opinions contain each expression, which bounds how much the local
extraction pipeline loses to markup handling or text-field precedence.
"""
import json, os, sys, time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FED_APP, OUT

TOKEN = os.environ.get("CL_TOKEN", "fdbac4bdfe9be4aedb04e3d4e9eaeb2b26a6dc0b")
S = requests.Session()
S.headers["Authorization"] = f"Token {TOKEN}"
BASE = "https://www.courtlistener.com/api/rest/v4/search/"

LITERAL = ["need not decide", "need not reach", "do not decide", "do not reach",
           "decline to decide", "decline to address", "without deciding",
           "assuming without deciding", "assume without deciding",
           "express no opinion", "reserve the question",
           "leave for another day", "not necessary to decide"]

PERIOD_RANGE = {"pre-1990": (None, "1989-12-31"), "1990-2004": ("1990-01-01", "2004-12-31"),
                "2005-2014": ("2005-01-01", "2014-12-31"), "2015-2026": ("2015-01-01", None)}


def count(**params):
    params.setdefault("type", "o")
    params["format"] = "json"
    for attempt in range(6):
        try:
            r = S.get(BASE, params=params, timeout=90)
            if r.status_code == 200:
                return r.json().get("count")
            if r.status_code in (429, 502, 503, 504):
                time.sleep(10 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(8 * (attempt + 1))
    return None


out = {"denominator_by_court": {}, "denominator_by_court_period": {},
       "phrase_by_court": {}, "phrase_total": {}}

for c in FED_APP:
    out["denominator_by_court"][c] = count(court=c)
    print("denom", c, out["denominator_by_court"][c], flush=True)
    for p, (lo, hi) in PERIOD_RANGE.items():
        kw = {"court": c}
        if lo:
            kw["filed_after"] = lo
        if hi:
            kw["filed_before"] = hi
        out["denominator_by_court_period"][f"{c}|{p}"] = count(**kw)

allc = " ".join(FED_APP)
for ph in LITERAL:
    out["phrase_total"][ph] = count(q=f'"{ph}"', court=allc)
    print("phrase", ph, out["phrase_total"][ph], flush=True)
    for c in FED_APP:
        out["phrase_by_court"][f"{c}|{ph}"] = count(q=f'"{ph}"', court=c)

out["union_total"] = count(q=" OR ".join(f'"{p}"' for p in LITERAL), court=allc)
print("union", out["union_total"])
json.dump(out, open(os.path.join(OUT, "api_crosscheck.json"), "w"), indent=1)
print("saved")
