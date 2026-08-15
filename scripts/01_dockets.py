"""Step 1: identify every docket belonging to a federal appellate court.

The bulk cluster table carries a docket foreign key but no court, so the docket
table is the only route from an opinion to the court that issued it.
"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, FED_APP_SET, stream_csv

# --- courts: record the full row for the fourteen target courts, and also note
# --- which historical courts we are deliberately excluding.
courts = {}
for r in stream_csv(os.path.join(DATA, "courts.csv.bz2"),
                    want=["id", "short_name", "full_name", "jurisdiction",
                          "start_date", "end_date", "citation_string"],
                    progress_every=0, label="courts"):
    courts[r["id"]] = r

federal_appellate_jurisdictions = {"F", "FS"}  # F = federal appellate, FS = SCOTUS
excluded = {k: v for k, v in courts.items()
            if v["jurisdiction"] in federal_appellate_jurisdictions
            and k not in FED_APP_SET}
print(f"target courts present: {sorted(FED_APP_SET & set(courts))}")
print(f"other F/FS courts in CourtListener (excluded): {sorted(excluded)}")
json.dump({"target": {k: courts[k] for k in sorted(FED_APP_SET & set(courts))},
           "excluded_federal": excluded},
          open(os.path.join(OUT, "01_courts.json"), "w"), indent=1)

# --- dockets: keep only the docket -> court edges we need.
docket_court = {}
for r in stream_csv(os.path.join(DATA, "dockets.csv.bz2"),
                    want=["id", "court_id"], progress_every=5_000_000,
                    label="dockets"):
    c = r["court_id"]
    if c in FED_APP_SET:
        try:
            docket_court[int(r["id"])] = c
        except ValueError:
            pass

print(f"federal appellate dockets: {len(docket_court):,}")
from collections import Counter
print(Counter(docket_court.values()).most_common())
with open(os.path.join(OUT, "01_docket_court.pkl"), "wb") as fh:
    pickle.dump(docket_court, fh, protocol=5)
