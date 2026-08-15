"""Step 2: federal appellate opinion clusters with normalized metadata."""
import os, pickle, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, stream_csv, period_of

docket_court = pickle.load(open(os.path.join(OUT, "01_docket_court.pkl"), "rb"))
print(f"loaded {len(docket_court):,} federal appellate dockets")

WANT = ["id", "docket_id", "date_filed", "date_filed_is_approximate",
        "precedential_status", "case_name", "citation_count", "judges",
        "source", "scdb_id", "blocked", "case_name_short"]

rows, n_bad = [], 0
for r in stream_csv(os.path.join(DATA, "opinion-clusters.csv.bz2"), want=WANT,
                    progress_every=2_000_000, label="clusters"):
    try:
        d = int(r["docket_id"])
    except (ValueError, TypeError):
        continue
    court = docket_court.get(d)
    if court is None:
        continue
    df = r["date_filed"]
    year = int(df[:4]) if df[:4].isdigit() else None
    try:
        cid, cc = int(r["id"]), int(r["citation_count"] or 0)
    except ValueError:
        n_bad += 1
        continue
    rows.append((cid, d, court, df, year, period_of(year),
                 r["date_filed_is_approximate"] == "t",
                 r["precedential_status"], r["case_name"],
                 cc, r["judges"], r["source"],
                 r["scdb_id"], r["blocked"] == "t"))

cl = pd.DataFrame(rows, columns=["cluster_id", "docket_id", "court", "date_filed",
                                 "year", "period", "date_approx",
                                 "precedential_status", "case_name",
                                 "citation_count", "judges", "source", "scdb_id",
                                 "blocked"])
print(f"federal appellate clusters: {len(cl):,} ({n_bad:,} unparseable rows dropped)")
print(cl.groupby("court").size().sort_values(ascending=False))
print(cl.groupby("period").size())
print(cl["precedential_status"].value_counts())
print("year range:", cl["year"].min(), cl["year"].max())
cl.to_parquet(os.path.join(OUT, "02_clusters.parquet"), index=False)
