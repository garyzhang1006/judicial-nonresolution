"""Step 4: the citation graph restricted to the federal appellate population.

The bulk citation map is an opinion-to-opinion edge list with a depth field
recording how many times the citing opinion cites the cited one. Edges are kept
when the cited opinion is in the population; whether the citing opinion is also
in the population is recorded rather than used as a filter, so that the share of
out-of-population citers can be reported.
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, stream_csv

meta = pd.read_parquet(os.path.join(OUT, "03_opinions_meta.parquet"),
                       columns=["opinion_id"])
pop = set(meta["opinion_id"].tolist())
print(f"population opinions: {len(pop):,}", flush=True)

rows = []
n = 0
for r in stream_csv(os.path.join(DATA, "citation-map.csv.bz2"),
                    want=["depth", "cited_opinion_id", "citing_opinion_id"],
                    progress_every=20_000_000, label="citations"):
    n += 1
    try:
        cited = int(r["cited_opinion_id"])
    except (ValueError, TypeError):
        continue
    if cited not in pop:
        continue
    try:
        citing = int(r["citing_opinion_id"])
    except (ValueError, TypeError):
        continue
    rows.append((citing, cited, int(r["depth"] or 1), citing in pop))

g = pd.DataFrame(rows, columns=["citing_id", "cited_id", "depth", "citing_in_pop"])
print(f"total edges scanned: {n:,}")
print(f"edges into population: {len(g):,}")
print(f"  citing opinion also in population: {g['citing_in_pop'].mean():.3f}")
g.to_parquet(os.path.join(OUT, "04_citations.parquet"), index=False)

indeg = g.groupby("cited_id").size().rename("n_citers")
indeg.to_frame().to_parquet(os.path.join(OUT, "04_indegree.parquet"))
print(indeg.describe())
