# Need Not Decide — code and paper

Measurement of explicit judicial non-resolution across 928,254 US federal
appellate opinions, a 306-item annotated benchmark, model evaluations, and a
citation-chain analysis of what later courts do with issues a court declined to
decide.

`judicial-nonresolution-NLLP.pdf` is the built paper. `paper/build/main.pdf` is
the same file.

## What is and isn't here

Included: all pipeline code, the paper source, and every small derived artefact
the paper's numbers come from (results CSVs, prevalence JSONs, annotation
worksheets and labels).

Excluded, because of size: the CourtListener bulk release (`data/`, 58 GB) and
the derived text shards and item parquets (`out/`, 19 GB). Those rebuild from
public data with steps 01–07 below. The `tectonic` binary used to typeset the
paper is also omitted.

## Layout

```
scripts/   pipeline, numbered in execution order
paper/     LaTeX source, generated tables/macros/figures, built PDF
out/       derived CSV/JSON artefacts and annotation worksheets
```

## Pipeline

Run in numeric order. Steps 01–05 stream the bulk release and are the only
expensive ones; everything after works on derived artefacts.

| Step | What it does |
|---|---|
| 01–04 | Parse the bulk dockets, clusters, opinions, and citation tables |
| 05 | Apply the frozen trigger dictionary, compute population prevalence |
| 06–07 | Build the item frame, draw the stratified sample |
| 08 | Render annotation worksheets; ingest labels |
| 09 / 09b / 09c | Sparse baselines / fine-tuned encoders / zero-shot LM |
| 10a–10c | Citation chains and analysis |
| 11–12 | Emit LaTeX tables, macros, and figures |
| 13–14 | Dictionary recall audit, error analysis |
| 15–19, 21 | Judicial characterizations, escalation, proposition matching |
| 22–23 | Additional figures, table pairing |

Support modules: `common.py` (bulk CSV streaming — note the export escapes with
backslashes, so `csv.reader(fh, escapechar="\\")` is required), `triggers.py`
(the frozen 13-expression dictionary), `issues.py`, `textstore.py`,
`guidelines.py`, `pagecheck.py`.

## Building the paper

```bash
cd paper && tectonic -X compile main.tex --outdir build
```

`scripts/pagecheck.py` verifies content ends on page 8 and back matter starts on
page 9, as the venue requires.

Tables, figures, and inline numbers are generated — `11_tables.py` writes
`paper/generated/macros.tex`, so figures in the prose stay tied to the pipeline
rather than being typed by hand. Re-running it after any change to the
evaluation updates the paper.

## Models

The zero-shot baseline defaults to `microsoft/Phi-3.5-mini-instruct`. The paper
run pins revision `2fe192450127e6a83f7441aef6e3ca586c338b77` and uses float16
weights on one Tesla T4; exact run metadata and output hashes are in
`out/09c_llm_run_manifest.json`. The model is scored by comparing the summed
log-probability of the ` DECIDED` and ` UNRESOLVED` continuations rather than by
generating, so refusals, verbose answers, and format drift cannot vary with
prompt length and contaminate the context comparison. Override with `LLM_MODEL`.

## Data source

CourtListener bulk release of 30 June 2026. The benchmark is distributed as
opinion identifiers with character offsets into the normalized text, together
with the normalization code, so upstream corrections and removals propagate.
