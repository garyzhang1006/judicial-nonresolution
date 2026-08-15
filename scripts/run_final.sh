#!/bin/bash
# Layer-two analysis, figures, tables, and the compiled paper.
set -e
cd /Users/austincheng/judicial-nonresolution
unset OUTDIR
echo "$(date +%T) === layer-two ingest and analysis ==="
python3 scripts/10b_worksheet.py ingest
python3 scripts/10c_analysis.py
echo "$(date +%T) === human context ablation ==="
python3 scripts/08_worksheet.py ablation || echo "(no ablation labels yet)"
echo "$(date +%T) === error analysis ==="
python3 scripts/14_errors.py || echo "(no error cases)"
echo "$(date +%T) === figures and tables ==="
python3 scripts/12_figures.py
python3 scripts/11_tables.py
echo "$(date +%T) === compile ==="
cd paper && ../tectonic -X compile main.tex --outdir build
cd .. && python3 scripts/pagecheck.py
echo "$(date +%T) FINAL_DONE"
