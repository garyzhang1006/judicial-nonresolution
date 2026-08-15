#!/bin/bash
set -e
cd /Users/austincheng/judicial-nonresolution
unset OUTDIR
echo "$(date +%T) === step 3: opinions pass ==="
python3 scripts/03_opinions.py
echo "$(date +%T) === step 4: citation graph ==="
python3 scripts/04_citations.py
echo "$(date +%T) === step 5: prevalence ==="
python3 scripts/05_prevalence.py
echo "$(date +%T) === step 6: item frame ==="
python3 scripts/06_frame.py
echo "$(date +%T) === step 7: benchmark sample ==="
python3 scripts/07_sample.py
echo "$(date +%T) === step 13: recall audit ==="
python3 scripts/13_recall.py
echo "$(date +%T) MAIN_PIPELINE_DONE"
