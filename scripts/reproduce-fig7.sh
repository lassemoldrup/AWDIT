#!/bin/bash
set -eu

export TIME_LIMIT=${TIME_LIMIT:-600} # seconds
export MEM_LIMIT=${MEM_LIMIT:-55} # GiB

echo "Reproducing Figure 7 from the paper..."
rm -r results/fig7 2> /dev/null || true
python scripts/run-benches.py txn all histories/bench/small results/fig7

echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/small results/fig7

echo "Packaging results..."
python scripts/create-scatterplots.py fig7 results/fig7
python scripts/extract-columns.py \
    "script,txns,ours (s),plume (s),polysi (s),dbcop (s),causalc+ (s),mono (s)" \
    results/fig7/*-fig7.csv \
    results/fig7/intermediate.csv
python scripts/split-rows.py script results/fig7/intermediate.csv results/fig7

echo "Cleaning up..."
rm results/fig7/*-stats.csv
rm results/fig7/*-mem.csv
rm results/fig7/*-time.csv
rm results/fig7/*-res.csv
rm results/fig7/intermediate.csv

echo "Done."