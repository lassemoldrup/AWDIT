#!/bin/bash
set -eu

export TIME_LIMIT=${TIME_LIMIT:-7200} # seconds
export MEM_LIMIT=${MEM_LIMIT:-55} # GiB

echo "Reproducing Figure 8 from the paper..."
rm -r results/fig8 2> /dev/null || true
python scripts/run-benches.py txn awdit-plume histories/bench/txn-exp results/fig8

echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/txn-exp results/fig8

echo "Packaging results..."
python scripts/create-scatterplots.py fig8 results/fig8
python scripts/sort-table.py events results/fig8/*-fig8.csv results/fig8/intermediate.csv
python scripts/extract-columns.py \
    "events,ours_rc (s),plume_rc (s)" \
    results/fig8/intermediate.csv \
    results/fig8-rc.csv
python scripts/extract-columns.py \
    "events,ours_ra (s),plume_ra (s)" \
    results/fig8/intermediate.csv \
    results/fig8-ra.csv
python scripts/extract-columns.py \
    "events,ours_cc (s),plume_cc (s)" \
    results/fig8/intermediate.csv \
    results/fig8-cc.csv

echo "Cleaning up..."
rm results/fig8/*-stats.csv
rm results/fig8/*-mem.csv
rm results/fig8/*-time.csv
rm results/fig8/*-res.csv
rm results/fig8/intermediate.csv

echo "Done."