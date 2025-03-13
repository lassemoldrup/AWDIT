#!/bin/bash
set -eu

export TIME_LIMIT=7200 # seconds
export MEM_LIMIT=45000000000 # bytes

echo "Reproducing Figure 8 from the paper..."
python scripts/run-benches.py txn awdit-plume histories/bench/txn-exp results/fig8
echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/txn-exp results/fig8
echo "Packaging results..."
python scripts/create-scatterplots.py fig8 results/fig8
echo "Cleaning up..."
rm results/fig8/*-stats.csv
rm results/fig8/*-mem.csv
rm results/fig8/*-time.csv
echo "Done."