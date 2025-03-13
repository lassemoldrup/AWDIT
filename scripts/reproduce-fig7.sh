#!/bin/bash
set -eu

export TIME_LIMIT=600 # seconds
export MEM_LIMIT=45000000000 # bytes

echo "Reproducing Figure 7 from the paper..."
python scripts/run-benches.py txn all histories/bench/small results/fig7
echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/small results/fig7
echo "Packaging results..."
python scripts/create-scatterplots.py fig7 results/fig7
echo "Cleaning up..."
rm results/fig7/*-stats.csv
rm results/fig7/*-mem.csv
rm results/fig7/*-time.csv
echo "Done."