#!/bin/bash
set -eu

export TIME_LIMIT=7200 # seconds
export MEM_LIMIT=45000000000 # bytes

echo "Reproducing Figure 9 from the paper..."
python scripts/run-benches.py txn awdit histories/bench/txn-linear results/fig9/txns
python scripts/run-benches.py sess awdit histories/bench/sess-2 results/fig9/sess
python scripts/run-benches.py ops awdit histories/bench/ops-2 results/fig9/ops
echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/txn-linear results/fig9/txns
python scripts/history-stats.py sess histories/bench/sess-2 results/fig9/sess
python scripts/history-stats.py ops histories/bench/ops-2 results/fig9/ops
echo "Packaging results..."
python scripts/create-scatterplots.py fig9 results/fig9/txns
python scripts/create-scatterplots.py fig9 results/fig9/sess
python scripts/create-scatterplots.py fig9 results/fig9/ops
echo "Cleaning up..."
rm results/fig9/*/*-stats.csv
rm results/fig9/*/*-mem.csv
rm results/fig9/*/*-time.csv
echo "Done."