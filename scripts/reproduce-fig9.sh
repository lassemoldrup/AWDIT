#!/bin/bash
set -eu

export TIME_LIMIT=${TIME_LIMIT:-600} # seconds
export MEM_LIMIT=${MEM_LIMIT:-55} # GiB

echo "Reproducing Figure 9 from the paper..."
rm -r results/fig9 2> /dev/null || true
python scripts/run-benches.py txn awdit histories/bench/scaling/txn results/fig9/txn
python scripts/run-benches.py sess awdit histories/bench/scaling/sess results/fig9/sess
python scripts/run-benches.py ops awdit histories/bench/scaling/ops results/fig9/ops

echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/scaling/txn results/fig9/txn
python scripts/history-stats.py sess histories/bench/scaling/sess results/fig9/sess
python scripts/history-stats.py ops histories/bench/scaling/ops results/fig9/ops

echo "Packaging results..."
python scripts/create-scatterplots.py fig9 results/fig9/txn
python scripts/create-scatterplots.py fig9 results/fig9/sess
python scripts/create-scatterplots.py fig9 results/fig9/ops
python scripts/extract-columns.py \
    "txns,ours_rc (s),ours_ra (s),ours_cc (s)" \
    results/fig9/txn/*-fig9.csv \
    results/fig9-txn.csv
python scripts/extract-columns.py \
    "sessions,ours_rc (s),ours_ra (s),ours_cc (s)" \
    results/fig9/sess/*-fig9.csv \
    results/fig9-sess.csv
python scripts/extract-columns.py \
    "ops_per_txn,ours_rc (s),ours_ra (s),ours_cc (s)" \
    results/fig9/ops/*-fig9.csv \
    results/fig9-ops.csv

echo "Cleaning up..."
rm results/fig9/*/*-stats.csv
rm results/fig9/*/*-mem.csv
rm results/fig9/*/*-time.csv
rm results/fig9/*/*-res.csv

echo "Done."