#!/bin/bash
set -eu

export TIME_LIMIT=${TIME_LIMIT:-7200} # seconds
export MEM_LIMIT=${MEM_LIMIT:-55} # GiB

echo "Comparing AWDIT and Plume on small dataset..."
rm -r results/small 2> /dev/null || true
python scripts/run-benches.py txn awdit-plume histories/bench/small results/small
echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/bench/small results/small
echo "Packaging results..."
python scripts/create-scatterplots.py small results/small
python scripts/extract-columns.py \
    "txns,ours_rc (s),plume_rc (s),ours_ra (s),plume_ra (s),ours_cc (s),plume_cc (s)" \
    results/small/*-small.csv \
    results/small.csv
echo "Cleaning up..."
rm results/small/*-stats.csv
rm results/small/*-mem.csv
rm results/small/*-time.csv
rm results/small/*-res.csv
echo "Done."