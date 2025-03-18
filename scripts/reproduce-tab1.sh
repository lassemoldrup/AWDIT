#!/bin/bash
set -eu

export TIME_LIMIT=${TIME_LIMIT:-600} # seconds
export MEM_LIMIT=${MEM_LIMIT:-55} # GiB

if [ ! -d "histories/violations" ]; then
    echo "Please extract the histories first."
    exit 1
fi

echo "Reproducing Table 1 from the paper..."
echo "Running AWDIT on all histories..."
rm results/tab1-output.txt 2> /dev/null || true
for hist in histories/violations/*; do
    hist_name=$(basename $hist)
    echo "Running AWDIT on $hist_name..."
    echo "## Output for $hist_name ##" >> results/tab1-output.txt
    target/release/awdit check -i read-committed $hist/plume/history.txt >> results/tab1-output.txt
    printf "\n##########################\n\n" >> results/tab1-output.txt
done

echo "Getting exhaustive results..."
rm -r results/tab1 2> /dev/null || true
python scripts/run-benches.py txn awdit-plume histories/violations results/tab1

echo "Gathering statistics about histories..."
python scripts/history-stats.py txn histories/violations results/tab1

echo "Packaging results..."
python scripts/create-scatterplots.py tab1 results/tab1
python scripts/extract-columns.py \
    "txns,sessions,database,script,ours_rc (res),ours_ra (res),ours_cc (res),plume_rc (res),plume_ra (res),plume_cc (res)" \
    results/tab1/*-tab1.csv \
    results/tab1.csv

echo "Cleaning up..."
rm results/tab1/*-stats.csv
rm results/tab1/*-mem.csv
rm results/tab1/*-time.csv
rm results/tab1/*-res.csv

echo "Done."