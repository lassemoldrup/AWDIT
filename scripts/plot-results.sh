#/bin/bash
set -eu

echo "Creating plots..."
if [ -f results/fig7-rubis.csv ]; then
    sed "s/DNF/600/g" results/fig7-rubis.csv > results/graphs/data/fig7-rubis.csv
fi
if [ -f results/fig7-twitter.csv ]; then
    sed "s/DNF/600/g" results/fig7-twitter.csv > results/graphs/data/fig7-twitter.csv
fi
if [ -f results/fig7-tpcc.csv ]; then
    sed "s/DNF/600/g" results/fig7-tpcc.csv > results/graphs/data/fig7-tpcc.csv
fi

if [ -f results/fig8-rc.csv ]; then
    sed "s/DNF/7200/g" results/fig8-rc.csv > results/graphs/data/fig8-rc.csv
fi
if [ -f results/fig8-ra.csv ]; then
    sed "s/DNF/7200/g" results/fig8-ra.csv > results/graphs/data/fig8-ra.csv
fi
if [ -f results/fig8-cc.csv ]; then
    sed "s/DNF/7200/g" results/fig8-cc.csv > results/graphs/data/fig8-cc.csv
fi

if [ -f results/fig9-txn.csv ]; then
    sed "s/DNF/600/g" results/fig9-txn.csv > results/graphs/data/fig9-txn.csv
fi
if [ -f results/fig9-sess.csv ]; then
    sed "s/DNF/600/g" results/fig9-sess.csv > results/graphs/data/fig9-sess.csv
fi
if [ -f results/fig9-ops.csv ]; then
    sed "s/DNF/600/g" results/fig9-ops.csv > results/graphs/data/fig9-ops.csv
fi

cd results/graphs
pdflatex -interaction nonstopmode main.tex > /dev/null
cd ../..
mv results/graphs/main.pdf results/plot.pdf

echo "Cleaning up..."
rm results/graphs/*.aux 2> /dev/null || true
rm results/graphs/*.log 2> /dev/null || true
rm results/graphs/*.out 2> /dev/null || true
rm results/graphs/*.toc 2> /dev/null || true
rm results/graphs/*.synctex.gz 2> /dev/null || true
