#/bin/bash
set -eu

echo "Creating plots..."
sed "s/DNF/600/g" results/fig7-rubis.csv > results/graphs/data/fig7-rubis.csv || true
sed "s/DNF/600/g" results/fig7-twitter.csv > results/graphs/data/fig7-twitter.csv || true
sed "s/DNF/600/g" results/fig7-tpcc.csv > results/graphs/data/fig7-tpcc.csv || true

sed "s/DNF/7200/g" results/fig8-rc.csv > results/graphs/data/fig8-rc.csv || true
sed "s/DNF/7200/g" results/fig8-ra.csv > results/graphs/data/fig8-ra.csv || true
sed "s/DNF/7200/g" results/fig8-cc.csv > results/graphs/data/fig8-cc.csv || true

sed "s/DNF/600/g" results/fig9-txn.csv > results/graphs/data/fig9-txn.csv || true
sed "s/DNF/600/g" results/fig9-sess.csv > results/graphs/data/fig9-sess.csv || true
sed "s/DNF/600/g" results/fig9-ops.csv > results/graphs/data/fig9-ops.csv || true

cd results/graphs
pdflatex main.tex
cd ../..
mv results/graphs/main.pdf results/plot.pdf

echo "Cleaning up..."
rm results/graphs/*.aux || true
rm results/graphs/*.log || true
rm results/graphs/*.out || true
rm results/graphs/*.toc || true
rm results/graphs/*.synctex.gz || true
