#!/bin/bash
set -eu

echo "Reproducing all figures from the paper..."
scripts/reproduce-fig7.sh
scripts/reproduce-fig8.sh
scripts/reproduce-fig9.sh
echo "All done."