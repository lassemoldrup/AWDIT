import sys
import os
import csv

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/create-scatterplots.py path/to/results')
        exit(1)
    path = sys.argv[1]

    for entry in os.listdir(path):
        if not entry.endswith('.csv'):
            continue

        with open(os.path.join(path, entry), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            rows[0] = [header.removeprefix('#') for header in rows[0]]
        with open(os.path.join(path, entry), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)