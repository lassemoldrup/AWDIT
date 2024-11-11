import sys
import os
import csv

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python scripts/replace-columns.py column path/to/source path/to/destination')
        exit(1)
    column = sys.argv[1]
    source = sys.argv[2]
    dest = sys.argv[3]

    sources = os.listdir(source)
    for entry in os.listdir(dest):
        if not entry.endswith('-time.csv'):
            continue

        parts = entry.split('-')
        key = '-'.join(parts[3:])

        with open(os.path.join(dest, entry), 'r') as csvfile:
            reader = csv.reader(csvfile)
            dest_rows = list(reader)
            dest_i = dest_rows[0].index(column)

        source_file = next(e for e in sources if e.endswith(key))
        with open(os.path.join(source, source_file), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            i = rows[0].index(column)
            vals = [row[i] for row in rows[1:]]

        for j, row in enumerate(dest_rows[1:]):
            row[dest_i] = vals[j]
            dest_rows[j+1] = row
                    
        with open(os.path.join(dest, entry), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(dest_rows)