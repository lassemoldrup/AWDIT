import sys
import csv

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python scripts/extract-columns.py columns path/to/source.csv path/to/dest.csv')
        exit(1)
    columns = sys.argv[1].split(',')
    source = sys.argv[2]
    dest = sys.argv[3]

    with open(source, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        idxs = [rows[0].index(column) for column in columns]
    
    with open(dest, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for row in rows[1:]:
            writer.writerow([row[i] for i in idxs])
