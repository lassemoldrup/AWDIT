import sys
import csv

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python scripts/extract-columns.py column path/to/source.csv path/to/dest.csv')
        exit(1)
    column = sys.argv[1]
    source = sys.argv[2]
    dest = sys.argv[3]

    with open(source, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        idx = rows[0].index(column)
    
    with open(dest, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(rows[0])
        writer.writerows(sorted(rows[1:], key=lambda row: float(row[idx])))
