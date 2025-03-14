import sys
import csv

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('usage: python scripts/split-rows.py column path/to/source.csv path/to/dest-prefix')
        exit(1)
    column = sys.argv[1]
    source = sys.argv[2]
    dest = sys.argv[3]

    output_rows = {}

    with open(source, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        idx = rows[0].index(column)
        del rows[0][idx]

        for row in rows[1:]:
            val = row[idx]
            if val not in output_rows:
                output_rows[val] = [rows[0]]
            del row[idx]
            output_rows[val].append(row)
    
    for key in output_rows:
        with open(f'{dest}-{key}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(output_rows[key])
