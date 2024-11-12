import sys
import os
import csv
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python scripts/create-scatterplots.py name path/to/results')
        exit(1)
    name = sys.argv[1]
    path = sys.argv[2]

    date = datetime.now().strftime('%Y-%m-%d')

    for isolation in ['rc', 'ra', 'cc']:
        time_headers = None
        res_headers = None
        headers = ['database', 'script', 'txns', 'sessions', 'events', 'keys', 'ops_per_txn']
        
        times = {}
        results = {}
        stats = {}
        for entry in os.listdir(path):
            if not entry.endswith('.csv') or (not f'-{isolation}-' in entry and not entry.endswith('-stats.csv')):
                continue
            
            if entry.endswith('-mem.csv'):
                continue
            with open(os.path.join(path, entry), 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                if entry.endswith('-time.csv'):
                    if time_headers is None:
                        time_headers = rows[0][1:]
                    times[entry.removesuffix('-time.csv')] = [row[1:] for row in rows[1:]]
                elif entry.endswith('-res.csv'):
                    if res_headers is None:
                        res_headers = rows[0][1:]
                    results[entry.removesuffix('-res.csv')] = [row[1:] for row in rows[1:]]
                elif entry.endswith('-stats.csv'):
                    elems = entry.split('-')[:-1]
                    elems.insert(5, isolation)
                    stats['-'.join(elems)] = [elems[3:5] + row[1:] for row in rows[1:]]
        
        if time_headers is None or res_headers is None:
            print(f'Missing headers for {isolation}, skipping')
            continue
        with open(os.path.join(path, f'{date}-{name}-{isolation}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(headers + time_headers + res_headers)
            for k, ts in times.items():
                res = results[k]
                sts = stats[k]
                writer.writerows([s + t + r for s, t, r in zip(sts, ts, res)])

