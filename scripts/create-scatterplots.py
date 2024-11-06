import sys
import os
import csv
from datetime import datetime

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/create-scatterplots.py path/to/results')
        exit(1)
    path = sys.argv[1]

    date = datetime.now().strftime('%Y-%m-%d')

    for isolation in ['rc', 'ra', 'cc']:
        with open(os.path.join(path, f'{date}-time-{isolation}.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if isolation == 'cc':
                writer.writerow(['#txns', 'ours (s)', 'plume (s)', 'dbcop (s)', 'ours (res)', 'plume (res)', 'dbcop (res)'])
            else:
                writer.writerow(['#txns', 'ours (s)', 'plume (s)', 'ours (res)', 'plume (res)'])
            
            times = {}
            results = {}
            for entry in os.listdir(path):
                if not entry.endswith('.csv') or not f'-{isolation}-' in entry:
                    continue
                
                if 'mem' in entry:
                    continue
                with open(os.path.join(path, entry), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                    if 'time' in entry:
                        times[entry.removesuffix('-time.csv')] = rows[1:]
                    elif 'res' in entry:
                        results[entry.removesuffix('-res.csv')] = [row[1:] for row in rows[1:]]
            for k, ts in times.items():
                res = results[k]
                writer.writerows([t + r for t, r in zip(ts, res)])

