import sys
import os
import csv
import itertools
from datetime import datetime

isolations = ['rc', 'ra', 'cc']
isolations_present = set()

def add_isolation(headers, iso):
    if len(isolations_present) == 1:
        return headers
    else:
        return [' '.join([f'{header.split()[0]}_{iso}'] + header.split()[1:]) for header in headers]

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python scripts/create-scatterplots.py name path/to/results')
        exit(1)
    name = sys.argv[1]
    path = sys.argv[2]

    date = datetime.now().strftime('%Y-%m-%d')

    time_headers = {iso: None for iso in isolations}
    mem_headers = {iso: None for iso in isolations}
    res_headers = {iso: None for iso in isolations}
    headers = ['database', 'script', 'txns', 'sessions', 'events', 'keys', 'ops_per_txn']
    
    times = {iso: {} for iso in isolations}
    mems = {iso: {} for iso in isolations}
    results = {iso: {} for iso in isolations}
    stats = {}

    files = list(os.listdir(path))
    for entry in files:
        for iso in isolations:
            if f'-{iso}' in entry:
                isolations_present.add(iso)
    
    for entry in files:
        if not entry.endswith('.csv'):
            continue
        
        with open(os.path.join(path, entry), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if entry.endswith('-time.csv'):
                _,_,_,db,script,iso,num,_ = entry.split('-')
                if time_headers[iso] is None:
                    time_headers[iso] = add_isolation(rows[0][1:], iso)
                times[iso][(db,script,num)] = [row[1:] for row in rows[1:]]
            elif entry.endswith('-mem.csv'):
                _,_,_,db,script,iso,num,_ = entry.split('-')
                if mem_headers[iso] is None:
                    mem_headers[iso] = add_isolation(rows[0][1:], iso)
                mems[iso][(db,script,num)] = [row[1:] for row in rows[1:]]
            elif entry.endswith('-res.csv'):
                _,_,_,db,script,iso,num,_ = entry.split('-')
                if res_headers[iso] is None:
                    res_headers[iso] = add_isolation(rows[0][1:], iso)
                results[iso][(db,script, num)] = [row[1:] for row in rows[1:]]
            elif entry.endswith('-stats.csv'):
                _,_,_,db,script,num,_ = entry.split('-')
                stats[(db,script,num)] = [[db, script] + row[1:] for row in rows[1:]]
    
    for iso in isolations:
        if time_headers[iso] is not None:
            headers += time_headers[iso]
    for iso in isolations:
        if mem_headers[iso] is not None:
            headers += mem_headers[iso]
    for iso in isolations:
        if res_headers[iso] is not None:
            headers += res_headers[iso]

    with open(os.path.join(path, f'{date}-{name}.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for k in times[next(iter(isolations_present))].keys():
            ts = [times[iso][k] for iso in isolations if iso in isolations_present]
            res = [results[iso][k] for iso in isolations if iso in isolations_present]
            mem = [mems[iso][k] for iso in isolations if iso in isolations_present]
            sts = stats[k]
            writer.writerows([itertools.chain.from_iterable(row) for row in zip(sts, *ts, *mem, *res)])

