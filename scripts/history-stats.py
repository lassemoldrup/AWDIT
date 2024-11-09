from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
import json
import csv
from datetime import datetime

date = datetime.now().strftime('%Y-%m-%d')

def get_stats(in_path, entry):
    in_path = os.path.join(in_path, entry, 'plume')
    result = subprocess.run(
        ['target/release/consistency', 'stats', '--json', in_path],
        check=True,
        stdout=subprocess.PIPE,
        text=True
    )
    decoder = json.decoder.JSONDecoder()
    return decoder.decode(result.stdout.strip())

def convert_txn_entry(in_path, out_path, entry):
    stats = get_stats(in_path, entry)

    _, db, script, _, _, txns, threads = entry.split('-')
    file_prefix = f'{date}-{db}-{script}-s{threads}'
    with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([txns, stats['num_transactions'], threads, stats['num_events'], stats['num_keys']])

def convert_session_entry(in_path, out_path, entry):
    stats = get_stats(in_path, entry)

    _, db, script, _, _, txns, threads = entry.split('-')
    file_prefix = f'{date}-{db}-{script}-t{txns}'
    with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([threads, stats['num_transactions'], stats['num_events'], stats['num_keys']])

def sort_results(results_path):
    for entry in os.listdir(results_path):
        with open(os.path.join(results_path, entry), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            headers = rows[0]
            rows = rows[1:]
            rows.sort(key=lambda x: int(x[0]))
        with open(os.path.join(results_path, entry), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

def convert_txn_series(in_path, out_path):
    entries = list(os.listdir(in_path))
    headers = set()
    for entry in entries:
        _, db, script, _, _, _, threads = entry.split('-')
        if (db, script, threads) in headers:
            continue
        file_prefix = f'{date}-{db}-{script}-s{threads}'
        with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['txns', 'actual_txns', 'sessions', 'events', 'keys'])
            headers.add((db, script, threads))

    with ThreadPoolExecutor() as executor:
        for entry in entries:
            executor.submit(convert_txn_entry, in_path, out_path, entry)
    sort_results(out_path)

def convert_session_series(in_path, out_path):
    entries = list(os.listdir(in_path))
    headers = set()
    for entry in entries:
        _, db, script, _, _, txns, _ = entry.split('-')
        if (db, script, txns) in headers:
            continue
        file_prefix = f'{date}-{db}-{script}-t{txns}'
        with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sessions', 'actual_txns', 'events', 'keys'])
            headers.add((db, script, txns))

    with ThreadPoolExecutor() as executor:
        for entry in entries:
            executor.submit(convert_session_entry, in_path, out_path, entry)
    sort_results(out_path)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python scripts/history-stats.py <txn OR sess> path/to/benches output/path')
        exit(1)
    txn_sess = sys.argv[1]
    in_path = sys.argv[2]
    out_path = sys.argv[3]

    print('Building our tool..')
    subprocess.run(
        ['cargo', 'build', '--release'],
        check=True
    )

    if txn_sess == 'txn':
        convert_txn_series(in_path, out_path)
    elif txn_sess == 'sess':
        convert_session_series(in_path, out_path)
    else:
        print('The first argument should be either `txn` or `sess`')
        exit(1)