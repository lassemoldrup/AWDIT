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

def sort_results(results_path, sort_key):
    for entry in os.listdir(results_path):
        if not entry.endswith('-stats.csv'):
            continue

        with open(os.path.join(results_path, entry), 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            headers = rows[0]
            rows = rows[1:]
            rows.sort(key=lambda row: int(row[sort_key]))
        with open(os.path.join(results_path, entry), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

def convert_entry(in_path, out_path, entry, file_prefix_fn):
    stats = get_stats(in_path, entry)

    _, _, _, _, ops, txns, threads = entry.split('-')
    if ops == '8' or ops == 'local':
        ops = 'N/A'
    file_prefix = file_prefix_fn(entry)
    with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([txns, stats['num_transactions'], threads, stats['num_events'], stats['num_keys'], ops])

def convert_series(in_path, out_path, file_prefix_fn):
    entries = list(os.listdir(in_path))
    headers = set()
    for entry in entries:
        file_prefix = file_prefix_fn(entry)
        if file_prefix in headers:
            continue
        with open(os.path.join(out_path, f'{file_prefix}-stats.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['txns', 'actual_txns', 'sessions', 'events', 'keys', 'ops_per_txn'])
            headers.add(file_prefix)

    with ThreadPoolExecutor() as executor:
        for entry in entries:
            executor.submit(convert_entry, in_path, out_path, entry, file_prefix_fn)

def txn_file_prefix(entry):
    _, db, script, _, _, _, threads = entry.split('-')
    return f'{date}-{db}-{script}-s{threads}'

def sess_file_prefix(entry):
    _, db, script, _, _, txn, _ = entry.split('-')
    return f'{date}-{db}-{script}-t{txn}'

def ops_file_prefix(entry):
    _, db, script, _, _, txn, threads = entry.split('-')
    return f'{date}-{db}-{script}-s{threads}-t{txn}'

def convert_txn_series(in_path, out_path):
    convert_series(in_path, out_path, txn_file_prefix)
    sort_results(out_path, 0)

def convert_sess_series(in_path, out_path):
    convert_series(in_path, out_path, sess_file_prefix)
    sort_results(out_path, 2)

def convert_ops_series(in_path, out_path):
    convert_series(in_path, out_path, ops_file_prefix)
    sort_results(out_path, 5)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python scripts/history-stats.py <txn OR sess OR ops> path/to/benches output/path')
        exit(1)
    txn_sess_ops = sys.argv[1]
    in_path = sys.argv[2]
    out_path = sys.argv[3]

    print('Building our tool..')
    subprocess.run(
        ['cargo', 'build', '--release'],
        check=True
    )

    if txn_sess_ops == 'txn':
        convert_txn_series(in_path, out_path)
    elif txn_sess_ops == 'sess':
        convert_sess_series(in_path, out_path)
    elif txn_sess_ops == 'ops':
        convert_ops_series(in_path, out_path)
    else:
        print('The first argument should be either `txn` or `sess`')
        exit(1)