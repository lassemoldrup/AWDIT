import subprocess
import os
import sys
import csv
from datetime import datetime

time_limit = '600' # seconds
mem_limit = str(55*10**9) # bytes

date = datetime.now().strftime('%Y-%m-%d')

def run_benchexec(cmd):
    result = subprocess.run(
        ['runexec', '--timelimit', time_limit, '--memlimit', mem_limit, '--read-only-dir', '/', '--overlay-dir', '/home', '--'] + cmd,
        check=True,
        stdout=subprocess.PIPE,
        text=True
    )
    memory = 'N/A'
    exitcode = None
    for line in result.stdout.splitlines():
        key, val = line.strip().split('=')
        if key == 'cputime':
            time = val.removesuffix('s')
        if key == 'memory':
            mb = int(val.removesuffix('B')) / 1024 / 1024
            memory = f'{mb:.2f}'
        if key == 'terminationreason' and val == 'cputime':
            return 'DNF', 'DNF', None
        elif key == 'terminationreason' and val == 'memory':
            return 'OOM', 'OOM', None
        if key == 'returnvalue':
            if val != '0':
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'WARN: Non-zero exit code {val}: {" ".join(cmd)}\n')
            exitcode = val
    return time, memory, exitcode
    
# Run with read-committed, read-atomic, or causal
def run_ours(history, isolation):
    time, memory, _ = run_benchexec(['target/release/awdit', 'check', '--isolation', isolation, os.path.join(history, 'plume/history.txt')])
    result = 'N/A'
    with open('output.log', 'r') as output:
        for line in output:
            if line.startswith('Consistent'):
                result = 'C'
            elif line.startswith('Inconsistent'):
                result = 'I'
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'Inconsistent. Ours ({isolation}): {history}\n')
    return time, memory, result

# Run with RC, RA, or TCC
def run_plume(history, isolation):
    time, memory, _ = run_benchexec(['java', '-jar', 'tools/Plume/Plume-1.0-SNAPSHOT-shaded.jar', '-i', isolation, '-t', 'PLUME', os.path.join(history, 'plume/history.txt')])
    result = 'N/A'
    with open('output.log', 'r') as output:
        for line in output:
            if line.startswith('ACCEPT'):
                result = 'C'
            elif line.startswith('REJECT'):
                result = 'I'
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'Inconsistent. Plume ({isolation}): {history}\n')
    return time, memory, result

def run_polysi(history):
    time, memory, _ = run_benchexec(['java', '-jar', 'tools/PolySI/PolySI-1.0.0-SNAPSHOT.jar', 'audit', '-t=TEXT', os.path.join(history, 'plume/history.txt')])
    result = 'N/A'
    with open('output.log', 'r') as output:
        for line in output:
            if line.startswith('[[[[ ACCEPT ]]]]'):
                result = 'C'
            elif line.startswith('[[[[ REJECT ]]]]'):
                result = 'I'
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'Inconsistent. PolySI: {history}\n')
    return time, memory, result

def run_dbcop(history):
    time, memory, exitcode = run_benchexec(['tools/dbcop/target/release/dbcop', 'verify', '--cons', 'cc', '--out_dir', 'dbcop-out', '--ver_dir', os.path.join(history, 'dbcop')])
    result = 'N/A'
    if exitcode == '0':
        result = 'C'
    elif exitcode != None:
        result = 'I'
        with open('errors.log', 'a') as err_file:
            err_file.write(f'Inconsistent. DBCop: {history}\n')
    return time, memory, result

def run_causalc_plus(history):
    time, memory, _ = run_benchexec(['python3', 'tools/CausalC+/clingo_txn.py', os.path.join(history, 'plume/history.txt')])
    result = 'N/A'
    with open('output.log', 'r') as output:
        for line in output:
            if line.startswith('ACCEPT'):
                result = 'C'
            elif line.startswith('REJECT'):
                result = 'I'
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'Inconsistent. CausalC+: {history}\n')
    return time, memory, result

def run_mono(history):
    time, memory, _ = run_benchexec(['python3', 'tools/mono/run_mono_txn.py', os.path.join(history, 'plume/history.txt')])
    result = 'N/A'
    with open('output.log', 'r') as output:
        for line in output:
            if line.startswith('ACCEPT'):
                result = 'C'
            elif line.startswith('REJECT'):
                result = 'I'
                with open('errors.log', 'a') as err_file:
                    err_file.write(f'Inconsistent. TCC-Mono: {history}\n')
    return time, memory, result

def run_tools(history, isolation, txns, tools):
    ours_iso_map = {'rc': 'read-committed', 'ra': 'read-atomic', 'cc': 'causal'}
    plume_iso_map = {'rc': 'RC', 'ra': 'RA', 'cc': 'TCC'}
    times = []
    mems = []
    results = []

    for tool in tools:
        if tool == 'ours':
            time, mem, res = run_ours(history, ours_iso_map[isolation])
        elif tool == 'plume':
            time, mem, res = run_plume(history, plume_iso_map[isolation])
        elif tool == 'polysi':
            time, mem, res = run_polysi(history)
        elif tool == 'dbcop':
            time, mem, res = run_dbcop(history)
        elif tool == 'causalc+':
            time, mem, res = run_causalc_plus(history)
        elif tool == 'mono':
            time, mem, res = run_mono(history)
        else:
            print('Unrecognized tool:', tool)
            exit(1)
        times.append(time)
        mems.append(mem)
        results.append(res)

    return times, mems, results

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

def txn_series(in_path, results_path, isolation, tools):
    headers = set()
    entries = list(os.listdir(in_path))
    for i, entry in enumerate(entries):
        print(f'{isolation} {i+1}/{len(entries)}: Running on {entry}')

        _, db, script, _, _, txns, threads = entry.split('-')
        times, mems, results = run_tools(os.path.join(in_path, entry), isolation, txns, tools)
        file_prefix = f'{date}-{db}-{script}-{isolation}-s{threads}'

        with open(os.path.join(results_path, f'{file_prefix}-time.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                writer.writerow(['txns'] + [tool + ' (s)' for tool in tools])
            writer.writerow([txns] + times)

        with open(os.path.join(results_path, f'{file_prefix}-mem.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                writer.writerow(['txns'] + [tool + ' (MiB)' for tool in tools])
            writer.writerow([txns] + mems)
        
        with open(os.path.join(results_path, f'{file_prefix}-res.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                writer.writerow(['txns'] + [tool + ' (res)' for tool in tools])
            writer.writerow([txns] + results)

        headers.add((db, script, threads))
        

def session_series(in_path, results_path, isolation, tools):
    headers = set()
    entries = list(os.listdir(in_path))
    for i, entry in enumerate(entries):
        print(f'{isolation} {i+1}/{len(entries)}: Running on {entry}')

        _, db, script, _, _, txns, threads = entry.split('-')
        times, mems, results = run_tools(os.path.join(in_path, entry), isolation, txns, tools)
        file_prefix = f'{date}-{db}-{script}-{isolation}-t{txns}'

        with open(os.path.join(results_path, f'{file_prefix}-time.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, txns) not in headers:
                writer.writerow(['sessions'] + [tool + ' (s)' for tool in tools])
            writer.writerow([threads] + times)

        with open(os.path.join(results_path, f'{file_prefix}-mem.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, txns) not in headers:
                    writer.writerow(['sessions'] + [tool + ' (MiB)' for tool in tools])
            writer.writerow([threads] + mems)
        
        with open(os.path.join(results_path, f'{file_prefix}-res.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, txns) not in headers:
                writer.writerow(['sessions'] + [tool + ' (res)' for tool in tools])
            writer.writerow([threads] + results)

        headers.add((db, script, txns))

def ops_series(in_path, results_path, isolation, tools):
    headers = set()
    entries = list(os.listdir(in_path))
    for i, entry in enumerate(entries):
        print(f'{isolation} {i+1}/{len(entries)}: Running on {entry}')

        _, db, script, _, ops, txns, threads = entry.split('-')
        times, mems, results = run_tools(os.path.join(in_path, entry), isolation, txns, tools)
        file_prefix = f'{date}-{db}-{script}-{isolation}-s{threads}'

        with open(os.path.join(results_path, f'{file_prefix}-time.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                writer.writerow(['ops_per_txn'] + [tool + ' (s)' for tool in tools])
            writer.writerow([ops] + times)

        with open(os.path.join(results_path, f'{file_prefix}-mem.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                    writer.writerow(['ops_per_txn'] + [tool + ' (MiB)' for tool in tools])
            writer.writerow([ops] + mems)
        
        with open(os.path.join(results_path, f'{file_prefix}-res.csv'), 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if (db, script, threads) not in headers:
                writer.writerow(['ops_per_txn'] + [tool + ' (res)' for tool in tools])
            writer.writerow([ops] + results)

        headers.add((db, script, threads))

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('usage: python scripts/run-benches.py <txn OR sess OR ops> path/to/benches results/path')
        exit(1)
    txn_sess_ops = sys.argv[1]
    in_path = sys.argv[2]
    results_path = sys.argv[3]

    print('Building our tool..')
    subprocess.run(
        ['cargo', 'build', '--release'],
        check=True
    )
    print('Building DBCop..')
    subprocess.run(
        ['cargo', 'build', '--release', '--manifest-path', 'tools/dbcop/Cargo.toml'],
        check=True
    )

    # tools = ['ours', 'plume', 'polysi', 'dbcop', 'causalc+', 'mono']
    tools = ['ours']
    if txn_sess_ops == 'txn':
        for isolation in ['rc', 'ra', 'cc']:
            txn_series(in_path, results_path, isolation, tools)
        sort_results(results_path)
    elif txn_sess_ops == 'sess':
        for isolation in ['rc', 'ra', 'cc']:
            session_series(in_path, results_path, isolation, tools)
        sort_results(results_path)
    elif txn_sess_ops == 'ops':
        for isolation in ['rc', 'ra', 'cc']:
            ops_series(in_path, results_path, isolation, tools)
        sort_results(results_path)
    else:
        print('The first argument should be either `txn` or `sess`')
        exit(1)
