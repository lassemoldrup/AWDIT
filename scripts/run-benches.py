import subprocess
import os
import csv
import itertools

def run_benchexec(cmd):
    result = subprocess.run(
        ['runexec', '--timelimit', '600', '--memlimit', str(48*10**9), '--read-only-dir', '/', '--overlay-dir', '/home', '--'] + cmd,
        check=True,
        stdout=subprocess.PIPE,
        text=True
    )
    for line in result.stdout.splitlines():
        [key, val] = line.strip().split('=')
        memory = 'N/A'
        if key == 'cputime':
            time = val.removesuffix('s')
        if key == 'memory':
            memory = str(int(val) // 1024 // 1024)
        if key == 'terminationreason' and val == 'cputime':
            return 'DNF', 'DNF'
        elif key == 'terminationreason' and val == 'memory':
            return 'OOM', 'OOM'
    return time, memory
    
# Run with read-committed, read-atomic, or causal
def run_ours(history, isolation):
    return run_benchexec(['target/release/consistency', 'check', '--isolation', isolation, os.path.join(history, 'plume')])

# Run with RC, RA, or TCC
def run_plume(history, isolation):
    return run_benchexec(['java', '-jar', 'tools/Plume/Plume-1.0-SNAPSHOT-shaded.jar', '-i', isolation, '-t', 'PLUME', os.path.join(history, 'plume/history.txt')])

def run_dbcop(history):
    return run_benchexec(['tools/dbcop/target/release/dbcop', 'verify', '--cons', 'cc', '--out_dir', 'dbcop-out', '--ver_dir', os.path.join(history, 'dbcop')])

def run_all_algs(history, isolation):
    ours_iso_map = {'rc': 'read-committed', 'ra': 'read-atomic', 'cc': 'causal'}
    plume_iso_map = {'rc': 'RC', 'ra': 'RA', 'cc': 'TCC'}
    times = []
    mems = []

    # Ours
    time, mem = run_ours(history, ours_iso_map[isolation])
    times.append(time)
    mems.append(mem)

    # Plume
    time, mem = run_plume(history, plume_iso_map[isolation])
    times.append(time)
    mems.append(mem)
    
    # DBCop
    if isolation == 'cc':
        time, mem = run_dbcop(history)
        times.append(time)
        mems.append(mem)

    return times, mems

if __name__ == '__main__':
    for isolation in ['rc', 'ra', 'cc']:
        headers = set()
        for entry in os.listdir('res/bench'):
            _, db, script, _, _, txns, threads = entry.split('-')
            times, mems = run_all_algs(os.path.join('res/bench', entry), isolation)

            with open(f'results/{db}-{script}-{isolation}-s{threads}-time.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if (db, script) not in headers:
                    if isolation == 'cc':
                        writer.writerow(['#txns', 'ours (s)', 'plume (s)', 'dbcop (s)'])
                    else:
                        writer.writerow(['#txns', 'ours (s)', 'plume (s)'])
                writer.writerow([txns] + times)

            with open(f'results/{db}-{script}-{isolation}-s{threads}-mem.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if (db, script) not in headers:
                    if isolation == 'cc':
                        writer.writerow(['#txns', 'ours (MiB)', 'plume (MiB)', 'dbcop (MiB)'])
                    else:
                        writer.writerow(['#txns', 'ours (MiB)', 'plume (MiB)'])
                writer.writerow([txns] + mems)

            headers.add((db, script))
        
        # Sort the results
        for entry in os.listdir('results'):
            with open(f'results/{entry}', 'r') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                headers = rows[0]
                rows = rows[1:]
                rows.sort(key=lambda x: int(x[0]))
            with open(f'results/{entry}', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                writer.writerows(rows)