from concurrent.futures import ThreadPoolExecutor
import sys
import subprocess
import os

def convert_entry(in_path, out_path, entry):
    full_path = os.path.join(in_path, entry, 'cobra/log')
    plume_path = os.path.join(out_path, entry, 'plume/history.txt')
    dbcop_path = os.path.join(out_path, entry, 'dbcop')

    # With TPC-C using Postgres and CockroachDB, the DB is loaded from an inital state,
    # hence reads of those initial writes will show up as thin-air reads. '-F' fixes this.
    fix_flag = []
    if 'postgres-tpcc' in entry or 'cockroachdb-tpcc' in entry:
        fix_flag = ['-F']

    try:
        if not os.path.exists(plume_path):
            if len(fix_flag) == 0:
                print(f'Running: target/release/awdit convert -m -t plume {full_path} {plume_path}')
            else:
                print(f'Running: target/release/awdit convert -m -t plume -F {full_path} {plume_path}')

            result = subprocess.run(
                ['target/release/awdit', 'convert', '-m', '-t', 'plume'] + fix_flag + [full_path, plume_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        else:
            print(f'Note: skipping {plume_path}')

        if not os.path.exists(dbcop_path):
            if len(fix_flag) == 0:
                print(f'Running: target/release/awdit convert -m -t dbcop {full_path} {dbcop_path}')
            else:
                print(f'Running: target/release/awdit convert -m -t dbcop -F {full_path} {dbcop_path}')

            result = subprocess.run(
                ['target/release/awdit', 'convert', '-m', '-t', 'dbcop'] + fix_flag + [full_path, dbcop_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        else:
            print(f'Note: skipping {dbcop_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error running tool in {entry}: {e.stderr}')
        exit(1)

def convert_all(in_path, out_path):
    entries = list(os.listdir(in_path))
    with ThreadPoolExecutor() as executor:
        for entry in entries:
            executor.submit(convert_entry, in_path, out_path, entry)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: python scripts/convert-benches.py path/to/benches output/path')
        exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    convert_all(in_path, out_path)