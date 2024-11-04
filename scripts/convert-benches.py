import sys
import subprocess
import os

def convert_all(base_path):
    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry, 'cobra/log')
        out_path = os.path.join('./res/bench', entry)
        plume_path = os.path.join(out_path, 'plume')
        dbcop_path = os.path.join(out_path, 'dbcop')
        try:
            print(f'Running: cargo run --release -- convert -m -t plume {full_path} {plume_path}')
            result = subprocess.run(
                ['cargo', 'run', '--release', '--', 'convert', '-m', '-t', 'plume', full_path, plume_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

            print(f'Running: cargo run --release -- convert -m -t dbcop {full_path} {dbcop_path}')
            result = subprocess.run(
                ['cargo', 'run', '--release', '--', 'convert', '-m', '-t', 'dbcop', full_path, dbcop_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f'Error running cargo in {entry}: {e.stderr}')
            exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/convert-benches path/to/benches')
        exit(1)
    path = sys.argv[1]

    convert_all(path)