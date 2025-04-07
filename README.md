# Getting started

We provide two options for evaluating AWDIT:
- A Docker image (preferred)
- A VirtualBox VM image

## Requirements

### Docker image

To run the Docker image you need
- A Linux system with kernel version >= 4.5
- Docker

**Important:** The Docker image will not work outside of Linux, since we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups provided by the host machine. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

### VM image

We also provide VirtualBox images for x86 and ARM capable of running the Docker image.
However, performance results may be skewed when running in a VM, so we recommend using the Docker image directly, if possible.

The images were tested with VirtualBox version 7.1.6.

## Running the VM image

To use the image, open VirtualBox and go to File > Import Appliance...
A wizard will guide you through the process of importing the VM.
By default this will allocate 8 GiB memory, 4 processor cores, and 55 GiB storage (benchmark histories will take up 32 GiB).
These values can be changed in VirtualBox settings afterwards, and we recommend increasing the amount of memory as much as possible.
Afterwards, boot up the VM and log in as `user` with password `user`.
Then follow along from the folder `AWDIT` on the desktop.

## Importing histories

The first step is to extract the supplied histories into the root of this folder. This can be done (from a shell in this directory) with the command

```shell
tar -xJf path/to/histories.tar.xz
```
This may take a while (10 min on our server), as the uncompressed size is around 32 GB.

### Importing histories in VM

You may be able to enable host-to-client drag-and-drop in VirtualBox to transfer the histories.
Otherwise, you can create a shared folder in VirtualBox settings, which can be auto-mounted on some location on the guest, e.g., `/mnt`.
If the histories are placed here, they can then be extracted into the `AWDIT` folder on the desktop using the above command.

## Running the Docker image

Running the Docker image will produce a shell capable of running AWDIT and all comparison tools.
To start a container from the Docker image, do the following.

### 1

Load the image. For x86 systems, run

```shell
sudo docker load -i docker/awdit-artifact-amd64.tar.gz
```

For ARM systems, run

```shell
sudo docker load -i docker/awdit-artifact-arm64.tar.gz
```

### 2

Run the container:

```shell
docker run --name awdit-container --privileged --cap-drop=all -ti --mount type=volume,dst=/home/user/awdit/results -v $(pwd)/histories:/home/user/awdit/histories awdit-artifact
```

To exit the container, simply run `exit`.

**Note:** The reason for `--privileged` is that we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups. To mitigate any risks when running like this, we use `--cap-drop=all` to disable as many capabilities of the container as possible, and the active user in the Docker container will be non-root. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

**Note:** To run the container again after stopping it, you must first delete the old container with the following command:

```shell
docker rm -v awdit-container
```

## Basic testing

To check that everything is working, inside the container, run

```shell
scripts/test-run.sh
```

This will run AWDIT and Plume at the RC isolation level on a set of small benchmarks and should take around half a minute.
When done, exit the container with `exit`. To retrieve the results from the container, run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

There should now be a file `results/small.csv` containing the results of the experiment.
This file should look like `results/small-expected.csv` (modulo performance difference).

## Note

- In the results, `DNF` means that the tool timed out, `OOM` means that the tool exceeded the memory limit.

- You may get occasional warnings like: `WARNING - System has swapped during benchmarking. Benchmark results are unreliable!`. This is safe to ignore, since it seems to happen even if the program under test is using little memory, and we have not observed it to affect performance. This is, of course, assuming that the system is not running out of memory, in which case a lower memory limit should be set.

# Step-by-step instructions

We first give a summary of our claims and then instructions for evaluating each.

Our main claim is that AWDIT significantly outperforms the state-of-the-art at the task of consistency checking histories under the isolation levels Read Committed (RC), Read Atomic (RA), and Causal Consistency (CC).
We support this claim with two experiments, the result of which are illustrated in Figure 7 and Figure 8 of the paper.

Furthermore, as illustrated in Figure 9, we support our theoretical statements about the scaling of AWDIT through a sequence of experiments that vary different parameters.

Finally, we report on consistency violations found while evaluating the performance in Table 1.

We provide scripts for automatically reproducing each of these results, but it may be necessary to tweak the time-out and memory limits in order to run them.
A summary of results and the time/memory requirements to reproduce them are shown below as well as time requirements with a 10 second time out and 25 GiB memory limit.

| Figure | Time (defaults) | Time (10 s, 25 GiB) | Memory (defaults) |
| :----- | :-------------- | ------------------: | ----------------: |
| fig7   | 5 hours         |              10 min |            32 GiB |
| fig8   | 71 hours        |              65 min |            37 GiB |
| fig9   | 2 min           |               2 min |           2.3 GiB |
| tab1   | 42 min          |              42 min |           5.7 GiB |

## Figure 7

This experiment compares AWDIT with the tools in the `tools` folder, all of which can check CC (or stronger).
The goal of this experiment is to show that tools other than AWDIT and Plume are much slower.
We test this by running all tools at the CC isolation level on a series of (relatively) small histories, collected from CockroachDB.
Concretely, we used three scripts, simulating different database workloads: TPC-C (`tpcc`) C-Twitter (`twitter`), and RUBiS (`rubis`).
For each of these, we graph the running time.

To reproduce Figure 7, run the Docker container as above, and inside run

```shell
scripts/reproduce-fig7.sh
```

By default, this runs with the time and memory limits used in the paper (10 min, 55 GiB).
To run the experiments with, e.g., a 10 second timeout and 25 GiB memory limit, run

```shell
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-fig7.sh
```

After the shell script finishes, exit the Docker container with `exit`, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

This will produce three files: `results/fig7-tpc.csv`, `results/fig7-twitter.csv`, and `results/fig7-rubis.csv`, containing the running times for all tools for the respective scripts.
These can be compared to the expected results: `results/fig7-tpc-expected.csv`, `results/fig7-twitter-expected.csv`, and `results/fig7-rubis-expected.csv`.

## Figure 8

This experiment compares AWDIT to Plume on 198 histories of sizes between 2^10 and 2^20, generated with different scripts.
Both AWDIT and Plume are able to check RC, RA, and CC, hence we compare running time on all three.
The results show that AWDIT always outperforms Plume, sometimes by 2-3 orders of magnitude.

To reproduce Figure 8, run the Docker container as above, and inside run

```shell
scripts/reproduce-fig8.sh
```

**Note:** Running this script with default parameters (2 hours, 55 GiB) takes around three days!
To run the experiments with, e.g., a 10 second timeout and 25 GiB memory limit, run

```shell
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-fig8.sh
```

After the shell scripts finishes, exit the docker container with `exit`, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

This will produce three files: `results/fig8-rc.csv`, `results/fig8-ra.csv`, and `results/fig8-cc.csv`, containing the running times for RC, RA, and CC, respectively.
These can be compared to the expected results: `results/fig8-rc-expected.csv`, `results/fig8-ra-expected.csv`, and `results/fig8-cc-expected.csv`.

## Figure 9

The final experiment tests the scaling of AWDIT, when varying the number of transactions, sessions, and operations per transaction, i.e., transaction size (while keeping the total history size fixed).

We find that AWDIT's performance scales linearly with transaction count, as expected with bounded size transactions.
As expected, number of sessions does not affect AWDIT's running time in RC and RA, whereas CC has a roughly linear scaling.
Finally, we find that transaction size only slightly affects RC and RA, as predicted.

To reproduce Figure 9, run the Docker container as above, and inside run

```shell
scripts/reproduce-fig9.sh
```

This should only take a few minutes, and use little memory.
After the shell scripts finishes, exit the docker container with `exit`, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

This will produce three files: `results/fig9-txn.csv`, `results/fig9-sess.csv`, and `results/fig9-ops.csv`, containing the running times for transaction scaling, sessions scaling, and transaction size scaling, respectively.
These can be compared to the expected results: `results/fig9-txn-expected.csv`, `results/fig9-sess-expected.csv`, and `results/fig9-ops-expected.csv`.

## Table 1

We found 8 consistency violations during testing. AWDIT correctly identifies all of these, whereas Plume fails to find some of them, due to crashing and time-outs (seemingly an infinite loop on some).

To reproduce Table 1, run the Docker container as above, and inside run

```shell
scripts/reproduce-tab1.sh
```

**Note:** All histories are run with a 10 min time-out, whereas the time-out from the paper depended on the type of benchmark. To use a 2 hour timeout, run 

```shell
TIME_LIMIT=7200 scripts/reproduce-tab1.sh
```

After the shell scripts finishes, exit the docker container with `exit`, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

This will produce two files: `results/tab1.csv` and `results/tab1-output.txt`.
`results/tab1.csv` contains information about the results of running AWDIT and Plume on the violating histories.
The result columns, e.g. `ours_rc (res)` (AWDIT RC Result), tells you whether the history is consistent `C`, inconsistent `I`, the solver ran out of time `DNF`, ... out of memory `OOM`, or crashed `Crash`.
This can be compared with `results/tab1-expected.csv`.
Finally, `results/tab1-output.txt` contains the output of running AWDIT on the history, which explains the exact violation detected.

## Plotting results

We also provide a script for plotting Figure 7-9 how they appear in the paper.
This can either be run inside the container or outside, once the data has been extracted, which requires a local installation of `pdflatex`.
To plot, run
```shell
scripts/plot-results.sh
```

## Structure of the artifact

For reference, the structure of the artifact is shown below:

```
AWDIT
│
└───docker
│   │   awdit-artifact-amd64.tar.gz # Docker image capable of running experiments (x86)
│   │   awdit-artifact-arm64.tar.gz # Docker image capable of running experiments (Arm64)
│   │   Dockerfile                  # The docker build script used to build the image
│   │   init-runexec.sh             # Init script for the Docker container
│
└───histories (from histories.tar.gz)
│   │   bench                       # Benchmarks
│   │   tests                       # Tests
│   │   violations                  # Violations found
│
└───results                         # Results of the benchmarks
│   │   fig7-tpcc-expected.csv      # Expected results for figure 7 TPC-C
│   │   ...
│
└───scripts                          Python and shell scripts to create and run benchmarks
│   │   test-run.sh                 # Test script
│   │   reproduce-fig7.sh           # Reproduce figure 7
│   │   reproduce-all.sh            # Reproduce all figures
│   │   ...
│
└───src                             # The Rust source code for AWDIT
│
└───tools                           # Other tools for comparison
│   │   CausalC+                    # Datalog implementation by Plume authors
│   │   dbcop                       # DBCop
│   │   mono                        # MonoSAT solver by Plume authors
│   │   Plume                       # Plume (artifact version)
│   │   PolySI                      # PolySI (artifact version)
```

# Manual usage of AWDIT

Other than using the provided scripts, AWDIT can be run as a standalone program.
Here, we provide information on how to do so.

## Dependencies

Building AWDIT requires Rust (>= 1.85).
The recommended way of installing Rust is through [rustup](https://rustup.rs).

If support for the DBCop format is desired (see the [formats](#formats) section), `libclang` is required.

## Building
To build, first update the submodules and run `cargo`:

```bash
git submodule update --init --recursive
cargo build --release
```

If support for the DBCop format is desired, add a feature flag:

```bash
cargo build --release --features dbcop
```

## Usage

We provide a brief introduction to the capabilities of the AWDIT tool.
For more information, run AWDIT with the `--help` flag:
```bash
target/release/awdit --help
```
Or, for information about a specific command:
```bash
target/release/awdit check --help
```

### Checking consistency

To check a history for consistency, use the `check` command:

```bash
target/release/awdit check -i <ISOLATION_LEVEL> path/to/history
```

The three possible values for `ISOLATION_LEVEL` are `read-committed`, `read-atomic`, and `causal`.
By default, the history will be assumed to be in the `plume` format (see the [formats](#formats) section for more information).

### Generating histories

To generate a random history, run

```bash
target/release/awdit generate output/path
```

By default, this will generate a history of 20 events in the `plume` format.

### Converting histories

To convert from one format to another, run

```bash
target/release/awdit convert -f <FROM_FORMAT> -t <TO_FORMAT> from/path to/path
```

### Getting statistics about a history

To get statistics about a history, run

```bash
target/release/awdit stats path/to/history
```

By default, the history is expected to be in the `plume` format, but the `--format` flag can be supplied to use a different format.

For JSON output, use the `--json` flag.

## Formats

The tool supports four history formats:

- `plume`: a text-based format used by Plume and PolySI. Histories in this format is a single `.txt` file.
  
- `dbcop`: a binary format used by DBCop. Histories in this format should be directories with a single file called `history.bincode`. Requires the `dbcop` feature.
  
- `cobra`: a binary format used by DBCobra. Histories in this format are directories with `.log` files (one for each session).
  
- `test`: a human-friendly text-based format useful for writing tests. A history in this format is a single `.txt` file.
