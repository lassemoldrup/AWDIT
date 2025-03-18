# Getting Started

We provide two options for evaluating AWDIT:
- A Docker image (preferred)
- A VirtualBox VM image

## Requirements

### Docker Image

To run the Docker image you need
- A Linux system with kernel version >= 4.5
- Docker

**Important:** The Docker image will not work outside of Linux, since we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups provided by the host machine. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

### VM Image

We also provide VirtualBox images for x86 and ARM capabel of running the Docker image.
However, VMs do not produce reliable performance results, so please only use this as a last resort. 

The image was tested with VirtualBox version X.X

To use the image, ... (TODO: how to start image) and log in as `user` with password `user`.
Then follow along from the folder `AWDIT` on the desktop.

## Running the Docker Image

Running the Docker image will produce a shell capable of running AWDIT and all comparison tools.
To run the image, do the following.

### 1

Load the image:

```shell
docker load -i docker/image.tar.gz
```

Optionally, you can pass a platform (`linux/amd64` or `linux/arm64`, depending on your system):

```shell
docker load -i docker/image.tar.gz --platform linux/amd64
```

### 2

Run the container:

```shell
docker run --name awdit-container --privileged --cap-drop=all -ti --mount type=volume,dst=/home/user/awdit/results -v $(pwd)/histories:/home/user/awdit/histories awdit-artifact
```

To exit the container, simply run `exit`.

**Note:** The reason for `--privileged` is that we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups. To mitigate any risks when running like this, we use `--cap-drop=all` to disable as many capabilities of the container a possible, and the active user in the Docker container will be non-root. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

**Note:** To run the container more than once, run the following to delete the old container:

```shell
docker rm -v awdit-container
```

## Basic Testing

To check that everything is working, inside the container, run

```shell
scripts/test-run.sh
```

This will run AWDIT and Plume on a set small benchmarks and should take a few minutes.
When done, exit the container with `exit`. To retrieve the results from the container, run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

There should now be a file `results/small.csv` containing the results of the experiment.
This file should `results/small-expected.csv` (modulo performance difference).

# Step-by-Step Instructions

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

**Note**: In the results, `DNF` means that the tool timed out, `OOM` means that the tool exceded the memory limit.

## Figure 7

This experiment compares AWDIT with the tools in the `tools` folder, all of which can check CC (or stronger).
The goal of this experiment is to show that tools other than AWDIT and Plume are much slower.
We test this by running all tools at the CC isolation level on a series of (relatively) small histories, collected from CockroachDB.
Concretely, we used three scripts, simulating different database workloads: TPC-C (`tpcc`) C-Twitter (`twitter`), and RUBiS (`rubis`).
For each of these, we produce a graph of running time.

To reproduce Figure 7, run the Docker container as above, and inside run

```shell
scripts/reproduce-fig7.sh
```

By default, this runs with the time and memory limits used in the paper (10 min, 55 GiB).
To run the experiments with, e.g., a 10 second timeout and 25 GiB memory limit, run

```shell
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-fig7.sh
```

After the shell scripts finishes, exit the docker image with `exit`, and run

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
TIME_LIMIT=10 MEM_LIMIT=25 scripts/reproduce-fig7.sh
```

After the shell scripts finishes, exit the docker image with `exit`, and run

```shell
docker cp awdit-container:/home/user/awdit/results .
```

This will produce three files: `results/fig8-rc.csv`, `results/fig8-ra.csv`, and `results/fig8-cc.csv`, containing the running times for RC, RA, and CC, respectively.
These can be compared to the expected results: `results/fig8-rc-expected.csv`, `results/fig8-ra-expected.csv`, and `results/fig8-cc-expected.csv`.

## Figure 9

The final experiment...

## Structure of the Artifact

The structure of the artifact is shown below:

```
AWDIT
│   README.md                   # This file
│   REPRODUCE.md                # Instructions for reproducing experiments
│   ...
│
└───docker
│   │   image.tar.gz            # Docker image capabable of running experiments
│   │   Dockerfile              # The docker build script used to build the image
│
└───histories
│   └───bench                   # Benchmarks
│   └───tests                   # Tests
│
└───results                     # Results of the benchmarks
│   │   fig7-tpcc-expected.csv  # Expected results for figure 7 TPC-C
│   │   ...
│
└───scripts                     # Python and shell scripts to create and run benchmarks
│   │   test-run.sh             # Test script
│   │   reproduce-fig7.sh       # Reproduce figure 7
│   │   reproduce-all.sh        # Reproduce all figures
│   │   ...
│
└───src                         # The Rust source code for AWDIT
│
└───tools                       # Other tools for comparison
```

## Dependencies

Running the tool requires Rust (>= 1.85).
The recommended way of installing Rust is through [rustup](https://rustup.rs).

## Running the tool

This section provides a quick overview of the features of the tool.
To see all options, use the `--help` flag.
Before running, first update the submodules and build the tool:

```bash
$ git submodule update --init --recursive
$ cargo build --release
```

TODO: explain how to check consistency.

To generate a random history, run

```bash
$ target/release/awdit generate output/path
```

By default, this will generate a history of 20 events in the `plume` format (see the [formats](#formats) section for more information).

To convert from one format to another, run

```bash
$ target/release/awdit convert -f <FROM_FORMAT> -t <TO_FORMAT> from/path to/path
```

To get statistics about a history, run

```bash
$ target/release/awdit stats path/to/history
```

By default, the history is expected to be in the `plume` format, but the `--format` flag can be supplied to use a different format.

## Formats

The tool supports four history formats:

- `plume`: a text-based format used by Plume and PolySI. Histories in this format is a single `.txt` file.
- `dbcop`: a binary format used by DBCop. Histories in this format should be directories with a single file called `history.bincode`.
- `cobra`: a binary format used by DBCobra. Histories in this format are directories with `.log` files (one for each session).
- `test`: a human-friendly text-based format useful for writing tests. A history in this format is a single `.txt` file.
