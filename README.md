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

We also provide VirtualBox images for x86 and ARM matching the above requirements.
However, VMs do not produce reliable performance results, so please only use this as a last resort. 

The image was tested with VirtualBox version X.X

To use the image, log in as `user` with password `user`.
Then follow along from the folder `AWDIT` on the desktop.

## Structure of the Artifact

The structure of the artifact is shown below:

```
AWDIT
│   README.md               # This file
│   REPRODUCE.md            # Instructions for reproducing experiments
│   ...
│
└───docker
│   │   image.tar.gz        # Docker image capabable of running experiments
│   │   Dockerfile          # The docker build script used to build the image
│
└───histories
│   └───bench               # Benchmarks
│   └───tests               # Tests
│
└───results                 # Results of the benchmarks
│
└───scripts                 # Python and shell scripts to create and run benchmarks
│   │   reproduce-fig7.sh   # Reproduce figure 7
│   │   reproduce-all.sh    # Reproduce all figures
│   │   ...
│
└───src                     # The Rust source code for AWDIT
│
└───tools                   # Other tools for comparison
```

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
docker run --name awdit-container --privileged --cap-drop=all -ti -v $(pwd)/results:/home/user/awdit/results -v $(pwd)/histories:/home/user/awdit/histories awdit-artifact
```

To exit the container, simply run `exit`.

**Note:** The reason for `--privileged` is that we use [BenchExec](https://github.com/sosy-lab/benchexec) for running benchmarks, which uses Linux cgroups. To mitigate any risks when running like this, we use `--cap-drop=all` to disable as many capabilities of the container a possible, and the active user in the Docker container will be non-root. See [this document](https://github.com/sosy-lab/benchexec/blob/26ea602ced1bf339db124efb3cb53bc5dd94098c/doc/benchexec-in-container.md) for more information.

**Note:** To run the container more than once, run the following to delete the old container:

```shell
docker rm -v awdit-container
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
