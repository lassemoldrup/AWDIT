# AWDIT Artifact

This README contains instructions on how to use the different capabilities of AWDIT.
For a concise description of how to reproduce experiments from the paper, see [REPRODUCE.md](./REPRODUCE.md).

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
└───res
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

## Dependencies

Running the tool requires Rust.
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
