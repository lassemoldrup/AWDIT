# AWDIT - A Weak Database Isolation Tester

AWDIT tests database histories for consistency with three isolation levels: Read Committed, Read Atomic, and Causal Consistency.

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
