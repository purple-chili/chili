![chili](https://github.com/purple-chili/chili/blob/main/icon.png?raw=true)

![PyPI - Version](https://img.shields.io/pypi/v/chili-pie?style=for-the-badge&color=%2321759b) ![Visual Studio Marketplace Version](https://img.shields.io/open-vsx/v/jshinonome/vscode-chili?style=for-the-badge&color=%23007ACC) ![PyPI - Version](https://img.shields.io/pypi/v/chiz?style=for-the-badge&label=chiz&color=%23FFA600)

# Chili

A programming language for data analysis and engineering, written in Rust.

Chili uses [Polars](https://www.pola.rs/) for data manipulation and [Arrow](https://arrow.apache.org/)/[Parquet](https://parquet.apache.org/) for data storage.

## Language Syntaxes

Chili provides two language syntaxes:

- **chili** — a modern syntax similar to JavaScript.
- **pepper** — a vintage syntax similar to q.

## Features

- **Polars-powered** — full Polars DataFrame support for high-performance data manipulation
- **Arrow / Parquet storage** — native support for columnar data formats
- **Partitioned tables** — date-partitioned Parquet storage with partition pruning
- **Lazy evaluation** — lazy DataFrames with `collect` for deferred computation
- **IPC / TCP** — built-in TCP listener for remote connections
- **Tick / Sub** — publish-subscribe infrastructure for real-time data
- **Job scheduler** — periodic job execution and memory monitoring
- **Cross-platform** — Linux, macOS, and Windows

## Installation

### Via pip (recommended)

```bash
pip install chili-pie
```

### From source

```bash
cargo build --release --bin chili
```

### Python bindings ([chili-sauce](https://pypi.org/project/chili-sauce/))

```bash
pip install chili-sauce
```

## Quick Start

Launch the REPL:

```bash
# chili syntax (default)
chili

# pepper syntax
chili -P
```

Evaluate a script:

```bash
chili script.chi
```

## Architecture

```
crates/
├── chili-bin      # CLI binary & REPL
├── chili-core     # Engine state, evaluator, IPC, and built-in functions
├── chili-op       # Data operations (partitioned I/O, scan, eval)
└── chili-parser   # Chumsky-based parser for both syntaxes
```

| Package | Description |
|---------|-------------|
| [chili-pie](https://pypi.org/project/chili-pie/) | Chili runtime distributed as a Python wheel |
| [chili-sauce](https://pypi.org/project/chili-sauce/) | Python bindings for `EngineState` via PyO3 |

## Development

### Appending while a snapshot is being written

For a subscriber buffer that is queried or flushed concurrently, use the symbol
form of `vstack` (Pepper syntax):

```pepper
.s.upd: {[data] vstack[`.s.buf; data]};
.s.flush: {[path; lo; hi]
  window: select from .s.buf where seq > lo, seq <= hi;
  wparquet[path; window; 3]
};
```

`vstack[table_symbol; dataframe]` atomically appends chunks and returns the incoming
row count. It creates a missing table and follows the existing table's dtypes and
timezones, like symbol `upsert`. A failed append preserves the table. Existing
`vstack[dataframe; dataframe]` calls still return a new dataframe without changing
either input.

This append path preserves the old buffers held by snapshots. Ordinary `upsert`
extends the backing storage and may copy the accumulated data when a snapshot is
alive. Use `vstack` consistently for such a buffer: switching back to `upsert` can
rechunk and copy its history.

Each input chunk adds metadata; batch incoming rows where practical and manage
retention. This operation does not compact chunks, reclaim flushed rows, or
guarantee a wall-clock latency bound. Filtering and Parquet preparation can still
allocate. Run the flush through `fn_call` from a separate calling thread; that
call releases the GIL and runs independently of the subscriber's apply thread.
Coordinate retention with the writer, and advance the flushed sequence only after
a successful write under the application's durability policy.

`wparquet[path; dataframe; level]` honors zstd levels 1–22. File sizes can still
differ from Python Polars because writer versions and row-group/page settings
can differ.

### Build and test

```bash
# Build
cargo build --bin chili

# Test
cargo test

# Clippy
cargo clippy --fix --allow-dirty

# Build & install Python runtime wheel
maturin build --release --out dist --manifest-path crates/chili-bin/Cargo.toml

# Build & install Python bindings
maturin develop --release --manifest-path crates/chili-py/Cargo.toml
```

A [Taskfile](https://taskfile.dev/) is provided for common workflows — run `task --list` to see all available tasks.

## References

- [Polars](https://www.pola.rs/) — an open-source library for data manipulation
- [chumsky](https://github.com/zesterer/chumsky) — a high-performance parser combinator library
- [kola](https://github.com/jshinonome/kola) — source code reused for interfacing with q
- [Nushell](https://www.nushell.sh/) — a new type of shell

Thanks to the authors of these projects and all other open-source projects for their great work.

## License

[MIT](LICENSE)
