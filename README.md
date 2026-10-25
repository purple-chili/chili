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

Each input chunk adds metadata. A snapshot is cheap to create, but the next
append may copy chunk descriptors while holding the variable lock. Batch incoming
rows where practical. Symbol-form `vstack` now automatically schedules background
compaction: a separate Rust worker starts on the first nonempty append, waits
250 ms between passes, and processes tables changed by `vstack`. It merges small
chunks with a 65,536-row target. No Python timer or job scheduler is required,
and compaction does not run on the subscriber's apply thread. Tables changed
only by `set_var` or `upsert` are not scheduled.

Use `.compact.configure[interval_ms; target_rows]` (Pepper) or
`engine.set_auto_compaction(interval_ms=250, target_rows=65536)` (Python) to tune
the worker. A zero interval disables future passes; a running pass may finish.
Python also accepts `ChiliEngine(compaction_interval=250,
compaction_target_rows=65536)`. `shutdown()` stops the worker and waits for any
running pass. Configuration after shutdown is rejected.

`compact[table_symbol; target_rows]` merges adjacent small chunks outside the
variable write lock. Already-large chunks keep their buffers, so compacting a
small tail does not repeatedly copy a large seeded prefix. The positive row
target controls grouping, not a maximum chunk or memory size: groups can exceed
the target and the final small group can fall below it. Columns may have
different chunk boundaries. `compact[dataframe; target_rows]` returns a compacted
dataframe without changing the input.

The symbol form publishes atomically and returns the compacted snapshot's row
count. Concurrent `vstack` appends are preserved in order after that snapshot.
Only one compaction can be pending for the current table state. An explicit
`compact` call reports an error if that table is already being compacted; the
automatic worker skips busy tables and retries on a later pass. A drain,
replacement, deletion, `upsert`,
`upsertn`, or other non-`vstack` mutation invalidates a pending compaction; it
returns a cancellation error instead of overwriting newer data. Retry later on
the current table. Old snapshots remain valid and retired buffers are released
outside the lock.

Retention by assignment, such as `.s.buf: neg[.s.keep] # .s.buf`, counts as a
replacement even though the result is a slice of the same table. It invalidates
an in-flight pass and removes the table from the pending compaction queue; a
subsequent `vstack` schedules it again. Frequent replacements can repeatedly
cancel compaction, leaving many small chunks and slow readers despite a bounded
row count. The worker cannot publish the old prefix after a trim because that
could restore rows the application removed.

For periodic bounded retention, ``upsertn[`.s.buf; 0 # data; .s.keep]`` appends an
empty dataframe with the incoming batch's schema and retains at most `.s.keep`
rows atomically. It also invalidates pending compaction. Its existing `extend`
path merges fragmented numeric columns itself; it does not preserve the
background worker's pass. This can copy the retained data while holding the
variable write lock, including on the apply thread when called from `upd`.
Choose the retained window and trim frequency against the application's latency
budget, especially while snapshots are alive. Keep ordinary incoming appends on
`vstack` when using this pattern.

Tune the interval for the incoming chunk rate. Automatic compaction does not
reclaim flushed rows or impose a latency guarantee. A very fragmented
buffer can still incur metadata work during concurrent appends and publication.
Filtering, compaction, and writing share CPU and memory bandwidth. Coordinate
retention with the writer, and advance the flushed sequence only after a
successful write under the application's durability policy.

`wparquet[path; dataframe; level]` honors zstd levels 1–22 and uses the ordered
streaming Parquet sink, as Python Polars does. In the pinned Rust Polars 0.55.2
implementation, unset row-group size uses an approximate 122,880-row streaming
target. `POLARS_IDEAL_SINK_MORSEL_SIZE_ROWS` and
`POLARS_IDEAL_SINK_MORSEL_SIZE_BYTES` can override the streaming targets; page
size defaults to 1 MiB. Min/max/null-count statistics are enabled, distinct counts
are disabled, and encoding selection is automatic. The returned integer is the
file size in bytes.

The streaming row target is approximate, not a fixed row-group size. Input chunk
layout affects row-group boundaries, file size, and file bytes, even for the same
ordered rows and compression level. Automatic compaction changes the live
table's chunk layout over time, so separate snapshots of the same logical rows
can produce different Parquet bytes. An existing snapshot remains unchanged by
later compaction. The sink environment variables adjust targets; they do not
select exact row-group sizing, which `wparquet` does not currently expose.
Do not use Parquet file hashes as a test of dataframe equality or expect byte
parity with Python Polars. Compare the schema and ordered values for logical
equality; exact-byte reproducibility requires controlling physical layout,
writer options, and library versions as well.

The Python extension and CLI use mimalloc for Rust allocations; the extension
does not replace Python's allocator.

`crates/chili-py/tests/bench_flush.py` provides a local synthetic acceptance
probe. Run it against a release wheel with NumPy, Polars, and PyArrow installed:

```bash
POLARS_MAX_THREADS=1 python crates/chili-py/tests/bench_flush.py writer --rows 8000000 --output tmp/writer.json
POLARS_MAX_THREADS=1 python crates/chili-py/tests/bench_flush.py stall --rows 12550000 --output tmp/stall.json
POLARS_MAX_THREADS=1 python crates/chili-py/tests/bench_flush.py reader --rows 0 --rate 0 --batch-rows 3 --max-frames 2000000 --read-interval 2 --loaded 32 --output tmp/reader.json
```

The TCP probes verify every received row and record subscriber apply timestamps,
append service times, Python progress observations, and publisher gaps separately.
JSON results include the environment, parameters, and raw operation timings;
adjacent Parquet/NPZ files retain the samples. The native compaction worker is
enabled by default; `--auto-compaction-ms 0` provides an uncompacted control.
For writer fragmentation checks,
use `--chunks 10000`; adding `--compact-interval .25` in writer mode compacts each
fresh buffer before timing and disables the native worker. In concurrent modes,
that option instead uses a separate Python maintenance thread for comparison.
Writer preparation time is reported separately and
excluded from the writer ratio. These probes use their own corpus and are not a
substitute for acceptance runs on the deployment hardware and workload.

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
