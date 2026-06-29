# Changelog

All notable changes to this project will be documented in this file.

## [0.9.4] - 2026-06-29

### Added

- `set_subscriber_queue_max` on `EngineState` and Python `ChiliEngine` — optionally shed Publishing subscribers whose outbound queue fills (`try_send` + dedicated writer thread); default `0` disables
- Post-eval audit hook — `set_post_eval_hook`, `get_post_eval_hook`, and `clear_post_eval_hook`; `eval_with_pre_hook` fires `(user; handle; query; result; error)` after inbound IPC evaluation (hook errors are logged and ignored)
- `set_post_eval_hook`, `clear_post_eval_hook`, and `get_post_eval_hook` on Python `ChiliEngine`
- `.handle.reply` built-in and `EngineState::reply` — fire-and-forget async write to any connected handle, including incoming caller connections
- `set_jobs_deactivate_on_error` and `jobs_deactivate_on_error` on `EngineState` and Python `ChiliEngine` — optionally deactivate scheduled jobs after a fire error instead of rescheduling every interval
- Pre-eval request hook — `EngineState::eval_with_pre_hook`, `set_pre_eval_hook`, and `get_pre_eval_hook`; inbound IPC requests can be routed through a registered `(user; handle; query) -> query'` function before evaluation (allow, rewrite, or deny via `raise`)
- `set_pre_eval_hook`, `clear_pre_eval_hook`, and `get_pre_eval_hook` on Python `ChiliEngine`
- `.broker.subscribeFiltered` built-in — register a subscriber on one topic with an optional per-handle row filter (`column` + allowed values); empty values means no filter
- `.tick.subscribeFiltered` and `.sub.initFiltered` pepper helpers — filtered live broadcast with unfiltered historical replay; filter state is restored on `.sub.recover`
- `filters` parameter on Python `ChiliEngine.subscribe()` — `{topic: (column, [values])}`; each filtered topic uses its own connection
- `bind_tcp_listener` and `run_accept_loop` on `EngineState` — separate synchronous bind from the blocking accept loop
- Integration tests for post-eval hook (`post_eval_hook_test.rs`), slow-subscriber shed (`slow_subscriber_shed_test.rs`), async reply (`async_reply_test.rs`), job quarantine (`job_quarantine_test.rs`), pre-eval hook (`pre_eval_hook_test.rs`), filtered subscribe (`test_tick_sub_filtered.py`), TCP listener bind behavior (`test_tcp_listener_bind.py`), and partition atomic overwrite / codec (`partition_atomic_codec_test.rs`)
- `wparc` built-in — customized partition write (`atomic` single-shard overwrite, per-call Parquet compression); `wpar` unchanged at 7 args
- `write_partition_native_full` exported from `chili-op`
- `write_partitioned_df_custom` on Python `ChiliEngine`

### Changed

- Publishing subscribers can use a bounded outbound queue: `publish` and `signal_eod` `try_send` whole frames and shed subscribers when the queue is full
- Handle I/O (`sync`, `async_`, `reply`, `publish`, `signal_eod`) runs blocking reads/writes under per-handle mutexes instead of holding the global handle map lock — slow connections no longer serialize all handle operations
- `execute_jobs` deactivates a failing scheduled job when `jobs_deactivate_on_error` is enabled; default behaviour (log and reschedule) is unchanged
- Inbound IPC conn handlers (`handle_q_conn`, `handle_chili_conn`) use `eval_with_pre_hook` instead of `eval` when a pre-eval hook is registered; local/REPL eval is unchanged
- `EngineState::publish` now accepts the update frame and applies per-subscriber row filters before serializing; each distinct filter is serialized once and shared across matching handles
- Python `start_tcp_listener` binds synchronously and raises `ChiliError` when the port is unavailable, instead of failing asynchronously via process exit
- TCP listener bind uses `socket2` with `SO_REUSEADDR` set before bind
- `SpicyObj::to_str_vec` accepts a `MixedList` of strings/symbols so Python symbol lists work as filter values

## [0.9.3] - 2026-06-20

### Added

- `.broker.validateSeqStrict` built-in — strict variant of `validateSeq` that returns an error on any corrupt or truncated frame instead of silently truncating; opens the file read-only

### Fixed

- `serde9::deserialize` now returns `Err` instead of panicking on truncated/torn frames — buffer-derived-length slices are bounds-checked via a `take` helper, so `count_seq_messages` torn-tail recovery works as designed
- Fixed Boolean Series deserialization mismatch — `length` (bit count) was incorrectly used as a byte count to slice the bitmap data; now reads remaining bytes (`&data_bytes[i..]`) to match what `serialize_bitmap` actually writes

## [0.9.2] - 2026-06-14

### Fixed

- `SyncFile` now recovers from sticky EIO by closing and reopening the underlying fd on `flush()` failure, fixing the `fsync_handle` / `rotate_handle` wedge on transient I/O errors
- `replay_chili_msgs_log` now tolerates torn trailing records (partial headers, short payloads, corrupt frames) — stops at the last valid frame with a warning instead of crashing replay
- `close_handle` now logs flush errors instead of silently swallowing them

## [0.9.1] - 2026-06-11

### Added

- `drain(id)` built-in — atomically take the accumulated DataFrame for a subscriber topic variable and reset it to a 0-row frame with the same schema, all under a single write-lock; eliminates the O(rows-so-far) cost of polling `get_var` on high-throughput subscribers
- `drain` method on Python `ChiliEngine` — `engine.drain(topic)` returns a `polars.DataFrame`

## [0.9.0] - 2026-05-25

### Added

- `.handle.fsync` built-in — explicitly flush a file handle's buffered data to disk (`fdatasync`), giving users on-demand durability control
- `SyncFile` wrapper in `utils.rs` — makes `Write::flush` call `sync_data()` so that file handle flushes issue `fdatasync` instead of the default no-op
- `detect_conn_type` utility in `utils.rs` — shared file-type detection (New/File/Sequence) from magic header bytes, used by `prepare_file_writer` and `validate_seq`
- `count_seq_messages` utility in `utils.rs` — walks sequence file frames and returns `(msg_count, valid_byte_size)`, used by `prepare_file_writer` and `validate_seq`
- `fsync_handle` method on Python `ChiliEngine` — exposes `.handle.fsync` via `fn_call`
- `async_` and `execute` on `EngineState` — positive handle numbers use sync IPC/file writes; negative handle numbers send async IPC without waiting for a response
- String literals in `eval_op` / `eval_call` are parsed and evaluated as Chili/Pepper query source (inline `eval_str` behavior)
- `py.typed` marker in `chili-py` for PEP 561 type checkers

### Changed

- `rotate_handle` skips rotation when the target URI already exists in the handle map, avoiding duplicate file handles for the same path
- `rotate_handle` now accepts non-empty files and sets `tick_count` to the existing message count for sequence files
- `close_handle` now flushes the writer (best-effort `fdatasync`) before dropping the handle
- `rotate_handle` now flushes the old handle's writer before replacing it, ensuring all data is durable on disk before the new file is opened
- `prepare_file_writer` returns `(writer, conn_type, msg_count)` — for sequence files, truncates to the last valid message boundary and reports the message count
- `validate_seq` refactored to use `detect_conn_type` and `count_seq_messages` utilities
- Handle sends in eval route through `execute` instead of always calling `sync`
- TCP incoming listener logs and drops bad connections instead of panicking on accept, auth, or handle setup failures

### Fixed

- `sync` no longer deadlocks when marking a handle disconnected after a failed write (sets `ConnType::Disconnected` inline instead of re-acquiring the handle lock)

## [0.8.2] - 2026-05-24

### Added

- `rotate_handle(handle_num, uri)` built-in (`.handle.rotate`) — swap a file handle's writer to a new `file://` path and reset its tick counter, enabling log rotation without closing/reopening handles
- `query_plan(query, hdb_path)` in Python bindings — returns the Polars logical plan string for a query without executing it, useful for query-tuning workflows
- `add_at_time(fn_name, start_time, description)` in Python bindings — schedule a pepper function to fire once at a given time on the chili job scheduler thread
- `table_count()` in Python bindings — return the number of partitioned tables currently loaded
- `lazy` parameter on `ChiliEngine.eval()` — when `True`, DataFrame results are returned as `polars.LazyFrame`
- `write_partitioned_df` now accepts `datetime.date` directly for the `date` parameter
- Default arguments for `tick(index=0, inc=1)` and `get_tick_count(index=0)`
- `test-py` task in Taskfile for running pytest on chili-py
- Parse cache unit tests and log-rotation integration tests
- Criterion benchmark for categorical eval in `chili-op`
- Register `LOG_FN` in Python engine for logging support
- Package import support in `import` — paths starting with `@` or alphabetic characters are resolved as chiz package imports (e.g. `import "@scope/dep-name/util"` resolves to `$CHIZPATH/@scope/dep-name/<version>/src/util.chi`)
- Version is resolved from local `chiz_index.json` or global `$CHIZPATH/.index`; import fails if the package is not found in either index
- File extension order follows the current language setting (`.chi` first in Chili mode, `.pep` first in Pepper mode)
- Supports deeper module paths (e.g. `import "@scope/pkg/sub/module"` → `src/sub/module.chi`)

### Changed

- Moved `prepare_file_writer` helper to `utils.rs`, shared between `open_handle` and `rotate_handle`
- `LazyCell` → `LazyLock` for the regex style constant (thread-safe)
- GIL released around `load_par_df`, `clear_par_df`, `add_at_time`, and `query_plan` in Python bindings
- `eval()` in Rust binding simplified — lazy/eager normalization moved to the Python layer
- Replaced `disconnect_handle` with `ConnType::Disconnected` flag on send errors to avoid killing subscriber handles
- `signal_eod` refactored for robustness
- Moved `logger.rs` from `chili-bin` to `chili-op`

### Fixed

- `upsert` / `insert` clippy lint fixes (unnecessary references)

## [0.8.0] - 2026-05-03

### Added

- Python bindings package "chili-sauce" for the Chili engine via PyO3
- `load_par_df` now recursively traverses subdirectories to discover tables, producing dot-separated qualified names (e.g. `load("/path/sub")` loads `sub.trade`, `sub.order`)
- `HandleOutOfRangeErr` error variant for handle numbers outside 0..1024
- Tick/Sub publish-subscribe system for Python bindings:
  - `init_tick(schema, log_dir, date)` — initialize the tick engine with table schemas
  - `publish(table, data)` — publish data to subscribers
  - `subscribe(tick_socket, topics)` — subscribe to a tick engine and replay log
  - Bundled `tick.pep` and `sub.pep` scripts for tick and subscriber logic
- `list_handle()` — return a DataFrame listing all active handles (num, socket, conn_type, ipc_type, is_local, on_disconnected)
- `.handle.exists` built-in function to check if a handle exists
- `job_interval` and `memory_limit` parameters to `EngineState` constructor (Python and Rust)
- `start_job_scheduler()` and `start_memory_monitor()` methods on `EngineState`
- `src_path` parameter to `eval()` in the Python binding for source location in error traces
- `tick(index, inc)` and `get_tick_count(index)` now accept an index parameter for multiple tick counters

### Changed

- Moved `check_memory_usage`, job scheduler, and memory monitor logic from `chili-bin` into `chili-core::EngineState` methods
- `sysinfo` dependency moved from `chili-bin` to `chili-core`
- `upsert` now supports `MixedList` in addition to `Series` for the collection argument

### Fixed

- **serde9 nested mixed list deserialization** — fixed a critical bug where nested `MixedList` items were deserialized from a hardcoded buffer offset (byte 16) instead of the current position, causing inner list elements to be read as the parent list's data. This was the root cause of broker topic misregistration in IPC pub/sub.
- Added bounds guard on `tick_count` access — handle numbers outside 0..1024 now return an error instead of panicking

## [0.7.5] - 2026-04-15

### Added

- Criterion benchmarks for `chili-op` (`load_par_df`, `scan`, `eval`, `write_partition`)
- Integration tests covering HDB partition loading + selection (R1 regression guard)
- `write_partition_native` helper for writing partitions programmatically

### Changed

- `load_par_df` now builds table metadata in parallel and shrinks the write-lock window
- `write_partition` optimizes parquet output for partition pruning by tuning row group size when sorting

### Fixed

- Partition discovery is now deterministic (partition vectors are sorted before binary-search-based lookups)
- Partition predicate handling now scans all where-clauses and preserves non-partition filters (fixes missing rows for `where date=...` / ranges)

## [0.7.4] - 2026-03-21

### Added

- Cache source code from REPL and IPC connections for printing error messages

## [0.7.2] - 2026-02-24

### Removed

- Removed the `%` operator
- Vim syntax highlighting support, chiz will support it soon

### Fixed

- Fixed the handling of Windows paths
- Fixed the symbol token for Windows paths
- Fixed `upsert` and `insert` to support DataFrame as the first argument

## [0.7.1] - 2026-02-22

### Added

- Supported Windows

## [0.7.0] - 2026-02-21

### Added

- Refactored to use chumsky as the parser
- Unified the binaries into one chili binary
- Added `-P` or `--pepper` flag to enable REPL using pepper syntax
- Supported macOS

## [0.6.4] - 2026-02-08

### Added

- Improved the handling of MixedList in the deserialization process to return an empty list when appropriate.
- Modified ListExp, SelectExp, ByExp, Table, Matrix, and Dict to support optional trailing commas.
- Enhanced BracketExp and ColNames to maintain consistency with the new syntax rules.

## [0.6.3] - 2026-01-25

### Added

- Lazy evaluation mode for the runtime
- New built-in function `collect` to collect a lazy DataFrame
- `os.pid` to retrieve the process ID
- `os.version` to retrieve the operating system version
- `os.syntax` to retrieve the syntax type (chili or pepper)

## [0.6.2] - 2025-12-14

### Added

- New built-in function `insert` to insert data from DataFrame or list and keep the last record for each group
- New built-in function `.os.mem` to retrieve memory statistics
- Memory limit command line option for the runtime

### Changed

- Enhanced upsert function to enforce DataFrame type for the first argument
- Updated parser to support new syntax for column definitions

## [0.6.1] - 2025-12-10

### Added

- Support for new syntax highlighting and validation in Chili language
- Short-circuit evaluation for logical operators (`||`, `&&`, `??`) in AST and evaluation logic
- Support for `if` statements with optional `else` blocks for `chili` language
- New binary operators and control keywords in grammar definitions

### Changed

- Enhanced startup banner with vintage feature support and updated graphics
- Updated job function name formatting in EngineState evaluation
- Updated parser to support new control flow structures
- Adjusted tests to reflect changes in parsing and evaluation structure

### Fixed

- Corrected comparison operator in 'lt' function

## [0.6.0] - 2025-12-07

### Added

- Initial release of Chili
- Support for two language syntaxes:
  - chili: a modern programming language similar to JavaScript
  - pepper: a vintage programming language similar to q
- Integration with Polars for data manipulation
- Support for Arrow/Parquet data storage
- Vim syntax highlighting support
