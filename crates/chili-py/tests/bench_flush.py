"""Local flush acceptance probe; this is not the customer's missing repro.

Run with an isolated built wheel on PYTHONPATH and POLARS_MAX_THREADS=1.
Examples:
  python bench_flush.py writer --rows 8000000 --output /tmp/writer.json
  python bench_flush.py stall --rows 12550000 --output /tmp/stall.json
  python bench_flush.py reader --rows 0 --rate 0 --batch-rows 3 --output /tmp/reader.json

All engines run in one process. TCP publisher and subscriber use real engine
threads. Subscriber timestamps record actual apply gaps and append service
times, separately from Python's sampled-progress gaps and publisher gaps.
Percentiles describe these local samples; they are not latency guarantees.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import statistics
import tempfile
import threading
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from chili import ChiliEngine


FUNCTIONS = """
.s.append: {[data] vstack[`.s.buf; data]};
.s.compact: {[] compact[`.s.buf; 65536]};
.s.window: {[lo; hi] select from .s.buf where seq >= lo, seq <= hi};
.s.count: {[lo; hi] count select from .s.buf where seq >= lo, seq <= hi};
.s.flush: {[path; lo; hi]
  window: select from .s.buf where seq >= lo, seq <= hi;
  wparquet[path; window; 3]
};
.s.whole: {[path] wparquet[path; .s.buf; 3]};
"""


def corpus(rows: int, start: int = 0, compressible: bool = False) -> pl.DataFrame:
    rng = np.random.default_rng(20261024)
    seq = np.arange(start, start + rows, dtype=np.int64)
    return pl.DataFrame(
        {
            "seq": seq,
            "px": np.full(rows, 123.45)
            if compressible
            else rng.integers(0, 1_000_000, rows).astype(float) / 100,
            "qty": np.ones(rows, dtype=np.int64)
            if compressible
            else rng.integers(0, 100_000, rows, dtype=np.int64),
            "time": seq + 1_700_000_000_000_000_000,
            "venue": np.zeros(rows, dtype=np.int64)
            if compressible
            else rng.integers(0, 8, rows, dtype=np.int64),
        }
    )


def stats(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"samples": 0}
    return {
        "samples": len(values),
        "median_ms": float(np.median(values)),
        "p99_ms": float(np.percentile(values, 99)),
        "p999_ms": float(np.percentile(values, 99.9)),
        "max_ms": float(values.max()),
    }


def environment(args) -> dict:
    return {
        "chili": importlib.metadata.version("chili-sauce"),
        "polars": pl.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "polars_threads": pl.thread_pool_size(),
        "parameters": vars(args) | {"output": str(args.output)},
        "scope": "local synthetic reproduction; not the original customer scripts or corpus",
        "compaction_mode": "manual caller"
        if args.compact_interval > 0
        else ("automatic worker" if args.auto_compaction_ms > 0 else "disabled"),
    }


def seed(engine, data, chunks):
    engine.set_var(".s.buf", data.head(0))
    if chunks <= 1:
        engine.set_var(".s.buf", data)
    else:
        for piece in data.iter_slices(max(1, (data.height + chunks - 1) // chunks)):
            engine.fn_call(".s.append", [piece])


def writer(args):
    data = corpus(args.rows, compressible=args.compressible)
    engine = ChiliEngine(
        pepper=True,
        compaction_interval=0 if args.compact_interval > 0 else args.auto_compaction_ms,
    )
    engine.eval(FUNCTIONS)
    records = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="chili-writer-", dir=args.output.parent
        ) as directory:
            for trial in range(args.trials):
                arms = [
                    "drain_python",
                    "select_wparquet",
                    "whole_wparquet",
                    "select_only",
                ]
                if trial % 2:
                    arms.reverse()
                for arm in arms:
                    seed(engine, data, args.chunks)
                    compact_seconds = 0
                    if args.compact_interval > 0:
                        compact_start = time.perf_counter()
                        engine.fn_call(".s.compact", [])
                        compact_seconds = time.perf_counter() - compact_start
                    input_chunks = engine.get_var(".s.buf").n_chunks("all")
                    path = str(Path(directory) / f"{arm}.parquet")
                    start = time.perf_counter()
                    if arm == "drain_python":
                        taken = engine.drain(".s.buf")
                        taken.write_parquet(
                            path, compression="zstd", compression_level=3
                        )
                    elif arm == "select_wparquet":
                        engine.fn_call(".s.flush", [path, 0, args.rows - 1])
                    elif arm == "whole_wparquet":
                        engine.fn_call(".s.whole", [path])
                    else:
                        assert (
                            engine.fn_call(".s.count", [0, args.rows - 1]) == args.rows
                        )
                    seconds = time.perf_counter() - start
                    record = {
                        "trial": trial,
                        "arm": arm,
                        "seconds": seconds,
                        "preparation_compaction_seconds": compact_seconds,
                        "input_chunks": input_chunks,
                    }
                    if arm != "select_only":
                        actual = pl.read_parquet(path)
                        assert actual.equals(data), arm
                        metadata = pq.ParquetFile(path).metadata
                        record.update(
                            bytes=Path(path).stat().st_size,
                            row_group_rows=[
                                metadata.row_group(i).num_rows
                                for i in range(metadata.num_row_groups)
                            ],
                            exact_readback=True,
                        )
                        del actual
                    records.append(record)
            medians = {
                arm: statistics.median(r["seconds"] for r in records if r["arm"] == arm)
                for arm in arms
            }
            sizes = {
                arm: next(r["bytes"] for r in records if r["arm"] == arm)
                for arm in arms
                if arm != "select_only"
            }
            ratio = medians["drain_python"] / medians["select_wparquet"]
            return {
                "environment": environment(args),
                "input_bytes": data.estimated_size(),
                "records": records,
                "median_seconds": medians,
                "bytes": sizes,
                "throughput_ratio": ratio,
                "throughput_gate_pass": ratio >= 0.9,
                "timing_scope": "write/select only; seeding and optional preparation compaction excluded",
                "size_difference_pct": 100
                * (sizes["select_wparquet"] / sizes["drain_python"] - 1),
            }
    finally:
        engine.shutdown()


def concurrent(args):
    initial = corpus(args.rows, compressible=args.compressible)
    total_time = args.warmup + args.baseline + args.loaded
    frames = int(args.rate * (total_time + 20)) if args.rate > 0 else args.max_frames
    incoming = corpus(frames * args.batch_rows, args.rows, args.compressible)
    publisher = ChiliEngine(pepper=True)
    subscriber = ChiliEngine(
        pepper=True,
        compaction_interval=0 if args.compact_interval > 0 else args.auto_compaction_ms,
    )
    stop = threading.Event()
    errors = []
    published = []
    observed = []
    compactions = []
    operations = []
    threads = []

    def guarded(fn):
        def run():
            try:
                fn()
            except BaseException as exc:
                errors.append(repr(exc))
                stop.set()

        return run

    try:
        with tempfile.TemporaryDirectory(
            prefix="chili-concurrent-", dir=args.output.parent
        ) as directory:
            publisher.init_tick({"trade": initial.head(0)}, directory + "/", "ticklog")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            publisher.start_tcp_listener(port)
            subscriber.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
            subscriber.set_var(".s.buf", initial)
            subscriber.set_var(".s.last", args.rows - 1)
            subscriber.set_var(
                ".s.metrics",
                pl.DataFrame(
                    schema={
                        "seq": pl.Int64,
                        "start": pl.Datetime("ns"),
                        "end": pl.Datetime("ns"),
                    }
                ),
            )
            subscriber.eval(FUNCTIONS)
            subscriber.eval(f"""
              upd: {{[table; data]
                start: now[`UTC]; {args.append}[`.s.buf; data]; end: now[`UTC];
                n: last data[`seq]; upsert[`.s.metrics; (n; start; end)];
                .s.last: n; tick[this.h; 1];
              }};
            """)

            def publish():
                deadline = time.perf_counter()
                for i in range(frames):
                    if stop.is_set():
                        break
                    piece = incoming.slice(i * args.batch_rows, args.batch_rows)
                    start = time.time_ns()
                    publisher.publish("trade", piece)
                    published.append((start, time.time_ns()))
                    if args.rate:
                        deadline += 1 / args.rate
                        delay = deadline - time.perf_counter()
                        if delay > 0:
                            stop.wait(delay)

            def observe():
                last = args.rows - 1
                while not stop.wait(0.0005):
                    current = subscriber.get_var(".s.last")
                    if current != last:
                        observed.append((time.time_ns(), current))
                        last = current

            def compact():
                while not stop.wait(args.compact_interval):
                    start = time.time_ns()
                    count = subscriber.fn_call(".s.compact", [])
                    compactions.append((start, time.time_ns(), count))

            for fn in [publish, observe] + (
                [compact] if args.compact_interval > 0 else []
            ):
                thread = threading.Thread(target=guarded(fn))
                thread.start()
                threads.append(thread)
            time.sleep(args.warmup)
            baseline_start = time.time_ns()
            time.sleep(args.baseline)
            loaded_start = time.time_ns()
            deadline = time.perf_counter() + args.loaded
            next_flush = next_read = time.perf_counter()
            last_written_hi = None
            output = str(Path(directory) / "flush.parquet")
            while time.perf_counter() < deadline and not errors:
                hi = subscriber.get_var(".s.last")
                if time.perf_counter() >= next_read and hi >= 1999:
                    start = time.time_ns()
                    selected = subscriber.fn_call(".s.window", [hi - 1999, hi])
                    end = time.time_ns()
                    assert selected.height == 2000
                    assert np.array_equal(
                        selected["seq"].to_numpy(), np.arange(hi - 1999, hi + 1)
                    )
                    operations.append(
                        {"kind": "read", "start": start, "end": end, "hi": hi}
                    )
                    del selected
                    next_read = time.perf_counter() + args.read_interval
                if args.shape == "stall" and time.perf_counter() >= next_flush:
                    start = time.time_ns()
                    size = subscriber.fn_call(".s.flush", [output, 0, hi])
                    operations.append(
                        {
                            "kind": "flush",
                            "start": start,
                            "end": time.time_ns(),
                            "hi": hi,
                            "bytes": size,
                        }
                    )
                    last_written_hi = hi
                    next_flush = time.perf_counter() + args.flush_interval
                time.sleep(0.001)
            loaded_end = time.time_ns()
            stop.set()
            for thread in threads:
                thread.join(timeout=30)
                assert not thread.is_alive(), "worker did not finish"
            assert not errors, errors
            assert len(published) < frames, (
                "publisher exhausted input; increase --max-frames or lower --rate"
            )
            expected_rows = args.rows + len(published) * args.batch_rows
            drain_deadline = time.perf_counter() + 30
            while (
                subscriber.get_var(".s.last") != expected_rows - 1
                and time.perf_counter() < drain_deadline
            ):
                time.sleep(0.01)
            metrics = subscriber.get_var(".s.metrics")
            live = subscriber.get_var(".s.buf")
            assert live.height == expected_rows, (live.height, expected_rows)
            assert metrics.height == len(published), (metrics.height, len(published))
            expected = pl.concat(
                [initial, incoming.head(len(published) * args.batch_rows)]
            )
            assert live.equals(expected), "lost, duplicated, reordered, or changed rows"
            if last_written_hi is not None:
                assert pl.read_parquet(output).equals(
                    expected.head(last_written_hi + 1)
                )
            metrics.write_parquet(args.output.with_suffix(".metrics.parquet"))
            starts = metrics["start"].cast(pl.Int64).to_numpy()
            ends = metrics["end"].cast(pl.Int64).to_numpy()
            published_array = np.asarray(published, dtype=np.int64).reshape(-1, 2)
            observed_array = np.asarray(observed, dtype=np.int64).reshape(-1, 2)
            np.savez_compressed(
                args.output.with_suffix(".raw.npz"),
                published=published_array,
                observed=observed_array,
                compactions=np.asarray(compactions, dtype=np.int64).reshape(-1, 3),
            )

            def phase(lo, hi):
                mask = (ends >= lo) & (ends < hi)
                gaps = np.diff(ends) / 1e6
                gap_mask = mask[1:] & mask[:-1]
                obs_mask = (observed_array[:, 0] >= lo) & (observed_array[:, 0] < hi)
                pub_mask = (published_array[:, 1] >= lo) & (published_array[:, 1] < hi)
                return {
                    "apply_gap": stats(gaps[gap_mask]),
                    "append_service": stats((ends[mask] - starts[mask]) / 1e6),
                    "observed_progress_gap": stats(
                        np.diff(observed_array[:, 0])[obs_mask[1:] & obs_mask[:-1]]
                        / 1e6
                    ),
                    "publisher_gap": stats(
                        np.diff(published_array[:, 1])[pub_mask[1:] & pub_mask[:-1]]
                        / 1e6
                    ),
                    "applied_frames": int(mask.sum()),
                    "seconds": (hi - lo) / 1e9,
                }

            baseline = phase(baseline_start, loaded_start)
            loaded = phase(loaded_start, loaded_end)

            def gate(metric):
                limit = min(50, 2 * baseline[metric]["p999_ms"])
                actual = loaded[metric]["p999_ms"]
                return {"limit_ms": limit, "actual_ms": actual, "pass": actual <= limit}

            return {
                "environment": environment(args),
                "initial_bytes": initial.estimated_size(),
                "baseline": baseline,
                "loaded": loaded,
                "apply_gate": gate("apply_gap"),
                "observer_gate": gate("observed_progress_gap"),
                "operations": operations,
                "compaction_latency": stats([(b - a) / 1e6 for a, b, _ in compactions]),
                "final_chunks": live.n_chunks("all"),
                "published_frames": len(published),
                "exact_live_readback": True,
                "exact_last_flush_readback": True
                if last_written_hi is not None
                else None,
                "phase_boundaries_ns": [baseline_start, loaded_start, loaded_end],
            }
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)
        subscriber.shutdown()
        publisher.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shape", choices=["writer", "stall", "reader"])
    parser.add_argument("--rows", type=int, default=8_000_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--append", choices=["vstack", "upsert"], default="vstack")
    parser.add_argument("--compact-interval", type=float, default=0)
    parser.add_argument(
        "--auto-compaction-ms",
        type=int,
        default=250,
        help="Native worker interval; 0 disables. Manual --compact-interval overrides this.",
    )
    parser.add_argument("--warmup", type=float, default=2)
    parser.add_argument("--baseline", type=float, default=6)
    parser.add_argument("--loaded", type=float, default=12)
    parser.add_argument("--rate", type=float, default=1000)
    parser.add_argument("--batch-rows", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=500_000)
    parser.add_argument("--read-interval", type=float, default=0.5)
    parser.add_argument("--flush-interval", type=float, default=2)
    parser.add_argument("--compressible", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.rows >= 0 and args.trials > 0 and args.batch_rows > 0
    assert args.rate >= 0 and args.compact_interval >= 0
    assert args.auto_compaction_ms >= 0
    assert args.chunks > 0 and args.max_frames > 0
    assert args.warmup >= 0 and args.baseline > 0 and args.loaded > 0
    assert args.read_interval > 0 and args.flush_interval > 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = writer(args) if args.shape == "writer" else concurrent(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
