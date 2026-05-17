"""Concurrent throughput bench for chili-py.

Measures GIL-release effectiveness across concurrent workloads.
Outputs JSON-line per shape × N.

Shapes:

- ``concurrent_eval``        — N threads call ``engine.eval(query)``. eval()
                               releases the GIL. Should scale near-linearly
                               with N.
- ``concurrent_load``        — N threads call ``engine.load_partitioned_df``.
                               Python wrapper routes through ``fn_call`` which
                               releases the GIL.
- ``concurrent_load_direct`` — N threads call ``engine.engine.load_par_df``
                               (the FFI binding directly). This path releases
                               the GIL as well.
- ``single_eval``            — 1 thread, same eval workload. Baseline reference
                               for the single-thread path.

Usage::

    python bench_concurrent.py                      # full sweep, JSON-line stdout
    python bench_concurrent.py --shape concurrent_load --workers 4 --duration 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from chili import ChiliEngine

# ----------------------------------------------------------------------------
# HDB fixture
# ----------------------------------------------------------------------------


def build_hdb(
    root: Path,
    table: str,
    n_partitions: int,
    n_symbols: int,
    rows_per_symbol: int,
) -> str:
    """Build a small partitioned HDB for the bench.

    Mirrors crates/chili-op/benches/common/mod.rs::build_hdb shape (categorical
    symbol, float price, int volume; partitions written newest-first to surface
    unsorted-fs behavior).
    """
    e = ChiliEngine()
    try:
        symbols = [f"SYM{i:04d}" for i in range(n_symbols)]
        for d_idx in reversed(range(n_partitions)):
            partition_date = date(2024, 1, 2) + timedelta(days=d_idx)
            sym_col: list[str] = []
            price_col: list[float] = []
            vol_col: list[int] = []
            for sym in symbols:
                for i in range(rows_per_symbol):
                    sym_col.append(sym)
                    price_col.append(i * 0.01)
                    vol_col.append(i * 100)
            df = pl.DataFrame(
                {
                    "symbol": pl.Series(sym_col, dtype=pl.Categorical),
                    "price": price_col,
                    "volume": vol_col,
                }
            )
            e.write_partitioned_df(df, str(root), table, partition_date)
    finally:
        e.shutdown()
    return str(root)


# ----------------------------------------------------------------------------
# Worker shapes
# ----------------------------------------------------------------------------


def _worker_eval(
    engine: ChiliEngine,
    queries: list[str],
    deadline: float,
) -> tuple[int, list[float]]:
    n = 0
    latencies_ms: list[float] = []
    q_idx = 0
    while time.monotonic() < deadline:
        q = queries[q_idx % len(queries)]
        q_idx += 1
        t0 = time.monotonic()
        engine.eval(q)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        n += 1
    return n, latencies_ms


def _worker_load_via_fn_call(
    engine: ChiliEngine,
    hdb_path: str,
    deadline: float,
) -> tuple[int, list[float]]:
    n = 0
    latencies_ms: list[float] = []
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        engine.load_partitioned_df(hdb_path)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        n += 1
    return n, latencies_ms


def _worker_load_direct_ffi(
    engine: ChiliEngine,
    hdb_path: str,
    deadline: float,
) -> tuple[int, list[float]]:
    n = 0
    latencies_ms: list[float] = []
    inner = engine.engine  # the pyo3 binding object; bypasses fn_call wrapper
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        inner.load_par_df(hdb_path)
        latencies_ms.append((time.monotonic() - t0) * 1000.0)
        n += 1
    return n, latencies_ms


# ----------------------------------------------------------------------------
# Shape runners
# ----------------------------------------------------------------------------

EVAL_QUERIES = [
    "select from t where date=2024.01.03",
    "select sum volume by symbol from t where date=2024.01.05",
    "select price, volume from t where date=2024.01.07, symbol=`SYM0005",
    "select mean price by symbol from t where date>=2024.01.02, date<=2024.01.06",
]


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _run_shape(
    shape: str,
    n_workers: int,
    duration: float,
    engine: ChiliEngine,
    hdb_path: str,
) -> dict:
    """Spawn ``n_workers`` threads on ``shape`` for ``duration`` seconds.

    Returns the result dict ready to be JSON-line-emitted.
    """
    deadline = time.monotonic() + duration
    t0_wall = time.monotonic()

    def submit(executor: ThreadPoolExecutor):
        if shape == "concurrent_eval" or shape == "single_eval":
            return [
                executor.submit(_worker_eval, engine, EVAL_QUERIES, deadline)
                for _ in range(n_workers)
            ]
        if shape == "concurrent_load":
            return [
                executor.submit(_worker_load_via_fn_call, engine, hdb_path, deadline)
                for _ in range(n_workers)
            ]
        if shape == "concurrent_load_direct":
            return [
                executor.submit(_worker_load_direct_ffi, engine, hdb_path, deadline)
                for _ in range(n_workers)
            ]
        raise ValueError(f"unknown shape: {shape}")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = submit(executor)
        total_n = 0
        all_latencies: list[float] = []
        for f in as_completed(futures):
            n, lats = f.result()
            total_n += n
            all_latencies.extend(lats)

    wall = time.monotonic() - t0_wall
    all_latencies.sort()
    return {
        "shape": shape,
        "n_workers": n_workers,
        "total_seconds": round(wall, 4),
        "total_calls": total_n,
        "calls_per_sec": round(total_n / wall, 2) if wall > 0 else 0.0,
        "p50_ms": round(_percentile(all_latencies, 50.0), 4),
        "p99_ms": round(_percentile(all_latencies, 99.0), 4),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        default="all",
        choices=[
            "all",
            "concurrent_eval",
            "concurrent_load",
            "concurrent_load_direct",
            "single_eval",
        ],
        help="bench shape (default: all = full sweep across N ∈ {1,2,4,8})",
    )
    p.add_argument("--workers", type=int, default=None, help="N workers (single-shape only)")
    p.add_argument("--duration", type=float, default=5.0, help="seconds per (shape, N)")
    p.add_argument("--hdb-partitions", type=int, default=50)
    p.add_argument("--hdb-symbols", type=int, default=20)
    p.add_argument("--hdb-rows-per-symbol", type=int, default=100)
    p.add_argument(
        "--hdb-path",
        default=None,
        help="reuse a prebuilt HDB at this path; if omitted, a temp HDB is built and torn down",
    )
    args = p.parse_args()

    # Build (or reuse) HDB
    cleanup_dir = None
    if args.hdb_path:
        hdb_path = args.hdb_path
        if not Path(hdb_path).exists():
            print(f"hdb path does not exist: {hdb_path}", file=sys.stderr)
            return 2
    else:
        cleanup_dir = tempfile.mkdtemp(prefix="chili_bench_concurrent_")
        hdb_path = build_hdb(
            Path(cleanup_dir),
            "t",
            args.hdb_partitions,
            args.hdb_symbols,
            args.hdb_rows_per_symbol,
        )
        print(
            f"# built HDB at {hdb_path} ({args.hdb_partitions}p × "
            f"{args.hdb_symbols}sym × {args.hdb_rows_per_symbol}rps)",
            file=sys.stderr,
        )

    # Set up engine and pre-load (so concurrent_eval has data to query)
    engine = ChiliEngine(pepper=True)
    engine.load_partitioned_df(hdb_path)

    try:
        if args.shape == "all":
            sweep_n = [1, 2, 4, 8]
            shapes = [
                "single_eval",
                "concurrent_eval",
                "concurrent_load",
                "concurrent_load_direct",
            ]
            for shape in shapes:
                if shape == "single_eval":
                    result = _run_shape(shape, 1, args.duration, engine, hdb_path)
                    print(json.dumps(result), flush=True)
                    continue
                for n in sweep_n:
                    result = _run_shape(shape, n, args.duration, engine, hdb_path)
                    print(json.dumps(result), flush=True)
        else:
            n = args.workers if args.workers is not None else 1
            result = _run_shape(args.shape, n, args.duration, engine, hdb_path)
            print(json.dumps(result), flush=True)
    finally:
        engine.shutdown()
        if cleanup_dir is not None:
            import shutil

            shutil.rmtree(cleanup_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
