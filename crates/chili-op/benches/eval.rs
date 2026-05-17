//! Eval-path hot path benchmarks.
//!
//! - `query_groupby_agg`: group-by + aggregation query.
//! - `query_select_star`: raw `select *` to measure eval overhead unrelated to polars compute.
//! - `query_select_one_col`, `query_select_three_cols`, `query_select_all_wide`:
//!   projection benchmarks against a 10-column wide-schema fixture. The diff
//!   between `select close` and `select *` measures whether projection pushdown
//!   is effective in chili's query path.

use std::time::Duration;

use chili_core::{EngineState, SpicyObj, Stack};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

mod common;
use common::{TempHdb, build_hdb, build_wide_hdb, make_engine};

fn eval(engine: &EngineState, query: &str) {
    let mut stack = Stack::new(None, 0, 0, "");
    let obj = engine
        .eval(&mut stack, &SpicyObj::String(query.to_owned()), "bench.pep")
        .unwrap();
    black_box(obj);
}

fn bench_eval(c: &mut Criterion) {
    let tmp = TempHdb::new("eval_100p");
    // 100 partitions × 50 symbols × 500 rows = 2.5M rows total — enough for group-by
    // to be meaningful but small enough that load is fast.
    build_hdb(&tmp, "t", 100, 50, 500);
    let engine = make_engine();
    engine.load_par_df(tmp.path()).unwrap();

    let mut group = c.benchmark_group("eval");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // B7 — group-by + aggregation across 10 days
    group.bench_function("query_groupby_agg", |b| {
        b.iter(|| {
            eval(
                &engine,
                "select mean price, sum volume by symbol from t where date>=2024.01.02, date<=2024.01.11",
            )
        });
    });

    // B8 — select * on a single partition (measures eval/IPC overhead with minimal compute)
    group.bench_function("query_select_star", |b| {
        b.iter(|| eval(&engine, "select from t where date=2024.01.03"));
    });

    group.finish();
}

/// Phase 9 — projection pushdown benches.
///
/// Wide fixture (10 OHLCV columns + injected `date`). Compares the wall time of
/// `select close from t where date=X` against `select from t where date=X`.
/// If polars projection pushdown is fully working, the 1-column query should
/// be visibly faster than the 11-column read. If they're similar, the
/// pushdown isn't reaching the parquet reader for chili's query shape.
fn bench_projection(c: &mut Criterion) {
    let tmp = TempHdb::new("eval_wide_100p");
    // 100 dates × 50 symbols × 200 rows = 1M rows × 11 columns = ~88 MB
    build_wide_hdb(&tmp, "wide", 100, 50, 200);
    let engine = make_engine();
    engine.load_par_df(tmp.path()).unwrap();

    let mut group = c.benchmark_group("projection");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Read all 11 columns (no projection)
    group.bench_function("select_all_wide", |b| {
        b.iter(|| eval(&engine, "select from wide where date=2024.01.03"));
    });

    // Project to a single column — if pushdown works, this should be ~10× faster
    group.bench_function("select_one_col", |b| {
        b.iter(|| eval(&engine, "select close from wide where date=2024.01.03"));
    });

    // Project to 3 columns — should be ~3× faster than select_all_wide
    group.bench_function("select_three_cols", |b| {
        b.iter(|| {
            eval(
                &engine,
                "select close, volume, vwap from wide where date=2024.01.03",
            )
        });
    });

    // Project + filter on a non-projected column — must include `symbol` for the filter
    group.bench_function("select_one_col_with_sym_filter", |b| {
        b.iter(|| {
            eval(
                &engine,
                "select close from wide where date=2024.01.03, symbol=`SYM0001",
            )
        });
    });

    group.finish();
}

criterion_group!(benches, bench_eval, bench_projection);
criterion_main!(benches);
