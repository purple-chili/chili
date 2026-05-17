//! Categorical filter eval benchmark.
//!
//! Measures the cost of filtering a partitioned table by a categorical
//! (symbol) column. Two shapes are drawn from the same fixture (single
//! partition, 100 symbols × `rows_per_symbol`):
//!
//! - **`categorical_filter_repeated`** — same query string each iteration
//!   (`select from t where symbol=`SYM0001`). Measures the steady-state cost
//!   when both the parse cache and any categorical mapping are warm.
//! - **`categorical_filter_distinct`** — each iteration picks a different
//!   symbol, cycling through all 100 symbols. Stresses the non-cached path:
//!   parse cache still hits the same query *shape* but the literal symbol
//!   varies, so any per-call categorical mapping rebuild surfaces here.
//!
//! If `distinct` is materially slower than `repeated`, a categorical mapping
//! cache would have a measurable impact. If similar, the rebuild cost is
//! already negligible.

use std::cell::Cell;
use std::time::Duration;

use chili_core::{EngineState, SpicyObj, Stack};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

mod common;
use common::{TempHdb, build_hdb, make_engine};

fn eval(engine: &EngineState, query: &str) {
    let mut stack = Stack::new(None, 0, 0, "");
    let obj = engine
        .eval(&mut stack, &SpicyObj::String(query.to_owned()), "bench.pep")
        .unwrap();
    black_box(obj);
}

fn bench_categorical_filter(c: &mut Criterion) {
    // Single partition, 100 symbols × 100 rows = 10K rows. Small enough that
    // each filter is fast (microseconds), so the 5K-iter criterion sweep is
    // dominated by per-iteration overhead — exactly what we want to measure.
    let tmp = TempHdb::new("categorical_eval_1p");
    build_hdb(&tmp, "t", 1, 100, 100);
    let engine = make_engine();
    engine.load_par_df(tmp.path()).unwrap();

    let mut group = c.benchmark_group("categorical_filter");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    // Repeated — same literal symbol every call. Parse cache hits + any
    // categorical mapping is fully warm by the time criterion's measurement
    // window starts.
    group.bench_function("repeated", |b| {
        let q = "select from t where date=2024.01.02, symbol=`SYM0001";
        b.iter(|| eval(&engine, q));
    });

    // Distinct — cycle through all 100 symbols. Parse cache still hits the
    // same compiled-query shape, but the symbol literal varies, so any
    // per-call mapping rebuild on the categorical column shows up as wall-
    // time delta versus `repeated`.
    group.bench_function("distinct", |b| {
        let queries: Vec<String> = (0..100)
            .map(|i| format!("select from t where date=2024.01.02, symbol=`SYM{:04}", i))
            .collect();
        let counter = Cell::new(0usize);
        b.iter(|| {
            let i = counter.get();
            counter.set(i + 1);
            eval(&engine, &queries[i % queries.len()]);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_categorical_filter);
criterion_main!(benches);
