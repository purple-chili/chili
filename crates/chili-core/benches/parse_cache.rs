//! B9 — Parse-cache benchmark.
//!
//! Measures the raw cost of `EngineState::parse` on a repeated query string.
//! Pre-D: every call re-parses. Post-D: first call parses, N-1 calls hit the
//! LRU. The pre/post diff on this bench is the clearest single signal that
//! proposal D is working.
//!
//! Also includes a "unique" variant that uses `format!("1 + {}", i)` to force
//! cache misses — useful to verify D didn't regress the uncached path.

use std::time::Duration;

use chili_core::EngineState;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state
}

fn bench_parse(c: &mut Criterion) {
    let engine = make_engine();

    let mut group = c.benchmark_group("parse");
    group.sample_size(100);
    group.measurement_time(Duration::from_secs(10));

    // B9a — 100% cache-hittable: same query every iteration
    group.bench_function("parse_repeat_same_query", |b| {
        let query = "select from t where date=2024.01.03, symbol=`AAPL";
        b.iter(|| {
            let nodes = engine.parse("bench.pep", query).unwrap();
            black_box(nodes);
        });
    });

    // B9b — 0% cache-hittable: unique query every iteration (stress the cold path)
    group.bench_function("parse_unique_query_per_iter", |b| {
        let mut i: u64 = 0;
        b.iter(|| {
            i += 1;
            let query = format!("x: {}; x + 1", i);
            let nodes = engine.parse("bench.pep", &query).unwrap();
            black_box(nodes);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
