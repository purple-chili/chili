//! B11 — Partition-write throughput.
//!
//! Measures `write_partition_py` loop, which is the pattern used by chili-py's
//! `Engine::wpar`. This bench surfaces the `fs::canonicalize` syscall cost
//! (proposal O) and any file-creation overhead.

use std::time::Duration;

use chili_core::SpicyObj;
use chili_op::write_partition_native;
use criterion::{Criterion, criterion_group, criterion_main};
use polars::prelude::*;

mod common;
use common::{TempHdb, make_row};

fn bench_wpar(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    let symbols = ["AAPL", "MSFT", "SPY", "GOOG", "TSLA"];
    let df: DataFrame = make_row(&symbols, 200); // 1000 rows per partition

    group.bench_function("wpar_1k_rows_fresh_hdb", |b| {
        b.iter_with_setup(
            || TempHdb::new("wpar_bench"),
            |tmp| {
                // Write 5 partitions per iteration — enough to amortize hdb creation
                // but small enough not to dominate the bench.
                for i in 0..5 {
                    write_partition_native(
                        tmp.path(),
                        &SpicyObj::Date(19724 + i),
                        "t",
                        &df,
                        &[],
                        false,
                        false,
                    )
                    .unwrap();
                }
            },
        );
    });

    group.finish();
}

criterion_group!(benches, bench_wpar);
criterion_main!(benches);
