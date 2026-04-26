//! Shared fixture helpers for chili-op benches.
//!
//! Each bench creates its own `TempHdb` to avoid cross-bench state leakage,
//! but the row-builder and partition-writer are reused here.
//!
//! Partitions are written in **non-ascending date order** to reliably surface
//! unsorted-filesystem behavior (the R1 regression condition). This means
//! benches also serve as a smoke test that the R1 fix remains in place.
//!
//! dead_code is allowed because each bench module only imports a subset of the
//! helpers, and rustc warns per-compilation-unit when the full module isn't
//! used — these are all genuinely used by at least one bench.

#![allow(dead_code)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use chili_core::{EngineState, SpicyObj};
use chili_op::{BUILT_IN_FN, write_partition_native};
use polars::prelude::*;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// RAII wrapper around a per-bench temp HDB root.
pub struct TempHdb {
    pub root: PathBuf,
}

impl TempHdb {
    pub fn new(prefix: &str) -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "chili_bench_{}_{}_{}",
            prefix,
            std::process::id(),
            id,
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    pub fn path(&self) -> &str {
        self.root.to_str().unwrap()
    }
}

impl Drop for TempHdb {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Build a DataFrame with N symbols × M rows-per-symbol worth of OHLCV-shaped data.
pub fn make_row(symbols: &[&str], rows_per_symbol: usize) -> DataFrame {
    let mut all_symbols = Vec::with_capacity(symbols.len() * rows_per_symbol);
    let mut all_values: Vec<f64> = Vec::with_capacity(symbols.len() * rows_per_symbol);
    let mut all_volumes: Vec<i64> = Vec::with_capacity(symbols.len() * rows_per_symbol);
    for sym in symbols {
        for i in 0..rows_per_symbol {
            all_symbols.push((*sym).to_string());
            all_values.push((i as f64) * 0.01);
            all_volumes.push((i as i64) * 100);
        }
    }

    DataFrame::new(
        symbols.len() * rows_per_symbol,
        vec![
            Column::new(PlSmallStr::from("symbol"), all_symbols)
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap(),
            Column::new(PlSmallStr::from("price"), all_values),
            Column::new(PlSmallStr::from("volume"), all_volumes),
        ],
    )
    .unwrap()
}

/// Build a wide DataFrame mirroring mdata's ohlcv schema: 10 columns total
/// (open/high/low/close/vwap floats, volume/transactions ints, symbol string,
/// plus the partition `date` column injected by `write_partition_py`). Used
/// to make projection-pushdown benchmarks meaningful — projecting from 10
/// columns to 1 should be visibly faster than reading all 10.
pub fn make_wide_row(symbols: &[&str], rows_per_symbol: usize) -> DataFrame {
    let n = symbols.len() * rows_per_symbol;
    let mut all_symbols: Vec<String> = Vec::with_capacity(n);
    let mut open: Vec<f64> = Vec::with_capacity(n);
    let mut high: Vec<f64> = Vec::with_capacity(n);
    let mut low: Vec<f64> = Vec::with_capacity(n);
    let mut close: Vec<f64> = Vec::with_capacity(n);
    let mut vwap: Vec<f64> = Vec::with_capacity(n);
    let mut volume: Vec<i64> = Vec::with_capacity(n);
    let mut transactions: Vec<i64> = Vec::with_capacity(n);
    let mut bid: Vec<f64> = Vec::with_capacity(n);
    let mut ask: Vec<f64> = Vec::with_capacity(n);
    for sym in symbols {
        for i in 0..rows_per_symbol {
            let f = (i as f64) * 0.01;
            all_symbols.push((*sym).to_string());
            open.push(f);
            high.push(f + 0.5);
            low.push(f - 0.5);
            close.push(f + 0.1);
            vwap.push(f + 0.05);
            volume.push((i as i64) * 100);
            transactions.push(i as i64);
            bid.push(f + 0.04);
            ask.push(f + 0.06);
        }
    }
    DataFrame::new(
        n,
        vec![
            Column::new(PlSmallStr::from("symbol"), all_symbols)
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap(),
            Column::new(PlSmallStr::from("open"), open),
            Column::new(PlSmallStr::from("high"), high),
            Column::new(PlSmallStr::from("low"), low),
            Column::new(PlSmallStr::from("close"), close),
            Column::new(PlSmallStr::from("vwap"), vwap),
            Column::new(PlSmallStr::from("volume"), volume),
            Column::new(PlSmallStr::from("transactions"), transactions),
            Column::new(PlSmallStr::from("bid"), bid),
            Column::new(PlSmallStr::from("ask"), ask),
        ],
    )
    .unwrap()
}

/// Build an HDB with `n_partitions` date partitions of the wide ohlcv schema,
/// each with `n_symbols` symbols × `rows_per_symbol` rows. Counterpart to
/// `build_hdb` but uses `make_wide_row`.
pub fn build_wide_hdb(
    tmp: &TempHdb,
    table: &str,
    n_partitions: usize,
    n_symbols: usize,
    rows_per_symbol: usize,
) {
    let dates = date_sequence(n_partitions);
    let syms: Vec<String> = (0..n_symbols).map(|i| format!("SYM{:04}", i)).collect();
    let sym_refs: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();
    for d in dates.iter().rev() {
        let df = make_wide_row(&sym_refs, rows_per_symbol);
        write_partition_native(
            tmp.path(),
            &SpicyObj::Date(*d),
            table,
            &df,
            &[],
            false,
            false,
        )
        .unwrap();
    }
}

/// Write one partition via chili-op's write_partition_native path.
pub fn write_one_partition(
    hdb: &str,
    table: &str,
    date_days: i32,
    symbols: &[&str],
    rows_per_symbol: usize,
) {
    let df = make_row(symbols, rows_per_symbol);
    write_partition_native(
        hdb,
        &SpicyObj::Date(date_days),
        table,
        &df,
        &[],
        false,
        false,
    )
    .unwrap();
}

/// Encoded dates: days since 1970-01-01 starting at 2024-01-02.
/// Each index `i` returns the date (2024-01-02 + i days), skipping weekends.
pub fn date_sequence(count: usize) -> Vec<i32> {
    // 2024-01-02 = 19724
    const START: i32 = 19724;
    (0..count as i32).map(|i| START + i).collect()
}

/// Build an HDB with `n_partitions` date partitions, each with `n_symbols` symbols
/// × `rows_per_symbol` rows. Writes partitions in **reverse order** to exercise
/// the unsorted-filesystem path.
pub fn build_hdb(
    tmp: &TempHdb,
    table: &str,
    n_partitions: usize,
    n_symbols: usize,
    rows_per_symbol: usize,
) {
    let dates = date_sequence(n_partitions);
    let syms: Vec<String> = (0..n_symbols).map(|i| format!("SYM{:04}", i)).collect();
    let sym_refs: Vec<&str> = syms.iter().map(|s| s.as_str()).collect();
    for d in dates.iter().rev() {
        write_one_partition(tmp.path(), table, *d, &sym_refs, rows_per_symbol);
    }
}

/// Build an HDB with multiple tables, each with `n_partitions_per_table`.
/// Tables are named `t0`, `t1`, `t2`, ...
pub fn build_multitable_hdb(
    tmp: &TempHdb,
    n_tables: usize,
    n_partitions_per_table: usize,
    n_symbols: usize,
    rows_per_symbol: usize,
) {
    for i in 0..n_tables {
        let table = format!("t{}", i);
        build_hdb(
            tmp,
            &table,
            n_partitions_per_table,
            n_symbols,
            rows_per_symbol,
        );
    }
}

/// Make a fresh, empty engine with chili-op built-ins registered and pepper enabled.
pub fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    state.enable_pepper();
    state
}
