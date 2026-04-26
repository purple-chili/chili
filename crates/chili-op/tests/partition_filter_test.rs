//! Integration tests for HDB partition loading + selection (R1 regression).
//!
//! These tests write real parquet files to a temp HDB, then exercise the
//! engine's load / query path through the public API to prove:
//!   1. `load_par_df` produces a sorted `pars` vector regardless of
//!      filesystem iteration order (the R1 root cause).
//!   2. `where date=X` returns the X partition and only the X partition.
//!   3. `where date>=X, date<=Y` returns rows for every date in [X, Y].
//!   4. `where date>=X` alone / `where date<=Y` alone pick the right side.
//!   5. Queries that mix a partition predicate with a non-partition
//!      predicate (e.g. `where symbol='AAPL', date=X`) are handled without
//!      silently dropping the non-partition filter.
//!
//! The unsorted directory entries are created in *reverse* order on
//! purpose — on APFS + Linux ext4 the on-disk iteration is effectively
//! non-deterministic, and writing in reverse gives us a reliable way to
//! surface the old bug if the sort is ever removed.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::{BUILT_IN_FN, write_partition_native};
use polars::prelude::*;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    state.enable_pepper();
    state
}

fn make_row(symbol: &str, value: f64) -> DataFrame {
    df![
        "symbol" => [symbol],
        "value"  => [value],
    ]
    .unwrap()
}

fn write_partition(engine_hdb: &str, date_days: i32, symbol: &str, value: f64) -> SpicyResult<()> {
    let df = make_row(symbol, value);
    write_partition_native(
        engine_hdb,
        &SpicyObj::Date(date_days),
        "ohlcv",
        &df,
        &[],
        false,
        false,
    )
    .map(|_| ())
}

fn eval_query(state: &EngineState, query: &str) -> DataFrame {
    let mut stack = Stack::new(None, 0, 0, "");
    let obj = state
        .eval(&mut stack, &SpicyObj::String(query.to_owned()), "test.chi")
        .expect("eval failed");
    match obj {
        SpicyObj::DataFrame(df) => df,
        SpicyObj::LazyFrame(lf) => lf.collect().expect("lf collect failed"),
        other => panic!("expected DataFrame, got {}", other.get_type_name()),
    }
}

struct TempHdb {
    root: PathBuf,
}

impl TempHdb {
    fn new() -> Self {
        // Unique per-test path: pid + monotonic counter. Avoids races between
        // parallel tests that would otherwise collide on shared filesystem
        // nanosecond resolution.
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("chili_r1_test_{}_{}", std::process::id(), id,));
        // Clean up any leftover from a previous aborted run, then create.
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }
    fn path(&self) -> &str {
        self.root.to_str().unwrap()
    }
}

impl Drop for TempHdb {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

// Encoded days since 1970-01-01 for dates we use in the tests.
const D_2024_01_02: i32 = 19724;
const D_2024_01_03: i32 = 19725;
const D_2024_01_04: i32 = 19726;
const D_2024_01_05: i32 = 19727;
const D_2024_01_08: i32 = 19730;

fn setup_hdb() -> TempHdb {
    let hdb = TempHdb::new();
    // Intentionally write in non-ascending order. On macOS APFS the
    // `fs::read_dir` iteration order matches creation order exactly, so
    // this guarantees `par_vec` would be unsorted without the load-time
    // sort — reliably surfacing the R1 bug if the fix ever regresses.
    for &d in &[
        D_2024_01_05,
        D_2024_01_02,
        D_2024_01_08,
        D_2024_01_03,
        D_2024_01_04,
    ] {
        write_partition(hdb.path(), d, "AAPL", d as f64).unwrap();
        write_partition(hdb.path(), d, "MSFT", (d as f64) * 2.0).unwrap();
    }
    hdb
}

#[test]
fn load_par_df_sorts_par_vec_regardless_of_fs_order() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();
    let par_df = engine.get_par_df("ohlcv").unwrap();
    assert_eq!(
        par_df.pars,
        vec![
            D_2024_01_02,
            D_2024_01_03,
            D_2024_01_04,
            D_2024_01_05,
            D_2024_01_08,
        ],
        "par_df.pars must be sorted ascending after load (R1 regression guard)"
    );
}

#[test]
fn equality_returns_matching_partition() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    // date=2024.01.03 used to return 0 rows (R1 sub-bug 1).
    let df = eval_query(&engine, "select from ohlcv where date=2024.01.03");
    assert_eq!(df.height(), 2, "date=2024.01.03 must return 2 rows");
}

#[test]
fn equality_returns_matching_partition_for_every_date() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    for (date_str, day) in &[
        ("2024.01.02", D_2024_01_02),
        ("2024.01.03", D_2024_01_03),
        ("2024.01.04", D_2024_01_04),
        ("2024.01.05", D_2024_01_05),
        ("2024.01.08", D_2024_01_08),
    ] {
        let df = eval_query(
            &engine,
            &format!("select from ohlcv where date={}", date_str),
        );
        assert_eq!(
            df.height(),
            2,
            "date={} should return 2 rows, got {}",
            date_str,
            df.height()
        );
        // Confirm the synthesized `date` column matches.
        let d = df
            .column("date")
            .unwrap()
            .date()
            .unwrap()
            .phys
            .get(0)
            .unwrap();
        assert_eq!(d, *day, "synthesized date column mismatch");
    }
}

#[test]
fn equality_missing_partition_returns_empty() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    let df = eval_query(&engine, "select from ohlcv where date=2024.01.06");
    assert_eq!(df.height(), 0, "missing partition should return 0 rows");
}

#[test]
fn narrow_range_returns_every_date_in_bounds() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    // Used to return 6 rows across [01-02, 01-04] (missing 01-03) — R1 sub-bug 2.
    let df = eval_query(
        &engine,
        "select from ohlcv where date>=2024.01.02, date<=2024.01.04",
    );
    assert_eq!(df.height(), 6, "3 dates × 2 symbols = 6 rows");
}

#[test]
fn lower_bound_only() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    let df = eval_query(&engine, "select from ohlcv where date>=2024.01.05");
    assert_eq!(
        df.height(),
        4,
        "2 dates (01-05, 01-08) × 2 symbols = 4 rows"
    );
}

#[test]
fn upper_bound_only() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    let df = eval_query(&engine, "select from ohlcv where date<=2024.01.04");
    assert_eq!(
        df.height(),
        6,
        "3 dates (01-02, 01-03, 01-04) × 2 symbols = 6 rows"
    );
}

#[test]
fn strict_range_bounds() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    // date > 2024.01.03 AND date < 2024.01.05 → only 01-04 matches.
    let df = eval_query(
        &engine,
        "select from ohlcv where date>2024.01.03, date<2024.01.05",
    );
    assert_eq!(df.height(), 2);
}

#[test]
fn partition_clause_then_non_partition_clause() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    // Partition clause first, row filter second. Classic case.
    let df = eval_query(
        &engine,
        "select from ohlcv where date=2024.01.03, symbol=`AAPL",
    );
    assert_eq!(df.height(), 1, "1 date × 1 symbol = 1 row");
}

#[test]
fn non_partition_clause_before_partition_clause_does_not_drop_filter() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    // Non-partition clause FIRST — used to cause the partition extractor
    // to skip where_exp[0] unconditionally, silently dropping the symbol
    // filter (Bug 2). Now it should work correctly.
    let df = eval_query(
        &engine,
        "select from ohlcv where symbol=`AAPL, date=2024.01.03",
    );
    assert_eq!(
        df.height(),
        1,
        "should scan 1 partition and keep 1 symbol row"
    );
}

#[test]
fn wide_range_returns_all_partitions() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    let df = eval_query(
        &engine,
        "select from ohlcv where date>=1900.01.01, date<=2099.12.31",
    );
    assert_eq!(df.height(), 10, "5 dates × 2 symbols = 10 rows");
}

#[test]
fn within_operator_partition_selection() {
    let hdb = setup_hdb();
    let engine = make_engine();
    engine.load_par_df(hdb.path()).unwrap();

    let df = eval_query(
        &engine,
        "select from ohlcv where date within 2024.01.03 2024.01.05",
    );
    assert_eq!(df.height(), 6, "3 dates × 2 symbols = 6 rows");
}
