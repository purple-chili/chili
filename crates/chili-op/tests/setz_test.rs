//! `setz` — attach timezone metadata without converting clock values.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use polars::datatypes::{DataType, TimeUnit, TimeZone};
use polars::prelude::NamedFrom;
use polars::series::Series;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
}

fn eval_pep(state: &EngineState, src: &str) -> SpicyObj {
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(&mut s, &SpicyObj::String(src.to_string()), "t.pep")
        .unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

#[test]
fn setz_attaches_utc_to_timestamp_literal() {
    let state = new_engine();
    let out = eval_pep(&state, "setz[`UTC; 2026.08.25D00:00:00]");
    let SpicyObj::Series(s) = out else {
        panic!("expected series, got {}", out.get_type_name());
    };
    assert_eq!(
        s.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
    );
}

#[test]
fn setz_preserves_physical_values() {
    let state = new_engine();
    let before = eval_pep(&state, "2026.08.25D12:34:56.123456789");
    let SpicyObj::Timestamp(ns) = before else {
        panic!("expected timestamp literal, got {}", before.get_type_name());
    };

    let out = eval_pep(&state, "setz[`UTC; 2026.08.25D12:34:56.123456789]");
    let SpicyObj::Series(s) = out else {
        panic!("expected series, got {}", out.get_type_name());
    };
    assert_eq!(
        s.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
    );
    let got = s
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(got, ns);
}

#[test]
fn setz_allows_compare_naive_literal_to_utc_column() {
    let state = new_engine();
    let cutoff = eval_pep(&state, "setz[`UTC; 2026.08.25D00:00:00]");
    let SpicyObj::Series(cutoff_s) = cutoff else {
        panic!("expected series cutoff");
    };
    let cutoff_ns = cutoff_s
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .get(0)
        .unwrap();

    let time = Series::new(
        "time".into(),
        vec![cutoff_ns - 1, cutoff_ns + 1],
    )
    .cast(&DataType::Datetime(
        TimeUnit::Nanoseconds,
        Some(TimeZone::UTC),
    ))
    .unwrap();
    let df = polars::frame::DataFrame::new(2, vec![time.into()]).unwrap();
    state.set_var("trade", SpicyObj::DataFrame(df)).unwrap();

    let out = eval_pep(
        &state,
        "select from trade where time > setz[`UTC; 2026.08.25D00:00:00]",
    );
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe, got {}", out.get_type_name());
    };
    assert_eq!(df.height(), 1);
    let got = df
        .column("time")
        .unwrap()
        .as_materialized_series()
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .get(0)
        .unwrap();
    assert_eq!(got, cutoff_ns + 1);
}
