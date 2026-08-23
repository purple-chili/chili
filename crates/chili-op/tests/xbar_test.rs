//! `xbar` — numeric and temporal bar bucketing, including tz-aware Datetime.

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
fn xbar_numeric() {
    let state = new_engine();
    let out = eval_pep(&state, "10 xbar 23 35 41");
    let SpicyObj::Series(s) = out else {
        panic!("expected series, got {}", out.get_type_name());
    };
    let got: Vec<i64> = s.i64().unwrap().into_no_null_iter().collect();
    assert_eq!(got, vec![20, 30, 40]);
}

#[test]
fn xbar_duration_on_datetime_ns_utc() {
    let state = new_engine();
    let time = Series::new(
        "time".into(),
        vec![1_000_000_000i64, 1_500_000_000, 2_500_000_000],
    )
    .cast(&DataType::Datetime(
        TimeUnit::Nanoseconds,
        Some(TimeZone::UTC),
    ))
    .unwrap();
    state.set_var("time", SpicyObj::Series(time)).unwrap();

    let out = eval_pep(&state, "0D00:00:01 xbar time");
    let SpicyObj::Series(s) = out else {
        panic!("expected series, got {}", out.get_type_name());
    };
    assert_eq!(
        s.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
    );
    let got: Vec<i64> = s
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(got, vec![1_000_000_000, 1_000_000_000, 2_000_000_000]);
}

#[test]
fn xbar_duration_on_datetime_ns_utc_in_select() {
    let state = new_engine();
    let time = Series::new(
        "time".into(),
        vec![1_000_000_000i64, 1_500_000_000, 2_500_000_000],
    )
    .cast(&DataType::Datetime(
        TimeUnit::Nanoseconds,
        Some(TimeZone::UTC),
    ))
    .unwrap();
    let sym = Series::new("sym".into(), vec!["a", "b", "c"]);
    let df = polars::frame::DataFrame::new(3, vec![time.into(), sym.into()]).unwrap();
    state.set_var("t", SpicyObj::DataFrame(df)).unwrap();

    let out = eval_pep(&state, "select time: 0D00:00:01 xbar time from t");
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe, got {}", out.get_type_name());
    };
    assert_eq!(df.height(), 3);
    let time = df.column("time").unwrap().as_materialized_series();
    assert_eq!(
        time.dtype(),
        &DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
    );
    let got: Vec<i64> = time
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect();
    assert_eq!(got, vec![1_000_000_000, 1_000_000_000, 2_000_000_000]);
}
