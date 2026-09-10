//! `nowz[`UTC]` carries timezone for query comparisons; `now` stays Timestamp.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    state.enable_pepper();
    state
}

fn eval(state: &EngineState, src: &str) -> SpicyObj {
    let mut stack = Stack::new(None, 0, 0, "");
    state
        .eval(&mut stack, &SpicyObj::String(src.to_owned()), "test.pep")
        .unwrap_or_else(|e| panic!("eval failed for `{src}`: {e}"))
}

fn eval_err(state: &EngineState, src: &str) -> String {
    let mut stack = Stack::new(None, 0, 0, "");
    state
        .eval(&mut stack, &SpicyObj::String(src.to_owned()), "test.pep")
        .expect_err(src)
        .to_string()
}

#[test]
fn nowz_utc_compares_against_tz_aware_column() {
    let engine = make_engine();
    eval(
        &engine,
        "q: ([] time: 2030.01.01D00:00:00 2030.01.02D00:00:00 2030.01.03D00:00:00)",
    );
    eval(&engine, "q: update time: setz[`UTC; time] from q");

    let obj = eval(&engine, "select from q where time < nowz[`UTC]");
    let df = match obj {
        SpicyObj::DataFrame(df) => df,
        SpicyObj::LazyFrame(lf) => lf.collect().unwrap(),
        other => panic!("expected DataFrame, got {}", other.get_type_name()),
    };
    assert_eq!(df.height(), 0);

    let obj = eval(&engine, "select from q where time > nowz[`UTC]");
    let df = match obj {
        SpicyObj::DataFrame(df) => df,
        SpicyObj::LazyFrame(lf) => lf.collect().unwrap(),
        other => panic!("expected DataFrame, got {}", other.get_type_name()),
    };
    assert_eq!(df.height(), 3);
}

#[test]
fn now_with_timezone_still_returns_timestamp() {
    let engine = make_engine();
    for src in ["now[`]", "now[`UTC]"] {
        let obj = eval(&engine, src);
        assert!(
            matches!(obj, SpicyObj::Timestamp(_)),
            "`{src}` expected Timestamp, got {}",
            obj.get_type_name()
        );
    }
}

#[test]
fn nowz_requires_timezone() {
    let engine = make_engine();
    let err = eval_err(&engine, "nowz[`]");
    assert!(
        err.contains("nowz requires a timezone"),
        "unexpected error: {err}"
    );
}
