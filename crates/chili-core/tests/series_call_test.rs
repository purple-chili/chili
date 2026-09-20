use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;
use polars::prelude::*;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
        .set_var(
            "v",
            SpicyObj::Series(Series::new("v".into(), &[10i64, 20, 30])),
        )
        .unwrap();
    state
}

fn eval(state: &EngineState, src: &str) -> SpicyResult<SpicyObj> {
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(src.into()),
        "series_call.pep",
    )
}

#[test]
fn scalar_calls_match_at_including_negative_and_missing_indices() {
    let state = new_engine();
    for (index, expected) in [
        ("1", SpicyObj::I64(20)),
        ("-1", SpicyObj::I64(30)),
        ("-3", SpicyObj::I64(10)),
        ("3", SpicyObj::Null),
        ("-4", SpicyObj::Null),
    ] {
        for src in [
            format!("v[{index}]"),
            format!("v {index}"),
            format!("v @ {index}"),
        ] {
            assert_eq!(eval(&state, &src).unwrap(), expected, "{src}");
        }
    }
}

#[test]
fn vector_calls_match_at_and_out_of_range_indices_become_null() {
    let state = new_engine();
    for src in [
        "v[2 0 -1 3 -4 4 0n]",
        "v 2 0 -1 3 -4 4 0n",
        "v @ 2 0 -1 3 -4 4 0n",
    ] {
        let result = eval(&state, src).unwrap();
        let series = result.series().unwrap();
        assert_eq!(series.name().as_str(), "v");
        assert_eq!(
            series.i64().unwrap().iter().collect::<Vec<_>>(),
            vec![Some(30), Some(10), Some(30), None, None, None, None],
            "{src}"
        );
    }
}

#[test]
fn empty_series_and_empty_integer_indices_are_supported() {
    let state = new_engine();
    state
        .set_var(
            "empty",
            SpicyObj::Series(Series::new_empty("empty".into(), &DataType::Int64)),
        )
        .unwrap();
    assert!(eval(&state, "empty[0]").unwrap().is_null());
    let result = eval(&state, "empty[0 -1]").unwrap();
    assert_eq!(result.series().unwrap().null_count(), 2);
    let result = eval(&state, "v[empty]").unwrap();
    assert_eq!(result.series().unwrap().len(), 0);
}

#[test]
fn non_numeric_series_and_null_values_are_preserved() {
    let state = new_engine();
    state
        .set_var(
            "s",
            SpicyObj::Series(Series::new("s".into(), &[Some("a"), None, Some("c")])),
        )
        .unwrap();
    assert_eq!(eval(&state, "s[0]").unwrap(), SpicyObj::String("a".into()));
    assert!(eval(&state, "s[1]").unwrap().is_null());
    assert_eq!(
        eval(&state, "s[2 0]").unwrap(),
        eval(&state, "s @ 2 0").unwrap()
    );
}

#[test]
fn series_parameters_and_chili_call_syntax_work() {
    let state = new_engine();
    assert_eq!(
        eval(&state, "f: {[x] x[1]}; f[v]").unwrap(),
        SpicyObj::I64(20)
    );
    let nodes = chili_core::parse("v(1)", 0, "series_call.chi").unwrap();
    assert_eq!(
        state.eval_ast(nodes, "", "v(1)").unwrap(),
        SpicyObj::I64(20)
    );
}

#[test]
fn invalid_indices_and_argument_counts_return_errors() {
    let state = new_engine();
    for src in ["v[1.5]", "v[`a]", "v[0n]", "v[]", "v[;1]", "v[0;1]"] {
        assert!(eval(&state, src).is_err(), "{src}");
    }
    let error = eval(&state, "v[0;1]").unwrap_err().to_string();
    assert!(error.contains("Expect 1 argument(s), 2 given"), "{error}");
}
