use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::{BUILT_IN_FN, LOG_FN};
use polars::prelude::*;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    for n in [4, 6] {
        state
            .set_var(
                &format!("row{n}"),
                SpicyObj::MixedList(vec![SpicyObj::I64(n)]),
            )
            .unwrap();
    }
    state
}

fn eval(state: &EngineState, src: &str) -> SpicyResult<SpicyObj> {
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(src.to_owned()),
        "upsertn.pep",
    )
}

fn values(obj: &SpicyObj) -> Vec<i64> {
    obj.df()
        .unwrap()
        .column("x")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

#[test]
fn named_table_retains_tail_and_returns_incoming_count() {
    let state = new_engine();
    let count = eval(&state, "upsertn[`t; ([]x: 1 2 3); 2]").unwrap();
    assert_eq!(count.to_i64().unwrap(), 3);
    assert_eq!(values(&state.get_var("t").unwrap()), vec![2, 3]);

    let count = eval(&state, "upsertn[`t; ([]x: 4 5); 3]").unwrap();
    assert_eq!(count.to_i64().unwrap(), 2);
    assert_eq!(values(&state.get_var("t").unwrap()), vec![3, 4, 5]);

    let count = eval(&state, "upsertn[`t; row6; 2]").unwrap();
    assert_eq!(count.to_i64().unwrap(), 1);
    assert_eq!(values(&state.get_var("t").unwrap()), vec![5, 6]);

    eval(&state, "upsertn[`t; ([]x: 7 8 9); 1]").unwrap();
    assert_eq!(values(&state.get_var("t").unwrap()), vec![9]);
}

#[test]
fn dataframe_values_append_without_mutating_the_input() {
    let state = new_engine();
    eval(&state, "t: ([]x: 1 2 3)").unwrap();
    let out = eval(&state, "upsertn[t; ([]x: 4 5); 3]").unwrap();
    assert_eq!(values(&out), vec![3, 4, 5]);
    let out = eval(&state, "upsertn[t; row4; 2]").unwrap();
    assert_eq!(values(&out), vec![3, 4]);
    let out = eval(&state, "upsertn[t; ([]x: 4 5); 10]").unwrap();
    assert_eq!(values(&out), vec![1, 2, 3, 4, 5]);
    assert_eq!(values(&state.get_var("t").unwrap()), vec![1, 2, 3]);

    // Regression: upsert used to extend a discarded clone for dataframe data.
    let out = eval(&state, "upsert[t; ([]x: 4 5)]").unwrap();
    assert_eq!(values(&out), vec![1, 2, 3, 4, 5]);
}

#[test]
fn zero_and_empty_batches_preserve_schema() {
    let state = new_engine();
    eval(&state, "t: ([]x: 1 2 3)").unwrap();
    let original = state.get_var("t").unwrap();
    state
        .set_var("empty", SpicyObj::DataFrame(original.df().unwrap().clear()))
        .unwrap();
    let count = eval(&state, "upsertn[`t; empty; 2]").unwrap();
    assert_eq!(count.to_i64().unwrap(), 0);
    assert_eq!(values(&state.get_var("t").unwrap()), vec![2, 3]);

    for src in [
        "upsertn[t; row4; 0]",
        "upsertn[`new; t; 0]; new",
        "upsertn[`t; row4; 0]; t",
    ] {
        let out = eval(&state, src).unwrap();
        assert_eq!(out.df().unwrap().height(), 0);
        assert_eq!(out.df().unwrap().schema(), original.df().unwrap().schema());
    }
}

#[test]
fn invalid_arguments_leave_table_unchanged() {
    let state = new_engine();
    eval(&state, "t: ([]x: 1 2 3)").unwrap();
    for src in [
        "upsertn[`t; row4; -1]",
        "upsertn[`t; row4; 1.5]",
        "upsertn[`t; row4; `bad]",
        "upsertn[`t; 4; 2]",
        "upsertn[42; row4; 2]",
        "upsertn[`t; ([]y: 4 5); 1]",
        "upsertn[`missing; row4; 2]",
    ] {
        let error = eval(&state, src).expect_err(src).to_string();
        assert!(!error.contains("syntax error"), "{src}: {error}");
        assert_eq!(values(&state.get_var("t").unwrap()), vec![1, 2, 3]);
    }
    assert!(state.get_var("missing").is_err());
    eval(&state, "scalar: 1").unwrap();
    assert!(eval(&state, "upsertn[`scalar; ([]x: 4 5); 1]").is_err());
    assert_eq!(state.get_var("scalar").unwrap().to_i64().unwrap(), 1);
}

#[test]
fn named_table_coerces_incoming_dtypes() {
    let state = new_engine();
    let target = DataFrame::new(2, vec![Column::new("x".into(), &[1i64, 2])]).unwrap();
    let incoming = DataFrame::new(2, vec![Column::new("x".into(), &[3i32, 4])]).unwrap();
    state.set_var("t", SpicyObj::DataFrame(target)).unwrap();
    state
        .set_var("data", SpicyObj::DataFrame(incoming))
        .unwrap();
    eval(&state, "upsertn[`t; data; 3]").unwrap();
    assert_eq!(values(&state.get_var("t").unwrap()), vec![2, 3, 4]);
}
