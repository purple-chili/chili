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

fn nullable_frame(start: i64, rows: usize) -> DataFrame {
    DataFrame::new(
        rows,
        vec![
            Column::new("x".into(), (start..start + rows as i64).collect::<Vec<_>>()),
            Column::new(
                "label".into(),
                (0..rows)
                    .map(|i| if i % 2 == 0 { Some("a") } else { None })
                    .collect::<Vec<_>>(),
            ),
        ],
    )
    .unwrap()
}

#[test]
fn pretrim_matches_append_then_tail_for_repeated_updates() {
    let state = new_engine();
    for old_rows in [0, 1, 5] {
        for batch_rows in [0, 1, 4, 8] {
            for limit in [0, 1, 3, 5, 12] {
                let original = nullable_frame(0, old_rows);
                let incoming = nullable_frame(10, batch_rows);
                state
                    .set_var("t", SpicyObj::DataFrame(original.clone()))
                    .unwrap();
                state
                    .set_var("data", SpicyObj::DataFrame(incoming.clone()))
                    .unwrap();
                let mut expected = original;
                for _ in 0..3 {
                    let before = state.get_var("t").unwrap();
                    expected.extend(&incoming).unwrap();
                    expected = expected.tail(Some(limit));
                    let value = eval(&state, &format!("upsertn[t; data; {limit}]")).unwrap();
                    assert!(value.df().unwrap().equals_missing(&expected));
                    assert!(
                        state
                            .get_var("t")
                            .unwrap()
                            .df()
                            .unwrap()
                            .equals_missing(before.df().unwrap())
                    );
                    let count = eval(&state, &format!("upsertn[`t; data; {limit}]")).unwrap();
                    assert_eq!(count.to_i64().unwrap(), batch_rows as i64);
                    assert!(
                        state
                            .get_var("t")
                            .unwrap()
                            .df()
                            .unwrap()
                            .equals_missing(&expected)
                    );
                    assert!(
                        state
                            .get_var("data")
                            .unwrap()
                            .df()
                            .unwrap()
                            .equals_missing(&incoming)
                    );
                }
            }
        }
    }
}

#[test]
fn rejected_batches_cannot_discard_old_rows_even_when_limit_is_zero() {
    let state = new_engine();
    let original = nullable_frame(0, 5);
    let mut wrong_second_name = nullable_frame(10, 4);
    wrong_second_name.rename("label", "wrong".into()).unwrap();
    let wrong_width =
        DataFrame::new(4, vec![Column::new("x".into(), [10i64, 11, 12, 13])]).unwrap();
    state
        .set_var("t", SpicyObj::DataFrame(original.clone()))
        .unwrap();
    for incoming in [wrong_second_name, wrong_width] {
        state
            .set_var("data", SpicyObj::DataFrame(incoming.clone()))
            .unwrap();
        // Exercise discard-all, replacement, partial retention and no trimming.
        for limit in [0, 1, 4, 6, 20] {
            for target in ["t", "`t"] {
                assert!(eval(&state, &format!("upsertn[{target}; data; {limit}]")).is_err());
                assert!(
                    state
                        .get_var("t")
                        .unwrap()
                        .df()
                        .unwrap()
                        .equals_missing(&original)
                );
                assert!(
                    state
                        .get_var("data")
                        .unwrap()
                        .df()
                        .unwrap()
                        .equals_missing(&incoming)
                );
            }
        }
    }
    for limit in [0, 1, 6] {
        assert!(eval(&state, &format!("upsertn[`t; row4; {limit}]")).is_err());
        assert!(
            state
                .get_var("t")
                .unwrap()
                .df()
                .unwrap()
                .equals_missing(&original)
        );
    }
}

#[test]
fn value_upsertn_still_rejects_late_dtype_mismatches_before_replacing_rows() {
    let state = new_engine();
    let original = nullable_frame(0, 5);
    let incoming = DataFrame::new(
        2,
        vec![
            Column::new("x".into(), [10i64, 11]),
            Column::new("label".into(), [1i64, 2]),
        ],
    )
    .unwrap();
    state
        .set_var("t", SpicyObj::DataFrame(original.clone()))
        .unwrap();
    state
        .set_var("data", SpicyObj::DataFrame(incoming))
        .unwrap();
    for limit in [0, 1, 2, 6] {
        assert!(eval(&state, &format!("upsertn[t; data; {limit}]")).is_err());
        assert!(
            state
                .get_var("t")
                .unwrap()
                .df()
                .unwrap()
                .equals_missing(&original)
        );
    }
}

#[test]
fn replacement_and_zero_limit_keep_named_dtype_coercion() {
    let state = new_engine();
    let original = DataFrame::new(2, vec![Column::new("x".into(), [1i64, 2])]).unwrap();
    let incoming = DataFrame::new(4, vec![Column::new("x".into(), [3i32, 4, 5, 6])]).unwrap();
    state
        .set_var("data", SpicyObj::DataFrame(incoming.clone()))
        .unwrap();
    for limit in [0, 2, 4] {
        state
            .set_var("t", SpicyObj::DataFrame(original.clone()))
            .unwrap();
        assert_eq!(
            eval(&state, &format!("upsertn[`t; data; {limit}]"))
                .unwrap()
                .to_i64()
                .unwrap(),
            4
        );
        let result = state.get_var("t").unwrap();
        assert_eq!(result.df().unwrap().schema(), original.schema());
        assert_eq!(values(&result), (3..7).skip(4 - limit).collect::<Vec<_>>());
        assert!(
            state
                .get_var("data")
                .unwrap()
                .df()
                .unwrap()
                .equals_missing(&incoming)
        );
    }
}

#[test]
fn pretrim_preserves_timezone_relabeling_without_shifting_values() {
    let state = new_engine();
    let frame = |values: &[i64], tz| {
        DataFrame::new(
            values.len(),
            vec![
                Int64Chunked::from_slice("x".into(), values)
                    .into_datetime(TimeUnit::Nanoseconds, tz)
                    .into_series()
                    .into_column(),
            ],
        )
        .unwrap()
    };
    let original = frame(&[10, 20], Some(TimeZone::UTC));
    let incoming = frame(&[30, 40], None);
    state
        .set_var("data", SpicyObj::DataFrame(incoming.clone()))
        .unwrap();
    for limit in [0, 1, 3, 5] {
        for target in ["t", "`t"] {
            state
                .set_var("t", SpicyObj::DataFrame(original.clone()))
                .unwrap();
            let out = eval(&state, &format!("upsertn[{target}; data; {limit}]")).unwrap();
            let result = if target == "t" {
                out
            } else {
                state.get_var("t").unwrap()
            };
            let expected = frame(&[10, 20, 30, 40], Some(TimeZone::UTC)).tail(Some(limit));
            assert!(result.df().unwrap().equals_missing(&expected));
            if target == "t" {
                assert!(
                    state
                        .get_var("t")
                        .unwrap()
                        .df()
                        .unwrap()
                        .equals_missing(&original)
                );
            }
            assert!(
                state
                    .get_var("data")
                    .unwrap()
                    .df()
                    .unwrap()
                    .equals_missing(&incoming)
            );
        }
    }
}

#[test]
fn replacement_preserves_string_to_categorical_coercion() {
    let state = new_engine();
    let dtype = DataType::Categorical(Categories::global(), Categories::global().mapping());
    let original = DataFrame::new(
        2,
        vec![Column::new("x".into(), ["a", "b"]).cast(&dtype).unwrap()],
    )
    .unwrap();
    let incoming = DataFrame::new(2, vec![Column::new("x".into(), ["c", "d"])]).unwrap();
    state
        .set_var("data", SpicyObj::DataFrame(incoming.clone()))
        .unwrap();
    for limit in [0, 1, 2, 3] {
        state
            .set_var("t", SpicyObj::DataFrame(original.clone()))
            .unwrap();
        let count = eval(&state, &format!("upsertn[`t; data; {limit}]")).unwrap();
        assert_eq!(count.to_i64().unwrap(), 2);
        let expected = DataFrame::new(
            4,
            vec![
                Column::new("x".into(), ["a", "b", "c", "d"])
                    .cast(&dtype)
                    .unwrap(),
            ],
        )
        .unwrap()
        .tail(Some(limit));
        assert!(
            state
                .get_var("t")
                .unwrap()
                .df()
                .unwrap()
                .equals_missing(&expected)
        );
        assert!(
            state
                .get_var("data")
                .unwrap()
                .df()
                .unwrap()
                .equals_missing(&incoming)
        );
    }
}
