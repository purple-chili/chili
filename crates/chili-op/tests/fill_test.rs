mod util;

use chili_core::{EngineState, SpicyError, SpicyObj, parse};
use chili_op::operator;
use polars::prelude::*;
use util::create_state;

fn eval(state: &EngineState, source: &str) -> SpicyObj {
    state
        .eval_ast(parse(source, 0, "fill.pep").unwrap(), "", source)
        .unwrap()
}

#[test]
fn scalar_fill_replaces_only_null_and_preserves_value_types() {
    for value in [
        SpicyObj::Null,
        SpicyObj::I64(0),
        SpicyObj::I64(5),
        SpicyObj::F64(1.5),
        SpicyObj::Boolean(false),
        SpicyObj::String("".into()),
        SpicyObj::Symbol("abc".into()),
        SpicyObj::Date(1),
    ] {
        assert_eq!(operator::fill(&[&value, &SpicyObj::Null]).unwrap(), value);
        assert_eq!(operator::fill(&[&SpicyObj::Null, &value]).unwrap(), value);
        if !value.is_null() {
            assert_eq!(
                operator::fill(&[&SpicyObj::I64(99), &value]).unwrap(),
                value
            );
        }
    }
    // NaN is a floating-point value, distinct from Chili's scalar null.
    let result = operator::fill(&[&SpicyObj::I64(5), &SpicyObj::F64(f64::NAN)]).unwrap();
    assert!(matches!(result, SpicyObj::F64(value) if value.is_nan()));
}

#[test]
fn scalar_fill_works_in_expressions_and_functions() {
    let state = create_state(false);
    for (source, expected) in [
        ("5^0n", SpicyObj::I64(5)),
        ("5^3", SpicyObj::I64(3)),
        ("0n^3", SpicyObj::I64(3)),
        ("0n^0n", SpicyObj::Null),
        ("f: {[x] 5^x}; f[0n]", SpicyObj::I64(5)),
    ] {
        assert_eq!(eval(&state, source), expected, "{source}");
    }
}

#[test]
fn existing_collection_fill_and_forward_fill_are_preserved() {
    let state = create_state(false);
    for (source, expected) in [
        ("0f ^ 1 2 0nf", "1 2 0f"),
        ("5 ^ (0n;3)", "(5;3)"),
        ("5 ^ {a: 0n; b: 3}", "{a: 5; b: 3}"),
        ("fill 1 0n 3", "1 1 3"),
    ] {
        assert_eq!(eval(&state, source), eval(&state, expected), "{source}");
    }
}

#[test]
fn query_fill_still_fills_column_nulls() {
    let state = create_state(false);
    let result = eval(&state, "t: ([]x: 1 0n 3); select x: 5^x from t");
    let expected = eval(&state, "([]x: 1 5 3)");
    assert_eq!(result, expected);
}

#[test]
fn unsupported_fill_reports_binary_operator_and_both_types() {
    let series = SpicyObj::Series(Series::new("".into(), &[1i64, 2]));
    for (left, right) in [
        (series.clone(), SpicyObj::I64(3)),
        (series, SpicyObj::Null),
        (SpicyObj::I64(5), SpicyObj::DataFrame(DataFrame::empty())),
    ] {
        let error = operator::fill(&[&left, &right]).unwrap_err();
        match error {
            SpicyError::UnsupportedBinaryOpErr(op, left_type, right_type) => {
                assert_eq!(op, "^");
                assert_eq!(left_type, left.get_type_name());
                assert_eq!(right_type, right.get_type_name());
            }
            other => panic!("expected binary fill error, got {other}"),
        }
    }
}
