//! Operator defects found in the 0.11.1 review: wrong results and panics
//! reachable from ordinary expressions.

use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
}

fn try_eval(state: &EngineState, src: &str) -> SpicyResult<SpicyObj> {
    state.eval(&mut Stack::new(None, 0, 0, ""), &SpicyObj::String(src.to_string()), "t.pep")
}

fn eval(state: &EngineState, src: &str) -> SpicyObj {
    try_eval(state, src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

fn i64s(obj: SpicyObj) -> Vec<Option<i64>> {
    let SpicyObj::Series(s) = obj else {
        panic!("expected series, got {}", obj.get_type_name());
    };
    s.i64().unwrap().iter().collect()
}

fn f64s(obj: SpicyObj) -> Vec<Option<f64>> {
    let SpicyObj::Series(s) = obj else {
        panic!("expected series, got {}", obj.get_type_name());
    };
    s.f64().unwrap().iter().collect()
}

#[test]
fn xbar_on_atoms_floors_like_the_series_form() {
    let st = new_engine();
    assert_eq!(eval(&st, "5 xbar 17"), SpicyObj::I64(15));
    assert_eq!(eval(&st, "5 xbar -3"), SpicyObj::I64(-5));
    assert_eq!(eval(&st, "1.0 xbar 1.7"), SpicyObj::F64(1.0));
    assert_eq!(i64s(eval(&st, "5 xbar 17 18")), vec![Some(15), Some(15)]);
    assert!(try_eval(&st, "0 xbar 5").is_err(), "zero bar is an error, not a panic");
}

#[test]
fn and_or_of_same_type_temporals() {
    let st = new_engine();
    assert_eq!(
        eval(&st, "2024.01.01 & 2024.01.02"),
        eval(&st, "2024.01.01"),
        "and is the minimum"
    );
    assert_eq!(
        eval(&st, "2024.01.01 | 2024.01.02"),
        eval(&st, "2024.01.02"),
        "or is the maximum"
    );
}

#[test]
fn appending_two_mixed_lists_keeps_both() {
    let st = new_engine();
    let SpicyObj::MixedList(l) = eval(&st, "(1;`a) ++ (2;`b)") else {
        panic!("expected mixed list");
    };
    assert_eq!(l.len(), 4);
    assert_eq!(l[2], SpicyObj::I64(2));
}

#[test]
fn integer_div_and_mod_floor_and_never_panic() {
    let st = new_engine();
    // atom and series forms agree
    assert_eq!(eval(&st, "-7 div 2"), SpicyObj::I64(-4));
    assert_eq!(i64s(eval(&st, "-7 -7 div 2")), vec![Some(-4), Some(-4)]);
    assert_eq!(eval(&st, "-7 mod 3"), SpicyObj::I64(2));
    assert_eq!(eval(&st, "7 div 2"), SpicyObj::I64(3));
    // zero divisor is null, not a process-level panic
    assert_eq!(eval(&st, "5 div 0"), SpicyObj::Null);
    assert_eq!(eval(&st, "5 mod 0"), SpicyObj::Null);
    assert_eq!(eval(&st, "1.5 mod 0n"), SpicyObj::Null);
}

#[test]
fn div_and_mod_across_dtypes() {
    let st = new_engine();
    assert_eq!(f64s(eval(&st, "1.5 2.5 div 2")), vec![Some(0.0), Some(1.0)]);
    assert_eq!(f64s(eval(&st, "1 2 3 div 2.0")), vec![Some(0.0), Some(1.0), Some(1.0)]);
    assert_eq!(f64s(eval(&st, "5.5 mod 1 2 3")), vec![Some(0.5), Some(1.5), Some(2.5)]);
}

#[test]
fn pow_with_negative_or_overflowing_exponent() {
    let st = new_engine();
    assert_eq!(eval(&st, "2 pow -1"), SpicyObj::F64(0.5));
    assert_eq!(eval(&st, "2 pow 10"), SpicyObj::I64(1024));
    assert_eq!(eval(&st, "2 pow 64"), SpicyObj::F64(2f64.powi(64)));
}

#[test]
fn rand_with_zero_or_negative_bound() {
    let st = new_engine();
    assert_eq!(i64s(eval(&st, "10 ? 0")).len(), 10, "0 means full range");
    assert_eq!(f64s(eval(&st, "10 ? 0.0")).len(), 10);
    assert!(try_eval(&st, "10 ? -1.0").is_err());
    let SpicyObj::MixedList(l) = eval(&st, "3 ? ()") else {
        panic!("expected mixed list");
    };
    assert!(l.is_empty());
}

#[test]
fn sum_and_product_of_a_dict_cover_every_value() {
    let st = new_engine();
    assert_eq!(eval(&st, "sum `a`b`c!1 2 3"), SpicyObj::I64(6));
    assert_eq!(eval(&st, "prod `a`b`c!2 3 4"), SpicyObj::I64(24));
}

#[test]
fn all_any_skip_null_children() {
    let st = new_engine();
    assert_eq!(eval(&st, "all (1b; 0n)"), SpicyObj::Boolean(true));
    assert_eq!(eval(&st, "any (0b; 0n)"), SpicyObj::Boolean(false));
}

#[test]
fn take_and_shift_edge_cases() {
    let st = new_engine();
    let SpicyObj::DataFrame(df) = eval(&st, "3 # 0 # ([] a: 1 2 3)") else {
        panic!("expected dataframe");
    };
    assert_eq!(df.height(), 0, "taking from an empty frame terminates");
    let SpicyObj::MixedList(l) = eval(&st, "-2 # ()") else {
        panic!("expected mixed list");
    };
    assert!(l.is_empty());
    let SpicyObj::MixedList(l) = eval(&st, "shift[-1; (1; \"a\"; `b)]") else {
        panic!("expected mixed list");
    };
    assert_eq!(l.len(), 3);
    assert_eq!(l[2], SpicyObj::Null);
    assert_eq!(l[0], SpicyObj::String("a".into()));
}

#[test]
fn function_operand_is_an_error_not_a_panic() {
    let st = new_engine();
    for src in ["{x} + 1.0", "{x} - 1.0", "{x} * 1.0", "{x} / 2"] {
        assert!(try_eval(&st, src).is_err(), "{src}");
    }
}

#[test]
fn single_key_asof_join_matches_an_exact_time() {
    let st = new_engine();
    eval(&st, "trades: ([] time: 1 2 3; px: 10 20 30)");
    eval(&st, "quotes: ([] time: 1 3; bid: 9 29)");
    let out = eval(&st, "aj[`time; trades; quotes]");
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe");
    };
    let bid: Vec<Option<i64>> = df.column("bid").unwrap().i64().unwrap().iter().collect();
    assert_eq!(bid, vec![Some(9), Some(9), Some(29)], "time 1 and 3 match exactly");
}

fn strs(obj: &SpicyObj, col: &str) -> Vec<String> {
    let SpicyObj::DataFrame(df) = obj else {
        panic!("expected dataframe, got {}", obj.get_type_name());
    };
    df.column(col)
        .unwrap()
        .cast(&polars::prelude::DataType::String)
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .map(|s| s.unwrap_or("").to_owned())
        .collect()
}

#[test]
fn string_functions_agree_inside_and_outside_a_query() {
    let st = new_engine();
    eval(&st, "t: ([] name: (\"  ab \"; \"a-b-c\"))");
    // trim
    assert_eq!(eval(&st, "trim \"  ab \""), SpicyObj::String("ab".into()));
    assert_eq!(strs(&eval(&st, "select trim name from t"), "name")[0], "ab");
    // replace: every occurrence
    assert_eq!(
        eval(&st, "replace[\"a-b-c\"; \"-\"; \"+\"]"),
        SpicyObj::String("a+b+c".into())
    );
    assert_eq!(
        strs(&eval(&st, "select replace[name; \"-\"; \"+\"] from t"), "name")[1],
        "a+b+c"
    );
    // pad: a positive length pads on the right
    assert_eq!(eval(&st, "pad[5; \"ab\"]"), SpicyObj::String("ab   ".into()));
    eval(&st, "u: ([] name: enlist \"ab\")");
    assert_eq!(strs(&eval(&st, "select pad[5; name] from u"), "name")[0], "ab   ");
}

#[test]
fn timezone_conversion_survives_daylight_saving_changes() {
    let st = new_engine();
    // 01:30 on 2024-11-03 happened twice in New York: the earlier one (EDT, UTC-4).
    assert_eq!(
        eval(&st, "utc[2024.11.03D01:30:00; \"America/New_York\"]"),
        eval(&st, "2024.11.03D05:30:00")
    );
    // 02:30 on 2024-03-10 never happened: shifted over the gap (EST offset, UTC-5).
    assert_eq!(
        eval(&st, "utc[2024.03.10D02:30:00; \"America/New_York\"]"),
        eval(&st, "2024.03.10D07:30:00")
    );
    // an ordinary time is unchanged by all this
    assert_eq!(
        eval(&st, "utc[2024.06.01D12:00:00; \"America/New_York\"]"),
        eval(&st, "2024.06.01D16:00:00")
    );
}
