//! In query call position, a builtin fn wins over a same-named column;
//! `'name'` / operand position still select the column. Non-builtins do not.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use polars::prelude::*;

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

fn as_df(obj: SpicyObj) -> DataFrame {
    match obj {
        SpicyObj::DataFrame(df) => df,
        SpicyObj::LazyFrame(lf) => lf.collect().unwrap(),
        other => panic!("expected DataFrame, got {}", other.get_type_name()),
    }
}

#[test]
fn builtin_count_wins_in_call_position_over_column() {
    let engine = make_engine();
    eval(&engine, "trap: ([] count: 1 2 3; sym: `a`b`c)");

    let df = as_df(eval(&engine, "select n: count sym from trap"));
    assert_eq!(df.height(), 1);
    assert_eq!(df.column("n").unwrap().u32().unwrap().get(0), Some(3));

    let df = as_df(eval(&engine, "select n: count[sym] from trap"));
    assert_eq!(df.height(), 1);
    assert_eq!(df.column("n").unwrap().u32().unwrap().get(0), Some(3));
}

#[test]
fn column_still_wins_in_operand_position_and_quotes() {
    let engine = make_engine();
    eval(&engine, "trap: ([] count: 1 2 3; sym: `a`b`c)");

    let df = as_df(eval(&engine, "select count from trap"));
    assert_eq!(df.column("count").unwrap().i64().unwrap().get(0), Some(1));

    let df = as_df(eval(&engine, "select n: sum count from trap"));
    assert_eq!(df.column("n").unwrap().i64().unwrap().get(0), Some(6));

    let df = as_df(eval(&engine, "select n: sum 'count' from trap"));
    assert_eq!(df.column("n").unwrap().i64().unwrap().get(0), Some(6));

    let df = as_df(eval(&engine, "select 'count' from trap"));
    assert_eq!(df.column("count").unwrap().i64().unwrap().get(0), Some(1));
}

#[test]
fn non_builtin_does_not_override_column_in_call_position() {
    let engine = make_engine();
    // Shadow builtin `count` with a user pepper fn — must not steal the column.
    eval(&engine, "count: {[x] 99}");
    eval(&engine, "trap: ([] count: 1 2 3; sym: `a`b`c)");

    let err = eval_err(&engine, "select n: count sym from trap");
    assert!(
        err.contains("fn call") || err.contains("expr") || err.contains("col"),
        "user fn must not override column; got: {err}"
    );

    // Operand / quoted forms still see the column.
    let df = as_df(eval(&engine, "select n: sum 'count' from trap"));
    assert_eq!(df.column("n").unwrap().i64().unwrap().get(0), Some(6));
}
