//! Evaluator and query defects found in the 0.11.1 review.

use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn engine(pepper: bool) -> EngineState {
    let mut state = EngineState::initialize();
    if pepper {
        state.enable_pepper();
    }
    state.register_fn(&BUILT_IN_FN);
    state
}

fn try_eval(state: &EngineState, src: &str) -> SpicyResult<SpicyObj> {
    let path = if state.is_repl_use_chili_syntax() { "t.chi" } else { "t.pep" };
    state.eval(&mut Stack::new(None, 0, 0, ""), &SpicyObj::String(src.to_string()), path)
}

fn eval(state: &EngineState, src: &str) -> SpicyObj {
    try_eval(state, src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

fn col_i64(obj: &SpicyObj, name: &str) -> Vec<Option<i64>> {
    let SpicyObj::DataFrame(df) = obj else {
        panic!("expected dataframe, got {}", obj.get_type_name());
    };
    df.column(name).unwrap().i64().unwrap().iter().collect()
}

#[test]
fn chili_else_block_runs() {
    let st = engine(false);
    assert_eq!(eval(&st, "x: 0; if (1 > 2) { x: 1; } else { x: 2; x: x + 10; } x"), SpicyObj::I64(12));
    assert_eq!(eval(&st, "y: 0; if (2 > 1) { y: 1; } else { y: 2; } y"), SpicyObj::I64(1));
    // else-if chains still work, and a return inside an else block propagates.
    assert_eq!(
        eval(&st, "z: 0; if (1 > 2) { z: 1; } else if (1 > 3) { z: 2; } else { z: 3; } z"),
        SpicyObj::I64(3)
    );
    assert_eq!(
        eval(&st, "f: function(a) { if (a > 1) { return 1; } else { return 2; } 3 }; f(0)"),
        SpicyObj::I64(2)
    );
}

#[test]
fn indexing_a_table_at_its_height_is_null_not_a_panic() {
    let st = engine(true);
    eval(&st, "t: ([] a: 1 2 3)");
    let out = eval(&st, "t[3]");
    assert_eq!(col_i64(&out, "a"), vec![None]);
    let out = eval(&st, "t[0 3]");
    assert_eq!(col_i64(&out, "a"), vec![Some(1), None]);
    assert_eq!(col_i64(&eval(&st, "t[2]"), "a"), vec![Some(3)]);
}

#[test]
fn delete_removes_only_rows_matching_every_clause() {
    let st = engine(true);
    eval(&st, "t: ([] a: 1 2 3 4; b: 1 1 5 5)");
    // Only a=2 has a>1 and b<2.
    let out = eval(&st, "delete from t where a>1, b<2");
    assert_eq!(col_i64(&out, "a"), vec![Some(1), Some(3), Some(4)]);
    // A null predicate does not match: the row stays.
    eval(&st, "u: ([] q: 1 0n 2)");
    let out = eval(&st, "delete from u where q=2");
    assert_eq!(col_i64(&out, "q"), vec![Some(1), None]);
}

#[test]
fn update_where_keeps_rows_the_clause_does_not_select() {
    let st = engine(true);
    eval(&st, "u: ([] s: `a`b`a; q: 1 2 3)");
    let out = eval(&st, "update q*2 from u where s=`a");
    assert_eq!(col_i64(&out, "q"), vec![Some(2), Some(2), Some(6)]);
    // Aliased onto an existing column: same.
    let out = eval(&st, "update q: q+10 from u where s=`b");
    assert_eq!(col_i64(&out, "q"), vec![Some(1), Some(12), Some(3)]);
    // A new column is null where the clause does not select.
    let out = eval(&st, "update r: q*2 from u where s=`a");
    assert_eq!(col_i64(&out, "r"), vec![Some(2), None, Some(6)]);
}

#[test]
fn empty_call_arguments_are_errors_not_panics() {
    let st = engine(true);
    assert!(try_eval(&st, "h: 5; h[]").is_err());
    assert!(try_eval(&st, "eval[()]").is_err());
}

#[test]
fn short_circuit_operators_skip_the_right_side() {
    for pepper in [true, false] {
        let st = engine(pepper);
        assert_eq!(eval(&st, "1b || 0b"), SpicyObj::Boolean(true));
        assert_eq!(eval(&st, "0b || 0b"), SpicyObj::Boolean(false));
        assert_eq!(eval(&st, "1b && 0b"), SpicyObj::Boolean(false));
        assert_eq!(eval(&st, "1b && 1b"), SpicyObj::Boolean(true));
        assert_eq!(eval(&st, "0n ?? 5"), SpicyObj::I64(5));
        assert_eq!(eval(&st, "3 ?? 5"), SpicyObj::I64(3));
        // The right side names nothing that exists: it must not be evaluated.
        assert_eq!(eval(&st, "1b || notDefinedAnywhere"), SpicyObj::Boolean(true));
        assert_eq!(eval(&st, "0b && notDefinedAnywhere"), SpicyObj::Boolean(false));
        assert_eq!(eval(&st, "3 ?? notDefinedAnywhere"), SpicyObj::I64(3));
        assert!(try_eval(&st, "0b || notDefinedAnywhere").is_err());
    }
}

#[test]
fn string_literal_escapes_are_decoded() {
    for pepper in [true, false] {
        let st = engine(pepper);
        assert_eq!(eval(&st, r#""a\nb""#), SpicyObj::String("a\nb".into()));
        assert_eq!(eval(&st, r#""tab\there""#), SpicyObj::String("tab\there".into()));
        assert_eq!(eval(&st, r#""\\""#), SpicyObj::String("\\".into()));
        assert_eq!(eval(&st, r#""say \"hi\"""#), SpicyObj::String("say \"hi\"".into()));
        assert_eq!(eval(&st, r#""plain""#), SpicyObj::String("plain".into()));
    }
}

#[test]
fn null_vector_literal_has_its_length() {
    let st = engine(true);
    assert_eq!(eval(&st, "count 0n 0n 0n"), SpicyObj::I64(3));
    assert_eq!(eval(&st, "0n"), SpicyObj::Null);
}

#[test]
fn list_of_datetime_atoms_stays_in_milliseconds() {
    use polars::prelude::{DataType, TimeUnit};
    let st = engine(true);
    let SpicyObj::Series(s) = eval(&st, "(2024.01.01T00:00:00; 2024.01.02T00:00:00)") else {
        panic!("expected a series");
    };
    assert_eq!(s.dtype(), &DataType::Datetime(TimeUnit::Milliseconds, None));
    // 2024-01-01 in ms since the epoch
    assert_eq!(s.to_physical_repr().i64().unwrap().get(0), Some(1_704_067_200_000));
}
