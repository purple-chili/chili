//! Query-context comparisons against fn operands must return an error,
//! not panic via `as_expr().unwrap()`.

use chili_core::{EngineState, SpicyError, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    state.enable_pepper();
    state
}

fn eval(state: &EngineState, src: &str) -> Result<SpicyObj, SpicyError> {
    let mut stack = Stack::new(None, 0, 0, "");
    state.eval(&mut stack, &SpicyObj::String(src.to_owned()), "test.pep")
}

#[test]
fn query_cmp_against_fn_operand_returns_error() {
    let engine = make_engine();
    eval(&engine, "t: ([] a: 1 2 3)").expect("seed table");

    for query in [
        "select from t where a < now[]",
        "select from t where a < sum",
        "select from t where a < count",
        "select from t where a > now[]",
        "select from t where a = sum",
    ] {
        let err = eval(&engine, query).expect_err(query);
        let msg = err.to_string();
        assert!(
            msg.contains("fn") || msg.contains("Unsupported"),
            "expected UnsupportedQueryJTypeErr-like error for `{query}`, got: {msg}"
        );
    }
}
