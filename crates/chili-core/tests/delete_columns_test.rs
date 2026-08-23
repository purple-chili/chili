//! Column-delete: bare ids and multi-column `delete qty, price from t`.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
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
fn delete_bare_column_id() {
    let state = new_engine();
    eval_pep(&state, "t: ([]qty: 1 2 3; price: 10 20 30; sym: `a`b`c);");
    let out = eval_pep(&state, "delete qty from t");
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe, got {}", out.get_type_name());
    };
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>(),
        vec!["price", "sym"]
    );
}

#[test]
fn delete_multiple_bare_column_ids() {
    let state = new_engine();
    eval_pep(&state, "t: ([]qty: 1 2 3; price: 10 20 30; sym: `a`b`c);");
    let out = eval_pep(&state, "delete qty, price from t");
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe, got {}", out.get_type_name());
    };
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>(),
        vec!["sym"]
    );
}

#[test]
fn delete_rows_where_still_works() {
    let state = new_engine();
    eval_pep(&state, "t: ([]qty: 1 2 3; price: 10 20 30; sym: `a`b`c);");
    let out = eval_pep(&state, "delete from t where qty = 2");
    let SpicyObj::DataFrame(df) = out else {
        panic!("expected dataframe, got {}", out.get_type_name());
    };
    assert_eq!(df.height(), 2);
}
