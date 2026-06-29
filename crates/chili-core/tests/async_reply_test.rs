//! Tests for `this.h` (caller handle) and `.handle.reply`.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    state
}

#[test]
fn returns_caller_handle_through_fn_call() {
    let state = new_engine();
    let mut def = Stack::new(None, 0, 0, "");
    state
        .eval(
            &mut def,
            &SpicyObj::String(".srv.whoami: {[] this.h}".into()),
            "t",
        )
        .unwrap();

    let mut conn = Stack::new(None, 0, 42, "alice");
    let call = SpicyObj::MixedList(vec![SpicyObj::Symbol(".srv.whoami".into())]);
    let got = state.eval(&mut conn, &call, "ipc42.pep").unwrap();
    assert_eq!(
        got.to_i64().unwrap(),
        42,
        "this.h must equal the caller handle"
    );

    let mut local = Stack::new(None, 0, 0, "");
    let call0 = SpicyObj::MixedList(vec![SpicyObj::Symbol(".srv.whoami".into())]);
    let got0 = state.eval(&mut local, &call0, "t").unwrap();
    assert_eq!(got0.to_i64().unwrap(), 0);
}

#[test]
fn handle_reply_registered_and_validates_target() {
    let state = new_engine();
    let mut s = Stack::new(None, 0, 0, "");
    let call = SpicyObj::MixedList(vec![
        SpicyObj::Symbol(".handle.reply".into()),
        SpicyObj::I64(999_999),
        SpicyObj::String("x".into()),
    ]);
    let err = state.eval(&mut s, &call, "t").unwrap_err();
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("handle"),
        "expected an invalid-handle error, got: {msg}"
    );
}
