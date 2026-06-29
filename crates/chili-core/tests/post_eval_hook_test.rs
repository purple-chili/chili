//! Post-eval audit hook tests for `eval_with_pre_hook`.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    state
}

/// Install a post-eval hook that records each fire into `.audit.*` engine vars.
fn install_audit_hook(state: &EngineState) {
    let mut s = Stack::new(None, 0, 0, "");
    for src in [
        ".audit.n: 0;",
        ".audit.q: `;",
        ".audit.r: `;",
        ".audit.e: `;",
        ".audit.hook: {[u; h; q; r; e]
            .audit.n: .audit.n + 1;
            .audit.q: q;
            .audit.r: r;
            .audit.e: e;
            0 };",
    ] {
        state
            .eval(&mut s, &SpicyObj::String(src.to_string()), "audit.pep")
            .unwrap_or_else(|e| panic!("setup eval failed for {src:?}: {e}"));
    }
    state.set_post_eval_hook(Some(".audit.hook".to_string()));
}

fn read_var(state: &EngineState, id: &str) -> SpicyObj {
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(&mut s, &SpicyObj::Symbol(id.into()), "read.pep")
        .unwrap()
}

#[test]
fn no_hook_is_zero_overhead_and_eval_unchanged() {
    let state = new_engine();
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(&mut s, &SpicyObj::String("x: 11;".to_string()), "t.pep")
        .unwrap();
    assert!(state.get_post_eval_hook().is_none());
    let q = SpicyObj::Symbol("x".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "t.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 11);
}

#[test]
fn post_hook_fires_with_result_on_success() {
    let state = new_engine();
    {
        let mut s = Stack::new(None, 0, 0, "");
        state
            .eval(&mut s, &SpicyObj::String("v: 42;".to_string()), "t.pep")
            .unwrap();
    }
    install_audit_hook(&state);

    let mut s = Stack::new(None, 0, 7, "alice");
    let q = SpicyObj::Symbol("v".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "ipc.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 42, "the request result is unchanged");

    assert_eq!(read_var(&state, ".audit.n").to_i64().unwrap(), 1);
    assert_eq!(
        read_var(&state, ".audit.r").to_i64().unwrap(),
        42,
        "the hook must receive the evaluated result"
    );
    assert!(
        matches!(read_var(&state, ".audit.e"), SpicyObj::Null),
        "the error arg must be Null on success"
    );
}

#[test]
fn post_hook_fires_with_error_on_failure() {
    let state = new_engine();
    install_audit_hook(&state);

    let mut s = Stack::new(None, 0, 7, "alice");
    let q = SpicyObj::Symbol("nope_undefined".into());
    let res = state.eval_with_pre_hook(&mut s, &q, "ipc.pep");
    assert!(res.is_err(), "the request error must propagate to the caller");

    assert_eq!(read_var(&state, ".audit.n").to_i64().unwrap(), 1);
    assert!(
        matches!(read_var(&state, ".audit.r"), SpicyObj::Null),
        "the result arg must be Null on error"
    );
    let err_obj = read_var(&state, ".audit.e");
    let err_str = match err_obj {
        SpicyObj::Symbol(sym) => sym,
        other => panic!("expected a symbol error arg, got {other:?}"),
    };
    assert!(
        !err_str.is_empty(),
        "the error arg must carry the error string"
    );
}

#[test]
fn clear_hook_stops_it_firing() {
    let state = new_engine();
    {
        let mut s = Stack::new(None, 0, 0, "");
        state
            .eval(&mut s, &SpicyObj::String("w: 5;".to_string()), "t.pep")
            .unwrap();
    }
    install_audit_hook(&state);
    state.set_post_eval_hook(None);
    assert!(state.get_post_eval_hook().is_none());

    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("w".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "ipc.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 5);
    assert_eq!(read_var(&state, ".audit.n").to_i64().unwrap(), 0);
}

#[test]
fn hook_error_is_swallowed_not_propagated() {
    let state = new_engine();
    {
        let mut s = Stack::new(None, 0, 0, "");
        state
            .eval(&mut s, &SpicyObj::String("u: 9;".to_string()), "t.pep")
            .unwrap();
        state
            .eval(
                &mut s,
                &SpicyObj::String(".bad.hook: {[u;h;q;r;e] raise \"boom\" };".to_string()),
                "t.pep",
            )
            .unwrap();
    }
    state.set_post_eval_hook(Some(".bad.hook".to_string()));

    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("u".into());
    let got = state
        .eval_with_pre_hook(&mut s, &q, "ipc.pep")
        .expect("a raising post-eval hook must NOT fail the request");
    assert_eq!(got.to_i64().unwrap(), 9, "the request result is unaffected");
}

#[test]
fn missing_hook_fn_is_swallowed() {
    let state = new_engine();
    {
        let mut s = Stack::new(None, 0, 0, "");
        state
            .eval(&mut s, &SpicyObj::String("z: 3;".to_string()), "t.pep")
            .unwrap();
    }
    state.set_post_eval_hook(Some(".nope.missing".to_string()));
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("z".into());
    let got = state
        .eval_with_pre_hook(&mut s, &q, "ipc.pep")
        .expect("an undefined post-eval hook must be skipped, not break the request");
    assert_eq!(got.to_i64().unwrap(), 3);
}
