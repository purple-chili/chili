//! Pre-eval request hook tests for `eval_with_pre_hook`.

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    // Register operator/built-in fns like `ChiliEngine(pepper=True)`.
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    state
}

/// Install a 3-arg ACL hook that denies, redirects, or allows by query symbol.
fn install_acl_hook(state: &EngineState) {
    let mut s = Stack::new(None, 0, 0, "");
    for src in [
        "allowedvar: 7;",
        "secretvar: 99;",
        "redirecttarget: 55;",
        ".acl.hook: {[u; h; q]
            $[ q ~ `secretvar;
               raise \"permission denied\";
               $[ q ~ `redirect; `redirecttarget; q ] ] };",
    ] {
        state
            .eval(&mut s, &SpicyObj::String(src.to_string()), "acl.pep")
            .unwrap_or_else(|e| panic!("setup eval failed for {src:?}: {e}"));
    }
    state.set_pre_eval_hook(Some(".acl.hook".to_string()));
}

#[test]
fn no_hook_is_identical_to_eval() {
    let state = new_engine();
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(&mut s, &SpicyObj::String("x: 11;".to_string()), "t.pep")
        .unwrap();
    assert!(state.get_pre_eval_hook().is_none());
    let q = SpicyObj::Symbol("x".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "t.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 11);
}

#[test]
fn hook_allows_passes_query_through() {
    let state = new_engine();
    install_acl_hook(&state);
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("allowedvar".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "ipc.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 7, "allow should resolve the var");
}

#[test]
fn hook_rewrites_redirects_to_another_request() {
    let state = new_engine();
    install_acl_hook(&state);
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("redirect".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "ipc.pep").unwrap();
    assert_eq!(
        got.to_i64().unwrap(),
        55,
        "rewrite should redirect to redirecttarget"
    );
}

#[test]
fn hook_deny_raises_and_propagates() {
    let state = new_engine();
    install_acl_hook(&state);
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("secretvar".into());
    let res = state.eval_with_pre_hook(&mut s, &q, "ipc.pep");
    let err = res.expect_err("deny must raise");
    assert!(
        err.to_string().contains("permission denied"),
        "deny error should carry the hook message, got: {err}"
    );
}

#[test]
fn clear_hook_restores_unfiltered_eval() {
    let state = new_engine();
    install_acl_hook(&state);
    state.set_pre_eval_hook(None);
    assert!(state.get_pre_eval_hook().is_none());
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("secretvar".into());
    let got = state.eval_with_pre_hook(&mut s, &q, "ipc.pep").unwrap();
    assert_eq!(got.to_i64().unwrap(), 99);
}

#[test]
fn missing_hook_fn_is_a_clear_error() {
    let state = new_engine();
    state.set_pre_eval_hook(Some(".nope.missing".to_string()));
    let mut s = Stack::new(None, 0, 1, "alice");
    let q = SpicyObj::Symbol("anything".into());
    let err = state
        .eval_with_pre_hook(&mut s, &q, "ipc.pep")
        .expect_err("a registered-but-undefined hook must error, not silently pass");
    assert!(err.to_string().contains("not defined"));
}
