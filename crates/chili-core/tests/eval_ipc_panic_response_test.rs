//! IPC eval must turn Rust panics into error results so Sync callers are not
//! parked forever when `eval_timeout_ms == 0`.

use std::sync::Arc;

use chili_core::{
    EngineState, Func, IpcEvalResult, SpicyObj, SpicyResult, Stack, eval_ipc_with_timeout,
};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> Arc<EngineState> {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    Arc::new(state)
}

fn panic_hook(
    _state: &EngineState,
    _stack: &mut Stack,
    _args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    panic!("deliberate ipc panic");
}

fn install_panic_pre_hook(state: &EngineState) {
    let f = Func::new_side_effect_built_in_fn(
        Some(Box::new(panic_hook)),
        3,
        "panic_hook",
        &["user", "handle", "query"],
    );
    state
        .set_var("panic_hook", SpicyObj::Fn(f))
        .expect("set panic_hook");
    state.set_pre_eval_hook(Some("panic_hook".to_string()));
}

#[test]
fn ipc_eval_panic_becomes_error_without_timeout() {
    let state = new_engine();
    state.set_eval_timeout_ms(0);
    install_panic_pre_hook(&state);

    let q = SpicyObj::String("1+1;".to_string());
    match eval_ipc_with_timeout(&state, "u", 1, &q, "t.pep") {
        IpcEvalResult::Finished(Err(e)) => {
            let msg = e.to_string();
            assert!(
                msg.contains("eval panicked") && msg.contains("deliberate ipc panic"),
                "unexpected error: {msg}"
            );
        }
        other => panic!("expected Finished(Err), got {other:?}"),
    }
}

#[test]
fn ipc_eval_panic_becomes_error_with_timeout_worker() {
    let state = new_engine();
    state.set_eval_timeout_ms(5_000);
    install_panic_pre_hook(&state);

    let q = SpicyObj::String("1+1;".to_string());
    match eval_ipc_with_timeout(&state, "u", 1, &q, "t.pep") {
        IpcEvalResult::Finished(Err(e)) => {
            let msg = e.to_string();
            assert!(
                msg.contains("eval panicked") && msg.contains("deliberate ipc panic"),
                "unexpected error: {msg}"
            );
        }
        other => panic!("expected Finished(Err), got {other:?}"),
    }
}
