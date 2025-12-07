use chili_core::EngineState;
use chili_op::BUILT_IN_FN;

pub fn create_state() -> EngineState {
    let state = EngineState::new();
    state.register_fn(&BUILT_IN_FN);
    state
}
