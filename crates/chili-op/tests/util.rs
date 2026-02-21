use chili_core::EngineState;
use chili_op::BUILT_IN_FN;

pub fn create_state(use_chili_syntax: bool) -> EngineState {
    let mut state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    if !use_chili_syntax {
        state.enable_pepper();
    }
    state
}
