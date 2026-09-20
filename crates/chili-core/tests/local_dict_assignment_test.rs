use chili_core::{EngineState, SpicyObj, SpicyResult, parse};

fn eval(state: &EngineState, src: &str, path: &str) -> SpicyResult<SpicyObj> {
    state.eval_ast(parse(src, 0, path)?, "", src)
}

#[test]
fn local_dictionary_supports_replacement_and_new_keys() {
    for (src, path) in [
        (
            "f: {[] d: {a: 1; b: 2}; key: `c; d[`a]:9; d[key]:3; d}; f[]",
            "repl.pep",
        ),
        (
            "f: function(){ d: {a: 1, b: 2}; key: `c; d(`a):9; d(key):3; d }; f()",
            "repl.chi",
        ),
    ] {
        let state = EngineState::initialize();
        let result = eval(&state, src, path).unwrap();
        let expected = eval(&state, "{a: 9; b: 2; c: 3}", "repl.pep").unwrap();
        assert_eq!(result, expected, "{path}");
        assert!(state.get_var("d").is_err());
    }
}

#[test]
fn dictionary_parameter_is_updated_without_changing_callers_value() {
    let state = EngineState::initialize();
    let original = eval(&state, "input: {a: 1; b: 2}", "repl.pep").unwrap();
    let result = eval(&state, "f: {[d] d[`a]:9; d}; f[input]", "repl.pep").unwrap();
    let expected = eval(&state, "{a: 9; b: 2}", "repl.pep").unwrap();
    assert_eq!(result, expected);
    assert_eq!(state.get_var("input").unwrap(), original);
    assert!(state.get_var("d").is_err());
}

#[test]
fn local_dictionary_shadows_same_named_global() {
    let state = EngineState::initialize();
    let global = eval(&state, "d: {a: 100; global: 200}", "repl.pep").unwrap();
    let result = eval(
        &state,
        "f: {[] d: {a: 1; local: 2}; d[`a]:9; d}; f[]",
        "repl.pep",
    )
    .unwrap();
    let expected = eval(&state, "{a: 9; local: 2}", "repl.pep").unwrap();
    assert_eq!(result, expected);
    assert_eq!(state.get_var("d").unwrap(), global);
}

#[test]
fn global_fallback_creates_local_copy_and_dotted_assignment_updates_global() {
    let state = EngineState::initialize();
    let original = eval(&state, "d: {a: 1; b: 2}", "repl.pep").unwrap();
    let expected = eval(&state, "{a: 9; b: 2}", "repl.pep").unwrap();
    let result = eval(&state, "f: {[] d[`a]:9; d}; f[]", "repl.pep").unwrap();
    assert_eq!(result, expected);
    assert_eq!(state.get_var("d").unwrap(), original);

    state.set_var(".d", original).unwrap();
    let result = eval(&state, "f: {[] .d[`a]:9}; f[]", "repl.pep").unwrap();
    assert!(result.is_null());
    assert_eq!(state.get_var(".d").unwrap(), expected);
    eval(&state, "d[`a]:9", "repl.pep").unwrap();
    assert_eq!(state.get_var("d").unwrap(), expected);
}

#[test]
fn missing_and_unsupported_local_targets_still_error() {
    let state = EngineState::initialize();
    let error = eval(&state, "f: {[] d[`a]:9}; f[]", "repl.pep")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Name 'd' is not defined"), "{error}");
    let global = eval(&state, "d: {a: 1}", "repl.pep").unwrap();
    let error = eval(&state, "f: {[] d: 42; d[`a]:9}; f[]", "repl.pep")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Not support index assignment"), "{error}");
    assert_eq!(state.get_var("d").unwrap(), global);
}
