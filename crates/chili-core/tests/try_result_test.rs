use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};

fn eval(src: &str) -> SpicyResult<SpicyObj> {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(src.to_owned()),
        "try_result.pep",
    )
}

#[test]
fn returns_last_value_of_successful_branch() {
    for src in [
        "try [1; 42] catch [99]",
        "f: {[] try [1; 42] catch [99]}; f[]",
        "try [42] catch [raise \"must not run\"]",
    ] {
        assert_eq!(eval(src).unwrap(), SpicyObj::I64(42), "{src}");
    }
}

#[test]
fn returns_last_value_of_catch_branch() {
    for src in [
        "try [1; raise \"boom\"; 42] catch [2; 99]",
        "f: {[] try [raise \"boom\"] catch [2; 99]}; f[]",
    ] {
        assert_eq!(eval(src).unwrap(), SpicyObj::I64(99), "{src}");
    }
}

#[test]
fn empty_branches_return_null_without_leaking_prior_values() {
    for src in [
        "try [] catch [99]",
        "try [42; raise \"boom\"] catch []",
        "f: {[] try [42; raise \"boom\"] catch []}; f[]",
    ] {
        assert!(eval(src).unwrap().is_null(), "{src}");
    }
}

#[test]
fn ordinary_branch_values_do_not_return_from_function() {
    for src in [
        "f: {[] try [42] catch [99]; 100}; f[]",
        "f: {[] try [raise \"boom\"] catch [99]; 100}; f[]",
    ] {
        assert_eq!(eval(src).unwrap(), SpicyObj::I64(100), "{src}");
    }
}

#[test]
fn explicit_returns_still_exit_function() {
    for (src, expected) in [
        ("f: {[] try [:42; 0] catch [:99]; 100}; f[]", 42),
        ("f: {[] try [raise \"boom\"] catch [:99; 0]; 100}; f[]", 99),
    ] {
        assert_eq!(eval(src).unwrap(), SpicyObj::I64(expected), "{src}");
    }
}

#[test]
fn nested_try_preserves_inner_result() {
    assert_eq!(
        eval("try [try [raise \"boom\"] catch [42]] catch [99]").unwrap(),
        SpicyObj::I64(42),
    );
}

#[test]
fn chili_syntax_also_preserves_branch_results() {
    let state = EngineState::initialize();
    for (src, expected) in [
        ("try { 1; 42 } catch(err) { 99 }", SpicyObj::I64(42)),
        (
            "try { 42; raise \"boom\"; } catch(err) { 2; 99 }",
            SpicyObj::I64(99),
        ),
        // Chili's trailing semicolon makes the block's final value null.
        ("try { 1; 42; } catch(err) { 99 }", SpicyObj::Null),
        ("try { 42; raise \"boom\"; } catch(err) {}", SpicyObj::Null),
    ] {
        let nodes = chili_core::parse(src, 0, "try_result.chi").unwrap();
        assert_eq!(state.eval_ast(nodes, "", src).unwrap(), expected, "{src}");
    }
}

#[test]
fn catch_errors_propagate_and_error_binding_is_preserved() {
    let error = eval("try [raise \"boom\"] catch [raise \"again\"]").unwrap_err();
    assert!(error.to_string().contains("again"));
    for src in [
        "try [raise \"boom\"] catch [err]",
        "f: {[] try [raise \"boom\"] catch [err]}; f[]",
    ] {
        let result = eval(src).unwrap();
        assert!(result.str().unwrap().contains("boom"), "{src}");
    }
}
