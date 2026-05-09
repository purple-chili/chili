use chili_core::parse;

mod util;

use crate::util::create_state;

#[test]
fn trace_mul_symbol_error() {
    let code = "1*`a";
    let state = create_state(true);
    let nodes = parse(code, 0, "repl.chi")
        .map_err(|e| {
            eprintln!("{}", e);
            e
        })
        .unwrap();
    let err = state.eval_ast(nodes, "", code).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Unsupported binary op"),
        "expected 'Unsupported binary op' in error, got: {msg}"
    );
    assert!(
        msg.contains("--> :1:"),
        "expected trace with line info in error, got: {msg}"
    );
    assert!(
        msg.contains("1*`a"),
        "expected source line in error, got: {msg}"
    );
    assert!(msg.contains("^"), "expected caret in error, got: {msg}");
}

#[test]
fn trace_mul_symbol_error_multiline() {
    let code = "a: 1;\nb: a*`x";
    let state = create_state(true);
    let nodes = parse(code, 0, "repl.chi")
        .map_err(|e| {
            eprintln!("{}", e);
            e
        })
        .unwrap();
    let err = state.eval_ast(nodes, "", code).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("--> :2:"),
        "expected error on line 2, got: {msg}"
    );
    assert!(
        msg.contains("b: a*`x"),
        "expected source line in error, got: {msg}"
    );
}

#[test]
fn trace_mul_symbol_error_pepper() {
    let code = "1*`a";
    let state = create_state(false);
    let nodes = parse(code, 0, "repl.pep")
        .map_err(|e| {
            eprintln!("{}", e);
            e
        })
        .unwrap();
    let err = state.eval_ast(nodes, "", code).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Unsupported binary op"),
        "expected 'Unsupported binary op' in error, got: {msg}"
    );
}
