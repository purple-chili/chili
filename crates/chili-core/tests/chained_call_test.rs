use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn new_engine(pepper: bool) -> EngineState {
    let mut state = EngineState::initialize();
    if pepper {
        state.enable_pepper();
    }
    state.register_fn(&BUILT_IN_FN);
    state
}

fn eval(state: &EngineState, src: &str, path: &str) -> SpicyResult<SpicyObj> {
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(src.into()),
        path,
    )
}

#[test]
fn pepper_chains_projections_left_to_right() {
    let state = new_engine(true);
    eval(&state, "f: {[a;b;c] a+b+c}", "chain.pep").unwrap();
    for src in ["f[1;2][3]", "f[1][2][3]", "f[;2;][1;3]", "f[1;;3][2]"] {
        assert_eq!(
            eval(&state, src, "chain.pep").unwrap(),
            SpicyObj::I64(6),
            "{src}"
        );
    }
}

#[test]
fn chili_chains_placeholder_projections_without_changing_arity_rules() {
    let state = new_engine(false);
    eval(&state, "f: function(a,b,c){a+b+c}", "chain.chi").unwrap();
    for src in ["f(1,2,)(3)", "f(1,,)(2,)(3)", "f(,2,)(1,3)"] {
        assert_eq!(
            eval(&state, src, "chain.chi").unwrap(),
            SpicyObj::I64(6),
            "{src}"
        );
    }
    let error = eval(&state, "f(1,2)(3)", "chain.chi")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Expect 3 argument(s), 2 given"), "{error}");
}

#[test]
fn repeated_projection_preserves_argument_positions_and_original_projection() {
    let state = new_engine(true);
    eval(
        &state,
        "f: {[a;b;c] x:100*a; y:10*b; x+y+c}; p: f[;2;]",
        "chain.pep",
    )
    .unwrap();
    for (src, expected) in [
        ("f[1][2][3]", 123),
        ("f[;2;][1;][3]", 123),
        ("f[;2;][;3][1]", 123),
        ("q: p[1]; q[3]", 123),
        ("p[4;5]", 425),
        ("add: +[;]; add[1][2]", 3),
    ] {
        assert_eq!(
            eval(&state, src, "chain.pep").unwrap(),
            SpicyObj::I64(expected),
            "{src}"
        );
    }
}

#[test]
fn returned_functions_and_empty_calls_can_be_chained() {
    for (pepper, path, src) in [
        (true, "chain.pep", "make: {[] {[] {[x] x+1}}}; make[][][3]"),
        (
            false,
            "chain.chi",
            "make: function(){ function(){ function(x){x+1} } }; make()()(3)",
        ),
    ] {
        let state = new_engine(pepper);
        assert_eq!(eval(&state, src, path).unwrap(), SpicyObj::I64(4));
    }
}

#[test]
fn chained_collection_lookups_and_series_results_work() {
    for (pepper, path, src) in [
        (true, "chain.pep", "d: {a: {b: 10 20 30}}; d[`a][`b][1]"),
        (false, "chain.chi", "d: {a: {b: 10 20 30}}; d(`a)(`b)(1)"),
    ] {
        let state = new_engine(pepper);
        assert_eq!(eval(&state, src, path).unwrap(), SpicyObj::I64(20));
    }
}

#[test]
fn chained_calls_work_inside_arguments_and_arithmetic() {
    let state = new_engine(true);
    eval(&state, "f: {[a;b;c] a+b+c}; twice: {[x] 2*x}", "chain.pep").unwrap();
    for (src, expected) in [("twice[f[1;2][3]]", 12), ("10+f[1;2][3]*2", 22)] {
        assert_eq!(
            eval(&state, src, "chain.pep").unwrap(),
            SpicyObj::I64(expected)
        );
    }
}

#[test]
fn first_call_side_effects_happen_once_and_errors_propagate() {
    let state = new_engine(true);
    let src = ".calls: 0; make: {[] .calls: .calls+1; {[x] x}}; make[][9]";
    assert_eq!(eval(&state, src, "chain.pep").unwrap(), SpicyObj::I64(9));
    assert_eq!(state.get_var(".calls").unwrap(), SpicyObj::I64(1));
    let error = eval(&state, "bad: {[] raise \"boom\"}; bad[][3]", "chain.pep")
        .unwrap_err()
        .to_string();
    assert!(error.contains("boom"), "{error}");
    let error = eval(&state, "f: {[a;b;c] a+b+c}; f[1;2][3;4]", "chain.pep")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Expect 1 argument(s), 2 given"), "{error}");
}

#[test]
fn indexed_assignment_remains_distinct_from_chained_calls() {
    for (pepper, path, src) in [
        (true, "chain.pep", "d: {a: {b: 1}}; d[`a]:{b: 9}; d[`a][`b]"),
        (
            false,
            "chain.chi",
            "d: {a: {b: 1}}; d(`a):{b: 9}; d(`a)(`b)",
        ),
    ] {
        let state = new_engine(pepper);
        assert_eq!(eval(&state, src, path).unwrap(), SpicyObj::I64(9));
    }
}

#[test]
fn pepper_parenthesized_dictionary_targets_support_string_and_symbol_keys() {
    let state = new_engine(true);
    eval(&state, "d: `a`b!(1;2)", "group.pep").unwrap();
    for src in [
        "d[\"b\"]",
        "d[`b]",
        "(d)[\"b\"]",
        "(d)[`b]",
        "((d))[\"b\"]",
        "(`a`b!(1;2))[\"b\"]",
        "(`a`b!(1;2))[`b]",
    ] {
        assert_eq!(
            eval(&state, src, "group.pep").unwrap(),
            SpicyObj::I64(2),
            "{src}"
        );
    }
    assert!(
        eval(&state, "(d)[\"missing\"]", "group.pep")
            .unwrap()
            .is_null()
    );
}

#[test]
fn pepper_grouped_calls_and_series_indexing_use_the_grouped_value() {
    let state = new_engine(true);
    eval(
        &state,
        "f: {[a;b;c] a+b+c}; make: {[] {[x] x+1}}",
        "group.pep",
    )
    .unwrap();
    for (src, expected) in [
        ("(f)[1;2;3]", 6),
        ("(f[1;2])[3]", 6),
        ("((f[1]))[2][3]", 6),
        ("(make[])[3]", 4),
        ("(+)[1;2]", 3),
        ("(10 20 30)[1]", 20),
    ] {
        assert_eq!(
            eval(&state, src, "group.pep").unwrap(),
            SpicyObj::I64(expected),
            "{src}"
        );
    }
}

#[test]
fn pepper_explicit_list_targets_remain_lists() {
    let state = new_engine(true);
    let dictionary = eval(&state, "d: {a: 1}", "group.pep").unwrap();
    for src in ["(d;)[0]", "(d;99)[0]", "(d;99;)[0]"] {
        assert_eq!(eval(&state, src, "group.pep").unwrap(), dictionary, "{src}");
    }
    assert!(eval(&state, "(d;)[1]", "group.pep").unwrap().is_null());
    assert!(eval(&state, "()[0]", "group.pep").unwrap().is_null());
    assert_eq!(
        eval(&state, "(d;99)[1]", "group.pep").unwrap(),
        SpicyObj::I64(99)
    );
}
