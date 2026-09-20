use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn eval(state: &EngineState, source: &str, path: &str) -> SpicyResult<SpicyObj> {
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(source.into()),
        path,
    )
}

fn new_engine(pepper: bool) -> EngineState {
    let mut state = EngineState::initialize();
    if pepper {
        state.enable_pepper();
    }
    state.register_fn(&BUILT_IN_FN);
    state
}

#[test]
fn compact_subtraction_matches_spaced_form_in_both_languages() {
    for (pepper, path) in [(true, "minus.pep"), (false, "minus.chi")] {
        let state = new_engine(pepper);
        eval(&state, "n: 5", path).unwrap();
        for (compact, spaced) in [
            ("5-1", "5 - 1"),
            ("n-1", "n - 1"),
            ("(5)-1", "(5) - 1"),
            ("5h-1h", "5h - 1h"),
            ("5.5-1.5", "5.5 - 1.5"),
            ("5.5-1.0e-3", "5.5 - 1.0e-3"),
            ("n-1 -2", "n - 1 -2"),
            ("5 6-1", "5 6 - 1"),
            ("2026.08.20-1", "2026.08.20 - 1"),
            ("2026.08.20-1D00:00:00", "2026.08.20 - 1D00:00:00"),
        ] {
            assert_eq!(
                eval(&state, compact, path).unwrap(),
                eval(&state, spaced, path).unwrap(),
                "{path}: {compact}"
            );
        }
        assert_eq!(eval(&state, "5-1", path).unwrap(), SpicyObj::I64(4));
        assert_eq!(eval(&state, "n*-1", path).unwrap(), SpicyObj::I64(-5));
        assert_eq!(eval(&state, "n - -1", path).unwrap(), SpicyObj::I64(6));
    }
}

#[test]
fn subtraction_after_calls_and_inside_query_predicates() {
    for (pepper, path, source) in [
        (true, "minus.pep", "f: {[] 5}; f[]-1"),
        (false, "minus.chi", "f: function(){5}; f()-1"),
    ] {
        let state = new_engine(pepper);
        assert_eq!(eval(&state, source, path).unwrap(), SpicyObj::I64(4));
    }
    let state = new_engine(true);
    let result = eval(
        &state,
        "t: ([]x: 1 2 3); select x:x-1 from t where (x-1)>0",
        "minus.pep",
    )
    .unwrap();
    assert_eq!(result, eval(&state, "([]x: 1 2)", "minus.pep").unwrap());
}

#[test]
fn negative_arguments_literals_and_whitespace_vectors_still_work() {
    let state = new_engine(true);
    assert_eq!(
        eval(&state, "f: {[x] x}; f -1", "minus.pep").unwrap(),
        SpicyObj::I64(-1)
    );
    assert_eq!(
        eval(&state, "f[-1]", "minus.pep").unwrap(),
        SpicyObj::I64(-1)
    );
    assert_eq!(
        eval(&state, "-9223372036854775808", "minus.pep").unwrap(),
        SpicyObj::I64(i64::MIN)
    );
    let result = eval(&state, "1 -1", "minus.pep").unwrap();
    assert_eq!(
        result.as_vec().unwrap(),
        vec![SpicyObj::I64(1), SpicyObj::I64(-1)]
    );
    assert!(
        eval(&state, "5--1", "minus.pep")
            .unwrap_err()
            .to_string()
            .contains("Name '--' is not defined")
    );
}
