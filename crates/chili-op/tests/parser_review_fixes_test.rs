//! Parser and lexer defects found in the 0.11.1 review.

use std::time::{Duration, Instant};

use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn engine(pepper: bool) -> EngineState {
    let mut state = EngineState::initialize();
    if pepper {
        state.enable_pepper();
    }
    state.register_fn(&BUILT_IN_FN);
    state
}

fn try_eval(state: &EngineState, src: &str) -> SpicyResult<SpicyObj> {
    let path = if state.is_repl_use_chili_syntax() { "t.chi" } else { "t.pep" };
    state.eval(&mut Stack::new(None, 0, 0, ""), &SpicyObj::String(src.to_string()), path)
}

fn eval(state: &EngineState, src: &str) -> SpicyObj {
    try_eval(state, src).unwrap_or_else(|e| panic!("eval failed for {src:?}: {e}"))
}

fn f64s(obj: SpicyObj) -> Vec<Option<f64>> {
    let SpicyObj::Series(s) = obj else {
        panic!("expected series, got {}", obj.get_type_name());
    };
    s.f64().unwrap().iter().collect()
}

#[test]
fn deeply_nested_expressions_parse_in_linear_time() {
    for pepper in [true, false] {
        let st = engine(pepper);
        let src = format!("{}1{}", "(".repeat(40), ")".repeat(40));
        let t0 = Instant::now();
        assert_eq!(eval(&st, &src), SpicyObj::I64(1));
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "40 nested parens took {:?} (pepper={pepper})",
            t0.elapsed()
        );
    }
    // nested calls
    let st = engine(true);
    eval(&st, "f: {[x] x + 1}");
    let src = format!("{}0{}", "f[".repeat(30), "]".repeat(30));
    let t0 = Instant::now();
    assert_eq!(eval(&st, &src), SpicyObj::I64(30));
    assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
}

#[test]
fn scientific_notation() {
    let st = engine(true);
    assert_eq!(eval(&st, "1e-5"), SpicyObj::F64(1e-5));
    assert_eq!(eval(&st, "1.5e+3"), SpicyObj::F64(1500.0));
    assert_eq!(eval(&st, "2.5E-2"), SpicyObj::F64(0.025));
    assert_eq!(eval(&st, "-1e-2"), SpicyObj::F64(-0.01));
    assert_eq!(eval(&st, "1.5e3"), SpicyObj::F64(1500.0));
    assert_eq!(f64s(eval(&st, "1e-5 2.0")), vec![Some(1e-5), Some(2.0)]);
    assert_eq!(f64s(eval(&st, "2.0 1e-5")), vec![Some(2.0), Some(1e-5)]);
    // subtraction is still subtraction when the operands are apart
    assert_eq!(eval(&st, "7 - 5"), SpicyObj::I64(2));
}

#[test]
fn float_vector_may_start_with_integers() {
    let st = engine(true);
    assert_eq!(f64s(eval(&st, "1 2.5")), vec![Some(1.0), Some(2.5)]);
    assert_eq!(f64s(eval(&st, "1 2 3.0")), vec![Some(1.0), Some(2.0), Some(3.0)]);
    assert_eq!(f64s(eval(&st, "1 0n 2.5 3")), vec![Some(1.0), None, Some(2.5), Some(3.0)]);
    // plain integer vectors are untouched
    let SpicyObj::Series(s) = eval(&st, "1 2 3") else {
        panic!("expected series");
    };
    assert_eq!(s.i64().unwrap().into_no_null_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
}
