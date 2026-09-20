mod util;

use chili_core::{EngineState, SpicyObj, Stack};
use util::create_state;

fn eval(state: &EngineState, source: &str) -> SpicyObj {
    state
        .eval(
            &mut Stack::new(None, 0, 0, ""),
            &SpicyObj::String(source.to_owned()),
            "cast.pep",
        )
        .unwrap()
}

#[test]
fn scalar_symbol_to_string_uses_value_not_display() {
    let state = create_state(false);
    assert_eq!(eval(&state, "`str$`abc"), SpicyObj::String("abc".into()));
    assert_eq!(eval(&state, "`abc").to_string(), "`abc");

    // Preserve the entire payload, including a backtick that is actual data.
    for value in ["", ".ns.name", "SOL/USDT", "`literal", "雪"] {
        state
            .set_var("s", SpicyObj::Symbol(value.to_owned()))
            .unwrap();
        assert_eq!(eval(&state, "`str$s"), SpicyObj::String(value.to_owned()));
    }
}

#[test]
fn scalar_and_series_symbol_to_string_casts_agree() {
    let state = create_state(false);
    let SpicyObj::Series(series) = eval(&state, "`str$`abc`def") else {
        panic!("expected a string series");
    };
    let values: Vec<_> = series.str().unwrap().iter().map(Option::unwrap).collect();
    assert_eq!(values, vec!["abc", "def"]);
    assert_eq!(
        eval(&state, "`str$`abc"),
        SpicyObj::String(values[0].into())
    );
    assert_eq!(
        eval(&state, "`str$`def"),
        SpicyObj::String(values[1].into())
    );
}

#[test]
fn recasting_symbols_preserves_their_values() {
    let state = create_state(false);
    assert_eq!(eval(&state, "`sym$`abc"), SpicyObj::Symbol("abc".into()));
    for value in ["abc", "", ".ns.name", "SOL/USDT", "`literal", "雪"] {
        let symbol = SpicyObj::Symbol(value.to_owned());
        state.set_var("s", symbol.clone()).unwrap();
        for source in ["`sym$s", "`cat$s", "`sym$`sym$s", "`cat$`cat$s"] {
            assert_eq!(eval(&state, source), symbol, "{source} on {value:?}");
        }
    }
}

#[test]
fn symbol_cast_preserves_mixed_list_shape() {
    let state = create_state(false);
    let input = SpicyObj::MixedList(vec![
        SpicyObj::Symbol("a".into()),
        SpicyObj::String("b".into()),
        SpicyObj::MixedList(vec![SpicyObj::Symbol("`literal".into())]),
    ]);
    let expected = SpicyObj::MixedList(vec![
        SpicyObj::Symbol("a".into()),
        SpicyObj::Symbol("b".into()),
        SpicyObj::MixedList(vec![SpicyObj::Symbol("`literal".into())]),
    ]);
    state.set_var("ks", input).unwrap();
    for source in ["`sym$ks", "`cat$ks"] {
        assert_eq!(eval(&state, source), expected);
    }
}
