mod util;
use crate::util::create_state;
use chili_core::SpicyObj;

#[test]
fn code_test() {
    let state = create_state(true);
    let code = "
    t: ([]sym:`a`a`b`b, time: 1 2 3 4, end: 3 5 7 9, price: 1 2 3 4);
    `t upsert [`c`c`c, 5 6 7, 9 9 9, 5 5 5];
    h0: count(t);
    insert(`t, `sym, [`c`c`c, 5 6 7, 9 9 9, 5 5 5]);
    h1: count(t);
    ";

    let nodes = state.parse("", code).unwrap();
    state.eval_ast(nodes, "", code).unwrap();
    assert_eq!(state.get_var("h0").unwrap(), SpicyObj::I64(7));
    assert_eq!(state.get_var("h1").unwrap(), SpicyObj::I64(3));
}
