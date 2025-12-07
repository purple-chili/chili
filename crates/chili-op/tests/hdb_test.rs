use std::path::PathBuf;

use chili_core::{SpicyObj, parse};

mod util;

use crate::util::create_state;

#[test]
fn hdb_case0() {
    let mut hdb_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    hdb_path.push("tests/data");
    let state = create_state();
    let _ = state.load_par_df(hdb_path.to_str().unwrap());
    assert_eq!(state.get_par_df("table1").unwrap().pars, vec![10957, 10958]);
    assert_eq!(state.get_par_df("table2").unwrap().pars, vec![2000]);
    assert_eq!(state.get_par_df("table3").unwrap().pars, vec![2000]);
    assert_eq!(state.get_par_df("table4").unwrap().pars, vec![] as Vec<i32>);

    let code = "
    r01: count(select from table1 where date<0000.01.01);
    r02: count(select from table1 where date<2000.01.01);
    r03: count(select from table1 where date<2000.01.02);
    r04: count(select from table1 where date<=2000.01.02);
    r05: count(select from table1 where date>=2000.01.02);
    r06: count(select from table1 where date>2000.01.02);
    r07: count(select from table1 where date>2500.01.02);
    r08: count(select from table2 where year=2000);
    r09: count(select from table2 where year<=2000);
    r10: count(select from table2 where year<2000);
    r11: count(select from table2 where year>1000);
    r12: count(select from table2 where year>=2000);
    r13: count(select from table2 where year>2000);
    r14: count(select from table3 where year=2000);
    r15: count(select from table4);

    r16: count(select from table1 where date in 2000.01.01 2000.01.02);
    r17: count(select from table1 where date in 2000.01.02);
    ";
    let nodes = parse(code, 0)
        .map_err(|e| {
            eprintln!("{}", e);
            e
        })
        .unwrap();
    state.eval_ast(nodes, "", code).unwrap();
    assert_eq!(state.get_var("r01").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r02").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r03").unwrap(), SpicyObj::I64(4));
    assert_eq!(state.get_var("r04").unwrap(), SpicyObj::I64(6));
    assert_eq!(state.get_var("r05").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r06").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r07").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r08").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r09").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r10").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r11").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r12").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r13").unwrap(), SpicyObj::I64(0));
    assert_eq!(state.get_var("r14").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r15").unwrap(), SpicyObj::I64(2));
    assert_eq!(state.get_var("r16").unwrap(), SpicyObj::I64(6));
    assert_eq!(state.get_var("r17").unwrap(), SpicyObj::I64(2));
}
