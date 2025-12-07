use std::path::PathBuf;

use chili_core::{SpicyObj, parse};

mod util;

use crate::util::create_state;

#[cfg(feature = "vintage")]
mod tests {
    use super::*;

    #[test]
    fn eval_case00() {
        let code = "total:sum 1.0 2.0f*3";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::F64(9.0));
        assert_eq!(state.get_var("total").unwrap(), SpicyObj::F64(9.0))
    }

    #[test]
    fn eval_case01() {
        let code = "
    f: {[x; y; z] x + y * z};
    r0: f[1; 2; 3];
    g: f[1; ; 9];
    h: {[]9};
    r1: h[];
    g 3
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(28));
        assert_eq!(state.get_var("r0").unwrap().to_i64().unwrap(), 7);
        assert_eq!(state.get_var("r1").unwrap().to_i64().unwrap(), 9);
        assert_eq!(
            state.get_var("f").unwrap().to_string(),
            "{[x; y; z] x + y * z}"
        );
        assert_eq!(
            state.get_var("g").unwrap().to_string(),
            "{[y]\n  {[x; y; z] x + y * z}\n}"
        )
    }

    #[test]
    fn eval_case02() {
        let code = "
    qty: 7 8 9h;
    t: ([]sym: `a`b`b; col1: 1 2 3; col2: 1.0 2.0 3.0f; 4 5 6i; qty);
    df1: select sum col1+col2, newCol: col2 by sym from t;
    count df1
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(2));
        assert_eq!(state.get_var("t").unwrap().df().unwrap().shape(), (3, 5));
        assert_eq!(state.get_var("df1").unwrap().df().unwrap().shape(), (2, 3))
    }

    #[test]
    fn eval_case03() {
        let code = "
    r1: eval (*; 9; 9);
    f: {[x;y] x + y};
    r2: eval[(`f; 9; 1)];
    t: timeit[(+; 1; 1); 1000];
    t
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert!(obj.to_f64().unwrap() > 0.0);
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::I64(81));
        assert_eq!(state.get_var("r2").unwrap(), SpicyObj::I64(10));
    }

    #[test]
    fn eval_case04() {
        let code = "
    d: {a: 1; b: 2; c: 3;};
    d2: {a: 4; b: 5; c: 6};
    d3: `a`b`c!(7;8;9);
    d[`d]:9;
    r1: d2[`c];
    d3[`c] + sum d[`a`d]
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(19));
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::I64(6));
    }

    #[test]
    fn eval_case05() {
        let code = "
    f: {[date]
        if[date>2020.01.01;
            :date;
            date: date + 1;
        ];
        2020.01.01
    };
    r1: f 2024.04.01;
    r2: f 2019.01.01;
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::Date(19814));
        assert_eq!(state.get_var("r2").unwrap(), SpicyObj::Date(18262));
    }

    #[test]
    fn eval_case07() {
        let code = "
    f: {[date]
        if[date>2020.01.01;
            'date;
            date: date + 1;
        ];
        2020.01.01
    };
    r1: f 2024.04.01;
    r2: f 2019.01.01;
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        assert!(
            state
                .eval_ast(nodes, "", code)
                .unwrap_err()
                .to_string()
                .contains("2024.04.01")
        );
    }

    #[test]
    fn eval_case08() {
        let code = "
    try [
        a: 1 * `a;
    ] catch [
        err ~ \"type\";
    ]
    ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", code).unwrap();
        assert!(
            state
                .get_var("err")
                .unwrap()
                .str()
                .unwrap()
                .contains("Unsupported binary op")
        );
    }

    #[test]
    fn eval_case09() {
        let mut src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        src_path.push("tests/src/main.pep");
        let code = format!("import \"{}\";", src_path.to_str().unwrap());
        let state = create_state();
        let nodes = parse(&code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", &code).unwrap();
        assert_eq!(state.get_var("n").unwrap(), SpicyObj::I64(314));
    }
}

#[cfg(not(feature = "vintage"))]
mod tests {
    use super::*;

    #[test]
    fn eval_case00() {
        let code = "total:sum(1.0 2.0f*3)";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::F64(9.0));
        assert_eq!(state.get_var("total").unwrap(), SpicyObj::F64(9.0))
    }

    #[test]
    fn eval_case01() {
        let code = "
        f: function(x,y,z){ x + y * z};
        r0: f(1, 2, 3);
        g: f(1, , 9);
        h: function(){9};
        r1: h();
        g(3)
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(36));
        assert_eq!(state.get_var("r0").unwrap().to_i64().unwrap(), 9);
        assert_eq!(state.get_var("r1").unwrap().to_i64().unwrap(), 9);
        assert_eq!(
            state.get_var("f").unwrap().to_string(),
            "function(x,y,z){ x + y * z}"
        );
        assert_eq!(
            state.get_var("g").unwrap().to_string(),
            "function(y)\n{\n  function(x,y,z){ x + y * z}\n}"
        )
    }

    #[test]
    fn eval_case02() {
        let code = "
        qty: 7 8 9h;
        t: ([]sym: `a`b`b, col1: 1 2 3, col2: 1.0 2.0 3.0f, 4 5 6i, qty);
        df1: select sum(col1+col2), newCol: col2 by sym from t;
        count(df1)
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(2));
        assert_eq!(state.get_var("t").unwrap().df().unwrap().shape(), (3, 5));
        assert_eq!(state.get_var("df1").unwrap().df().unwrap().shape(), (2, 3))
    }

    #[test]
    fn eval_case03() {
        let code = "
        r1: eval([*, 9, 9]);
        f: function(x,y){ x + y};
        r2: eval([`f, 9, 1]);
        t: timeit([+, 1, 1], 1000);
        t
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert!(obj.to_f64().unwrap() > 0.0);
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::I64(81));
        assert_eq!(state.get_var("r2").unwrap(), SpicyObj::I64(10));
    }

    #[test]
    fn eval_case04() {
        let code = "
        d: {a: 1, b: 2, c: 3,};
        d2: {a: 4, b: 5, c: 6};
        d3: `a`b`c![7,8,9];
        d(`d):9;
        r1: d2(`c);
        d3(`c) + sum(d(`a`d))
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        let obj = state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(obj, SpicyObj::I64(19));
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::I64(6));
    }

    #[test]
    fn eval_case05() {
        let code = "
        f: function(date){
            if(date>2020.01.01){
                return date;
                date: date + 1;
            }
            2020.01.01
        };
        r1: f(2024.04.01);
        r2: f(2019.01.01);
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", code).unwrap();
        assert_eq!(state.get_var("r1").unwrap(), SpicyObj::Date(19814));
        assert_eq!(state.get_var("r2").unwrap(), SpicyObj::Date(18262));
    }

    #[test]
    fn eval_case07() {
        let code = "
        f: function(date){
            if(date>2020.01.01){
                raise date;
                date: date + 1;
            }
            2020.01.01
        };
        r1: f(2024.04.01);
        r2: f(2019.01.01);
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        assert!(
            state
                .eval_ast(nodes, "", code)
                .unwrap_err()
                .to_string()
                .contains("2024.04.01")
        );
    }

    #[test]
    fn eval_case08() {
        let code = "
        try {
            a: 1 * `a;
        } catch {
            err ~ \"type\";
        }
        ";
        let state = create_state();
        let nodes = parse(code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", code).unwrap();
        assert!(
            state
                .get_var("err")
                .unwrap()
                .str()
                .unwrap()
                .contains("Unsupported binary op")
        );
    }

    #[test]
    fn eval_case09() {
        let mut src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        src_path.push("tests/src/main.chi");
        let code = format!("import(\"{}\");", src_path.to_str().unwrap());
        let state = create_state();
        let nodes = parse(&code, 0)
            .map_err(|e| {
                eprintln!("{}", e);
                e
            })
            .unwrap();
        state.eval_ast(nodes, "", &code).unwrap();
        assert_eq!(state.get_var("n").unwrap(), SpicyObj::I64(314));
    }
}
