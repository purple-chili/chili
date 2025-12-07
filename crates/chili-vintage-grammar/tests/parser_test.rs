use chili_vintage_grammar::{ChiliParser, Rule};
use pest::Parser;

use crate::util::pretty_format_rules;

#[path = "./util.rs"]
mod util;

#[test]
fn parse_comments() {
    let code = "/*
    block of comment
*/

\"string❤️\"; // comment

// comment
\"string\";

/* */
    ";
    let pairs = ChiliParser::parse(Rule::Program, code).unwrap();
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(vec!["Exp -> String", "Exp -> String", "EOI", ""], actual)
}

#[test]
fn parse_case00() {
    let code = "total:sum 1.0 2.0f*3";
    let pairs = ChiliParser::parse(Rule::Program, code).unwrap();
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> UnaryExp",
            "       -> Id",
            "       -> BinaryExp",
            "         -> Floats",
            "         -> BinaryOp",
            "         -> Exp -> I64",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case01() {
    let code = "
    f: {[x; y; z] x + y * z};
    r: f[1; 2; 3];
    g: f[1; ; 9];
    h: {[]9};
    g 3
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Fn",
            "       -> Params",
            "         -> Id",
            "         -> Id",
            "         -> Id",
            "       -> Exp -> BinaryExp",
            "           -> Id",
            "           -> BinaryOp",
            "           -> Exp -> BinaryExp",
            "               -> Id",
            "               -> BinaryOp",
            "               -> Exp -> Id",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> FnCall",
            "       -> Id",
            "       -> Arg -> Exp -> I64",
            "       -> Arg -> Exp -> I64",
            "       -> Arg -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> FnCall",
            "       -> Id",
            "       -> Arg -> Exp -> I64",
            "       -> Arg -> DelayedArg",
            "       -> Arg -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Fn",
            "       -> Params",
            "       -> Exp -> I64",
            "Exp -> UnaryExp",
            "   -> Id",
            "   -> I64",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case02() {
    let code = "
    qty: 7 8 9h;
    t: ([]sym: `a`b`b; col1: 1 2 3; col2: 1.0 2.0 3.0f; 4 5 6i; qty);
    df1: select sum col1+col2, newCol: col2 by sym from t where sym=`a;
    count df1
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Integers",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Table",
            "       -> ColExp -> RenameColExp",
            "           -> ColName",
            "           -> Syms",
            "       -> ColExp -> RenameColExp",
            "           -> ColName",
            "           -> Integers",
            "       -> ColExp -> RenameColExp",
            "           -> ColName",
            "           -> Floats",
            "       -> ColExp -> Integers",
            "       -> ColExp -> Id",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Query",
            "       -> SelectOp",
            "       -> SelectExp",
            "         -> ColExp -> UnaryQueryExp",
            "             -> Id",
            "             -> BinaryQueryExp",
            "               -> Id",
            "               -> BinaryOp",
            "               -> Id",
            "         -> ColExp -> RenameColExp",
            "             -> ColName",
            "             -> Id",
            "       -> ByExp -> ColExp -> Id",
            "       -> FromExp -> Id",
            "       -> WhereExp -> BinaryQueryExp",
            "           -> Id",
            "           -> BinaryOp",
            "           -> Sym",
            "Exp -> UnaryExp",
            "   -> Id",
            "   -> Id",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case02_01() {
    let code = "
    delete from df1 where col1=1;
    delete from df1;
    delete col1, col2, col3 from df1;
    select col1++col2 from df1;
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> Query",
            "   -> DeleteOp",
            "   -> FromExp -> Id",
            "   -> WhereExp -> BinaryQueryExp",
            "       -> Id",
            "       -> BinaryOp",
            "       -> I64",
            "Exp -> Query",
            "   -> DeleteOp",
            "   -> FromExp -> Id",
            "Exp -> Query",
            "   -> DeleteOp",
            "   -> ColNames",
            "     -> ColName",
            "     -> ColName",
            "     -> ColName",
            "   -> FromExp -> Id",
            "Exp -> Query",
            "   -> SelectOp",
            "   -> SelectExp -> ColExp -> BinaryQueryExp",
            "         -> Id",
            "         -> BinaryOp",
            "         -> Id",
            "   -> FromExp -> Id",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case03() {
    let code = "
    r1: eval (*; 9; 9);
    f: {[x;y] x - y};
    r2: eval[(`f; 9; 1)];
    t: timeit[(+; 1; 1); 1000];
    t
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> UnaryExp",
            "       -> Id",
            "       -> ListExp",
            "         -> BinaryOp",
            "         -> Exp -> I64",
            "         -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Fn",
            "       -> Params",
            "         -> Id",
            "         -> Id",
            "       -> Exp -> BinaryExp",
            "           -> Id",
            "           -> BinaryOp",
            "           -> Exp -> Id",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> FnCall",
            "       -> Id",
            "       -> Arg -> Exp -> ListExp",
            "             -> Exp -> Sym",
            "             -> Exp -> I64",
            "             -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> FnCall",
            "       -> Id",
            "       -> Arg -> Exp -> ListExp",
            "             -> BinaryOp",
            "             -> Exp -> I64",
            "             -> Exp -> I64",
            "       -> Arg -> Exp -> I64",
            "Exp -> Id",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case04() {
    let code = "
    d: {a: 1; b: 2; c: 3;};
    d2: {a: 4; b: 5; c: 6};
    d3: `a`b`c!(7;8;9);
    d[`d]:9;
    r1: d2[`c];
    d3[`c] + sum d[`a`d]
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Dict",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Dict",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "       -> KeyValueExp",
            "         -> Id",
            "         -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> BinaryExp",
            "       -> Syms",
            "       -> BinaryOp",
            "       -> Exp -> ListExp",
            "           -> Exp -> I64",
            "           -> Exp -> I64",
            "           -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> FnCall",
            "     -> Id",
            "     -> Arg -> Exp -> Sym",
            "   -> Exp -> I64",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> FnCall",
            "       -> Id",
            "       -> Arg -> Exp -> Sym",
            "Exp -> BinaryExp",
            "   -> FnCall",
            "     -> Id",
            "     -> Arg -> Exp -> Sym",
            "   -> BinaryOp",
            "   -> Exp -> UnaryExp",
            "       -> Id",
            "       -> FnCall",
            "         -> Id",
            "         -> Arg -> Exp -> Syms",
            "EOI",
            ""
        ],
        actual
    )
}

#[test]
fn parse_case05() {
    let code = "
    f: {[date]
        if[date>2020.01.01;
            'error;
            :date;
            date: date + 1;
        ];
        2020.01.01
    };
    r1: f 2024.04.01;
    r2: f 2019.01.01;
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> Fn",
            "       -> Params -> Id",
            "       -> IfExp",
            "         -> Exp -> BinaryExp",
            "             -> Id",
            "             -> BinaryOp",
            "             -> Exp -> Date",
            "         -> RaiseExp -> Exp -> Id",
            "         -> ReturnExp -> Exp -> Id",
            "         -> Exp -> AssignmentExp",
            "             -> Id",
            "             -> Exp -> BinaryExp",
            "                 -> Id",
            "                 -> BinaryOp",
            "                 -> Exp -> I64",
            "       -> Exp -> Date",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> UnaryExp",
            "       -> Id",
            "       -> Date",
            "Exp -> AssignmentExp",
            "   -> Id",
            "   -> Exp -> UnaryExp",
            "       -> Id",
            "       -> Date",
            "EOI",
            ""
        ],
        actual
    )
}

// #[test]
// fn parse_case06() {
//     let code = "
//     10 {[x]x ++ sum -2#x}/ 1 1
//     ";
//     let pairs = match JParser::parse(Rule::Program, code) {
//         Ok(p) => p,
//         Err(e) => {
//             eprintln!("{}", e);
//             panic!("failed to parse")
//         }
//     };
//     let binding = pretty_format_rules(pairs);
//     let actual: Vec<&str> = binding.split("\n").collect();
//     assert_eq!(
//         vec![
//             "Exp -> BinaryIteratorExp",
//             "   -> I64",
//             "   -> Fn",
//             "     -> Params -> Id",
//             "     -> Exp -> BinaryExp",
//             "         -> Id",
//             "         -> BinaryOp",
//             "         -> Exp -> UnaryExp",
//             "             -> UnaryKeyword",
//             "             -> Exp -> BinaryExp",
//             "                 -> I64",
//             "                 -> BinaryOp",
//             "                 -> Exp -> Id",
//             "   -> Iterator",
//             "   -> Exp -> I64s",
//             "EOI",
//             ""
//         ],
//         actual
//     )
// }

#[test]
fn parse_case07() {
    let code = "
    try [
        a: 1 + `a;
    ] catch [
        err ~ \"type\";
    ]
    ";
    let pairs = match ChiliParser::parse(Rule::Program, code) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{}", e);
            panic!("failed to parse")
        }
    };
    let binding = pretty_format_rules(pairs);
    let actual: Vec<&str> = binding.split("\n").collect();
    assert_eq!(
        vec![
            "TryExp",
            " -> TryStatements -> Exp -> AssignmentExp",
            "       -> Id",
            "       -> Exp -> BinaryExp",
            "           -> I64",
            "           -> BinaryOp",
            "           -> Exp -> Sym",
            " -> CatchStatements -> Exp -> BinaryExp",
            "       -> Id",
            "       -> BinaryOp",
            "       -> Exp -> String",
            "EOI",
            ""
        ],
        actual
    )
}
