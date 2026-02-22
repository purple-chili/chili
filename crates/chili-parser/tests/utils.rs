use chili_parser::Token;
use chili_parser::utils::print_errs;
use chumsky::prelude::*;

#[track_caller]
pub fn assert_eq_chili_expr(src: &str, expected: Vec<&str>) {
    let (tokens, _) = chili_parser::Token::lexer().parse(src).into_output_errors();
    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(t, _)| !matches!(t, Token::Comment(_)))
        .collect::<Vec<_>>();
    let (program, errs) = chili_parser::Expr::parser_chili()
        .parse(
            tokens
                .as_slice()
                .map((src.len()..src.len()).into(), |(t, s)| (t, s)),
        )
        .into_output_errors();

    if errs.len() > 0 {
        print_errs(errs, "repl", src);
    }

    assert_eq!(program.unwrap().pretty_print(0), expected);
}

#[track_caller]
pub fn should_fail_chili_expr(src: &str, msg: &str) {
    let (tokens, _) = chili_parser::Token::lexer().parse(src).into_output_errors();
    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(t, _)| !matches!(t, Token::Comment(_)))
        .collect::<Vec<_>>();
    let (program, errs) = chili_parser::Expr::parser_chili()
        .parse(
            tokens
                .as_slice()
                .map((src.len()..src.len()).into(), |(t, s)| (t, s)),
        )
        .into_output_errors();

    let has_err = errs.len() > 0;

    if errs.len() > 0 {
        print_errs(errs, "repl", src);
    }

    if program.is_some() {
        println!("Program:");
        for stmt in program.unwrap().pretty_print(2) {
            println!("{}", stmt);
        }
    }

    assert!(has_err, "Should fail - {}", msg)
}

#[track_caller]
pub fn assert_eq_pepper_expr(src: &str, expected: Vec<&str>) {
    let (tokens, _) = chili_parser::Token::lexer().parse(src).into_output_errors();
    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(t, _)| !matches!(t, Token::Comment(_)))
        .collect::<Vec<_>>();
    let (program, errs) = chili_parser::Expr::parser_pepper()
        .parse(
            tokens
                .as_slice()
                .map((src.len()..src.len()).into(), |(t, s)| (t, s)),
        )
        .into_output_errors();

    if errs.len() > 0 {
        print_errs(errs, "repl", src);
    }

    assert_eq!(program.unwrap().pretty_print(0), expected);
}

#[track_caller]
pub fn should_fail_pepper_expr(src: &str, msg: &str) {
    let (tokens, _) = chili_parser::Token::lexer().parse(src).into_output_errors();
    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(t, _)| !matches!(t, Token::Comment(_)))
        .collect::<Vec<_>>();
    let (program, errs) = chili_parser::Expr::parser_pepper()
        .parse(
            tokens
                .as_slice()
                .map((src.len()..src.len()).into(), |(t, s)| (t, s)),
        )
        .into_output_errors();

    let has_err = errs.len() > 0;

    if errs.len() > 0 {
        print_errs(errs, "repl", src);
    }

    if program.is_some() {
        println!("Program:");
        for stmt in program.unwrap().pretty_print(2) {
            println!("{}", stmt);
        }
    }

    assert!(has_err, "Should fail - {}", msg)
}

#[track_caller]
pub fn assert_eq_tokens(src: &str, src_path: &str, expected: Vec<&str>, include_punc: bool) {
    let (tokens, errs) = chili_parser::Token::lexer()
        .parse(&src)
        .into_output_errors();

    if errs.len() > 0 {
        print_errs(errs, src_path, &src);
    }

    let tokens = tokens
        .unwrap()
        .into_iter()
        .filter(|(token, _)| match token {
            Token::Punc(_) => include_punc,
            _ => true,
        })
        .map(|(token, span)| format!("{}|{}", token, span.end - span.start))
        .collect::<Vec<String>>();
    assert_eq!(tokens, expected);
}
