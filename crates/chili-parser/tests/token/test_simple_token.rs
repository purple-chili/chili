use crate::assert_eq_tokens;
use chili_parser::Token;
use chumsky::Parser;

#[test]
fn line_comments_end_at_newline_or_end_of_input() {
    for src in [
        "//",
        "// comment",
        "// comment\n",
        "// comment\r\n",
        "// 雪 /* text",
    ] {
        let tokens = Token::lexer().parse(src).into_result().unwrap();
        assert_eq!(
            tokens,
            vec![(Token::Comment(src.into()), (0..src.len()).into())]
        );
    }
    let src = "// comment\n42";
    let tokens = Token::lexer().parse(src).into_result().unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[1], (Token::Int("42".into()), (11..13).into()));
}

#[test]
fn test_float() {
    assert_eq_tokens(
        "1 2 3 4 5f;",
        "repl.chi",
        vec!["Float'1 2 3 4 5f'|10", "Punc';'|1"],
        true,
    );
    assert_eq_tokens(
        "1 2 3 4 5f64;",
        "repl.chi",
        vec!["Float'1 2 3 4 5f64'|12", "Punc';'|1"],
        true,
    );
}

#[test]
fn test_windows_path() {
    assert_eq_tokens(
        "`C:\\Users\\chili\\Documents\\test.chi",
        "repl.chi",
        vec!["Symbol'`C:\\Users\\chili\\Documents\\test.chi'|34"],
        true,
    );
}

#[test]
fn test_short_duration() {
    assert_eq_tokens(
        "0D00:05;",
        "repl.chi",
        vec!["Duration'0D00:05'|7", "Punc';'|1"],
        true,
    );
    assert_eq_tokens(
        "0D00:05:00;",
        "repl.chi",
        vec!["Duration'0D00:05:00'|10", "Punc';'|1"],
        true,
    );
}
