use std::path::PathBuf;

use crate::assert_eq_tokens;
use chili_parser::Token;
use chumsky::Parser;

#[test]
fn adjacent_negative_literals_become_subtraction_with_correct_spans() {
    for (source, expected) in [
        (
            "5-1",
            vec![
                (Token::Int("5".into()), (0..1).into()),
                (Token::Op("-".into()), (1..2).into()),
                (Token::Int("1".into()), (2..3).into()),
            ],
        ),
        (
            "n-1 -2",
            vec![
                (Token::Id("n".into()), (0..1).into()),
                (Token::Op("-".into()), (1..2).into()),
                (Token::Int("1 -2".into()), (2..6).into()),
            ],
        ),
    ] {
        assert_eq!(
            Token::lexer().parse(source).into_result().unwrap(),
            expected,
            "{source}"
        );
    }
    for (source, expected) in [
        ("n-1h", Token::Int("1h".into())),
        ("n-1.5", Token::Float("1.5".into())),
        ("n-1.0e-3", Token::Float("1.0e-3".into())),
        ("n-0w", Token::Float("0w".into())),
        ("n-1D00:00:00", Token::Duration("1D00:00:00".into())),
    ] {
        let tokens = Token::lexer().parse(source).into_result().unwrap();
        assert_eq!(tokens.len(), 3, "{source}");
        assert_eq!(tokens[1], (Token::Op("-".into()), (1..2).into()));
        assert_eq!(tokens[2], (expected, (2..source.len()).into()));
    }
}

#[test]
fn signs_in_literals_arguments_and_vectors_are_preserved() {
    for source in [
        "-1",
        "-9223372036854775808",
        "1 -1",
        "-1 -2",
        "-1.0e-3",
        "-1D00:00:00",
    ] {
        let tokens = Token::lexer().parse(source).into_result().unwrap();
        assert_eq!(tokens.len(), 1, "{source}");
        assert_eq!(tokens[0].0.str(), Some(source));
    }
    for (source, last) in [
        ("f -1", "-1"),
        ("n*-1", "-1"),
        ("n - -1", "-1"),
        ("x:-1", "-1"),
    ] {
        let tokens = Token::lexer().parse(source).into_result().unwrap();
        assert_eq!(
            tokens.last().unwrap().0,
            Token::Int(last.into()),
            "{source}"
        );
    }
    for source in ["\"a-1\"", "`a-1", "// a-1"] {
        assert_eq!(
            Token::lexer().parse(source).into_result().unwrap().len(),
            1,
            "{source}"
        );
    }
}

#[test]
fn operator_repetition_only_consumes_matching_characters() {
    for op in ["**", "++", "--", "&&", "||", "??", "***"] {
        let tokens = Token::lexer().parse(op).into_result().unwrap();
        assert_eq!(tokens, vec![(Token::Op(op.into()), (0..op.len()).into())]);
    }
    for src in ["*-", "+-", "*+", "+*", ":-"] {
        let tokens = Token::lexer().parse(src).into_result().unwrap();
        assert_eq!(tokens.len(), 2, "{src}");
        for (index, (token, span)) in tokens.iter().enumerate() {
            assert_eq!(token, &Token::Op(src[index..index + 1].into()));
            assert_eq!(*span, (index..index + 1).into());
        }
    }
}

#[test]
fn mixed_comparisons_remain_single_tokens() {
    for op in ["!=", "<=", ">="] {
        let tokens = Token::lexer().parse(op).into_result().unwrap();
        assert_eq!(tokens, vec![(Token::Op(op.into()), (0..op.len()).into())]);
    }
}

#[test]
fn negative_literals_follow_operators_without_spaces() {
    for op in ["*", "+", "++", ":", "!=", "<=", ">="] {
        let src = format!("n{op}-1");
        let tokens = Token::lexer().parse(&src).into_result().unwrap();
        assert_eq!(
            tokens,
            vec![
                (Token::Id("n".into()), (0..1).into()),
                (Token::Op(op.into()), (1..1 + op.len()).into()),
                (Token::Int("-1".into()), (1 + op.len()..src.len()).into()),
            ]
        );
    }
}

#[test]
fn test_special_token() {
    let mut src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src_path.push("tests/chili/src/special.chi");
    let src_path = src_path.to_str().unwrap();
    let src = std::fs::read_to_string(src_path).unwrap();
    let expected = vec![
        "Comment|21",
        "Int'1'|1",
        "Op'-'|1",
        "Int'1'|1",
        "Punc';'|1",
        "Int'1 -1'|4",
        "Punc';'|1",
        "Int'1'|1",
        "Op'-'|1",
        "Int'1'|1",
        "Punc';'|1",
        "Int'1'|1",
        "Op'-'|1",
        "Int'-1'|2",
        "Punc';'|1",
        "Comment|22",
        "Int'1'|1",
        "Op'--'|2",
        "Int'1'|1",
        "Punc';'|1",
        "Int'1'|1",
        "Op'--'|2",
        "Int'1'|1",
        "Punc';'|1",
        "Comment|15",
        "Id'f'|1",
        "Punc'('|1",
        "Punc','|1",
        "Int'1'|1",
        "Punc','|1",
        "Int'2'|1",
        "Punc')'|1",
        "Punc';'|1",
        "Id'f'|1",
        "Punc'('|1",
        "Punc','|1",
        "Punc','|1",
        "Int'3'|1",
        "Punc')'|1",
        "Punc';'|1",
        "Id'f'|1",
        "Punc'('|1",
        "Int'1'|1",
        "Punc','|1",
        "Punc','|1",
        "Punc')'|1",
        "Punc';'|1",
    ];
    assert_eq_tokens(&src, src_path, expected, true);
}
