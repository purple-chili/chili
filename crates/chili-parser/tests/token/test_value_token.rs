use chili_parser::Token;
use chili_parser::utils::print_errs;
use chumsky::prelude::*;
use std::path::PathBuf;

#[test]
fn test_value_token() {
    let mut src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src_path.push("tests/chili/src/value.chi");
    let src_path = src_path.to_str().unwrap();
    let src = std::fs::read_to_string(src_path).unwrap();
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
            Token::Punc(_) => false,
            _ => true,
        })
        .map(|(token, span)| format!("{}|{}", token, span.end - span.start))
        .collect::<Vec<String>>();
    assert_eq!(
        tokens,
        vec![
            "Comment|8",
            "Null'0n'|2",
            "Float'0nf32'|5",
            "Float'0ne'|3",
            "Float'0n 0nf32'|8",
            "Comment|18",
            "Float'0w -0w 0we'|10",
            "Float'-0wf'|4",
            "Float'0wf32'|5",
            "Float'-0wf64'|6",
            "Comment|18",
            "Float'0wf11'|5",
            "Comment|8",
            "Bool'0b'|2",
            "Bool'1b'|2",
            "Bool'true'|4",
            "Bool'false'|5",
            "Bool'01000b'|6",
            "Bool'1111b'|5",
            "Comment|7",
            "Hex'0x01ff03'|8",
            "Hex'0x01'|4",
            "Comment|13",
            "Hex'0x01f'|5",
            "Comment|13",
            "Timestamp'2025.01.01D 2025.01.01D12:34:56 2025.01.01D12:34:56.123456789'|61",
            "Timestamp'2025.01.01D\n2025.01.01D12:34:56'|31",
            "Comment|12",
            "Datetime'2025.01.01T12:34:56 2025.01.01T12:34:56.123'|43",
            "Comment|21",
            "Datetime'2025.01.01T12:34:56.123456789'|29",
            "Comment|12",
            "Duration'1D 0D12:34:56 0D12:34:56.123 0D12:34:56.123456789'|49",
            "Duration'-1D 0D00:00:00'|14",
            "Duration'0D -12D'|7",
            "Duration'-123D'|5",
            "Comment|39",
            "Date'2025.01.01 262144.12.31'|23",
            "Comment|14",
            "Date'2025.22.31'|10",
            "Comment|8",
            "Time'00:12:34 00:12:34.123 00:12:34.123456789'|40",
            "Comment|14",
            "Time'00:92:34.123456789'|18",
            "Comment|7",
            "Int'1 2 3'|5",
            "Int'-1 0 0n 3 0n'|12",
            "Int'0n 0n 0n 1'|10",
            "Int'1'|1",
            "Int'-5'|2",
            "Int'0 0ni32'|7",
            "Int'0n 1 3u8'|8",
            "Int'3u8'|3",
            "Comment|9",
            "Float'0n 1.0 0n -0w 1 3'|17",
            "Float'0n 0n 0n 0ne'|12",
            "Float'1.0 0n 2f32'|11",
            "Float'1.0E10 0n 0w'|12",
            "Float'0w'|2",
            "Float'-0w'|3",
            "Float'3.14e-10'|8",
            "Float'3.0E10'|6",
            "Comment|10",
            "Symbol'`a'|2",
            "Symbol'`a:_//-`b`c'|11",
            "Symbol'````c'|5",
            "Symbol'`c````'|6",
            "Comment|10",
            "Str'abc'|5",
            "Str'\\n'|4",
            "Str'\\\"abc\\\"'|9",
            "Str'\\n\\t\\\\\\r'|10",
            "Str'辣椒'|8"
        ]
    )
}
