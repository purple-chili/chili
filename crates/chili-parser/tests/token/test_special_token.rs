use std::path::PathBuf;

use crate::assert_eq_tokens;

#[test]
fn test_special_token() {
    let mut src_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    src_path.push("tests/chili/src/special.chi");
    let src_path = src_path.to_str().unwrap();
    let src = std::fs::read_to_string(src_path).unwrap();
    let expected = vec![
        "Comment|21",
        "Int'1'|1",
        "Int'-1'|2",
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
