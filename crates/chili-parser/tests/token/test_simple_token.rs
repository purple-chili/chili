use crate::assert_eq_tokens;

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
