use anyhow::Result;

use thiserror::Error;

use crate::{ArgType, SpicyObj, arg_type::raise_arg_type_err};

#[derive(Debug, Error)]
pub enum SpicyError {
    #[error("{0}")]
    Err(String),

    #[error("Failed to eval: {0}")]
    EvalErr(String),

    #[error("Err: {0}")]
    RaiseErr(String),

    #[error("Invalid Handle err {0}")]
    InvalidHandleErr(i64),

    #[error("Handle number {0} out of range, allowed range is 0..{1}")]
    HandleOutOfRangeErr(i64, usize),

    #[error("Parser err: {0}")]
    ParserErr(String),

    #[error("Failed to refer {0} from {1}")]
    MismatchedTypeErr(String, String),

    #[error("Unsupported unary op '{0}' for '{1}'")]
    UnsupportedUnaryOpErr(String, String),

    #[error("Unsupported binary op '{0}' between '{1}' and '{2}'")]
    UnsupportedBinaryOpErr(String, String, String),

    #[error("Unsupported J type '{0}' as query expression")]
    UnsupportedQueryJTypeErr(String),

    #[error("Unsupported query unary op '{0}'")]
    UnsupportedQueryUnaryOpErr(String),

    #[error("Unsupported query binary op '{0}'")]
    UnsupportedQueryBinaryOpErr(String),

    #[error("Not supported series type {0:?}.")]
    NotSupportedSeriesTypeErr(polars::datatypes::DataType),

    #[error("Not able to deserialize type code '{0}'")]
    NotAbleToDeserializeErr(u8),

    #[error("Not series")]
    NotSeriesErr(),

    #[error("Not yet implemented '{0}'")]
    NotYetImplemented(String),

    #[error("Expect {0} argument(s), {1} given")]
    MismatchedArgNumErr(usize, usize),

    #[error("Expect {0} argument(s) function, {1} argument(s) function given")]
    MismatchedArgNumFnErr(usize, usize),

    #[error("Expect '{0}' for '{1}' argument , got '{2}'")]
    MismatchedArgTypeErr(String, usize, String),

    #[error("Name '{0}' is not defined")]
    NameErr(String),

    #[error("Length error '{0}' vs '{1}'")]
    MismatchedLengthErr(usize, usize),

    #[error("Forbidden '{0}' keyword")]
    ForbiddenKeywordErr(String),

    #[error("OS Err: '{0}'")]
    OsErr(String),

    #[error("Failed to read lock '{0}'")]
    ReadLockErr(String),

    #[error("Failed to write lock '{0}'")]
    WriteLockErr(String),
    // <- serde
    #[error("Not able to serialize {0}")]
    NotAbleToSerializeErr(String),

    #[error("Not supported k type {0:?}.")]
    NotSupportedKTypeErr(u8),

    #[error("Not supported k list - k type {0:?}.")]
    NotSupportedKListErr(u8),

    #[error("Not supported nested list - k type {0:?}.")]
    NotSupportedKNestedListErr(u8),

    #[error("Not supported mixed list - expected k type {0:?}, but got {1:?}.")]
    NotSupportedKMixedListErr(u8, u8),

    #[error("{0}")]
    DeserializationErr(String),

    #[error("Not supported k operator - k value {0:?}.")]
    NotSupportedKOperatorErr(u8),

    #[error("Internal Server Error - {0:?}")]
    ServerErr(String),

    #[error("Length over i32::MAX.")]
    OverLengthErr(),

    #[error("Not supported polars nested list type {0:?}.")]
    NotSupportedPolarsNestedListTypeErr(polars::datatypes::DataType),
    // -> serde
    #[error("Requires '{0}' condition for this partitioned dataframe")]
    MissingParCondErr(String),
}

pub type SpicyResult<T> = Result<T, SpicyError>;

impl SpicyError {
    pub fn new_arg_type_err(args: &SpicyObj, arg_pos: usize, arg_type: &ArgType) -> Self {
        SpicyError::Err(raise_arg_type_err(args, arg_pos, arg_type))
    }
}

pub fn trace(source: &str, path: &str, pos: usize, msg: &str) -> String {
    if pos >= source.len() {
        return msg.to_string();
    }
    let mut start = 0;
    let mut r = 1;
    let mut display_col = 0;
    let mut chars = source.chars().peekable();
    let mut i = 0;
    while i < pos {
        match chars.next() {
            Some('\r') => {
                if let Some(&'\n') = chars.peek() {
                    chars.next();
                    i += 2;
                } else {
                    i += 1;
                }
                r += 1;
                display_col = 0;
                start = i;
            }
            Some('\n') => {
                i += 1;
                r += 1;
                display_col = 0;
                start = i;
            }
            Some(ch) => {
                i += ch.len_utf8();
                display_col += unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            }
            None => return msg.to_string(),
        }
    }
    let c = display_col + 1;
    let end = source[pos..]
        .find(['\n', '\r'])
        .map_or(source.len(), |i| pos + i);
    let line = &source[start..end];
    let underline = " ".repeat(display_col) + "^";

    format!(
        "--> {path}:{r}:{c}\n\
        \n\
        {line}\n\
        {underline}\n\
        {msg}"
    )
}

#[test]
fn display_trace() {
    let input = "1+1;\r\n1;\n`a + 1;";

    assert_eq!(
        trace(input, "", 12, "type"),
        ["--> :3:4", "", "`a + 1;", "   ^", "type"].join("\n")
    );
}

#[test]
fn display_trace_standalone_cr() {
    let input = "1+1;\r1;\r`a + 1;";

    assert_eq!(
        trace(input, "", 11, "type"),
        ["--> :3:4", "", "`a + 1;", "   ^", "type"].join("\n")
    );
}
