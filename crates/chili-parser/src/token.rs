use chumsky::prelude::*;
use indexmap::IndexMap;
use std::{cmp::Ordering, fmt};

pub type Span = SimpleSpan<usize, ()>;

pub trait Length {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Length for Span {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Eq)]
pub struct Position(pub usize, pub usize);

impl Ord for Position {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

const BINARY_OPERATORS: &[&str] = &[
    "as",
    "bottom",
    "corr",
    "cov0",
    "cov1",
    "cross",
    "differ",
    "div",
    "each",
    "emean",
    "equal",
    "estd",
    "evar",
    "explode",
    "extend",
    "fby",
    "gather",
    "hstack",
    "in",
    "intersect",
    "join",
    "like",
    "log",
    "matches",
    "mmax",
    "mmean",
    "mmedian",
    "mmin",
    "mod",
    "mskew",
    "mstd0",
    "mstd1",
    "msum",
    "mvar0",
    "mvar1",
    "pad",
    "parallel",
    "pow",
    "quantile",
    "reshape",
    "rotate",
    "round",
    "set",
    "shift",
    "split",
    "ss",
    "ssr",
    "sub",
    "top",
    "union",
    "upsert",
    "vstack",
    "within",
    "wmean",
    "wsum",
    "xasc",
    "xbar",
    "xdesc",
    "xrename",
    "xreorder",
];

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    Null(String),
    Bool(String),
    Hex(String),
    Timestamp(String),
    Datetime(String),
    Duration(String),
    Date(String),
    Time(String),
    Int(String),
    Float(String),
    Symbol(String),
    Str(String),
    Column(String),
    Op(String),
    Id(String),
    Comment(String),
    Punc(char),
    // delayed argument
    DelayedArg,
    // nil expression at the end of the block statements
    Nil,
    Fn,
    If,
    Else,
    While,
    Try,
    Catch,
    Return,
    Raise,
    Select,
    Update,
    Delete,
    By,
    From,
    Where,
    Limit,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Null(n) => write!(f, "Null'{n}'"),
            Token::Hex(h) => write!(f, "Hex'{h}'"),
            Token::Timestamp(t) => write!(f, "Timestamp'{t}'"),
            Token::Datetime(d) => write!(f, "Datetime'{d}'"),
            Token::Duration(d) => write!(f, "Duration'{d}'"),
            Token::Date(d) => write!(f, "Date'{d}'"),
            Token::Time(t) => write!(f, "Time'{t}'"),
            Token::Bool(x) => write!(f, "Bool'{x}'"),
            Token::Int(n) => write!(f, "Int'{n}'"),
            Token::Float(n) => write!(f, "Float'{n}'"),
            Token::Symbol(s) => write!(f, "Symbol'{s}'"),
            Token::Str(s) => write!(f, "Str'{s}'"),
            Token::Column(c) => write!(f, "Column'{c}'"),
            Token::Op(s) => write!(f, "Op'{s}'"),
            Token::Punc(c) => write!(f, "Punc'{c}'"),
            Token::Id(s) => write!(f, "Id'{s}'"),
            Token::Comment(_) => write!(f, "Comment"),
            Token::DelayedArg => write!(f, "DelayedArg"),
            Token::Nil => write!(f, "Nil"),
            Token::Fn => write!(f, "Fn"),
            Token::If => write!(f, "If"),
            Token::Else => write!(f, "Else"),
            Token::While => write!(f, "While"),
            Token::Try => write!(f, "Try"),
            Token::Catch => write!(f, "Catch"),
            Token::Return => write!(f, "Return"),
            Token::Raise => write!(f, "Raise"),
            Token::Select => write!(f, "Select"),
            Token::Update => write!(f, "Update"),
            Token::Delete => write!(f, "Delete"),
            Token::By => write!(f, "By"),
            Token::From => write!(f, "From"),
            Token::Where => write!(f, "Where"),
            Token::Limit => write!(f, "Limit"),
        }
    }
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Token::Fn
                | Token::If
                | Token::Else
                | Token::While
                | Token::Try
                | Token::Catch
                | Token::Return
                | Token::Raise
                | Token::Select
                | Token::Update
                | Token::Delete
                | Token::By
                | Token::From
                | Token::Where
                | Token::Limit
        )
    }

    pub fn is_operator(&self) -> bool {
        matches!(self, Token::Op(_))
    }

    pub fn str(&self) -> Option<&str> {
        let str = match self {
            Token::Null(n) => n,
            Token::Hex(h) => h,
            Token::Timestamp(t) => t,
            Token::Datetime(d) => d,
            Token::Duration(d) => d,
            Token::Date(d) => d,
            Token::Time(t) => t,
            Token::Bool(b) => b,
            Token::Int(n) => n,
            Token::Float(n) => n,
            Token::Symbol(s) => s,
            Token::Str(s) => s,
            Token::Column(c) => c,
            Token::Op(s) => s,
            Token::Id(s) => s,
            Token::Comment(c) => c,
            _ => return None,
        };
        Some(str)
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Token::Str(_))
    }

    pub fn is_column(&self) -> bool {
        matches!(self, Token::Column(_))
    }

    pub fn is_symbol(&self) -> bool {
        matches!(self, Token::Symbol(_))
    }

    // parse token
    pub fn lexer<'a>() -> impl Parser<
        'a,
        &'a str,
        Vec<(Token, SimpleSpan<usize, ()>)>,
        extra::Err<Rich<'a, char, SimpleSpan<usize, ()>>>,
    > {
        // 0n 0ne 0nf32
        let null_ = just("0n")
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(just("0n"))
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Null(s.to_string()))
            .boxed();

        // 0xFFFF
        let hex = just("0x")
            .then(text::digits(16))
            .to_slice()
            .map(|s: &str| Token::Hex(s.to_string()))
            .boxed();

        let bool_ = text::digits(2)
            .at_least(1)
            .then(just('b'))
            .to_slice()
            .map(|s: &str| Token::Bool(s.to_string()))
            .boxed();

        // parse infinity float, 0we, 0wf, 0wf32, 0wf64
        let infinity = just('-')
            .or_not()
            .then(just("0w"))
            .to_slice()
            .map(|s: &str| Token::Float(s.to_string()))
            .boxed();

        // parse temporals
        let date = text::digits(10)
            .at_least(4)
            .then(just('.'))
            .then(text::digits(10).exactly(2))
            .then(just('.'))
            .then(text::digits(10).exactly(2))
            .to_slice()
            .map(|s: &str| Token::Date(s.to_string()))
            .boxed();

        let dates = date
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(date.clone())
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Date(s.to_string()))
            .boxed();

        let time = text::digits(10)
            .exactly(2)
            .then(just(':'))
            .then(text::digits(10).exactly(2))
            .then(just(':'))
            .then(text::digits(10).exactly(2))
            .then(
                just('.')
                    .then(text::digits(10).at_most(9).or_not())
                    .or_not(),
            )
            .to_slice()
            .map(|s: &str| Token::Time(s.to_string()))
            .boxed();

        let times = time
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(time.clone())
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Time(s.to_string()))
            .boxed();

        let timestamp = date
            .clone()
            .then(just('D'))
            .then(time.clone().or_not())
            .to_slice()
            .map(|s: &str| Token::Timestamp(s.to_string()))
            .boxed();

        let timestamps = timestamp
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(timestamp.clone())
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Timestamp(s.to_string()))
            .boxed();

        let datetime = date
            .clone()
            .then(just('T'))
            .then(time.clone().or_not())
            .to_slice()
            .map(|s: &str| Token::Datetime(s.to_string()))
            .boxed();

        let datetimes = datetime
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(datetime.clone())
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Datetime(s.to_string()))
            .boxed();

        let duration = just('-')
            .or_not()
            .then(text::int(10).then(just('D')).then(time.clone().or_not()))
            .to_slice()
            .map(|s: &str| Token::Duration(s.to_string()))
            .boxed();

        let durations = duration
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(duration.clone())
                    .repeated(),
            )
            .to_slice()
            .map(|s: &str| Token::Duration(s.to_string()))
            .boxed();

        // parse numbers
        let int = just('-')
            .or_not()
            .then(text::int(10))
            .to_slice()
            .map(|s: &str| Token::Int(s.to_string()))
            .boxed();

        let int_follow_null = just("0n")
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(just("0n"))
                    .repeated(),
            )
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(int.clone())
                    .repeated()
                    .at_least(1),
            )
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(int.clone().or(null_.clone()))
                    .repeated(),
            )
            .then(one_of("hiu").then(text::digits(10).or_not()).or_not())
            .to_slice()
            .map(|s: &str| Token::Int(s.to_string()))
            .boxed();

        let ints = int
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(null_.clone().or(int.clone()))
                    .repeated(),
            )
            .then(one_of("hiu").then(text::digits(10).or_not()).or_not())
            .to_slice()
            .map(|s: &str| Token::Int(s.to_string()))
            .boxed();

        let typed_int = null_
            .clone()
            .or(int.clone())
            .then(one_of("hiu").then(text::digits(10).or_not()))
            .to_slice()
            .map(|s: &str| Token::Int(s.to_string()))
            .boxed();

        let finity = just('-')
            .or_not()
            .then(
                text::int(10)
                    .then(just('.').then(text::digits(10).at_least(1)))
                    .then(
                        one_of("eE")
                            .then(just('-').or_not())
                            .then(text::int(10))
                            .or_not(),
                    ),
            )
            .to_slice()
            .map(|s: &str| Token::Float(s.to_string()))
            .boxed();

        let float = infinity.clone().or(finity.clone()).boxed();

        let float_follow_null = just("0n")
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(just("0n"))
                    .repeated(),
            )
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(float.clone())
                    .repeated()
                    .at_least(1),
            )
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(null_.clone().or(float.clone()).or(int.clone()))
                    .repeated(),
            )
            .then(one_of("ef").then(text::digits(10).or_not()).or_not())
            .to_slice()
            .map(|s: &str| Token::Float(s.to_string()))
            .boxed();

        let floats = float
            .clone()
            .then(
                any()
                    .filter(|c: &char| c.is_whitespace())
                    .then(null_.clone().or(float.clone()).or(int.clone()))
                    .repeated(),
            )
            .then(one_of("ef").then(text::digits(10).or_not()).or_not())
            .to_slice()
            .map(|s: &str| Token::Float(s.to_string()))
            .boxed();

        let typed_float = null_
            .clone()
            .or(float.clone())
            .or(int)
            .then(one_of("ef").then(text::digits(10).or_not()))
            .to_slice()
            .map(|s: &str| Token::Float(s.to_string()))
            .boxed();

        let double_quote_escape = just('\\').then_ignore(one_of("\\/\"bfnrt"));

        let single_quote_escape = just('\\').then_ignore(one_of("\\/'bfnrt"));

        // parse strings
        let str_ = none_of("\\\"")
            .or(double_quote_escape)
            .repeated()
            .to_slice()
            .delimited_by(just('"'), just('"'))
            .map(|s: &str| Token::Str(s.to_string()))
            .boxed();

        let column = none_of("\\'")
            .or(single_quote_escape)
            .repeated()
            .to_slice()
            .delimited_by(just('\''), just('\''))
            .map(|s: &str| Token::Column(s.to_string()))
            .boxed();

        // parse symbols
        let sym = just('`')
            .then(
                any()
                    .filter(|c: &char| {
                        c.is_ascii_alphanumeric()
                            || *c == '.'
                            || *c == ':'
                            || *c == '/'
                            || *c == '_'
                            || *c == '-'
                    })
                    .repeated(),
            )
            .repeated()
            .at_least(1)
            .to_slice()
            .map(|s: &str| Token::Symbol(s.to_string()))
            .boxed();

        // parse identifiers and keywords
        let id = text::ascii::ident().map(|id: &str| match id {
            "function" => Token::Fn,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "try" => Token::Try,
            "catch" => Token::Catch,
            "return" => Token::Return,
            "raise" => Token::Raise,
            "true" => Token::Bool("true".to_string()),
            "false" => Token::Bool("false".to_string()),
            "select" => Token::Select,
            "update" => Token::Update,
            "delete" => Token::Delete,
            "by" => Token::By,
            "from" => Token::From,
            "where" => Token::Where,
            "limit" => Token::Limit,
            _ => {
                if BINARY_OPERATORS.contains(&id) {
                    Token::Op(id.to_string())
                } else {
                    Token::Id(id.to_string())
                }
            }
        });

        let global_id = just('.')
            .then(id)
            .repeated()
            .at_least(1)
            .to_slice()
            .map(|s: &str| Token::Id(s.to_string()))
            .boxed();

        let local_id = id
            .then(just('.').then(id).repeated().at_least(1))
            .to_slice()
            .map(|s: &str| Token::Id(s.to_string()))
            .boxed();

        // parse operators
        let op = one_of("+*-/!<>=:.$?@!~_&|#^%~")
            .repeated()
            .at_least(1)
            .to_slice()
            .map(|s: &str| Token::Op(s.to_string()))
            .boxed();

        // parse control characters (delimiters, semicolons, etc.)
        let punc = one_of("()[]{};,").map(Token::Punc);

        let comment = just("//")
            .then(any().and_is(just('\n').not()).repeated())
            .then(just('\n'))
            .to_slice()
            .map(|s: &str| Token::Comment(s.to_string()))
            .boxed();

        let block_comment = just("/*")
            .then(any().and_is(just("*/").not()).repeated().then(just("*/")))
            .map(|s| Token::Comment(s.0.to_string()))
            .boxed();

        // the priority of tokens
        let token = choice((
            comment,
            block_comment,
            typed_float,
            typed_int,
            float_follow_null,
            int_follow_null,
            null_,
            hex,
            bool_,
            durations,
            timestamps,
            datetimes,
            dates,
            times,
            floats,
            ints,
            sym,
            str_,
            column,
            global_id,
            local_id,
            op,
            punc,
            id,
        ));

        token
            .map_with(|token, e| (token, e.span()))
            .padded()
            // If we encounter an error, skip and attempt to lex the next character as a token instead
            .recover_with(skip_then_retry_until(any().ignored(), end()))
            .repeated()
            .collect()
    }
}

pub fn calculate_line_col(tokens: &[(Token, Span)], src: &str) -> IndexMap<usize, (usize, usize)> {
    let mut line_col_map = IndexMap::new();

    let mut chars = src.chars().peekable();

    // language server line and column start from 0
    let mut line_col = (0, 0);

    let mut i = 0;

    line_col_map.insert(0, (line_col.0, line_col.1));

    let mut j = 1;

    while let Some(c) = chars.next() {
        match c {
            '\r' => {
                if let Some(&'\n') = chars.peek() {
                    chars.next();

                    i += 2;

                    line_col = (line_col.0 + 1, 0);
                } else {
                    i += 1;
                    line_col = (line_col.0, line_col.1 + 1);
                }
            }
            '\n' => {
                i += 1;
                line_col = (line_col.0 + 1, 0);
            }
            _ => {
                i += c.len_utf8();
                line_col = (line_col.0, line_col.1 + 1);
            }
        }

        while j < tokens.len() * 2 {
            let token = tokens[j / 2].1;
            if token.start == i {
                line_col_map.insert(i, (line_col.0, line_col.1));
                j += 1;
                break;
            }
            if token.end == i {
                line_col_map.insert(i, (line_col.0, line_col.1));
                j += 1;
            } else {
                break;
            }
        }
    }
    // add the last line and column
    line_col_map.insert(src.len(), (line_col.0, line_col.1));
    line_col_map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_line_col() {
        let src = [
            "upd: function(table, data) {",
            "  table upsert data;",
            "  tick(1);",
            "};",
        ]
        .join("\n");
        let tokens = Token::lexer().parse(&src).unwrap();
        let line_col_map = calculate_line_col(&tokens, &src);
        assert_eq!(line_col_map.len(), 29);
    }
}
