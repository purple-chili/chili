use chili_parser::{Expr, Token};
use chumsky::prelude::*;
use reedline::{ValidationResult, Validator};
pub struct ChiliValidator {
    pub use_chili_syntax: bool,
}

impl Validator for ChiliValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        if line.ends_with(";") {
            return ValidationResult::Complete;
        }

        let (tokens, errs) = Token::lexer().parse(line).into_output_errors();
        if !errs.is_empty() {
            return ValidationResult::Complete;
        }
        let tokens = tokens.unwrap();
        let (_, errs) = if self.use_chili_syntax {
            Expr::parser_chili()
                .parse(
                    tokens
                        .as_slice()
                        .map((line.len()..line.len()).into(), |(t, s)| (t, s)),
                )
                .into_output_errors()
        } else {
            Expr::parser_pepper()
                .parse(
                    tokens
                        .as_slice()
                        .map((line.len()..line.len()).into(), |(t, s)| (t, s)),
                )
                .into_output_errors()
        };

        if errs.is_empty() {
            ValidationResult::Complete
        } else {
            let err_end = errs.last().unwrap().span().end;
            if err_end == line.len() {
                ValidationResult::Incomplete
            } else {
                ValidationResult::Complete
            }
        }
    }
}
