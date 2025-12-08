#[cfg(not(feature = "vintage"))]
use chili_grammar::{ChiliParser, Rule};
#[cfg(feature = "vintage")]
use chili_vintage_grammar::{ChiliParser, Rule};
use pest::{Parser, error::InputLocation};
use reedline::{ValidationResult, Validator};

pub struct ChiliValidator {}

impl Validator for ChiliValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        if line.ends_with(";") {
            return ValidationResult::Complete;
        }

        match ChiliParser::parse(Rule::Program, line) {
            Ok(_) => ValidationResult::Complete,
            Err(e) => {
                let end = match e.location {
                    InputLocation::Pos(pos) => pos,
                    InputLocation::Span(span) => span.1,
                };
                if end == line.len() {
                    ValidationResult::Incomplete
                } else {
                    ValidationResult::Complete
                }
            }
        }
    }
}
