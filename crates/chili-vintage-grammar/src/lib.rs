use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "chili.pest"]
pub struct ChiliParser;
