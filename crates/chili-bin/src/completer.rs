use std::collections::HashMap;

use chili_core::EngineState;
use reedline::{Completer, Span, Suggestion};

pub(crate) struct ChiliCompleter {
    suggestions: HashMap<String, String>,
}

impl ChiliCompleter {
    pub fn new(state: &EngineState) -> Self {
        let suggestions = state.get_displayed_vars().unwrap();
        Self { suggestions }
    }
}

impl Completer for ChiliCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let trimmed_line = line.trim_end();
        let last_word = trimmed_line.split_whitespace().last().unwrap_or("");
        if trimmed_line.len() < line.len() || last_word.is_empty() {
            return vec![];
        }
        self.suggestions
            .iter()
            .filter(|(s, _)| s.starts_with(last_word))
            .map(|(_, d)| Suggestion {
                value: d.to_string(),
                description: Some(d.to_string()),
                span: Span::new(pos - last_word.len(), pos),
                ..Default::default()
            })
            .collect()
    }
}
