use std::sync::Arc;

use chili_core::EngineState;
use reedline::{Completer, Span, Suggestion};

pub(crate) struct ChiliCompleter {
    state: Arc<EngineState>,
}

impl ChiliCompleter {
    pub fn new(state: &Arc<EngineState>) -> Self {
        Self {
            state: Arc::clone(state),
        }
    }
}

impl Completer for ChiliCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let trimmed_line = line.trim_end();
        let last_word = trimmed_line.split_whitespace().last().unwrap_or("");
        if trimmed_line.len() < line.len() || last_word.is_empty() {
            return vec![];
        }
        let suggestions = match self.state.get_displayed_vars() {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        suggestions
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
