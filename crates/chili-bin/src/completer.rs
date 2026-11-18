use std::sync::Arc;

use chili_core::EngineState;
use reedline::{Completer, CompletionResult, Span, Suggestion};

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
    fn complete(&mut self, line: &str, pos: usize) -> CompletionResult {
        let trimmed_line = line.trim_end();
        let last_word = trimmed_line.split_whitespace().last().unwrap_or("");
        if trimmed_line.len() < line.len() || last_word.is_empty() {
            return CompletionResult::fresh(Vec::<Suggestion>::new());
        }
        let suggestions = match self.state.get_displayed_vars() {
            Ok(s) => s,
            Err(_) => return CompletionResult::fresh(Vec::<Suggestion>::new()),
        };
        let suggestions = suggestions
            .iter()
            .filter(|(s, _)| s.starts_with(last_word))
            .map(|(_, d)| Suggestion {
                value: d.to_string(),
                description: Some(d.to_string()),
                span: Span::new(pos - last_word.len(), pos),
                ..Default::default()
            })
            .collect::<Vec<_>>();
        CompletionResult::fresh(suggestions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_are_fresh_and_replace_the_typed_prefix() {
        let state = Arc::new(EngineState::initialize());
        let mut completer = ChiliCompleter::new(&state);
        let line = ".handle.op";
        let result = completer.complete(line, line.len());
        assert!(!result.is_provisional());
        let suggestions = result.suggestions();
        assert!(!suggestions.is_empty());
        assert!(
            suggestions
                .iter()
                .any(|s| s.value.starts_with(".handle.open"))
        );
        assert!(
            suggestions
                .iter()
                .all(|s| s.span == Span::new(0, line.len()))
        );
        let result = completer.complete(" ", 1);
        assert!(!result.is_provisional());
        assert!(result.suggestions().is_empty());
    }
}
