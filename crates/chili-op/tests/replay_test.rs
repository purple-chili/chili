use std::path::PathBuf;

use chili_core::parse;

mod util;

use crate::util::create_state;

mod tests {
    use super::*;

    #[test]
    fn replay6_case00() {
        let mut tp_log_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tp_log_path.push("tests/tp_log/20250707.data");
        let state = create_state(false);
        let src = "
        upd: {[table; data]
            table upsert data;
            tick[1]
        };
        ";
        let nodes = parse(src, 0, "repl.pep").unwrap();
        state.eval_ast(nodes, "", src).unwrap();
        let nodes = parse(
            &format!(
                "replay_q[\"{}\"; 0; 5; `trade]",
                &tp_log_path.to_str().unwrap()
            ),
            0,
            "repl.pep",
        )
        .unwrap();
        state.eval_ast(nodes, "", "").unwrap();
        assert_eq!(state.get_var("trade").unwrap().size(), 10);
        let nodes = parse(
            &format!(
                "replay_q[\"{}\"; 5; 81; `trade]",
                &tp_log_path.to_str().unwrap()
            ),
            0,
            "repl.pep",
        )
        .unwrap();
        state.eval_ast(nodes, "", "").unwrap();
        assert_eq!(state.get_var("trade").unwrap().size(), 412);
    }

    #[test]
    fn replay6_case01() {
        let mut tp_log_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        tp_log_path.push("tests/tp_log/20250707.data");
        let state = create_state(false);
        let src = "
        upd: {[table; data]
            table upsert data;
            tick[1]
        };
        ";
        let nodes = parse(src, 0, "repl.pep").unwrap();
        state.eval_ast(nodes, "", src).unwrap();
        let nodes = parse(
            &format!(
                "replay_q[\"{}\"; 5; 81; `trade]",
                &tp_log_path.to_str().unwrap()
            ),
            0,
            "repl.pep",
        )
        .unwrap();
        state.eval_ast(nodes, "", "").unwrap();
        assert_eq!(state.get_var("trade").unwrap().size(), 402);
    }
}
