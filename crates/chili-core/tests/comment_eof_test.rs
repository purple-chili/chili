use chili_core::{EngineState, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn eval(state: &EngineState, src: &str, path: &str) -> SpicyResult<SpicyObj> {
    state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(src.into()),
        path,
    )
}

#[test]
fn trailing_line_comments_work_in_both_syntaxes() {
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    for path in ["comment.pep", "comment.chi"] {
        for src in [
            "1 + 1 // c",
            "1 + 1 // c\n",
            "1 + 1 // c\r\n",
            "1 + 1 //",
            "1 + 1 // ) ]; raise \"ignored\"",
            "// heading\n1 + 1 // end",
        ] {
            assert_eq!(
                eval(&state, src, path).unwrap(),
                SpicyObj::I64(2),
                "{path}: {src}"
            );
        }
    }
}

#[test]
fn pepper_comment_only_input_returns_null() {
    let state = EngineState::initialize();
    for src in ["//", "// comment", "// comment\n"] {
        assert!(eval(&state, src, "comment.pep").unwrap().is_null());
    }
}

#[test]
fn division_strings_and_block_comments_are_unchanged() {
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    for path in ["comment.pep", "comment.chi"] {
        assert_eq!(eval(&state, "6 / 2", path).unwrap(), SpicyObj::F64(3.0));
        assert_eq!(
            eval(&state, "42 /* comment */", path).unwrap(),
            SpicyObj::I64(42)
        );
        assert_eq!(
            eval(&state, "\"// literal\"", path).unwrap(),
            SpicyObj::String("// literal".into())
        );
    }
}

#[test]
fn files_without_final_newline_import_successfully() {
    let dir = tempfile::tempdir().unwrap();
    let state = EngineState::initialize();
    for extension in ["pep", "chi"] {
        let path = dir.path().join(format!("comment.{extension}"));
        std::fs::write(&path, "42 // no trailing newline").unwrap();
        assert_eq!(
            state
                .import_source_path("", path.to_str().unwrap())
                .unwrap(),
            SpicyObj::I64(42)
        );
    }
}
