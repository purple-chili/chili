use std::{fs, panic::AssertUnwindSafe, path::Path};

use chili_core::{EngineState, Func, SpicyObj, SpicyResult, Stack};
use chili_op::BUILT_IN_FN;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
}

fn import(state: &EngineState, path: &Path) -> SpicyResult<SpicyObj> {
    state.import_source_path("", path.to_str().unwrap())
}

#[test]
fn unchanged_parse_errors_are_reported_on_every_import() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.pep");
    fs::write(&path, "x: (").unwrap();

    for _ in 0..2 {
        let error = import(&state, &path).unwrap_err().to_string();
        assert!(error.contains("failed to parse"), "{error}");
    }
    assert_eq!(state.parse_cache_len(), 0);

    fs::write(&path, "x: 42").unwrap();
    assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(42));
    assert!(import(&state, &path).unwrap().is_null());
}

#[test]
fn runtime_failure_retries_and_can_succeed_without_source_changes() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retry.pep");
    fs::write(&path, "attempts: attempts+1; result: dependency").unwrap();
    state.set_var("attempts", SpicyObj::I64(0)).unwrap();

    for expected in [1, 2] {
        assert!(
            import(&state, &path)
                .unwrap_err()
                .to_string()
                .contains("dependency")
        );
        assert_eq!(state.get_var("attempts").unwrap(), SpicyObj::I64(expected));
        assert!(state.get_var("result").is_err());
    }

    state.set_var("dependency", SpicyObj::I64(42)).unwrap();
    assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(42));
    assert_eq!(state.get_var("attempts").unwrap(), SpicyObj::I64(3));
    assert!(import(&state, &path).unwrap().is_null());
    assert_eq!(state.get_var("attempts").unwrap(), SpicyObj::I64(3));
}

#[test]
fn source_registration_and_parsing_do_not_mark_imports_successful() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    for (name, parse_first) in [("registered.pep", false), ("parsed.pep", true)] {
        let path = dir.path().join(name);
        let source = "result: 42";
        fs::write(&path, source).unwrap();
        let canonical = path.canonicalize().unwrap();
        let canonical = canonical.to_str().unwrap();
        let source_id = state.set_source(canonical, source).unwrap();
        if parse_first {
            state.parse(canonical, source).unwrap();
        }

        assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(42));
        assert_eq!(
            state.get_source(source_id).unwrap(),
            (canonical.into(), source.into())
        );
        assert_eq!(state.set_source(canonical, source).unwrap(), source_id);
    }
}

#[test]
fn changed_source_is_imported_and_a_failed_revision_is_retryable() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("revisions.pep");
    fs::write(&path, "value: 1").unwrap();
    assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(1));
    fs::write(&path, "value: missing").unwrap();
    for _ in 0..2 {
        assert!(import(&state, &path).is_err());
    }
    fs::write(&path, "value: 2").unwrap();
    assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(2));
    assert!(import(&state, &path).unwrap().is_null());
}

#[test]
fn circular_imports_terminate_and_failed_parent_can_retry() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.pep");
    let b = dir.path().join("b.pep");
    fs::write(&a, "import \"./b.pep\"; result: dependency").unwrap();
    fs::write(&b, "import \"./a.pep\"; loaded: 7").unwrap();

    for _ in 0..2 {
        assert!(
            import(&state, &a)
                .unwrap_err()
                .to_string()
                .contains("dependency")
        );
    }
    assert_eq!(state.get_var("loaded").unwrap(), SpicyObj::I64(7));
    state.set_var("dependency", SpicyObj::I64(42)).unwrap();
    assert_eq!(import(&state, &a).unwrap(), SpicyObj::I64(42));
    assert!(import(&state, &a).unwrap().is_null());
    assert!(import(&state, &b).unwrap().is_null());
}

#[test]
fn failed_nested_import_can_be_retried_after_dependency_is_fixed() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.pep");
    let b = dir.path().join("b.pep");
    fs::write(&a, "import \"./b.pep\"; value").unwrap();
    fs::write(&b, "value: (").unwrap();
    for _ in 0..2 {
        assert!(
            import(&state, &a)
                .unwrap_err()
                .to_string()
                .contains("failed to parse")
        );
    }
    fs::write(&b, "value: 42").unwrap();
    assert_eq!(import(&state, &a).unwrap(), SpicyObj::I64(42));
}

fn concurrent_import_probe(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let path = args[0].str()?;
    let error = std::thread::scope(|scope| {
        scope
            .spawn(|| state.import_source_path("", path))
            .join()
            .unwrap()
    })
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("import already in progress on another thread")
    );
    Ok(SpicyObj::I64(42))
}

#[test]
fn concurrent_import_does_not_report_unfinished_source_as_loaded() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("concurrent.pep");
    fs::write(&path, "probe[path]").unwrap();
    state
        .set_var("path", SpicyObj::String(path.to_str().unwrap().into()))
        .unwrap();
    state
        .set_var(
            "probe",
            SpicyObj::Fn(Func::new_side_effect_built_in_fn(
                Some(Box::new(concurrent_import_probe)),
                1,
                "probe",
                &["path"],
            )),
        )
        .unwrap();
    assert_eq!(import(&state, &path).unwrap(), SpicyObj::I64(42));
    assert!(import(&state, &path).unwrap().is_null());
}

fn panic_probe(_args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    panic!("deliberate import panic");
}

#[test]
fn panic_does_not_leave_import_permanently_in_progress() {
    let state = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("panic.pep");
    fs::write(&path, "probe[]").unwrap();
    state
        .set_var(
            "probe",
            SpicyObj::Fn(Func::new_built_in_fn(
                Some(Box::new(panic_probe)),
                0,
                "probe",
                &[],
            )),
        )
        .unwrap();
    for _ in 0..2 {
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| import(&state, &path))).is_err());
    }
}
