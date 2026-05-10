use chili_core::EngineState;
use serial_test::serial;
use std::env;
use std::fs;
use std::path::Path;

/// Helper: create a temp directory tree mimicking a CHIZPATH with a package installed.
fn setup_chiz_env(chiz_root: &Path, pkg_name: &str, version: &str, module: &str, ext: &str) {
    let src_dir = chiz_root.join(pkg_name).join(version).join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let file_path = src_dir.join(format!("{}.{}", module, ext));
    fs::write(&file_path, format!("// {} module\n", module)).unwrap();
}

/// Helper: write a chiz_index.json in the given directory.
fn write_local_index(dir: &Path, entries: &[(&str, &str)]) {
    let mut map = serde_json::Map::new();
    for (name, version) in entries {
        map.insert(
            name.to_string(),
            serde_json::json!({"name": name, "version": version, "dependencies": {}}),
        );
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    fs::write(dir.join("chiz_index.json"), json).unwrap();
}

/// Helper: write a global .index in the CHIZPATH root.
fn write_global_index(chiz_root: &Path, entries: &[(&str, &str)]) {
    let mut map = serde_json::Map::new();
    for (name, version) in entries {
        let key = format!("{}@{}", name, version);
        map.insert(
            key,
            serde_json::json!({"name": name, "version": version, "dependencies": {}}),
        );
    }
    let json = serde_json::to_string_pretty(&map).unwrap();
    fs::write(chiz_root.join(".index"), json).unwrap();
}

#[test]
#[serial]
fn test_resolve_scoped_package_import_chi() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    setup_chiz_env(&chiz_root, "@myorg/utils", "1.0.0", "helper", "chi");
    write_global_index(&chiz_root, &[("@myorg/utils", "1.0.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::initialize();
    let result = state.resolve_package_import("@myorg/utils/helper");
    unsafe { env::remove_var("CHIZPATH") };

    let resolved = result.unwrap();
    assert!(resolved.exists());
    assert!(resolved.to_string_lossy().ends_with("helper.chi"));
}

#[test]
#[serial]
fn test_resolve_scoped_package_import_pep() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    setup_chiz_env(&chiz_root, "@myorg/tools", "0.2.0", "render", "pep");
    write_global_index(&chiz_root, &[("@myorg/tools", "0.2.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::initialize();
    let result = state.resolve_package_import("@myorg/tools/render");
    unsafe { env::remove_var("CHIZPATH") };

    let resolved = result.unwrap();
    assert!(resolved.exists());
    assert!(resolved.to_string_lossy().ends_with("render.pep"));
}

#[test]
#[serial]
fn test_resolve_pepper_mode_prefers_pep() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    setup_chiz_env(&chiz_root, "@myorg/dual", "1.0.0", "api", "chi");
    setup_chiz_env(&chiz_root, "@myorg/dual", "1.0.0", "api", "pep");
    write_global_index(&chiz_root, &[("@myorg/dual", "1.0.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::new(false, false, true);
    let result = state.resolve_package_import("@myorg/dual/api");
    unsafe { env::remove_var("CHIZPATH") };

    let resolved = result.unwrap();
    assert!(resolved.to_string_lossy().ends_with("api.pep"));
}

#[test]
#[serial]
fn test_resolve_unscoped_package_import() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    setup_chiz_env(&chiz_root, "common-lib", "2.0.0", "util", "chi");
    write_global_index(&chiz_root, &[("common-lib", "2.0.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::initialize();
    let result = state.resolve_package_import("common-lib/util");
    unsafe { env::remove_var("CHIZPATH") };

    let resolved = result.unwrap();
    assert!(resolved.exists());
    assert!(resolved.to_string_lossy().contains("common-lib"));
    assert!(resolved.to_string_lossy().ends_with("util.chi"));
}

#[test]
#[serial]
fn test_resolve_deeper_module_path() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    let src_dir = chiz_root
        .join("@myorg/deep")
        .join("1.0.0")
        .join("src")
        .join("sub");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(src_dir.join("module.chi"), "// nested\n").unwrap();

    write_global_index(&chiz_root, &[("@myorg/deep", "1.0.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::initialize();
    let result = state.resolve_package_import("@myorg/deep/sub/module");
    unsafe { env::remove_var("CHIZPATH") };

    let resolved = result.unwrap();
    assert!(resolved.exists());
    assert!(resolved.to_string_lossy().ends_with("sub/module.chi"));
}

#[test]
#[serial]
fn test_resolve_package_version_from_local_index() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    let cwd = env::current_dir().unwrap();
    write_local_index(&cwd, &[("@myorg/utils", "1.0.0")]);

    let state = EngineState::initialize();
    let result = state.resolve_package_version("@myorg/utils", &chiz_root);

    fs::remove_file(cwd.join("chiz_index.json")).ok();

    assert_eq!(result.unwrap(), "1.0.0");
}

#[test]
fn test_resolve_package_version_from_global_index() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    write_global_index(&chiz_root, &[("@myorg/analytics", "2.3.0")]);

    let state = EngineState::initialize();
    let result = state.resolve_package_version("@myorg/analytics", &chiz_root);

    assert_eq!(result.unwrap(), "2.3.0");
}

#[test]
fn test_resolve_package_version_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    let state = EngineState::initialize();
    let result = state.resolve_package_version("@nonexistent/pkg", &chiz_root);

    assert!(result.is_err());
}

#[test]
#[serial]
fn test_resolve_module_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let chiz_root = tmp.path().join("chiz");
    fs::create_dir_all(&chiz_root).unwrap();

    let src_dir = chiz_root.join("@myorg/empty").join("1.0.0").join("src");
    fs::create_dir_all(&src_dir).unwrap();
    write_global_index(&chiz_root, &[("@myorg/empty", "1.0.0")]);

    unsafe { env::set_var("CHIZPATH", chiz_root.to_str().unwrap()) };
    let state = EngineState::initialize();
    let result = state.resolve_package_import("@myorg/empty/missing");
    unsafe { env::remove_var("CHIZPATH") };

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("package module not found"));
}

#[test]
fn test_invalid_scoped_import_missing_module() {
    let state = EngineState::initialize();
    let result = state.resolve_package_import("@scope/pkg");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expected '@scope/pkg-name/module'"));
}

#[test]
fn test_invalid_unscoped_import_missing_module() {
    let state = EngineState::initialize();
    let result = state.resolve_package_import("pkg");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expected 'pkg-name/module'"));
}
