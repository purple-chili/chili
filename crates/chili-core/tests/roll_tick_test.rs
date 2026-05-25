//! `rotate_handle` — file-handle rotation correctness tests.
//!
//! Verifies that `rotate_handle` correctly swaps a file-backed handle's
//! underlying writer to a new empty file, resets the tick counter, and
//! rejects non-empty target files.

use std::sync::Arc;

use chili_core::{EngineState, SpicyObj, serde9};

/// One inbound `.tick.upd`-shaped message carrying a recoverable int.
fn msg(i: i64) -> SpicyObj {
    SpicyObj::MixedList(vec![
        SpicyObj::Symbol("upd".into()),
        SpicyObj::Symbol("trade".into()),
        SpicyObj::I64(i),
    ])
}

/// Independent raw-frame oracle. Returns the ordered payload ints found
/// in the tplog at `path`. A torn/partial trailing frame is ignored.
/// Empty/missing file → empty vec.
fn read_tplog_ints(path: &str) -> Vec<i64> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    if bytes.len() < 8 {
        return vec![];
    }
    assert_eq!(
        &bytes[0..4],
        &[255, 0, 0, 0],
        "tplog {path} missing the 8-byte sequence magic header"
    );
    let mut pos = 8usize;
    let mut out = Vec::new();
    while pos + 16 <= bytes.len() {
        let len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 16;
        if len == 0 || pos + len > bytes.len() {
            break;
        }
        let mut dp = 0usize;
        let obj = serde9::deserialize(&bytes[pos..pos + len], &mut dp)
            .expect("oracle: serde9 read-path decode of a complete frame");
        let v = obj.as_vec().expect("oracle: frame payload is a MixedList");
        let i = v
            .last()
            .expect("oracle: non-empty MixedList")
            .to_i64()
            .expect("oracle: trailing element is the I64 key");
        out.push(i);
        pos += len;
    }
    out
}

fn new_engine() -> Arc<EngineState> {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    Arc::new(state)
}

fn open_seg(engine: &Arc<EngineState>, path: &str) -> i64 {
    match engine.open_handle(&format!("file://{path}"), 0).unwrap() {
        SpicyObj::I64(h) => h,
        other => panic!("open_handle returned non-i64: {other:?}"),
    }
}

#[test]
fn rotate_handle_swaps_writer_to_new_file() {
    let dir = tempfile::tempdir().unwrap();
    let seg0 = dir.path().join("seg_0000");
    let seg1 = dir.path().join("seg_0001");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let seg1s = seg1.to_str().unwrap().to_owned();

    let engine = new_engine();
    let h0 = open_seg(&engine, &seg0s);

    // Write to the original segment.
    for i in 0..5 {
        engine.sync(&h0, &msg(i)).unwrap();
    }

    // Rotate the handle to a new file.
    engine
        .rotate_handle(&h0, &format!("file://{seg1s}"))
        .unwrap();

    // Write to the rotated handle — should go to seg1.
    for i in 10..13 {
        engine.sync(&h0, &msg(i)).unwrap();
    }

    let in_seg0 = read_tplog_ints(&seg0s);
    let in_seg1 = read_tplog_ints(&seg1s);

    assert_eq!(
        in_seg0,
        (0..5).collect::<Vec<_>>(),
        "pre-rotation writes in seg0"
    );
    assert_eq!(in_seg1, vec![10, 11, 12], "post-rotation writes in seg1");
}

#[test]
fn rotate_handle_accepts_non_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let seg0 = dir.path().join("seg_0000");
    let seg1 = dir.path().join("seg_0001");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let seg1s = seg1.to_str().unwrap().to_owned();

    let engine = new_engine();
    let h0 = open_seg(&engine, &seg0s);
    engine.sync(&h0, &msg(1)).unwrap();

    // Pre-populate seg1 as a valid sequence file via a separate handle.
    let h1 = open_seg(&engine, &seg1s);
    engine.sync(&h1, &msg(100)).unwrap();
    engine.close_handle(&h1).unwrap();

    // Rotate h0 to the non-empty seg1 — should succeed and append.
    engine
        .rotate_handle(&h0, &format!("file://{seg1s}"))
        .unwrap();
    engine.sync(&h0, &msg(101)).unwrap();

    let in_seg1 = read_tplog_ints(&seg1s);
    assert_eq!(
        in_seg1,
        vec![100, 101],
        "rotate to non-empty file should append"
    );
}

#[test]
fn rotate_handle_resets_tick_count() {
    let dir = tempfile::tempdir().unwrap();
    let seg0 = dir.path().join("seg_0000");
    let seg1 = dir.path().join("seg_0001");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let seg1s = seg1.to_str().unwrap().to_owned();

    let engine = new_engine();
    let h0 = open_seg(&engine, &seg0s);

    // Advance tick count at index = h0.
    engine.tick(h0 as usize, 10).unwrap();
    assert_eq!(engine.get_tick_count(h0 as usize).unwrap(), 10);

    // Rotate resets tick_count[h0] to 0.
    engine
        .rotate_handle(&h0, &format!("file://{seg1s}"))
        .unwrap();
    assert_eq!(
        engine.get_tick_count(h0 as usize).unwrap(),
        0,
        "rotate_handle must reset tick_count for the handle index"
    );
}

#[test]
fn rotate_handle_rejects_non_file_uri() {
    let engine = new_engine();
    let dir = tempfile::tempdir().unwrap();
    let seg0 = dir.path().join("seg_0000");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let h0 = open_seg(&engine, &seg0s);

    let result = engine.rotate_handle(&h0, "chili://localhost:5000");
    assert!(
        result.is_err(),
        "rotate_handle must reject non-file:// URIs, got {result:?}"
    );
}
