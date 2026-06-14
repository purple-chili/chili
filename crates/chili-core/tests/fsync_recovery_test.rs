//! `fsync_handle` — fsync recovery regression tests.
//!
//! Verifies that `fsync_handle` on a healthy handle behaves identically
//! to today (no happy-path change), and that the tplog is intact after
//! write + fsync sequences.

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
fn fsync_handle_on_healthy_handle_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let seg = dir.path().join("seg_healthy");
    let segs = seg.to_str().unwrap().to_owned();

    let engine = new_engine();
    let h = open_seg(&engine, &segs);

    // Write several messages and fsync after each — all should succeed.
    for i in 0..5 {
        engine.sync(&h, &msg(i)).unwrap();
        engine
            .fsync_handle(&h)
            .expect("fsync_handle on a healthy handle must succeed");
    }

    let ints = read_tplog_ints(&segs);
    assert_eq!(
        ints,
        (0..5).collect::<Vec<_>>(),
        "all writes should be recoverable from the tplog"
    );
}

#[test]
fn fsync_handle_on_invalid_handle_returns_error() {
    let engine = new_engine();
    let result = engine.fsync_handle(&9999);
    assert!(
        result.is_err(),
        "fsync_handle on a non-existent handle must return an error"
    );
}

#[test]
fn fsync_handle_after_rotate_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let seg0 = dir.path().join("seg_0000");
    let seg1 = dir.path().join("seg_0001");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let seg1s = seg1.to_str().unwrap().to_owned();

    let engine = new_engine();
    let h = open_seg(&engine, &seg0s);

    // Write + fsync on original segment.
    engine.sync(&h, &msg(1)).unwrap();
    engine.fsync_handle(&h).unwrap();

    // Rotate to new segment.
    engine
        .rotate_handle(&h, &format!("file://{seg1s}"))
        .unwrap();

    // Write + fsync on rotated segment — must succeed.
    engine.sync(&h, &msg(2)).unwrap();
    engine
        .fsync_handle(&h)
        .expect("fsync_handle after rotate must succeed");

    assert_eq!(read_tplog_ints(&seg0s), vec![1]);
    assert_eq!(read_tplog_ints(&seg1s), vec![2]);
}
