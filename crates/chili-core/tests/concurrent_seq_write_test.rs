//! Concurrent sequence-log writes on one handle must not corrupt the tplog.
//!
//! Regression for a New→Sequence race: `sync()` copied `conn_type` before
//! acquiring the per-handle mutex, so a second writer could still take the
//! `New` path after the first frame and insert a spurious magic header.

use std::sync::Arc;
use std::thread;

use chili_core::{EngineState, SpicyObj, utils};

fn msg(i: i64) -> SpicyObj {
    SpicyObj::MixedList(vec![
        SpicyObj::Symbol("upd".into()),
        SpicyObj::Symbol("t1".into()),
        SpicyObj::I64(i),
    ])
}

#[test]
fn concurrent_sync_writes_on_one_sequence_handle() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap();

    let engine = Arc::new(EngineState::initialize());
    let h = match engine
        .open_handle(&format!("file://{path_str}"), 0)
        .unwrap()
    {
        SpicyObj::I64(h) => h,
        other => panic!("open_handle returned {other:?}"),
    };

    let threads = 4;
    let per_thread = 100;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                for i in 0..per_thread {
                    let key = (t * per_thread + i) as i64;
                    engine.sync(&h, &msg(key)).expect("sync write");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    let (count, _, _) =
        utils::count_sequence_file_messages(path_str, false, false).expect("validate");
    assert_eq!(
        count,
        (threads * per_thread) as i64,
        "tplog must contain every concurrent frame"
    );
}
