//! `lpt` — log-write + publish + tick[tick_index; 1] under one lock.

use std::sync::Arc;
use std::thread;

use chili_core::{utils, EngineState, SpicyObj};

fn open_log(engine: &EngineState, path: &str) -> i64 {
    match engine.open_handle(&format!("file://{path}"), 0).unwrap() {
        SpicyObj::I64(h) => h,
        other => panic!("open_handle returned {other:?}"),
    }
}

#[test]
fn lpt_requires_msg_handle() {
    let engine = EngineState::initialize();
    let err = engine
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(1), 0)
        .unwrap_err();
    assert!(
        err.to_string().contains(".tick.msgHandle"),
        "expected missing-handle error, got {err}"
    );
}

#[test]
fn lpt_writes_log_and_increments_tick0() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap();

    let engine = EngineState::initialize();
    let h = open_log(&engine, path_str);
    engine.set_var(".tick.msgHandle", SpicyObj::I64(h)).unwrap();

    let n = engine
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(7), 0)
        .unwrap();
    assert_eq!(n, SpicyObj::I64(1));
    assert_eq!(engine.get_tick_count(0).unwrap(), 1);

    let (count, _, _) =
        utils::count_sequence_file_messages(path_str, false, false).expect("validate");
    assert_eq!(count, 1);
}

#[test]
fn concurrent_lpt_serializes_log_and_tick() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap().to_owned();

    let engine = Arc::new(EngineState::initialize());
    let h = open_log(&engine, &path_str);
    engine.set_var(".tick.msgHandle", SpicyObj::I64(h)).unwrap();

    let threads = 4;
    let per_thread = 50;
    let expected = (threads * per_thread) as i64;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                for i in 0..per_thread {
                    let v = (t * per_thread + i) as i64;
                    engine
                        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(v), 0)
                        .expect("lpt");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(engine.get_tick_count(0).unwrap(), expected);
    let (count, _, _) =
        utils::count_sequence_file_messages(&path_str, false, false).expect("validate");
    assert_eq!(count, expected);
}

#[test]
fn lpt_tick_index_selects_counter_slot() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap();

    let engine = EngineState::initialize();
    let log_h = open_log(&engine, path_str);
    engine
        .set_var(".tick.msgHandle", SpicyObj::I64(log_h))
        .unwrap();

    let n = engine
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(1), log_h as usize)
        .unwrap();
    assert_eq!(n.to_i64().unwrap(), 1);
    assert_eq!(engine.get_tick_count(0).unwrap(), 0);
    assert_eq!(engine.get_tick_count(log_h as usize).unwrap(), 1);
}
