//! `lpt` — log-write + publish + tick under one lock.
//!
//! Last argument is an int (`tick_index`, increment by 1) or a symbol
//! (`stamp_col`: per-row `tick[0] + i`, then increment by row count).

use std::sync::Arc;
use std::thread;

use chili_core::{serde9, utils, EngineState, SpicyObj};
use polars::prelude::{DataType, IntoColumn, NamedFrom};
use polars::{frame::DataFrame, series::Series};

fn open_log(engine: &EngineState, path: &str) -> i64 {
    match engine.open_handle(&format!("file://{path}"), 0).unwrap() {
        SpicyObj::I64(h) => h,
        other => panic!("open_handle returned {other:?}"),
    }
}

fn idx(n: i64) -> SpicyObj {
    SpicyObj::I64(n)
}

fn col_seq() -> SpicyObj {
    SpicyObj::Symbol("seq".into())
}

fn frame(vals: Vec<i64>) -> SpicyObj {
    let n = vals.len();
    let s = Series::new("v".into(), vals);
    let df = DataFrame::new(n, vec![s.into_column()]).expect("test frame");
    SpicyObj::DataFrame(df)
}

fn read_tplog_seq_cols(path: &str) -> Vec<Vec<u64>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    if bytes.len() < 8 {
        return vec![];
    }
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
            .expect("deserialize tplog frame");
        let v = obj.as_vec().expect("frame is MixedList");
        let SpicyObj::DataFrame(df) = &v[2] else {
            panic!("expected DataFrame payload, got {:?}", v[2].get_type_name());
        };
        let s = df
            .column("seq")
            .expect("seq column")
            .as_materialized_series();
        assert_eq!(s.dtype(), &DataType::UInt64);
        let seqs: Vec<u64> = s.u64().unwrap().into_no_null_iter().collect();
        out.push(seqs);
        pos += len;
    }
    out
}

#[test]
fn lpt_requires_msg_handle() {
    let engine = EngineState::initialize();
    let err = engine
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(1), &idx(0))
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
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(7), &idx(0))
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
                        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(v), &idx(0))
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
        .lpt(
            &SpicyObj::Symbol("t1".into()),
            &SpicyObj::I64(1),
            &idx(log_h),
        )
        .unwrap();
    assert_eq!(n.to_i64().unwrap(), 1);
    assert_eq!(engine.get_tick_count(0).unwrap(), 0);
    assert_eq!(engine.get_tick_count(log_h as usize).unwrap(), 1);
}

#[test]
fn lpt_stamp_col_uses_row_count_and_contiguous_seq() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap();

    let engine = EngineState::initialize();
    let h = open_log(&engine, path_str);
    engine.set_var(".tick.msgHandle", SpicyObj::I64(h)).unwrap();

    let n = engine
        .lpt(
            &SpicyObj::Symbol("t1".into()),
            &frame(vec![10, 11, 12, 13, 14]),
            &col_seq(),
        )
        .unwrap();
    assert_eq!(n.to_i64().unwrap(), 5);
    assert_eq!(engine.get_tick_count(0).unwrap(), 5);

    let n = engine
        .lpt(
            &SpicyObj::Symbol("t1".into()),
            &frame(vec![20, 21, 22, 23, 24]),
            &col_seq(),
        )
        .unwrap();
    assert_eq!(n.to_i64().unwrap(), 10);

    let frames = read_tplog_seq_cols(path_str);
    assert_eq!(frames, vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]]);
}

#[test]
fn concurrent_lpt_stamp_seqs_are_unique_and_gap_free() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let path_str = path.to_str().unwrap().to_owned();

    let engine = Arc::new(EngineState::initialize());
    let h = open_log(&engine, &path_str);
    engine.set_var(".tick.msgHandle", SpicyObj::I64(h)).unwrap();

    let threads = 4;
    let per_thread = 10;
    let rows = 5i64;
    let expected = threads * per_thread * rows;
    let handles: Vec<_> = (0..threads)
        .map(|_| {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                for _ in 0..per_thread {
                    engine
                        .lpt(
                            &SpicyObj::Symbol("t1".into()),
                            &frame(vec![1, 2, 3, 4, 5]),
                            &col_seq(),
                        )
                        .expect("lpt stamp");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(engine.get_tick_count(0).unwrap(), expected);
    let mut seqs: Vec<u64> = read_tplog_seq_cols(&path_str).into_iter().flatten().collect();
    seqs.sort_unstable();
    let want: Vec<u64> = (0..expected as u64).collect();
    assert_eq!(seqs, want);
}

#[test]
fn lpt_stamp_requires_dataframe() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("2026.08.14");
    let engine = EngineState::initialize();
    let h = open_log(&engine, path.to_str().unwrap());
    engine.set_var(".tick.msgHandle", SpicyObj::I64(h)).unwrap();

    let err = engine
        .lpt(&SpicyObj::Symbol("t1".into()), &SpicyObj::I64(1), &col_seq())
        .unwrap_err();
    assert!(
        err.to_string().contains("dataframe"),
        "expected dataframe type error, got {err}"
    );
}
