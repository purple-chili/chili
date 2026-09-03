//! Regression: handle-map vs per-handle I/O locks must not deadlock, and
//! `rotate_handle` must not race with `lpt`/`sync`.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chili_core::{EngineState, SpicyObj, utils};
use polars::frame::DataFrame;
use polars::prelude::{IntoColumn, NamedFrom};
use polars::series::Series;

fn open_log(engine: &EngineState, path: &str) -> i64 {
    match engine.open_handle(&format!("file://{path}"), 0).unwrap() {
        SpicyObj::I64(h) => h,
        other => panic!("open_handle returned {other:?}"),
    }
}

fn tiny_frame() -> SpicyObj {
    let s = Series::new("v".into(), vec![1i64]);
    SpicyObj::DataFrame(DataFrame::new(1, vec![s.into_column()]).unwrap())
}

fn upd(i: i64) -> SpicyObj {
    SpicyObj::MixedList(vec![
        SpicyObj::Symbol("upd".into()),
        SpicyObj::Symbol("t".into()),
        SpicyObj::I64(i),
    ])
}

#[test]
fn concurrent_lpt_fsync_and_stats_do_not_deadlock() {
    let engine = Arc::new(EngineState::initialize());
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tick.log");
    let log_h = open_log(&engine, path.to_str().unwrap());
    engine
        .set_var(".tick.msgHandle", SpicyObj::I64(log_h))
        .unwrap();

    let started = Instant::now();
    let workers: Vec<_> = (0..6)
        .map(|n| {
            let state = Arc::clone(&engine);
            thread::spawn(move || {
                for i in 0..40 {
                    match n % 3 {
                        0 => {
                            let _ = state.lpt(
                                &SpicyObj::Symbol("t".into()),
                                &tiny_frame(),
                                &SpicyObj::I64(i % 2),
                            );
                        }
                        1 => {
                            let _ = state.fsync_handle(&log_h);
                        }
                        _ => {
                            let _ = state.stats();
                        }
                    }
                }
            })
        })
        .collect();

    for w in workers {
        w.join().expect("worker panicked");
    }
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "stress test took too long; possible lock contention hang"
    );
}

#[test]
fn concurrent_lpt_rotate_and_fsync_do_not_deadlock() {
    let engine = Arc::new(EngineState::initialize());
    let dir = tempfile::tempdir().expect("tempdir");
    let seg0 = dir.path().join("seg0");
    let log_h = open_log(&engine, seg0.to_str().unwrap());
    engine
        .set_var(".tick.msgHandle", SpicyObj::I64(log_h))
        .unwrap();

    let started = Instant::now();
    let writers: Vec<_> = (0..4)
        .map(|_| {
            let state = Arc::clone(&engine);
            thread::spawn(move || {
                for i in 0..80 {
                    let _ = state.lpt(
                        &SpicyObj::Symbol("t".into()),
                        &tiny_frame(),
                        &SpicyObj::I64(0),
                    );
                    if i % 20 == 19 {
                        let _ = state.fsync_handle(&log_h);
                    }
                }
            })
        })
        .collect();

    let rotator = {
        let state = Arc::clone(&engine);
        let dir_path = dir.path().to_path_buf();
        thread::spawn(move || {
            for n in 1..=8 {
                thread::sleep(Duration::from_millis(5));
                let seg = dir_path.join(format!("seg{n}"));
                let _ = state.rotate_handle(&log_h, &format!("file://{}", seg.display()));
            }
        })
    };

    for w in writers {
        w.join().expect("writer panicked");
    }
    rotator.join().expect("rotator panicked");
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "lpt+rotate stress hung; possible lock-order deadlock"
    );
}

#[test]
fn rotate_under_lpt_traffic_keeps_sequence_magic() {
    let engine = Arc::new(EngineState::initialize());
    let dir = tempfile::tempdir().expect("tempdir");
    let seg0 = dir.path().join("day0");
    let seg1 = dir.path().join("day1");
    let seg0s = seg0.to_str().unwrap().to_owned();
    let seg1s = seg1.to_str().unwrap().to_owned();
    let log_h = open_log(&engine, &seg0s);
    engine
        .set_var(".tick.msgHandle", SpicyObj::I64(log_h))
        .unwrap();

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let publisher = {
        let state = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut i = 0i64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                state
                    .lpt(
                        &SpicyObj::Symbol("t".into()),
                        &tiny_frame(),
                        &SpicyObj::I64(0),
                    )
                    .expect("lpt");
                i += 1;
                if i > 10_000 {
                    break;
                }
            }
        })
    };

    thread::sleep(Duration::from_millis(20));
    engine
        .rotate_handle(&log_h, &format!("file://{seg1s}"))
        .expect("rotate");
    thread::sleep(Duration::from_millis(30));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    publisher.join().expect("publisher");

    // A few more writes after rotate must land on seg1 with a valid header.
    for i in 0..20 {
        engine.sync(&log_h, &upd(i)).expect("post-rotate sync");
    }

    let bytes = std::fs::read(&seg1s).expect("read seg1");
    assert!(
        bytes.len() >= 8,
        "rotated log should not be empty, got {} bytes",
        bytes.len()
    );
    assert_eq!(
        &bytes[0..8],
        &[255, 0, 0, 0, 0, 0, 0, 0],
        "rotated log must start with sequence magic, got {:02x?}",
        &bytes[0..8.min(bytes.len())]
    );
    let (count, _, _) =
        utils::count_sequence_file_messages(&seg1s, false, false).expect("validate seg1");
    assert!(count >= 20, "expected post-rotate frames, got {count}");
}
