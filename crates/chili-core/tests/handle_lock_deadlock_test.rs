//! Regression: `sync`/`lpt` must not invert handle-table vs per-handle I/O locks.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chili_core::{EngineState, SpicyObj};
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
