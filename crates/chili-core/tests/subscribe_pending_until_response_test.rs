//! #10: `.broker.subscribe` registers pending until sync Response is written,
//! so `publish` cannot interleave Async frames into the handshake.

use std::io::{Read, Result as IoResult, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chili_core::{ConnType, EngineState, IpcType, SpicyObj};
use polars::frame::DataFrame;
use polars::prelude::{IntoColumn, NamedFrom};
use polars::series::Series;

fn tiny_frame() -> SpicyObj {
    let s = Series::new("v".into(), vec![1i64]);
    SpicyObj::DataFrame(DataFrame::new(1, vec![s.into_column()]).unwrap())
}

/// Counts how many times publish attempted a direct write by wrapping the stream.
struct CountingStream {
    writes: Arc<AtomicBool>,
}

impl Read for CountingStream {
    fn read(&mut self, _buf: &mut [u8]) -> IoResult<usize> {
        Ok(0)
    }
}

impl Write for CountingStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if !buf.is_empty() {
            self.writes.store(true, Ordering::SeqCst);
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

#[test]
fn pending_subscriber_does_not_receive_publish() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let wrote = Arc::new(AtomicBool::new(false));
    let h = match engine
        .set_handle(
            Some(Box::new(CountingStream {
                writes: Arc::clone(&wrote),
            })),
            "127.0.0.1:0",
            "tcp://test",
            true,
            IpcType::Chili,
            ConnType::Incoming,
            0,
        )
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("{other:?}"),
    };
    engine.handle_subscriber(&h).unwrap();
    engine.add_subscriber_ex("trade", h, None, false).unwrap();

    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("trade".into()),
            "trade",
            &tiny_frame(),
        )
        .unwrap();

    assert!(
        !wrote.load(Ordering::SeqCst),
        "pending (non-live) subscriber must not receive publish frames"
    );

    engine.activate_subscribers(h);
    wrote.store(false, Ordering::SeqCst);

    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("trade".into()),
            "trade",
            &tiny_frame(),
        )
        .unwrap();

    assert!(
        wrote.load(Ordering::SeqCst),
        "after activate_subscribers, publish must reach the handle"
    );
}

#[test]
fn live_add_subscriber_still_receives_immediately() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let wrote = Arc::new(AtomicBool::new(false));
    let h = match engine
        .set_handle(
            Some(Box::new(CountingStream {
                writes: Arc::clone(&wrote),
            })),
            "127.0.0.1:0",
            "tcp://test",
            true,
            IpcType::Chili,
            ConnType::Incoming,
            0,
        )
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("{other:?}"),
    };
    engine.handle_subscriber(&h).unwrap();
    engine.add_subscriber("trade", h).unwrap(); // live=true

    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("trade".into()),
            "trade",
            &tiny_frame(),
        )
        .unwrap();

    assert!(
        wrote.load(Ordering::SeqCst),
        "direct add_subscriber must remain live for tests/tooling"
    );
}

#[test]
fn drop_pending_removes_inactive_only() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let wrote_trade = Arc::new(AtomicBool::new(false));
    let h = match engine
        .set_handle(
            Some(Box::new(CountingStream {
                writes: Arc::clone(&wrote_trade),
            })),
            "127.0.0.1:0",
            "tcp://test",
            true,
            IpcType::Chili,
            ConnType::Incoming,
            0,
        )
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("{other:?}"),
    };
    engine.handle_subscriber(&h).unwrap();
    engine.add_subscriber_ex("trade", h, None, false).unwrap();
    engine.activate_subscribers(h);
    engine.add_subscriber_ex("quote", h, None, false).unwrap();

    engine.drop_pending_subscribers(h);

    // trade (live) still receives; quote was pending and dropped.
    wrote_trade.store(false, Ordering::SeqCst);
    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("trade".into()),
            "trade",
            &tiny_frame(),
        )
        .unwrap();
    assert!(wrote_trade.load(Ordering::SeqCst), "live trade sub kept");

    wrote_trade.store(false, Ordering::SeqCst);
    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("quote".into()),
            "quote",
            &tiny_frame(),
        )
        .unwrap();
    assert!(
        !wrote_trade.load(Ordering::SeqCst),
        "pending quote sub must be dropped"
    );
}

// ---------------------------------------------------------------------------
// Handshake gap: frames published while a handle is pending must be buffered
// and written after activation, in publish order, before any live frame.
// ---------------------------------------------------------------------------

use std::sync::Mutex as StdMutex;

use chili_core::serde9;

/// Records every byte written so tests can parse the Async frames back.
#[derive(Clone)]
struct RecordingStream {
    buf: Arc<StdMutex<Vec<u8>>>,
}

impl Read for RecordingStream {
    fn read(&mut self, _buf: &mut [u8]) -> IoResult<usize> {
        Ok(0)
    }
}

impl Write for RecordingStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        self.buf.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

fn frame(v: i64) -> SpicyObj {
    let s = Series::new("v".into(), vec![v]);
    SpicyObj::DataFrame(DataFrame::new(1, vec![s.into_column()]).unwrap())
}

/// Parse `[1, type, 0×6][u64 len][payload]` frames and return each payload's `v`.
fn written_values(bytes: &[u8]) -> Vec<i64> {
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos + 16 <= bytes.len() {
        assert_eq!(bytes[pos], 1, "frame header byte");
        let len = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
        pos += 16;
        let mut dp = 0usize;
        let obj = serde9::deserialize(&bytes[pos..pos + len], &mut dp).expect("deserialize");
        let parts = obj.as_vec().expect("MixedList");
        assert_eq!(parts[0], SpicyObj::Symbol("upd".into()));
        let SpicyObj::DataFrame(df) = &parts[2] else {
            panic!("expected DataFrame payload");
        };
        let s = df.column("v").unwrap().as_materialized_series();
        out.extend(s.i64().unwrap().into_no_null_iter());
        pos += len;
    }
    assert_eq!(pos, bytes.len(), "trailing partial frame");
    out
}

fn recording_handle(engine: &EngineState) -> (i64, RecordingStream) {
    let stream = RecordingStream {
        buf: Arc::new(StdMutex::new(Vec::new())),
    };
    let h = match engine
        .set_handle(
            Some(Box::new(stream.clone())),
            "127.0.0.1:0",
            "tcp://test",
            true,
            IpcType::Chili,
            ConnType::Incoming,
            0,
        )
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("{other:?}"),
    };
    engine.handle_subscriber(&h).unwrap();
    (h, stream)
}

fn publish(engine: &EngineState, table: &str, v: i64) {
    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol(table.into()),
            table,
            &frame(v),
        )
        .unwrap();
}

#[test]
fn pending_frames_are_flushed_on_activation_in_order_before_live() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let (h, stream) = recording_handle(&engine);
    let bound = engine
        .subscribe_pending(&["trade", "quote"], h, None)
        .unwrap();
    assert_eq!(bound, 0);

    publish(&engine, "trade", 1);
    publish(&engine, "quote", 2);
    publish(&engine, "trade", 3);
    assert!(
        stream.buf.lock().unwrap().is_empty(),
        "pending handle must not be written to"
    );
    assert_eq!(engine.pending_frame_count(h), 3);

    engine.activate_subscribers(h);
    assert_eq!(engine.pending_frame_count(h), 0);
    assert_eq!(written_values(&stream.buf.lock().unwrap()), vec![1, 2, 3]);

    publish(&engine, "quote", 4);
    assert_eq!(
        written_values(&stream.buf.lock().unwrap()),
        vec![1, 2, 3, 4]
    );
}

#[test]
fn lpt_between_bound_and_activation_is_delivered_exactly_once() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("tplog");
    let log = match engine
        .open_handle(&format!("file://{}", log_path.display()), 0)
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("{other:?}"),
    };
    let lpt = |v: i64| {
        engine
            .lpt(
                &SpicyObj::Symbol("trade".into()),
                &frame(v),
                &SpicyObj::I64(0),
                &SpicyObj::I64(log),
            )
            .unwrap()
    };

    // Frame 10 is fully logged + ticked before the subscribe: replay covers it.
    lpt(10);
    let (h, stream) = recording_handle(&engine);
    let bound = engine.subscribe_pending(&["trade"], h, None).unwrap();
    assert_eq!(
        bound, 1,
        "bound counts the frame that landed before subscribe"
    );

    // Frame 11 runs between bound read and activation: the reported gap.
    lpt(11);
    assert_eq!(engine.get_tick_count(0).unwrap(), 2);
    assert!(stream.buf.lock().unwrap().is_empty());

    engine.activate_subscribers(h);
    lpt(12);

    // Not 10 (replayed from the log up to `bound`), 11 exactly once, then live 12.
    assert_eq!(written_values(&stream.buf.lock().unwrap()), vec![11, 12]);
}

#[test]
fn drop_pending_discards_buffered_frames() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let (h, stream) = recording_handle(&engine);
    engine.subscribe_pending(&["trade"], h, None).unwrap();
    publish(&engine, "trade", 1);
    assert_eq!(engine.pending_frame_count(h), 1);

    engine.drop_pending_subscribers(h);
    assert_eq!(engine.pending_frame_count(h), 0);
    engine.activate_subscribers(h);
    assert!(stream.buf.lock().unwrap().is_empty());

    // Subscription is gone too: a later publish reaches nobody.
    publish(&engine, "trade", 2);
    assert!(stream.buf.lock().unwrap().is_empty());
}

#[test]
fn pending_buffer_is_capped_by_subscriber_queue_max() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let (h, stream) = recording_handle(&engine);
    engine.set_subscriber_queue_max(2);
    engine.subscribe_pending(&["trade"], h, None).unwrap();

    publish(&engine, "trade", 1);
    publish(&engine, "trade", 2);
    assert_eq!(engine.pending_frame_count(h), 2);
    // Third frame overflows: handle is shed and the buffer released.
    publish(&engine, "trade", 3);
    assert_eq!(engine.pending_frame_count(h), 0);

    engine.activate_subscribers(h);
    assert!(
        stream.buf.lock().unwrap().is_empty(),
        "shed handle must not be written to"
    );
}

#[test]
fn filtered_pending_frames_keep_the_filter() {
    let engine = EngineState::initialize();
    engine.set_subscriber_queue_max(-1); // direct writes so the stream can be asserted synchronously
    let (h, stream) = recording_handle(&engine);
    let filter = chili_core::SubFilter::new("v".into(), vec!["2".into()]);
    engine
        .subscribe_pending(&["trade"], h, Some(filter))
        .unwrap();

    // Filter on an i64 column: cast comparison is string-based via filtered_message;
    // rows not matching are dropped, matching rows kept.
    let s = Series::new("v".into(), vec![1i64, 2, 3]);
    let df = SpicyObj::DataFrame(DataFrame::new(3, vec![s.into_column()]).unwrap());
    engine
        .publish(
            &SpicyObj::Symbol("upd".into()),
            &SpicyObj::Symbol("trade".into()),
            "trade",
            &df,
        )
        .unwrap();
    engine.activate_subscribers(h);
    let got = written_values(&stream.buf.lock().unwrap());
    assert_eq!(got.len(), 1, "one frame written");
    assert_eq!(got, vec![2]);
}
