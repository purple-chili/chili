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
    engine
        .add_subscriber_ex("trade", h, None, false)
        .unwrap();

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
    engine
        .add_subscriber_ex("trade", h, None, false)
        .unwrap();
    engine.activate_subscribers(h);
    engine
        .add_subscriber_ex("quote", h, None, false)
        .unwrap();

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
