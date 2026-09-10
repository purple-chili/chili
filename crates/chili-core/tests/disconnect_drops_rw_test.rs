//! Per-peer disconnect must drop the accept-path writer Arc (and the
//! shutdown clone), otherwise two CLOSED fds stay stranded per departed peer.

use std::io::{Read, Result as IoResult, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chili_core::{ConnType, EngineState, IpcType, SpicyError, SpicyObj};

struct TrackedStream {
    dropped: Arc<AtomicBool>,
}

impl Drop for TrackedStream {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

impl Read for TrackedStream {
    fn read(&mut self, _buf: &mut [u8]) -> IoResult<usize> {
        Ok(0)
    }
}

impl Write for TrackedStream {
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        Ok(buf.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

#[test]
fn disconnect_handle_drops_rw_and_clears_writer() {
    let dropped = Arc::new(AtomicBool::new(false));
    let engine = EngineState::initialize();

    let h = match engine
        .set_handle(
            Some(Box::new(TrackedStream {
                dropped: Arc::clone(&dropped),
            })),
            "127.0.0.1:0",
            "tcp://127.0.0.1:0",
            true,
            IpcType::Chili,
            ConnType::Incoming,
            0,
        )
        .unwrap()
    {
        SpicyObj::I64(n) => n,
        other => panic!("set_handle returned {other:?}"),
    };

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let _client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    engine.set_shutdown_handle(&h, server.try_clone().unwrap());

    assert!(
        !dropped.load(Ordering::SeqCst),
        "writer must still be alive before disconnect"
    );

    engine.disconnect_handle(&h).unwrap();

    assert!(
        dropped.load(Ordering::SeqCst),
        "disconnect_handle must drop the map-held writer Arc"
    );

    let err = engine
        .sync(&h, &SpicyObj::String("1+1".into()))
        .expect_err("sync after disconnect must fail");
    assert!(
        matches!(err, SpicyError::InvalidHandleErr(_)),
        "expected InvalidHandleErr after rw cleared, got {err:?}"
    );

    // Handle entry stays for onDisconnected / close-hook callback lookup.
    assert!(engine.get_callback(&h).is_ok());
}
