//! A connection's replies and the Async frames `publish` sends to it share one
//! socket. They must go through one serialized writer: a Response spliced into
//! the middle of a published frame corrupts the subscriber's framing.

use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use chili_core::{
    EngineState, SpicyObj, serde9,
    utils::{MessageType, send_auth, write_chili_ipc_msg},
};

fn start_server(queue_max: i64) -> (Arc<EngineState>, u16) {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    engine.set_subscriber_queue_max(queue_max);
    let listener = EngineState::bind_tcp_listener(0, false).expect("bind");
    let port = listener.local_addr().unwrap().port();
    let srv = Arc::clone(&engine);
    std::thread::spawn(move || srv.run_accept_loop(listener, vec![]));
    (engine, port)
}

fn incoming_handle(engine: &Arc<EngineState>) -> i64 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let df = engine.list_handle().unwrap();
        let nums = df.column("num").unwrap().i64().unwrap();
        let conn = df.column("conn_type").unwrap().str().unwrap();
        for i in 0..df.height() {
            if conn.get(i) == Some("Incoming") {
                return nums.get(i).unwrap();
            }
        }
        assert!(Instant::now() < deadline, "no Incoming handle");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Read frames until `want_responses` Responses were seen; every header must be
/// well-formed. Returns (responses, async frames).
fn read_frames(mut stream: TcpStream, want_responses: usize) -> Result<(usize, usize), String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(20)))
        .unwrap();
    let (mut responses, mut asyncs) = (0usize, 0usize);
    let mut header = [0u8; 16];
    while responses < want_responses {
        stream
            .read_exact(&mut header)
            .map_err(|e| format!("read header after {responses} responses: {e}"))?;
        if header[0] != 1 || header[1] > 2 || header[2..8] != [0u8; 6] {
            return Err(format!("corrupt frame header {header:?}"));
        }
        let len = u64::from_le_bytes(header[8..].try_into().unwrap()) as usize;
        if len > 4 * 1024 * 1024 {
            return Err(format!("implausible frame length {len}"));
        }
        let mut body = vec![0u8; len];
        stream
            .read_exact(&mut body)
            .map_err(|e| format!("read body: {e}"))?;
        match header[1] {
            0 => asyncs += 1,
            2 => responses += 1,
            t => return Err(format!("unexpected message type {t}")),
        }
    }
    Ok((responses, asyncs))
}

fn run(queue_max: i64) {
    let (engine, port) = start_server(queue_max);
    let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
    assert_eq!(send_auth(&mut client, "", "", 9).unwrap(), 9);
    let h = incoming_handle(&engine);
    engine.handle_subscriber(&h).unwrap();
    engine.add_subscriber("trade", h).unwrap(); // live

    const REQUESTS: usize = 150;
    let reader = {
        let stream = client.try_clone().unwrap();
        std::thread::spawn(move || read_frames(stream, REQUESTS))
    };

    // Publisher: large frames, so a write spans many socket writes.
    let stop = Arc::new(AtomicBool::new(false));
    let publisher = {
        let engine = Arc::clone(&engine);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let payload = SpicyObj::String("x".repeat(256 * 1024));
            while !stop.load(Ordering::Relaxed) {
                let _ = engine.publish(
                    &SpicyObj::Symbol("upd".into()),
                    &SpicyObj::Symbol("trade".into()),
                    "trade",
                    &payload,
                );
            }
        })
    };

    // The same connection keeps sending Sync requests while frames stream in.
    let query = serde9::serialize(&SpicyObj::String("1+1".into()), false).unwrap();
    for _ in 0..REQUESTS {
        write_chili_ipc_msg(&mut client, &query, MessageType::Sync).unwrap();
        client.flush().unwrap();
        std::thread::sleep(Duration::from_millis(2));
    }

    let result = reader.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    publisher.join().unwrap();
    let (responses, asyncs) = result.expect("stream stayed well-framed");
    assert_eq!(responses, REQUESTS);
    assert!(asyncs > 0, "published frames must have been interleaved in time");
}

#[test]
fn replies_never_splice_into_published_frames_direct_mode() {
    run(-1);
}

#[test]
fn replies_never_splice_into_published_frames_queued_mode() {
    run(0);
}
