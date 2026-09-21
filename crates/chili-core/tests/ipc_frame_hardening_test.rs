//! Peer-controlled IPC header bytes must never panic a connection thread or
//! abort the process, and teardown must always run.

use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    time::{Duration, Instant},
};

use chili_core::{EngineState, utils::send_auth};

fn start_server() -> (Arc<EngineState>, u16) {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    let listener = EngineState::bind_tcp_listener(0, false).expect("bind");
    let port = listener.local_addr().unwrap().port();
    let srv = Arc::clone(&engine);
    std::thread::spawn(move || srv.run_accept_loop(listener, vec![]));
    (engine, port)
}

fn connect(port: u16, version: u8) -> TcpStream {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let remote = send_auth(&mut stream, "", "", version).expect("auth");
    assert_eq!(remote, version);
    stream
}

/// conn_type of the only handle whose role is not `want_not`, polled until it
/// equals `want` or the deadline passes.
fn await_conn_type(engine: &Arc<EngineState>, want: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let df = engine.list_handle().expect("list_handle");
        let conn = df.column("conn_type").unwrap().str().unwrap();
        if (0..df.height()).any(|i| conn.get(i) == Some(want)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// The server closed its side: a read returns 0 (or a reset), not a timeout.
fn assert_closed_by_server(stream: &mut TcpStream) {
    let mut buf = [0u8; 64];
    match stream.read(&mut buf) {
        Ok(0) => {}
        Ok(n) => panic!("expected close, got {n} bytes"),
        Err(e) => assert!(
            !matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
            "server kept the connection open: {e}"
        ),
    }
}

fn chili_header(msg_type: u8, len: u64) -> [u8; 16] {
    let mut h = [0u8; 16];
    h[0] = 1;
    h[1] = msg_type;
    h[8..].copy_from_slice(&len.to_le_bytes());
    h
}

#[test]
fn unknown_chili_message_type_disconnects_and_runs_teardown() {
    let (engine, port) = start_server();
    let mut s = connect(port, 9);
    s.write_all(&chili_header(3, 0)).unwrap();
    assert_closed_by_server(&mut s);
    assert!(
        await_conn_type(&engine, "Disconnected"),
        "teardown must mark the handle Disconnected"
    );
    // The listener and engine are still healthy.
    let _again = connect(port, 9);
}

#[test]
fn absurd_chili_length_does_not_allocate_or_abort() {
    let (engine, port) = start_server();
    let mut s = connect(port, 9);
    s.write_all(&chili_header(1, 1u64 << 62)).unwrap();
    s.write_all(&[0u8; 32]).unwrap();
    s.shutdown(std::net::Shutdown::Write).unwrap();
    assert!(
        await_conn_type(&engine, "Disconnected"),
        "short body must end the connection cleanly"
    );
    let _again = connect(port, 9);
}

#[test]
fn zero_length_chili_frame_gets_an_error_response() {
    let (_engine, port) = start_server();
    let mut s = connect(port, 9);
    s.write_all(&chili_header(1, 0)).unwrap();
    let mut header = [0u8; 16];
    s.read_exact(&mut header)
        .expect("server must answer a Sync frame, not die");
    assert_eq!(header[1], 2, "Response frame");
    let len = u64::from_le_bytes(header[8..].try_into().unwrap()) as usize;
    let mut body = vec![0u8; len];
    s.read_exact(&mut body).expect("error body");
}

#[test]
fn q_length_shorter_than_header_disconnects() {
    let (engine, port) = start_server();
    let mut s = connect(port, 6);
    s.write_all(&[1, 1, 0, 0, 4, 0, 0, 0]).unwrap();
    assert_closed_by_server(&mut s);
    assert!(await_conn_type(&engine, "Disconnected"));
    let _again = connect(port, 6);
}

#[test]
fn q_bad_message_error_reply_is_a_framed_response() {
    let (_engine, port) = start_server();
    let mut s = connect(port, 6);
    // Sync, body = month vector (type 13, attr 0, count 1, one i32): unsupported.
    let body = [13u8, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let mut msg = vec![1u8, 1, 0, 0];
    msg.extend_from_slice(&((body.len() + 8) as u32).to_le_bytes());
    msg.extend_from_slice(&body);
    s.write_all(&msg).unwrap();

    let mut header = [0u8; 8];
    s.read_exact(&mut header).expect("framed reply");
    assert_eq!(&header[..4], &[1, 2, 0, 0], "q Response header");
    let total = u32::from_le_bytes(header[4..].try_into().unwrap()) as usize;
    let mut rest = vec![0u8; total - 8];
    s.read_exact(&mut rest).expect("reply body matches declared length");
    assert_eq!(rest[0], 128, "q error object");
    assert_eq!(*rest.last().unwrap(), 0, "NUL-terminated error text");
}

#[test]
fn silent_client_does_not_block_new_connections() {
    let (_engine, port) = start_server();
    let _silent = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    std::thread::sleep(Duration::from_millis(100));
    let t0 = Instant::now();
    let _ok = connect(port, 9);
    assert!(
        t0.elapsed() < Duration::from_secs(2),
        "a client that never authenticates must not stall the accept loop"
    );
}

/// Send one Sync frame with `body`, expect an error Response, and check the
/// server still accepts and serves afterwards.
fn expect_error_response(body: &[u8]) {
    let (_engine, port) = start_server();
    let mut s = connect(port, 9);
    s.write_all(&chili_header(1, body.len() as u64)).unwrap();
    s.write_all(body).unwrap();
    let mut header = [0u8; 16];
    s.read_exact(&mut header)
        .expect("server must answer, not crash or hang");
    assert_eq!(header[1], 2, "Response frame");
    let len = u64::from_le_bytes(header[8..].try_into().unwrap()) as usize;
    let mut reply = vec![0u8; len];
    s.read_exact(&mut reply).expect("error body");
    let _again = connect(port, 9);
}

#[test]
fn deeply_nested_message_is_rejected_without_overflowing_the_stack() {
    // 20,000 levels of "mixed list of one mixed list": code 90, count 1, byte length.
    let mut body = Vec::new();
    for _ in 0..20_000 {
        body.extend_from_slice(&[0x5a, 0, 0, 0, 1, 0, 0, 0]);
        body.extend_from_slice(&8u64.to_le_bytes());
    }
    expect_error_response(&body);
}

#[test]
fn huge_element_count_does_not_preallocate() {
    // mixed list claiming 2^32 - 1 elements, with nothing behind it
    let mut body = vec![0x5a, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];
    body.extend_from_slice(&8u64.to_le_bytes());
    body.extend_from_slice(&[0u8; 8]);
    expect_error_response(&body);
    // dict with the same claim
    let mut body = vec![0x5b, 0, 0, 0, 0xff, 0xff, 0xff, 0xff];
    body.extend_from_slice(&8u64.to_le_bytes());
    body.extend_from_slice(&[0u8; 8]);
    expect_error_response(&body);
}
