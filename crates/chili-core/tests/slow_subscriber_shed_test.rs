//! Slow-subscriber shed tests (write timeout + socket shutdown).

use std::{
    io::Read,
    net::TcpStream,
    sync::Arc,
    time::{Duration, Instant},
};

use chili_core::{EngineState, SpicyObj, utils::send_auth};

fn start_server(write_timeout_ms: i64) -> (Arc<EngineState>, u16) {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    engine.set_write_timeout_ms(write_timeout_ms);

    let listener = EngineState::bind_tcp_listener(0, false).expect("bind on ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();

    let srv = Arc::clone(&engine);
    std::thread::spawn(move || {
        srv.run_accept_loop(listener, vec![]);
    });
    (engine, port)
}

fn connect_subscriber(port: u16) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to listener");
    let sref = socket2::SockRef::from(&stream);
    let _ = sref.set_recv_buffer_size(2048);

    let mut stream = stream;
    let remote_version = send_auth(&mut stream, "", "", 9).expect("auth handshake");
    assert_eq!(remote_version, 9, "server must negotiate chili v9");
    stream
}

fn await_incoming_handle(engine: &Arc<EngineState>) -> i64 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let df = engine.list_handle().expect("list_handle");
        if df.height() > 0 {
            let nums = df.column("num").unwrap().i64().unwrap();
            let conn = df.column("conn_type").unwrap().str().unwrap();
            for i in 0..df.height() {
                if conn.get(i) == Some("Incoming") {
                    return nums.get(i).unwrap();
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "server never registered the incoming subscriber handle"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn conn_type_of(engine: &Arc<EngineState>, h: i64) -> Option<String> {
    let df = engine.list_handle().ok()?;
    let nums = df.column("num").ok()?.i64().ok()?;
    let conn = df.column("conn_type").ok()?.str().ok()?;
    for i in 0..df.height() {
        if nums.get(i) == Some(h) {
            return conn.get(i).map(|s| s.to_owned());
        }
    }
    None
}

fn big_message() -> (SpicyObj, SpicyObj, SpicyObj) {
    let payload = "x".repeat(512 * 1024);
    (
        SpicyObj::Symbol("upd".into()),
        SpicyObj::Symbol("trade".into()),
        SpicyObj::String(payload),
    )
}

#[test]
fn slow_subscriber_is_shed_on_write_timeout() {
    let (engine, port) = start_server(200);
    let mut peer = connect_subscriber(port);
    let h = await_incoming_handle(&engine);

    engine.handle_subscriber(&h).expect("promote to Publishing");
    engine
        .add_subscriber("trade", h)
        .expect("register subscriber on topic");

    let (upd, table, _msg) = big_message();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut shed = false;
    for _ in 0..200 {
        let payload = SpicyObj::String("x".repeat(512 * 1024));
        let _ = engine.publish(&upd, &table, "trade", &payload);

        if conn_type_of(&engine, h).as_deref() == Some("Disconnected") {
            shed = true;
            break;
        }
        assert!(
            Instant::now() < deadline,
            "write-timeout shed never fired within the guard window"
        );
    }

    assert!(
        shed,
        "subscriber that stopped reading must be shed (handle marked Disconnected) on write timeout"
    );

    peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    let mut buf = [0u8; 64];
    loop {
        match peer.read(&mut buf) {
            Ok(0) => break,
            Ok(_) => continue,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                panic!("peer socket was not shut down — read blocked past the shed")
            }
            Err(_) => break,
        }
    }
}

#[test]
fn write_timeout_off_by_default_leaves_handle_live() {
    let (engine, port) = start_server(0);
    let _peer = connect_subscriber(port);
    let h = await_incoming_handle(&engine);
    assert_eq!(
        conn_type_of(&engine, h).as_deref(),
        Some("Incoming"),
        "with the timeout off, the accepted handle stays a live Incoming handle"
    );
}
