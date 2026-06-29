//! Slow-subscriber shed tests via bounded outbound queues.

use std::{
    io::Read,
    net::TcpStream,
    sync::Arc,
    time::{Duration, Instant},
};

use chili_core::{EngineState, SpicyObj, utils::send_auth};

/// Bind a listener on an ephemeral port, run the accept loop on a background
/// thread, and return the engine handle + the port it is listening on.
fn start_server(queue_max: i64) -> (Arc<EngineState>, u16) {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    engine.set_subscriber_queue_max(queue_max);

    let listener = EngineState::bind_tcp_listener(0, false).expect("bind on ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();

    let srv = Arc::clone(&engine);
    std::thread::spawn(move || {
        srv.run_accept_loop(listener, vec![]);
    });
    (engine, port)
}

/// Connect a chili (v9) subscriber socket to `port`, completing the auth
/// handshake. When `shrink_rcvbuf` is set, shrinks `SO_RCVBUF` so a non-reading
/// subscriber's kernel buffer fills fast and the server-side writer thread
/// blocks, backing the bounded channel up to `Full` quickly + deterministically.
fn connect_subscriber(port: u16, shrink_rcvbuf: bool) -> TcpStream {
    let stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to listener");
    if shrink_rcvbuf {
        let sref = socket2::SockRef::from(&stream);
        let _ = sref.set_recv_buffer_size(2048);
    }

    let mut stream = stream;
    let remote_version = send_auth(&mut stream, "", "", 9).expect("auth handshake");
    assert_eq!(remote_version, 9, "server must negotiate chili v9");
    stream
}

/// Poll `list_handle()` until at least `want` chili Incoming handles show up,
/// returning their handle numbers in connection order. Bounded so a missed
/// accept can't hang the test.
fn await_incoming_handles(engine: &Arc<EngineState>, want: usize) -> Vec<i64> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let df = engine.list_handle().expect("list_handle");
        let nums = df.column("num").unwrap().i64().unwrap();
        let conn = df.column("conn_type").unwrap().str().unwrap();
        let mut found: Vec<i64> = Vec::new();
        for i in 0..df.height() {
            if conn.get(i) == Some("Incoming") {
                found.push(nums.get(i).unwrap());
            }
        }
        if found.len() >= want {
            found.sort();
            return found;
        }
        assert!(
            Instant::now() < deadline,
            "server never registered {} incoming subscriber handle(s)",
            want
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

/// `(upd; table)` heads for a publish. The payload is built per-iteration since
/// `publish` borrows `&SpicyObj`.
fn upd_table() -> (SpicyObj, SpicyObj) {
    (
        SpicyObj::Symbol("upd".into()),
        SpicyObj::Symbol("trade".into()),
    )
}

fn big_payload() -> SpicyObj {
    SpicyObj::String("x".repeat(512 * 1024)) // 512 KiB
}

#[test]
fn full_queue_subscriber_is_shed_publisher_never_blocks() {
    let (engine, port) = start_server(4);

    let _slow = connect_subscriber(port, true);
    let h = await_incoming_handles(&engine, 1)[0];

    engine.handle_subscriber(&h).expect("promote to Publishing");
    engine
        .add_subscriber("trade", h)
        .expect("register subscriber on topic");

    let (upd, table) = upd_table();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut shed = false;
    for _ in 0..200 {
        let t0 = Instant::now();
        let _ = engine.publish(&upd, &table, "trade", &big_payload());
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "publish blocked on a slow subscriber — the bounded-queue path must never block"
        );

        if conn_type_of(&engine, h).as_deref() == Some("Disconnected") {
            shed = true;
            break;
        }
        assert!(
            Instant::now() < deadline,
            "queue-depth shed never fired within the guard window"
        );
    }

    assert!(
        shed,
        "a subscriber whose bounded outbound queue fills (stopped draining) must be shed"
    );
}

#[test]
fn healthy_subscriber_keeps_receiving_while_slow_one_is_shed() {
    let (engine, port) = start_server(4);

    let healthy = connect_subscriber(port, false);
    let _slow = connect_subscriber(port, true);

    let handles = await_incoming_handles(&engine, 2);
    let (h_healthy, h_slow) = (handles[0], handles[1]);

    let mut healthy_reader = healthy.try_clone().expect("clone healthy socket");
    let (got_tx, got_rx) = std::sync::mpsc::channel::<usize>();
    std::thread::spawn(move || {
        let mut buf = [0u8; 64 * 1024];
        let mut total = 0usize;
        loop {
            match healthy_reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    total += n;
                    let _ = got_tx.send(total);
                }
                Err(_) => break,
            }
        }
    });

    for h in [h_healthy, h_slow] {
        engine.handle_subscriber(&h).expect("promote to Publishing");
        engine
            .add_subscriber("trade", h)
            .expect("register on topic");
    }

    let (upd, table) = upd_table();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut slow_shed = false;
    for _ in 0..400 {
        let t0 = Instant::now();
        let _ = engine.publish(&upd, &table, "trade", &big_payload());
        assert!(
            t0.elapsed() < Duration::from_secs(2),
            "publish blocked — a slow subscriber must not wedge the publisher"
        );
        if conn_type_of(&engine, h_slow).as_deref() == Some("Disconnected") {
            slow_shed = true;
            break;
        }
        assert!(Instant::now() < deadline, "slow subscriber never shed");
    }

    assert!(slow_shed, "the slow subscriber must be shed");

    // The healthy subscriber must NOT have been shed and must have received data.
    assert_ne!(
        conn_type_of(&engine, h_healthy).as_deref(),
        Some("Disconnected"),
        "the healthy (draining) subscriber must NOT be shed"
    );
    let mut received = 0usize;
    while let Ok(n) = got_rx.try_recv() {
        received = n;
    }
    // Give the writer thread a beat to flush a few frames to the healthy peer.
    let drain_deadline = Instant::now() + Duration::from_secs(3);
    while received == 0 && Instant::now() < drain_deadline {
        if let Ok(n) = got_rx.recv_timeout(Duration::from_millis(200)) {
            received = n;
        }
    }
    assert!(
        received > 0,
        "the healthy subscriber must keep receiving published frames"
    );
}

/// Control: with the queue bound OFF (the default, 0), a Publishing subscriber
/// uses the Direct (blocking-write) path — no writer thread, no `queued`
/// shape — exactly the pre-2b behaviour. We just assert the opt-in is genuinely
/// opt-in: a fresh handle is Incoming, not pre-shed.
#[test]
fn queue_bound_off_by_default_leaves_handle_live() {
    let (engine, port) = start_server(0);
    let _peer = connect_subscriber(port, false);
    let h = await_incoming_handles(&engine, 1)[0];
    assert_eq!(
        conn_type_of(&engine, h).as_deref(),
        Some("Incoming"),
        "with the queue bound off, the accepted handle stays a live Incoming handle"
    );
    // Promotion with the bound off must succeed and stay on the Direct path.
    engine.handle_subscriber(&h).expect("promote to Publishing");
    assert_eq!(
        conn_type_of(&engine, h).as_deref(),
        Some("Publishing"),
        "promotion with the bound off keeps the handle live (Direct path)"
    );
}
