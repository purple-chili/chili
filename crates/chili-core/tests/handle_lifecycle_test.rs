//! Handle lifecycle: errors must not lose handles, closing must disconnect the
//! peer, and a dead subscriber must leave the topic map.

use std::{
    io::{Read, Seek, SeekFrom, Write},
    net::TcpStream,
    sync::Arc,
    time::{Duration, Instant},
};

use chili_core::{EngineState, SpicyObj, utils, utils::send_auth};
use polars::prelude::*;

fn start_server() -> (Arc<EngineState>, u16) {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    let listener = EngineState::bind_tcp_listener(0, false).expect("bind");
    let port = listener.local_addr().unwrap().port();
    let srv = Arc::clone(&engine);
    std::thread::spawn(move || srv.run_accept_loop(listener, vec![]));
    (engine, port)
}

fn connect(port: u16) -> TcpStream {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    assert_eq!(send_auth(&mut s, "", "", 9).unwrap(), 9);
    s
}

fn incoming(engine: &Arc<EngineState>) -> i64 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let df = engine.list_handle().unwrap();
        let nums = df.column("num").unwrap().i64().unwrap();
        let conn = df.column("conn_type").unwrap().str().unwrap();
        for i in 0..df.height() {
            if matches!(conn.get(i), Some("Incoming") | Some("Publishing")) {
                return nums.get(i).unwrap();
            }
        }
        assert!(Instant::now() < deadline, "no inbound handle");
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn handle_nums(engine: &EngineState) -> Vec<i64> {
    let df = engine.list_handle().unwrap();
    df.column("num")
        .unwrap()
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

fn subscriber_count(engine: &EngineState) -> usize {
    let df = engine.list_topic_map().unwrap();
    let lists = df.column("subscribers").unwrap().list().unwrap();
    (0..lists.len())
        .map(|i| lists.get_as_series(i).map(|s| s.len()).unwrap_or(0))
        .sum()
}

fn frame(v: i64) -> SpicyObj {
    SpicyObj::DataFrame(DataFrame::new(1, vec![Series::new("v".into(), vec![v]).into_column()]).unwrap())
}

fn open_log(engine: &EngineState, path: &std::path::Path) -> i64 {
    match engine
        .open_handle(&format!("file://{}", path.display()), 0)
        .unwrap()
    {
        SpicyObj::I64(h) => h,
        other => panic!("{other:?}"),
    }
}

#[test]
fn subscribing_on_a_non_outgoing_handle_keeps_the_handle() {
    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let log = open_log(&engine, &dir.path().join("tplog"));

    assert!(engine.handle_publisher(&log).is_err());
    assert!(engine.handle_publisher_ex(&log, true).is_err());
    assert!(
        handle_nums(&engine).contains(&log),
        "a rejected .handle.subscribing must not delete the log handle"
    );
    engine
        .lpt(
            &SpicyObj::Symbol("trade".into()),
            &frame(1),
            &SpicyObj::I64(0),
            &SpicyObj::I64(log),
        )
        .expect("the log handle still writes");
}

#[test]
fn close_handle_disconnects_the_peer() {
    let (engine, port) = start_server();
    let mut client = connect(port);
    let h = incoming(&engine);
    engine.close_handle(&h).unwrap();
    let mut buf = [0u8; 8];
    match client.read(&mut buf) {
        Ok(0) => {}
        Ok(n) => panic!("expected close, got {n} bytes"),
        Err(e) => assert!(
            !matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            ),
            "closing a handle must disconnect its peer: {e}"
        ),
    }
}

#[test]
fn a_departed_subscriber_leaves_the_topic_map() {
    let (engine, port) = start_server();
    let client = connect(port);
    let h = incoming(&engine);
    engine.handle_subscriber(&h).unwrap();
    engine.add_subscriber("trade", h).unwrap();
    engine.add_subscriber("quote", h).unwrap();
    assert_eq!(subscriber_count(&engine), 2);

    drop(client);
    let deadline = Instant::now() + Duration::from_secs(5);
    while subscriber_count(&engine) != 0 {
        assert!(
            Instant::now() < deadline,
            "subscriptions of a disconnected handle must be purged"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_log_cannot_be_opened_twice() {
    let engine = EngineState::initialize();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tplog");
    let h = open_log(&engine, &path);
    let err = engine
        .open_handle(&format!("file://{}", path.display()), 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("already open"), "{err}");
    // After closing it can be opened again.
    engine.close_handle(&h).unwrap();
    open_log(&engine, &path);
}

#[test]
fn reopened_log_handle_counter_is_absolute() {
    let engine = EngineState::initialize();
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    let h = open_log(&engine, &a);
    for v in 0..3 {
        engine
            .lpt(
                &SpicyObj::Symbol("t".into()),
                &frame(v),
                &SpicyObj::I64(h as i64),
                &SpicyObj::I64(h),
            )
            .unwrap();
    }
    engine.close_handle(&h).unwrap();
    // Same number is reused for a fresh, empty log: its count must be 0, not 3.
    let h2 = open_log(&engine, &dir.path().join("b"));
    assert_eq!(h2, h, "test relies on handle number reuse");
    assert_eq!(engine.get_tick_count(h2 as usize).unwrap(), 0);
}

#[test]
fn undecodable_complete_frame_is_not_treated_as_a_torn_tail() {
    let engine = EngineState::initialize();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tplog");
    let h = open_log(&engine, &path);
    for v in 0..3 {
        engine
            .lpt(
                &SpicyObj::Symbol("t".into()),
                &frame(v),
                &SpicyObj::I64(0),
                &SpicyObj::I64(h),
            )
            .unwrap();
    }
    engine.close_handle(&h).unwrap();

    // Corrupt the type code of the second frame's payload; lengths stay intact.
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let total = file.metadata().unwrap().len();
    let mut len = [0u8; 8];
    file.seek(SeekFrom::Start(8)).unwrap();
    file.read_exact(&mut len).unwrap();
    let first = u64::from_le_bytes(len);
    let second_payload = 8 + 16 + first + 16;
    file.seek(SeekFrom::Start(second_payload)).unwrap();
    file.write_all(&[0x7f]).unwrap();
    file.sync_all().unwrap();

    // The walker starts after the 8-byte magic header, as `validateSeq` calls it.
    file.seek(SeekFrom::Start(8)).unwrap();
    let (count, valid_size) = utils::count_seq_messages(&mut file, true).unwrap();
    assert_eq!(count, 3, "frames after the bad one are still valid");
    assert_eq!(valid_size, total, "nothing after the bad frame may be truncated");
}

#[test]
fn pending_subscription_on_a_dead_handle_is_refused() {
    let (engine, port) = start_server();
    let client = connect(port);
    let h = incoming(&engine);
    drop(client);
    // wait for teardown to mark the handle
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let df = engine.list_handle().unwrap();
        let conn = df.column("conn_type").unwrap().str().unwrap();
        if (0..df.height()).any(|i| conn.get(i) == Some("Disconnected")) {
            break;
        }
        assert!(Instant::now() < deadline, "handle never disconnected");
        std::thread::sleep(Duration::from_millis(20));
    }
    // What a timed-out request's worker thread would do after its connection is gone.
    assert!(engine.subscribe_pending(&["trade"], h, None).is_err());
    assert!(engine.subscribe_pending(&["trade"], 9999, None).is_err());
    assert_eq!(subscriber_count(&engine), 0, "nothing may be left registered");
}

#[test]
fn a_log_held_by_another_engine_cannot_be_opened() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tplog");
    let first = EngineState::initialize();
    let h = open_log(&first, &path);
    first
        .lpt(
            &SpicyObj::Symbol("t".into()),
            &frame(1),
            &SpicyObj::I64(0),
            &SpicyObj::I64(h),
        )
        .unwrap();

    let second = EngineState::initialize();
    let err = second
        .open_handle(&format!("file://{}", path.display()), 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("another process"), "{err}");

    // Released with the handle.
    first.close_handle(&h).unwrap();
    open_log(&second, &path);
}
