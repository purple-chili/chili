//! Server-side CHILI_IPC_TOKEN check. Own test binary: it sets a process-wide
//! environment variable.

use std::{net::TcpStream, sync::Arc, time::Duration};

use chili_core::{EngineState, utils::send_auth};

#[test]
fn server_token_is_required_when_set() {
    // SAFETY: single test in this binary; set before any thread reads it.
    unsafe { std::env::set_var("CHILI_IPC_TOKEN", "s3cret") };

    let engine = Arc::new(EngineState::initialize());
    engine.set_arc_self(Arc::clone(&engine)).unwrap();
    let listener = EngineState::bind_tcp_listener(0, false).expect("bind");
    let port = listener.local_addr().unwrap().port();
    let srv = Arc::clone(&engine);
    std::thread::spawn(move || srv.run_accept_loop(listener, vec![]));

    let attempt = |password: &str| {
        let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        send_auth(&mut s, "alice", password, 9)
    };

    assert!(attempt("wrong").is_err(), "wrong token must be rejected");
    assert!(attempt("").is_err(), "missing token must be rejected");
    assert!(attempt("s3cret1").is_err(), "longer token must be rejected");
    assert_eq!(attempt("s3cret").expect("right token"), 9);
}
