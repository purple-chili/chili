//! `replay_chili_msgs_log` — torn-tail tolerance tests.
//!
//! Verifies that `replay_chili_msgs_log` correctly handles:
//! - Clean tplogs (no torn tail)
//! - Torn-tail headers (partial 16-byte header at end)
//! - Header OK + partial payload
//! - Valid frame then garbage payload (corrupt deserialize)
//! - Skip-path seek overshoot (torn size field)

use std::io::Write;
use std::sync::Arc;

use chili_core::{EngineState, SpicyObj, serde9};

fn new_engine() -> Arc<EngineState> {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    Arc::new(state)
}

/// Write `count` valid sequence messages to a file and return the path.
/// Each message is `[upd, trade, i]` for i in 0..count.
fn write_valid_tplog(dir: &tempfile::TempDir, name: &str, count: usize) -> String {
    let path = dir.path().join(name);
    let paths = path.to_str().unwrap().to_owned();

    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .unwrap();

    // 8-byte sequence magic header
    file.write_all(&[255, 0, 0, 0, 0, 0, 0, 0]).unwrap();

    for i in 0..count {
        let msg = SpicyObj::MixedList(vec![
            SpicyObj::Symbol("upd".into()),
            SpicyObj::Symbol("trade".into()),
            SpicyObj::I64(i as i64),
        ]);
        let payload = serde9::serialize(&msg, false).unwrap();
        let payload_bytes: Vec<u8> = payload.iter().flat_map(|v| v.iter().copied()).collect();
        let size = payload_bytes.len() as u64;
        let utc_time = 1000u64 + i as u64;

        // 16-byte frame header: [8-byte size][8-byte timestamp]
        file.write_all(&size.to_le_bytes()).unwrap();
        file.write_all(&utc_time.to_le_bytes()).unwrap();
        file.write_all(&payload_bytes).unwrap();
    }

    file.flush().unwrap();
    file.sync_all().unwrap();
    paths
}

#[test]
fn replay_clean_tplog() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "clean.seq", 10);

    let engine = new_engine();
    let result = engine
        .replay_chili_msgs_log(&path, 0, 10, 0, &vec![], false, 0)
        .unwrap();

    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(list.len(), 10, "should replay all 10 messages");
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}

#[test]
fn replay_torn_tail_partial_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "torn_header.seq", 5);

    // Append a partial header (less than 16 bytes) to simulate a torn tail.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03])
        .unwrap();
    file.flush().unwrap();
    drop(file);

    let engine = new_engine();
    let result = engine
        .replay_chili_msgs_log(&path, 0, 100, 0, &vec![], false, 0)
        .unwrap();

    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(
                list.len(),
                5,
                "should replay exactly the 5 valid messages, ignoring torn header"
            );
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}

#[test]
fn replay_torn_tail_partial_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "torn_payload.seq", 3);

    // Append a valid 16-byte header claiming a large payload, but only write a few bytes.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let fake_size: u64 = 1024;
    let fake_time: u64 = 9999;
    file.write_all(&fake_size.to_le_bytes()).unwrap();
    file.write_all(&fake_time.to_le_bytes()).unwrap();
    // Only write 10 bytes of the claimed 1024
    file.write_all(&[0u8; 10]).unwrap();
    file.flush().unwrap();
    drop(file);

    let engine = new_engine();
    let result = engine
        .replay_chili_msgs_log(&path, 0, 100, 0, &vec![], false, 0)
        .unwrap();

    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(
                list.len(),
                3,
                "should replay exactly the 3 valid messages, stopping at torn payload"
            );
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}

#[test]
fn replay_corrupt_frame_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "corrupt_frame.seq", 4);

    // Append a frame with a valid header but garbage payload that serde9 cannot decode.
    // Use a short buffer (3 bytes) to guarantee a deserialization failure.
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let garbage = vec![0xFEu8; 3];
    let size = garbage.len() as u64;
    let utc_time: u64 = 9999;
    file.write_all(&size.to_le_bytes()).unwrap();
    file.write_all(&utc_time.to_le_bytes()).unwrap();
    file.write_all(&garbage).unwrap();
    file.flush().unwrap();
    drop(file);

    let engine = new_engine();
    let result = engine
        .replay_chili_msgs_log(&path, 0, 100, 0, &vec![], false, 0)
        .unwrap();

    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(
                list.len(),
                4,
                "should replay exactly the 4 valid messages, stopping at corrupt frame"
            );
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}

#[test]
fn replay_zero_size_sentinel() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "zero_size.seq", 2);

    // Append a frame header with size=0 (torn-tail marker).
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap(); // size = 0
    file.write_all(&1000u64.to_le_bytes()).unwrap(); // timestamp (doesn't matter)
    file.flush().unwrap();
    drop(file);

    let engine = new_engine();
    let result = engine
        .replay_chili_msgs_log(&path, 0, 100, 0, &vec![], false, 0)
        .unwrap();

    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(
                list.len(),
                2,
                "should replay exactly the 2 valid messages, stopping at zero-size sentinel"
            );
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}

#[test]
fn replay_skip_path_seek_overshoot() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_tplog(&dir, "seek_overshoot.seq", 3);

    // Append a frame with a size that overshoots EOF (but valid 16-byte header).
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let huge_size: u64 = u64::MAX / 2; // way past EOF
    let utc_time: u64 = 0; // time < start_time triggers skip path
    file.write_all(&huge_size.to_le_bytes()).unwrap();
    file.write_all(&utc_time.to_le_bytes()).unwrap();
    file.flush().unwrap();
    drop(file);

    let engine = new_engine();
    // Use start_time=500 so the overshooting frame (utc_time=0) goes through the skip path.
    let result = engine
        .replay_chili_msgs_log(&path, 0, 100, 500, &vec![], false, 0)
        .unwrap();

    // All 3 valid messages have utc_time >= 1000 > 500, so they're replayed.
    // The 4th frame has utc_time=0 < 500, hits the skip path, seek overshoots → breaks.
    match result {
        SpicyObj::MixedList(list) => {
            assert_eq!(
                list.len(),
                3,
                "should replay the 3 valid messages, stopping at skip-path seek overshoot"
            );
        }
        _ => panic!("expected MixedList, got {:?}", result),
    }
}
