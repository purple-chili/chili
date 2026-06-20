//! `count_seq_messages_strict` — strict validation tests.
//!
//! Verifies that the strict variant returns `Err` (not silent break) on:
//! - Torn-tail headers (partial 16-byte header at end)
//! - Header OK + partial payload
//! - Valid frame then garbage payload (corrupt deserialize)
//! - Zero-size frame sentinel
//!
//! And that a clean file passes without error.

use std::io::{Read, Write};

use chili_core::{SpicyObj, serde9, utils};

/// Write `count` valid sequence messages to a file and return the path.
fn write_valid_seq(dir: &tempfile::TempDir, name: &str, count: usize) -> String {
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

/// Open a seq file, skip the 8-byte magic header, and call count_seq_messages_strict.
fn strict_validate(
    path: &str,
    must_deserialize: bool,
) -> Result<(i64, u64), chili_core::SpicyError> {
    let mut file = std::fs::OpenOptions::new().read(true).open(path).unwrap();

    // Skip 8-byte magic header
    let mut header = [0u8; 8];
    file.read_exact(&mut header).unwrap();

    utils::count_seq_messages_strict(&mut file, must_deserialize)
}

#[test]
fn strict_clean_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "clean.seq", 10);

    let (count, _) = strict_validate(&path, false).unwrap();
    assert_eq!(count, 10);
}

#[test]
fn strict_clean_file_with_deserialize() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "clean_deser.seq", 5);

    let (count, _) = strict_validate(&path, true).unwrap();
    assert_eq!(count, 5);
}

#[test]
fn strict_torn_tail_partial_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "torn_header.seq", 3);

    // Append a partial header (7 bytes < 16)
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03])
        .unwrap();
    file.flush().unwrap();
    drop(file);

    let result = strict_validate(&path, false);
    assert!(result.is_err(), "should error on torn header");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("frame header"),
        "error should mention frame header: {}",
        err_msg
    );
}

#[test]
fn strict_torn_tail_partial_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "torn_payload.seq", 3);

    // Append a valid header claiming 1024 bytes but only write 10
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let fake_size: u64 = 1024;
    let fake_time: u64 = 9999;
    file.write_all(&fake_size.to_le_bytes()).unwrap();
    file.write_all(&fake_time.to_le_bytes()).unwrap();
    file.write_all(&[0u8; 10]).unwrap();
    file.flush().unwrap();
    drop(file);

    let result = strict_validate(&path, true);
    assert!(result.is_err(), "should error on truncated payload");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("truncated frame payload"),
        "error should mention truncated payload: {}",
        err_msg
    );
}

#[test]
fn strict_corrupt_frame_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "corrupt_frame.seq", 4);

    // Append a frame with garbage payload
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let garbage = vec![0x50u8; 8]; // type code 0x50 is unrecognized by serde9
    let size = garbage.len() as u64;
    let utc_time: u64 = 9999;
    file.write_all(&size.to_le_bytes()).unwrap();
    file.write_all(&utc_time.to_le_bytes()).unwrap();
    file.write_all(&garbage).unwrap();
    file.flush().unwrap();
    drop(file);

    let result = strict_validate(&path, true);
    assert!(result.is_err(), "should error on corrupt frame");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("corrupt frame"),
        "error should mention corrupt frame: {}",
        err_msg
    );
}

#[test]
fn strict_zero_size_frame() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "zero_size.seq", 2);

    // Append a zero-size frame header
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(&0u64.to_le_bytes()).unwrap();
    file.write_all(&1000u64.to_le_bytes()).unwrap();
    file.flush().unwrap();
    drop(file);

    let result = strict_validate(&path, false);
    assert!(result.is_err(), "should error on zero-length frame");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("zero-length frame"),
        "error should mention zero-length frame: {}",
        err_msg
    );
}

#[test]
fn strict_skip_path_does_not_mask_errors() {
    // Without must_deserialize, a valid-header + seek-overshoot should still error
    let dir = tempfile::tempdir().unwrap();
    let path = write_valid_seq(&dir, "seek_overshoot.seq", 2);

    // Append a frame with a size that overshoots EOF
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    let huge_size: u64 = u64::MAX / 2;
    let utc_time: u64 = 9999;
    file.write_all(&huge_size.to_le_bytes()).unwrap();
    file.write_all(&utc_time.to_le_bytes()).unwrap();
    file.flush().unwrap();
    drop(file);

    let result = strict_validate(&path, false);
    assert!(result.is_err(), "should error when seek overshoots EOF");
}
