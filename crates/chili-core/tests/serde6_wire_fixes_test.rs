//! q wire format fixes: compression framing, multi-chunk series, nested lists.

use std::io::Cursor;

use chili_core::{SpicyObj, serde6, utils};
use polars::prelude::*;

fn roundtrip(obj: &SpicyObj) -> SpicyObj {
    let bytes = serde6::serialize(obj).expect("serialize");
    serde6::deserialize(&bytes, &mut 0, false).expect("deserialize")
}

fn i64s(obj: &SpicyObj) -> Vec<Option<i64>> {
    let SpicyObj::Series(s) = obj else {
        panic!("expected series, got {}", obj.get_type_name());
    };
    s.i64().unwrap().iter().collect()
}

#[test]
fn large_remote_message_is_a_valid_compressed_q_message() {
    // 200k zeros: ~1.6 MB, far above the 1 MiB threshold, and very compressible.
    let obj = SpicyObj::Series(Series::new("".into(), vec![0i64; 200_000]));
    let body = serde6::serialize(&obj).unwrap();
    let msg = serde6::q_message(&body, 2, true);

    assert!(msg.len() < body.len() / 2, "message must actually be compressed");
    let header: [u8; 8] = msg[..8].try_into().unwrap();
    let (msg_type, total, compression) = utils::decode_header6(&header).expect("header");
    assert_eq!(msg_type, utils::MessageType::Response);
    assert_eq!(total, msg.len(), "header length covers the whole message");
    assert_eq!(compression, 1, "compression flag must be set in the header");

    let mut wire = Cursor::new(msg[8..].to_vec());
    let decoded = utils::read_q_msg(&mut wire, total - 8, compression).expect("read_q_msg");
    assert_eq!(i64s(&decoded), vec![Some(0i64); 200_000]);
}

#[test]
fn small_or_local_message_is_plain() {
    let body = serde6::serialize(&SpicyObj::I64(42)).unwrap();
    for compress in [false, true] {
        let msg = serde6::q_message(&body, 1, compress);
        assert_eq!(&msg[..4], &[1, 1, 0, 0]);
        assert_eq!(
            u32::from_le_bytes(msg[4..8].try_into().unwrap()) as usize,
            msg.len()
        );
        assert_eq!(&msg[8..], &body[..]);
    }
}

#[test]
fn multi_chunk_series_serializes_every_value_once() {
    let mut s = Series::new("".into(), vec![1i64, 2, 3]);
    s.append(&Series::new("".into(), vec![4i64, 5])).unwrap();
    s.append(&Series::new("".into(), vec![6i64])).unwrap();
    assert!(s.n_chunks() > 1, "test needs a multi-chunk series");
    let decoded = roundtrip(&SpicyObj::Series(s));
    assert_eq!(
        i64s(&decoded),
        (1..=6).map(Some).collect::<Vec<Option<i64>>>()
    );
}

#[test]
fn nested_numeric_list_with_more_values_than_lists() {
    // Two lists, four values: the value buffer is longer than the list count.
    let a = Series::new("".into(), vec![1i64, 2]);
    let b = Series::new("".into(), vec![3i64, 4, 5]);
    let s = Series::new("".into(), vec![a, b]);
    let bytes = serde6::serialize(&SpicyObj::Series(s)).expect("serialize");
    // mixed list of 2, then long vector [1,2], then long vector [3,4,5]
    let mut expected = vec![0u8, 0, 2, 0, 0, 0];
    for list in [&[1i64, 2][..], &[3i64, 4, 5][..]] {
        expected.extend_from_slice(&[7, 0]);
        expected.extend_from_slice(&(list.len() as i32).to_le_bytes());
        for v in list {
            expected.extend_from_slice(&v.to_le_bytes());
        }
    }
    assert_eq!(bytes, expected);
}

#[test]
fn nested_bool_list_writes_zero_for_false() {
    let a = Series::new("".into(), vec![false; 20]);
    let b = Series::new("".into(), vec![true, false, true]);
    let s = Series::new("".into(), vec![a, b]);
    let bytes = serde6::serialize(&SpicyObj::Series(s)).expect("serialize");
    let mut expected = vec![0u8, 0, 2, 0, 0, 0];
    expected.extend_from_slice(&[1, 0]);
    expected.extend_from_slice(&20i32.to_le_bytes());
    expected.extend_from_slice(&[0u8; 20]);
    expected.extend_from_slice(&[1, 0]);
    expected.extend_from_slice(&3i32.to_le_bytes());
    expected.extend_from_slice(&[1, 0, 1]);
    assert_eq!(bytes, expected);
}

fn decode(bytes: &[u8]) -> Result<SpicyObj, chili_core::SpicyError> {
    serde6::deserialize(bytes, &mut 0, false)
}

#[test]
fn q_char_atom_is_the_character() {
    assert_eq!(decode(&[0xf6, b'a']).unwrap(), SpicyObj::String("a".into()));
}

#[test]
fn q_symbols_need_not_be_utf8_or_terminated() {
    // Latin-1 é: not valid UTF-8 on its own
    assert!(decode(&[0xf5, 0xe9, 0x00]).is_ok());
    // no NUL terminator before the buffer ends: must not index past it
    let _ = decode(&[0xf5, b'a']);
}

#[test]
fn unsupported_q_values_are_errors_not_panics() {
    // `a`b!"xy": symbol keys, char vector values
    let dict = [
        0x63, 0x0b, 0x00, 0x02, 0, 0, 0, b'a', 0, b'b', 0, 0x0a, 0x00, 0x02, 0, 0, 0, b'x', b'y',
    ];
    assert!(decode(&dict).is_err());
    // flip of a dict with one month column (type 13)
    let table = [
        0x62, 0x00, 0x63, 0x0b, 0x00, 0x01, 0, 0, 0, b'm', 0, 0x00, 0x00, 0x01, 0, 0, 0, 0x0d,
        0x00, 0x01, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert!(decode(&table).is_err());
    // same table shape with an enum column (type 20): past the size table
    let mut enum_table = table;
    enum_table[17] = 0x14;
    assert!(decode(&enum_table).is_err());
}

#[test]
fn q_null_temporal_atoms_decode_to_null() {
    let i64_min = i64::MIN.to_le_bytes();
    let i32_min = i32::MIN.to_le_bytes();
    let nan = f64::NAN.to_le_bytes();
    let atom = |code: u8, payload: &[u8]| {
        let mut v = vec![code];
        v.extend_from_slice(payload);
        decode(&v).unwrap()
    };
    assert_eq!(atom(244, &i64_min), SpicyObj::Null, "0Np");
    assert_eq!(atom(243, &i32_min), SpicyObj::Null, "0Nm");
    assert_eq!(atom(243, &i32::MAX.to_le_bytes()), SpicyObj::Null, "0Wm");
    assert_eq!(atom(242, &i32_min), SpicyObj::Null, "0Nd");
    assert_eq!(atom(241, &nan), SpicyObj::Null, "0Nz");
    assert_eq!(atom(240, &i64_min), SpicyObj::Null, "0Nn");
    // ordinary values are unchanged: 2000.01m and 1999.12m
    assert_eq!(atom(243, &0i32.to_le_bytes()), atom(242, &0i32.to_le_bytes()));
    let dec_1999 = atom(243, &(-1i32).to_le_bytes());
    let SpicyObj::Date(d) = dec_1999 else { panic!("expected a date") };
    let SpicyObj::Date(jan_2000) = atom(242, &0i32.to_le_bytes()) else { panic!() };
    assert_eq!(jan_2000 - d, 31, "1999.12.01 is 31 days before 2000.01.01");
}
