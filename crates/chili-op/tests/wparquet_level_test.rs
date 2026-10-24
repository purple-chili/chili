use std::fs::{self, File};

use chili_core::{EngineState, SpicyObj};
use chili_op::BUILT_IN_FN;
use polars::io::parquet::write::ParquetCompression;
use polars::prelude::*;
use polars_parquet::write::ZstdLevel;

#[test]
fn wparquet_matches_explicit_zstd_levels_and_roundtrips() {
    let dir = tempfile::tempdir().unwrap();
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    let df = df!(
        "seq" => (0..50_000i64).collect::<Vec<_>>(),
        "qty" => (0..50_000i64).map(|i| (i * 104729) % 10007).collect::<Vec<_>>(),
        "name" => (0..50_000).map(|i| format!("symbol-{}-{}", i % 97, i % 13)).collect::<Vec<_>>()
    )
    .unwrap();
    let data = SpicyObj::DataFrame(df.clone());
    let mut written = vec![];
    for level in [1, 3, 9] {
        let path = dir.path().join(format!("chili-{level}.parquet"));
        let path_arg = SpicyObj::String(path.to_str().unwrap().into());
        let result = state
            .fn_call("wparquet", &[&path_arg, &data, &SpicyObj::I64(level)])
            .unwrap();
        let actual = fs::read(&path).unwrap();
        assert_eq!(result.to_i64().unwrap(), actual.len() as i64);
        let roundtrip = ParquetReader::new(File::open(&path).unwrap())
            .finish()
            .unwrap();
        assert_eq!(roundtrip, df);

        let mut reference = vec![];
        ParquetWriter::new(&mut reference)
            .with_compression(ParquetCompression::Zstd(Some(
                ZstdLevel::try_new(level as i32).unwrap(),
            )))
            .finish(&mut df.clone())
            .unwrap();
        assert_eq!(
            actual, reference,
            "requested zstd level {level} was not honored"
        );
        written.push(actual);
    }
    assert_ne!(
        written[0], written[2],
        "fixture must distinguish zstd levels 1 and 9"
    );
}

#[test]
fn invalid_level_does_not_truncate_an_existing_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.parquet");
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    let path_arg = SpicyObj::String(path.to_str().unwrap().into());
    let data = SpicyObj::DataFrame(df!("seq" => [1i64]).unwrap());
    fs::write(&path, b"preserve this file").unwrap();
    for level in [-1, 0, 23, i64::MAX] {
        assert!(
            state
                .fn_call("wparquet", &[&path_arg, &data, &SpicyObj::I64(level)])
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"preserve this file");
    }
}
