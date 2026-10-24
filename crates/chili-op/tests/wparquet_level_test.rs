use std::fs::{self, File};

use chili_core::{EngineState, SpicyObj};
use chili_op::BUILT_IN_FN;
use polars::io::parquet::write::{ParquetCompression, ParquetWriteOptions};
use polars::prelude::*;
use polars_parquet::write::ZstdLevel;
use std::sync::Arc;

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

        let reference_path = dir.path().join(format!("reference-{level}.parquet"));
        df.clone()
            .lazy()
            .sink(
                SinkDestination::File {
                    target: SinkTarget::Path(PlRefPath::new(reference_path.to_str().unwrap())),
                },
                FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions {
                    compression: ParquetCompression::Zstd(Some(
                        ZstdLevel::try_new(level as i32).unwrap(),
                    )),
                    ..Default::default()
                })),
                UnifiedSinkArgs::default(),
            )
            .unwrap()
            .collect_with_engine(Engine::Streaming)
            .unwrap();
        let reference = fs::read(reference_path).unwrap();
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
fn wparquet_preserves_order_nulls_and_empty_schema_with_many_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("streaming.parquet");
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    let path_arg = SpicyObj::String(path.to_str().unwrap().into());
    let mut df = df!("seq" => Vec::<i64>::new(), "value" => Vec::<Option<&str>>::new()).unwrap();
    for i in 0..300i64 {
        df.vstack_mut(
            &df!("seq" => [i], "value" => [if i % 7 == 0 { None } else { Some("value") }]).unwrap(),
        )
        .unwrap();
    }
    for frame in [df.clone(), df.clear()] {
        let result = state
            .fn_call(
                "wparquet",
                &[
                    &path_arg,
                    &SpicyObj::DataFrame(frame.clone()),
                    &SpicyObj::I64(3),
                ],
            )
            .unwrap();
        assert_eq!(
            result.to_i64().unwrap(),
            fs::metadata(&path).unwrap().len() as i64
        );
        let read = ParquetReader::new(File::open(&path).unwrap())
            .finish()
            .unwrap();
        assert_eq!(read, frame);
    }
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
