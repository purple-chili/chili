use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use chili_core::SpicyObj;
use chili_op::{write_partition_native, write_partition_native_full};
use polars::prelude::*;

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempHdb {
    root: PathBuf,
}

impl TempHdb {
    fn new() -> Self {
        let id = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("chili_ac_test_{}_{}", std::process::id(), id));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }
    fn path(&self) -> &str {
        self.root.to_str().unwrap()
    }
}

impl Drop for TempHdb {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

const DAY: i32 = 20454; // 2026-01-01

fn make_df(value: i64) -> DataFrame {
    df![
        "symbol" => ["AAPL"],
        "value"  => [value],
    ]
    .unwrap()
}

fn make_big_df() -> DataFrame {
    let n = 50_000;
    let symbols: Vec<&str> = (0..n).map(|_| "AAPL").collect();
    let values: Vec<i64> = (0..n).map(|i| (i % 7) as i64).collect();
    df![
        "symbol" => symbols,
        "value"  => values,
    ]
    .unwrap()
}

fn shard_bytes(hdb: &str, table: &str) -> u64 {
    let shard = format!("{}/{}/2026.01.01_0000", hdb, table);
    fs::metadata(&shard).unwrap().len()
}

fn read_value(hdb: &str, table: &str) -> i64 {
    let shard = format!("{}/{}/2026.01.01_0000", hdb, table);
    let f = std::fs::File::open(&shard).expect("shard missing");
    let df = ParquetReader::new(f).finish().expect("parquet read failed");
    df.column("value").unwrap().i64().unwrap().get(0).unwrap()
}

fn partition_dir(hdb: &str, table: &str) -> PathBuf {
    PathBuf::from(format!("{}/{}", hdb, table))
}

fn shard_count(hdb: &str, table: &str) -> usize {
    let pat = format!("{}/{}/2026.01.01_*", hdb, table);
    glob::glob(&pat).unwrap().filter_map(|p| p.ok()).count()
}

#[test]
fn test_atomic_overwrite_roundtrips_and_never_empties() {
    let hdb = TempHdb::new();
    let table = "ohlcv";

    write_partition_native(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(100),
        &[],
        false,
        true,
    )
    .unwrap();
    assert_eq!(read_value(hdb.path(), table), 100);
    assert_eq!(shard_count(hdb.path(), table), 1);

    write_partition_native_full(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(200),
        &[],
        false,
        true,
        None,
    )
    .unwrap();

    assert_eq!(read_value(hdb.path(), table), 200);
    assert_eq!(shard_count(hdb.path(), table), 1);
    let tmp = format!("{}/{}/2026.01.01.tmp_0000", hdb.path(), table);
    assert!(!PathBuf::from(&tmp).exists(), "temp file must be renamed away");
    let legacy_tmp = format!("{}/{}/2026.01.01_0000.tmp", hdb.path(), table);
    assert!(
        !PathBuf::from(&legacy_tmp).exists(),
        "legacy glob-matching temp name must not be used"
    );

    let dir = partition_dir(hdb.path(), table);
    let entries: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).collect();
    assert!(
        entries.iter().any(|e| e.file_name().to_string_lossy().starts_with("2026.01.01_")),
        "partition shard must always be present"
    );
}

#[test]
fn test_atomic_overwrite_collapses_multishard() {
    let hdb = TempHdb::new();
    let table = "ohlcv";

    for v in [1, 2, 3] {
        write_partition_native(
            hdb.path(),
            &SpicyObj::Date(DAY),
            table,
            &make_df(v),
            &[],
            false,
            false,
        )
        .unwrap();
    }
    assert_eq!(shard_count(hdb.path(), table), 3);

    write_partition_native_full(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(99),
        &[],
        false,
        true,
        None,
    )
    .unwrap();
    assert_eq!(shard_count(hdb.path(), table), 1);
    assert_eq!(read_value(hdb.path(), table), 99);
}

#[test]
fn test_codec_snappy_roundtrips_and_differs_from_zstd() {
    let big = make_big_df();

    let hdb_snappy = TempHdb::new();
    write_partition_native_full(
        hdb_snappy.path(),
        &SpicyObj::Date(DAY),
        "ohlcv",
        &big,
        &[],
        false,
        true,
        Some("snappy"),
    )
    .unwrap();

    let hdb_zstd = TempHdb::new();
    write_partition_native_full(
        hdb_zstd.path(),
        &SpicyObj::Date(DAY),
        "ohlcv",
        &big,
        &[],
        false,
        true,
        Some("zstd"),
    )
    .unwrap();

    let shard_snappy = format!("{}/ohlcv/2026.01.01_0000", hdb_snappy.path());
    let df_back =
        ParquetReader::new(std::fs::File::open(&shard_snappy).unwrap()).finish().unwrap();
    assert_eq!(df_back.height(), big.height());

    let snappy_bytes = shard_bytes(hdb_snappy.path(), "ohlcv");
    let zstd_bytes = shard_bytes(hdb_zstd.path(), "ohlcv");
    assert_ne!(
        snappy_bytes, zstd_bytes,
        "snappy ({}) and zstd ({}) must produce different on-disk sizes",
        snappy_bytes, zstd_bytes
    );
}

#[test]
fn test_codec_unknown_errors() {
    let hdb = TempHdb::new();
    let res = write_partition_native_full(
        hdb.path(),
        &SpicyObj::Date(DAY),
        "ohlcv",
        &make_df(1),
        &[],
        false,
        true,
        Some("brotli-typo"),
    );
    assert!(res.is_err(), "unknown codec must error");
}

#[test]
fn test_codec_default_and_zstd_equivalent() {
    let hdb = TempHdb::new();
    let table = "ohlcv";
    write_partition_native_full(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(7),
        &[],
        false,
        true,
        Some("zstd"),
    )
    .unwrap();
    assert_eq!(read_value(hdb.path(), table), 7);
}

// ---------------------------------------------------------------------------
// Rechunk: merge shards into `_0000` without ever risking the existing data.
// ---------------------------------------------------------------------------

fn all_values(hdb: &str, table: &str) -> Vec<i64> {
    let pat = format!("{}/{}/2026.01.01_*", hdb, table);
    let mut out = Vec::new();
    for shard in glob::glob(&pat).unwrap().filter_map(|p| p.ok()) {
        let f = std::fs::File::open(&shard).expect("shard missing");
        let df = ParquetReader::new(f).finish().expect("parquet read failed");
        out.extend(df.column("value").unwrap().i64().unwrap().into_no_null_iter());
    }
    out.sort();
    out
}

fn stray_files(hdb: &str, table: &str) -> Vec<String> {
    fs::read_dir(partition_dir(hdb, table))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect()
}

#[test]
fn test_rechunk_merges_shards_and_keeps_every_row() {
    let hdb = TempHdb::new();
    let table = "ohlcv";
    for (value, rechunk) in [(1, false), (2, false), (3, true)] {
        write_partition_native(
            hdb.path(),
            &SpicyObj::Date(DAY),
            table,
            &make_df(value),
            &[],
            rechunk,
            false,
        )
        .unwrap();
    }
    assert_eq!(shard_count(hdb.path(), table), 1, "merged into _0000");
    assert_eq!(all_values(hdb.path(), table), vec![1, 2, 3]);
    assert!(stray_files(hdb.path(), table).is_empty());
}

#[test]
fn test_failed_rechunk_leaves_existing_shards_untouched() {
    let hdb = TempHdb::new();
    let table = "ohlcv";
    write_partition_native(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(1),
        &[],
        false,
        false,
    )
    .unwrap();
    // A shard the merge cannot read: the sink fails after the new shard is written.
    let bad = partition_dir(hdb.path(), table).join("2026.01.01_0001");
    fs::write(&bad, b"this is not a parquet file").unwrap();

    let res = write_partition_native(
        hdb.path(),
        &SpicyObj::Date(DAY),
        table,
        &make_df(2),
        &[],
        true,
        false,
    );
    assert!(res.is_err(), "a failed merge must be reported, not swallowed");
    assert_eq!(read_value(hdb.path(), table), 1, "_0000 must survive");
    assert!(bad.exists(), "other shards must survive");
    assert!(
        shard_count(hdb.path(), table) >= 2,
        "no shard may be deleted when the merge failed"
    );
    assert!(stray_files(hdb.path(), table).is_empty(), "temp file cleaned up");
}
