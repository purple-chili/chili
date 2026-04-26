use polars::{
    chunked_array::ops::SortMultipleOptions,
    io::{
        SerReader, SerWriter,
        csv::read::{CsvParseOptions, CsvReadOptions},
        parquet::write::ParquetWriteOptions,
    },
    lazy::{
        dsl::col,
        frame::{LazyFrame, ScanArgsParquet},
    },
    prelude::{
        Categories, CsvWriter, FileWriteFormat, IntoLazy, JsonFormat, JsonReader, JsonWriter,
        ParquetReader, PlRefPath, PlSmallStr, SinkDestination, SinkTarget, UnifiedSinkArgs,
    },
};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{Arc, RwLock},
};

use polars::{
    datatypes::{DataType, Field, TimeUnit},
    prelude::{Schema, SchemaRef},
};
use std::sync::LazyLock;

use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};

/// Proposal O — process-wide cache of `fs::canonicalize` results for HDB
/// paths, keyed by the input string. `wpar` is called in tight loops by
/// some downstream users (mdata's batch ingest, partition migration
/// scripts), and `fs::canonicalize` is a syscall that hits the filesystem
/// every time. The canonicalized path of an HDB root is invariant for the
/// lifetime of the process (no one is going to `mv` the HDB root mid-run),
/// so caching is safe and lock-light. RwLock because the read path is
/// massively dominant — write only happens once per unique HDB path.
static CANON_CACHE: LazyLock<RwLock<HashMap<String, PathBuf>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn canon_cached(hdb_path: &str) -> SpicyResult<PathBuf> {
    if let Ok(cache) = CANON_CACHE.read()
        && let Some(p) = cache.get(hdb_path)
    {
        return Ok(p.clone());
    }
    // Slow path: canonicalize and insert.
    let canon = fs::canonicalize(hdb_path).map_err(|e| SpicyError::Err(e.to_string()))?;
    if let Ok(mut cache) = CANON_CACHE.write() {
        cache.insert(hdb_path.to_string(), canon.clone());
    }
    Ok(canon)
}

use crate::util;

// path, has_header, separator, ignore_errors, dtypes
pub fn read_csv(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::StrOrSym,
            ArgType::Boolean,
            ArgType::Str,
            ArgType::Boolean,
            ArgType::Any,
        ],
    )?;

    let file = args[0].str()?;
    let has_header = args[1].to_bool()?;
    let separator = args[2].str()?;
    let ignore_errors = args[3].to_bool()?;
    let dtypes = args[4];

    let mut schema_ref: Option<SchemaRef> = None;
    let mut columns = None;

    // support dict only, as slice of datatype could fail
    if dtypes.is_dict() && dtypes.size() > 0 {
        let dict = dtypes.dict().unwrap();
        let mut fields: Vec<Field> = Vec::with_capacity(dtypes.size());
        let mut cols: Vec<PlSmallStr> = Vec::with_capacity(dtypes.size());
        for i in 0..dtypes.size() {
            let (key, value) = dict.get_index(i).unwrap();
            fields.push(Field::new(
                key.into(),
                map_str_to_polars_dtype(value.str()?)?,
            ));
            cols.push(key.into());
        }
        let schema: Schema = fields.into_iter().collect();
        schema_ref = Some(schema.into());
        columns = Some(Arc::from(cols));
    }

    let parse_options = CsvParseOptions::default()
        .with_separator(separator.as_bytes()[0])
        .with_missing_is_null(true)
        .with_try_parse_dates(true);

    let df = CsvReadOptions::default()
        .with_has_header(has_header)
        .with_schema_overwrite(schema_ref)
        .with_ignore_errors(ignore_errors)
        .with_columns(columns)
        .with_parse_options(parse_options)
        .try_into_reader_with_file_path(Some(file.into()))
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?
        .finish()
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?;

    Ok(SpicyObj::DataFrame(df))
}

// path, dtypes
pub fn read_json(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Any])?;

    let file = args[0].str().unwrap();
    let dtypes = args[1];

    let mut schema_ref: Option<SchemaRef> = None;

    // support dict only, as slice of datatype could fail
    if dtypes.is_dict() && dtypes.size() > 0 {
        let dict = dtypes.dict().unwrap();
        let mut fields: Vec<Field> = Vec::with_capacity(dtypes.size());
        let mut cols: Vec<String> = Vec::with_capacity(dtypes.size());
        for i in 0..dtypes.size() {
            let (key, value) = dict.get_index(i).unwrap();
            fields.push(Field::new(
                key.into(),
                map_str_to_polars_dtype(value.str()?)?,
            ));
            cols.push(key.to_owned());
        }
        let schema: Schema = fields.into_iter().collect();
        schema_ref = Some(schema.into());
    }

    let mut file = File::open(file).map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut reader = JsonReader::new(&mut file);
    let df = if let Some(schema) = schema_ref {
        reader = reader.with_schema_overwrite(&schema);
        reader
            .finish()
            .map_err(|e| SpicyError::Err(e.to_string()))?
    } else {
        reader
            .finish()
            .map_err(|e| SpicyError::Err(e.to_string()))?
    };

    Ok(SpicyObj::DataFrame(df))
}

// file, n_rows, rechunk, columns
pub fn read_parquet(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::StrOrSym,
            ArgType::Int,
            ArgType::Boolean,
            ArgType::Any,
        ],
    )?;
    let file = args[0].str().unwrap();
    let n_rows = args[1].to_i64().unwrap();
    let rechunk = args[2].to_bool().unwrap();
    let columns = if args[3].size() > 0 {
        match args[3] {
            SpicyObj::Symbol(s) | SpicyObj::String(s) => vec![col(s)],
            SpicyObj::Series(series) => match series.dtype() {
                DataType::String => series
                    .str()
                    .unwrap()
                    .into_iter()
                    .map(|s| col(s.unwrap_or("")))
                    .collect(),
                DataType::Categorical(_, _) => series
                    .cat32()
                    .unwrap()
                    .iter_str()
                    .map(|s| col(s.unwrap_or("")))
                    .collect(),
                _ => vec![],
            },
            _ => vec![],
        }
    } else {
        vec![]
    };
    let mut args = ScanArgsParquet {
        cache: false,
        ..Default::default()
    };
    if n_rows > 0 {
        args.n_rows = Some(n_rows as usize);
    }
    if rechunk {
        args.rechunk = rechunk;
    }
    let mut lazy_df = LazyFrame::scan_parquet(
        PlRefPath::new(Path::new(file).to_str().unwrap_or_default()),
        args,
    )
    .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
    if !columns.is_empty() {
        lazy_df = lazy_df.select(columns);
    }
    lazy_df
        .collect()
        .map_err(|e| SpicyError::EvalErr(e.to_string()))
        .map(SpicyObj::DataFrame)
}

// file path
pub fn read_txt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let file_path = args[0].str().unwrap();
    let contents = fs::read_to_string(file_path).map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::String(contents))
}

// file path, df, sep
pub fn write_csv(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::DataFrame, ArgType::Str])?;
    let file = args[0].str().unwrap();
    let df = args[1].df().unwrap();
    let sep = args[2].str().unwrap();
    if sep.len() > 1 || !sep.is_ascii() {
        return Err(SpicyError::Err(format!(
            "expect len 1 ascii string, got '{}'",
            sep
        )));
    }
    let sep = sep.as_bytes()[0];

    let mut file = File::create(file)
        .map_err(|e| SpicyError::Err(format!("failed to create file '{}': {}", file, e)))?;

    CsvWriter::new(&mut file)
        .with_separator(sep)
        .finish(&mut df.clone())
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::Null)
}

// file path, df
pub fn write_json(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::DataFrame])?;
    let file = args[0].str().unwrap();
    let df = args[1].df().unwrap();
    let mut file = File::create(file)
        .map_err(|e| SpicyError::Err(format!("failed to create file '{}': {}", file, e)))?;
    JsonWriter::new(&mut file)
        .with_json_format(JsonFormat::JsonLines)
        .finish(&mut df.clone())
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::Null)
}

// file, df, compress level
pub fn write_parquet(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::DataFrame, ArgType::Int])?;

    let file = args[0].str()?;
    let df = args[1].df()?;
    let level = args[2].to_i64()?;

    if !(1..=22).contains(&level) {
        Err(SpicyError::Err(
            "Invalid compression level: valid compression level 1-22".to_owned(),
        ))
    } else {
        util::write_parquet_to_filepath(file, df).map(|size| SpicyObj::I64(size as i64))
    }
}

// hdb_path, partition, table, df, columns, rechunk
pub fn write_partition(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::StrOrSym,
            ArgType::Any,
            ArgType::Sym,
            ArgType::DataFrame,
            ArgType::SymOrSyms,
            ArgType::Boolean,
            ArgType::Boolean,
        ],
    )?;
    let hdb_path = args[0].str().unwrap();
    let partition = args[1];
    let table_name = args[2].str().unwrap();
    let df = args[3].df().unwrap();
    let columns = args[4].to_str_vec().unwrap();
    // rechunk | append
    let rechunk = *args[5].bool().unwrap();
    let overwrite = *args[6].bool().unwrap();
    write_partition_native(
        hdb_path, partition, table_name, &df, &columns, rechunk, overwrite,
    )
}

pub fn write_partition_native(
    hdb_path: &str,
    partition: &SpicyObj,
    table_name: &str,
    df: &polars::prelude::DataFrame,
    sort_columns: &[&str],
    rechunk: bool,
    overwrite: bool,
) -> SpicyResult<SpicyObj> {
    let sort_options = SortMultipleOptions::default();
    // Proposal O — same canonicalize cache as the public `write_partition`.
    let hdb_path = canon_cached(hdb_path)?;

    // When sort_columns is non-empty, force a smaller row group
    // so polars can later prune row groups by min/max stats on the sort key.
    //
    // polars' default row_group_size is 512*512 = 262144 rows; mdata-shaped
    // partitions (10k-50k rows) end up as a single row group covering the
    // entire symbol range, defeating pruning.
    //
    // Critically: polars' `chunk_df_for_writing` uses floor division
    // (`n_splits = height / row_group_size`). To get N splits we need
    // row_group_size <= height/N. We aim for at least ~10 row groups per
    // partition, with a minimum of 1024 rows per group (otherwise polars
    // rechunks small groups back together at line 154 of chunks.rs:
    // "if df.estimated_size() / n_chunks < 128 * 1024 { rechunk }").
    //
    // Compute the target row_group_size based on the actual DataFrame
    // height — this gives good selectivity for partitions of any size.
    let row_group_size: Option<usize> = if !sort_columns.is_empty() {
        let n_rows = df.height();
        // Target ~16 row groups, with floor 1024 and ceiling 32768
        let target = (n_rows / 16).max(1024).min(32768);
        Some(target)
    } else {
        None
    };

    let mut column_names = df.get_column_names_owned();
    let mut lf = df.clone().lazy();
    if !sort_columns.is_empty() {
        lf = lf.sort(sort_columns.to_vec(), sort_options.clone());
    }

    let par_str = match partition {
        SpicyObj::Date(_) => {
            if !column_names.contains(&"date".into()) {
                column_names.insert(0, "date".into());
                lf = lf.with_column(partition.as_expr().unwrap().alias("date"));
                lf = lf.select(column_names.into_iter().map(col).collect::<Vec<_>>());
            }
            partition.to_string()
        }
        SpicyObj::I64(year) => {
            if *year >= 1000 && *year <= 3999 {
                if !column_names.contains(&"year".into()) {
                    column_names.insert(0, "year".into());
                    lf = lf.with_column(partition.as_expr().unwrap().alias("year"));
                    lf = lf.select(column_names.into_iter().map(col).collect::<Vec<_>>());
                }
                year.to_string()
            } else {
                return Err(SpicyError::Err(format!(
                    "Requires year between 1000 and 3999, got '{}'",
                    year
                )));
            }
        }
        SpicyObj::Null => {
            let table_path = hdb_path.join(table_name);
            if table_path.exists() && table_path.is_dir() {
                return Err(SpicyError::Err(format!(
                    "{} is a directory, cannot write single dataframe to it",
                    table_name
                )));
            } else {
                if table_path.exists() && !overwrite {
                    return Err(SpicyError::Err(format!(
                        "{} already exists, skip writing as overwrite is false",
                        table_name
                    )));
                }
                return util::write_parquet_to_filepath(table_path.to_string_lossy().as_ref(), df)
                    .map(|size| SpicyObj::I64(size as i64));
            }
        }
        _ => {
            return Err(SpicyError::Err(format!(
                "Requires date or i64 or 0n as partition, got '{}'",
                partition.get_type_name()
            )));
        }
    };

    let df = lf
        .collect()
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?;

    let table_path = hdb_path.join(table_name);
    if !table_path.exists() {
        fs::create_dir(&table_path).map_err(|e| SpicyError::Err(e.to_string()))?;
    }
    let par_path = if table_path.is_dir() {
        table_path.join(&par_str)
    } else {
        return Err(SpicyError::Err(format!(
            "{} is not a directory",
            table_path.display()
        )));
    };

    let schema_path = table_path.join("schema");
    if !schema_path.exists() {
        let schema = df.clear();
        util::write_parquet_to_filepath(schema_path.to_string_lossy().as_ref(), &schema)?;
    } else {
        let f = File::open(schema_path).map_err(|e| SpicyError::Err(e.to_string()))?;
        let schema_df = ParquetReader::new(f)
            .finish()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        if !schema_df.get_column_names().eq(&df.get_column_names()) {
            return Err(SpicyError::Err(format!(
                "Column names mismatch:\n\
                   - expected: {:?}\n\
                   - got:      {:?}",
                schema_df.get_column_names(),
                df.get_column_names()
            )));
        }
        let mut err_msg = String::new();
        schema_df
            .columns()
            .iter()
            .zip(df.columns().iter())
            .for_each(|(col0, col1)| {
                if col0.dtype() != col1.dtype() {
                    err_msg.push_str(&format!(
                        "Column '{}' has different data type: expected {:?}, got {:?}",
                        col0.name(),
                        col0.dtype(),
                        col1.dtype()
                    ));
                }
            });
        if !err_msg.is_empty() {
            return Err(SpicyError::Err(err_msg));
        }
    }

    let par_wild_path = format!("{}_*", par_path.display());
    let existing_sub_parts: Vec<PathBuf> = glob::glob(&par_wild_path)
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .map(|p| p.map_err(|e| SpicyError::Err(e.to_string())))
        .collect::<SpicyResult<Vec<_>>>()?;

    if overwrite && !existing_sub_parts.is_empty() {
        for path in &existing_sub_parts {
            fs::remove_file(path).map_err(|e| SpicyError::Err(e.to_string()))?;
        }
    }

    if existing_sub_parts.is_empty() || overwrite {
        let sub_par_path = format!("{}_0000", par_path.display());
        util::write_parquet_to_filepath_with_row_group_size(&sub_par_path, &df, row_group_size)
            .map(|size| SpicyObj::I64(size as i64))
    } else {
        let mut par = existing_sub_parts.len();
        let mut sub_par_path = format!("{}_{:04}", par_path.display(), par);
        while PathBuf::from_str(&sub_par_path).unwrap().exists() && par < 10000 {
            par += 1;
            sub_par_path = format!("{}_{:04}", par_path.display(), par);
        }
        if par == 10000 {
            Err(SpicyError::Err(
                "Exceed maximum sub partition number 9999".to_string(),
            ))
        } else {
            let size = util::write_parquet_to_filepath_with_row_group_size(
                &sub_par_path,
                &df,
                row_group_size,
            )?;
            if rechunk {
                let tmp_path = table_path.join("tmp");
                let args = ScanArgsParquet::default();
                let file_format =
                    FileWriteFormat::Parquet(Arc::new(ParquetWriteOptions::default()));
                let mut rechunk_lf = LazyFrame::scan_parquet(
                    PlRefPath::new(Path::new(&par_wild_path).to_str().unwrap_or_default()),
                    args,
                )
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
                if !sort_columns.is_empty() {
                    rechunk_lf = rechunk_lf.sort(sort_columns.to_vec(), sort_options);
                }
                let _ = rechunk_lf
                    .sink(
                        SinkDestination::File {
                            target: SinkTarget::Path(PlRefPath::new(
                                Path::new(&tmp_path).to_str().unwrap_or_default(),
                            )),
                        },
                        file_format,
                        UnifiedSinkArgs::default(),
                    )
                    .map_err(|e| SpicyError::EvalErr(e.to_string()))?
                    .collect();
                for path in glob::glob(&par_wild_path).unwrap() {
                    match path {
                        Ok(path) => {
                            fs::remove_file(path).map_err(|e| SpicyError::Err(e.to_string()))?
                        }
                        Err(e) => return Err(SpicyError::Err(e.to_string())),
                    }
                }
                fs::rename(tmp_path, table_path.join(format!("{}_0000", par_str)))
                    .map_err(|e| SpicyError::Err(e.to_string()))
                    .map(|_| SpicyObj::I64(size as i64))
            } else {
                Ok(SpicyObj::I64(size as i64))
            }
        }
    }
}

// file path, txt, append
pub fn write_txt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Str, ArgType::Boolean])?;
    let file_path = args[0].str().unwrap();
    let contents = args[1].str().unwrap();
    let append = args[2].bool().unwrap();
    let mut file = OpenOptions::new()
        .write(true)
        .append(*append)
        .create(true)
        .open(file_path)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    file.write_all(contents.as_bytes())
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::Null)
}

pub fn map_str_to_polars_dtype(s: &str) -> SpicyResult<DataType> {
    match DATATYPE_MAP.get(s) {
        Some(data_type) => Ok(data_type.clone()),
        None => Err(SpicyError::EvalErr(format!(
            "Not supported data type '{}'",
            s
        ))),
    }
}

static DATATYPE_MAP: LazyLock<HashMap<String, DataType>> = LazyLock::new(|| {
    [
        ("bool".to_owned(), DataType::Boolean),
        ("u8".to_owned(), DataType::UInt8),
        ("u16".to_owned(), DataType::UInt16),
        ("u32".to_owned(), DataType::UInt32),
        ("u64".to_owned(), DataType::UInt64),
        ("i8".to_owned(), DataType::Int8),
        ("i16".to_owned(), DataType::Int16),
        ("i32".to_owned(), DataType::Int32),
        ("i64".to_owned(), DataType::Int64),
        ("i128".to_owned(), DataType::Int128),
        ("f32".to_owned(), DataType::Float32),
        ("f64".to_owned(), DataType::Float64),
        ("date".to_owned(), DataType::Date),
        (
            "timestamp".to_owned(),
            DataType::Datetime(TimeUnit::Nanoseconds, None),
        ),
        (
            "datetime".to_owned(),
            DataType::Datetime(TimeUnit::Milliseconds, None),
        ),
        ("time".to_owned(), DataType::Time),
        (
            "duration".to_owned(),
            DataType::Duration(TimeUnit::Nanoseconds),
        ),
        (
            "sym".to_owned(),
            DataType::Categorical(Categories::global(), Categories::global().mapping()),
        ),
        (
            "cat".to_owned(),
            DataType::Categorical(Categories::global(), Categories::global().mapping()),
        ),
        ("str".to_owned(), DataType::String),
    ]
    .into_iter()
    .collect()
});

pub fn h_del(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let filepath = args[0].str().unwrap();
    fs::remove_file(filepath)
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(|_| SpicyObj::Null)
}

pub fn exists(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let path = args[0].str().unwrap();
    Ok(SpicyObj::Boolean(
        Path::new(path)
            .try_exists()
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}
