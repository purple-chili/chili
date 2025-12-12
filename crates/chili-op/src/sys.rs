use crate::random::set_global_random_seed;
use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};
use indexmap::IndexMap;
use polars::{
    frame::DataFrame,
    prelude::{Categories, Column, DataType, TimeUnit},
};

use std::{env, process::Command, time::SystemTime};

// rows, cols
pub fn console(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::Int])?;
    let rows = args[0].to_i64()?;
    let cols = args[1].to_i64()?;
    unsafe {
        env::set_var("POLARS_FMT_MAX_ROWS", rows.to_string());
        env::set_var("POLARS_FMT_MAX_COLS", cols.to_string());
    };
    Ok(SpicyObj::Null)
}

pub fn seed(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int])?;
    let arg0 = args[0];
    if arg0.is_integer() {
        let seed = arg0.to_i64().unwrap();
        set_global_random_seed(seed as u64);
        Ok(SpicyObj::I64(seed))
    } else {
        Err(SpicyError::Err("Requires int for 'seed'".to_owned()))
    }
}

pub fn getenv(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let key = args[0].str().unwrap();
    Ok(SpicyObj::String(env::var(key).unwrap_or("".to_owned())))
}

pub fn system(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Str])?;
    let cmd: Vec<&str> = args[0].str().unwrap().split_whitespace().collect();
    if cmd.is_empty() {
        Err(SpicyError::Err("Empty cmd".to_owned()))
    } else {
        let output = match Command::new(cmd[0]).args(&cmd[1..]).output() {
            Ok(output) => output,
            Err(e) => return Err(SpicyError::Err(e.to_string())),
        };
        Ok(SpicyObj::String(format!(
            "{}",
            String::from_utf8_lossy(&output.stdout)
        )))
    }
}

pub fn setenv(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Str, ArgType::Str])?;
    let name = args[0].str()?;
    let value = args[1].str()?;
    unsafe { env::set_var(name, value) };
    Ok(SpicyObj::Null)
}

pub fn nyi(_: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    Err(SpicyError::Err("Not yet implemented".to_owned()))
}

pub fn sleep(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int])?;
    let ms = args[0].to_i64()?;
    std::thread::sleep(std::time::Duration::from_millis(ms as u64));
    Ok(SpicyObj::Null)
}

pub fn glob(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let path = args[0].str()?;
    let files = glob::glob(path).map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut paths = Vec::new();
    let mut names = Vec::new();
    let mut sizes = Vec::new();
    let mut types = Vec::new();
    let mut mod_times = Vec::new();
    for entry in files {
        match entry {
            Ok(entry) => {
                paths.push(entry.display().to_string());
                names.push(entry.file_name().unwrap().to_string_lossy().to_string());
                let metadata = entry.metadata().unwrap();
                let mod_time = metadata
                    .modified()
                    .unwrap()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap();
                mod_times.push(mod_time.as_nanos() as i64);
                sizes.push(metadata.len() as i64);
                let filepath = if metadata.is_file() {
                    "file"
                } else if metadata.is_dir() {
                    "dir"
                } else if metadata.is_symlink() {
                    "symlink"
                } else {
                    "unknown"
                };
                types.push(filepath);
            }
            Err(e) => return Err(SpicyError::Err(e.to_string())),
        }
    }
    Ok(SpicyObj::DataFrame(
        DataFrame::new(vec![
            Column::new("path".into(), paths),
            Column::new("name".into(), names),
            Column::new("size".into(), sizes),
            Column::new("type".into(), types)
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap(),
            Column::new("mod_time".into(), mod_times)
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
        ])
        .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn mem(_args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let sys = sysinfo::System::new_all();
    let process =
        sys.process(sysinfo::get_current_pid().map_err(|e| SpicyError::Err(e.to_string()))?);
    if let Some(process) = process {
        let memory_usage = process.memory();
        let memory_usage_gb = memory_usage as f64 / 1048576.0;
        let mut dict = IndexMap::new();
        let memory_limit = std::env::var("CHILI_MEMORY_LIMIT").unwrap_or("0".to_owned());
        dict.insert("used".to_owned(), SpicyObj::F64(memory_usage_gb));
        dict.insert(
            "limit".to_owned(),
            SpicyObj::F64(memory_limit.parse::<f64>().unwrap_or(0.0)),
        );
        dict.insert(
            "total".to_owned(),
            SpicyObj::F64(sys.total_memory() as f64 / 1048576.0),
        );
        dict.insert(
            "avail".to_owned(),
            SpicyObj::F64(sys.free_memory() as f64 / 1048576.0),
        );
        Ok(SpicyObj::Dict(dict))
    } else {
        Err(SpicyError::Err("Process not found".to_owned()))
    }
}
