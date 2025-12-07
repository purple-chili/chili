use std::{
    collections::HashMap,
    fs::{self},
    io::{Read, Write},
    sync::LazyLock,
};

use crate::{ArgType, Func, SpicyError, SpicyObj, SpicyResult, serde9, validate_args};

fn write_binary(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Any])?;
    let path = args[0].str().unwrap();
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| SpicyError::Err(format!("failed to create file '{}': {}", path, e)))?;
    let bytes = serde9::serialize(args[1], true)?;
    for b in bytes {
        file.write_all(&b)
            .map_err(|e| SpicyError::Err(format!("failed to write to file '{}': {}", path, e)))?;
    }
    Ok(args[0].clone())
}

fn read_binary(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let path = args[0].str().unwrap();
    let mut file = fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| SpicyError::Err(format!("failed to open file '{}': {}", path, e)))?;
    let size = file
        .metadata()
        .map_err(|e| SpicyError::Err(format!("failed to get file size '{}': {}", path, e)))?
        .len();
    let mut bytes = vec![0u8; size as usize];
    file.read_exact(&mut bytes)
        .map_err(|e| SpicyError::Err(format!("failed to read file '{}': {}", path, e)))?;
    let obj = serde9::deserialize(&bytes, &mut 0)?;
    Ok(obj)
}

pub static IO_FN: LazyLock<HashMap<String, Func>> = LazyLock::new(|| {
    [
        (
            "wbin".to_owned(),
            Func::new_built_in_fn(Some(Box::new(write_binary)), 2, "wbin", &["path", "data"]),
        ),
        (
            "rbin".to_owned(),
            Func::new_built_in_fn(Some(Box::new(read_binary)), 1, "rbin", &["path"]),
        ),
    ]
    .into_iter()
    .collect()
});
