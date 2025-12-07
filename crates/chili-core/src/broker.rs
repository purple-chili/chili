use std::{
    collections::HashMap,
    fs::{self},
    io::{Read, Seek},
    sync::LazyLock,
};

use log::warn;

use crate::{
    ArgType, EngineState, Func, SpicyError, SpicyObj, SpicyResult, Stack, serde9, validate_args,
};

// message broker functions
fn publish(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::StrOrSym, ArgType::Any])?;
    let table = args[1].str().unwrap();
    let message = args[2];
    let bytes = serde9::serialize(
        &SpicyObj::MixedList(vec![args[0].clone(), args[1].clone(), message.clone()]),
        false,
    )?;
    state.publish(table, &bytes)?;
    Ok(SpicyObj::Null)
}

fn subscribe(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::StrLike])?;
    let handle = args[0].to_i64().unwrap();
    let topics = args[1].to_str_vec().unwrap();
    for topic in topics {
        state.add_subscriber(topic, handle)?;
    }
    // update connection type to publishing
    state.handle_subscriber(&handle)?;
    Ok(SpicyObj::Null)
}

fn unsubscribe(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::StrLike])?;
    let handle = args[0].to_i64().unwrap();
    let topics = args[1].to_str_vec().unwrap();
    for topic in topics {
        state.remove_subscriber(topic, handle)?;
    }
    Ok(SpicyObj::Null)
}

fn validate_seq(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Boolean])?;
    let path = args[0].str().unwrap();
    let must_deserialize = args[1].to_bool().unwrap();
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .map_err(|e| SpicyError::Err(format!("failed to open file '{}': {}", path, e)))?;

    if file
        .metadata()
        .map_err(|e| SpicyError::Err(format!("failed to get file size '{}': {}", path, e)))?
        .len()
        == 0
    {
        return Ok(SpicyObj::I64(0));
    }

    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|e| SpicyError::Err(format!("failed to read header '{}': {}", path, e)))?;
    if header[..4] != [255, 0, 0, 0] {
        return Err(SpicyError::Err(format!("not a sequence file '{}'", path)));
    }
    let mut i: i64 = 0;
    let mut valid_size = 8;
    let total_size = file
        .metadata()
        .map_err(|e| SpicyError::Err(format!("failed to get file size '{}': {}", path, e)))?
        .len();
    while valid_size < total_size {
        let mut header = [0u8; 16];
        let res = file.read_exact(&mut header);
        if res.is_err() {
            break;
        }
        let msg_size = u64::from_le_bytes(header[..8].try_into().unwrap());
        if msg_size == 0 {
            break;
        }
        if must_deserialize {
            let mut buffer = vec![0u8; msg_size as usize];
            file.read_exact(&mut buffer).map_err(|e| {
                warn!("failed to read message at index {}: {}", i, e);
                SpicyError::Err(e.to_string())
            })?;
            if serde9::deserialize(&buffer, &mut 0).is_err() {
                break;
            };
            i += 1;
            valid_size += msg_size + 16;
        } else {
            match file.seek_relative(msg_size as i64) {
                Ok(_) => {
                    i += 1;
                    valid_size += msg_size + 16;
                }
                Err(e) => {
                    warn!("failed to seek to next msg '{}': {}", path, e);
                    break;
                }
            }
        }
    }
    file.set_len(valid_size)
        .map_err(|e| SpicyError::Err(format!("failed to set valid size '{}': {}", path, e)))?;
    Ok(SpicyObj::I64(i))
}

fn eod(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let message = args[0];
    state.signal_eod(message)?;
    Ok(SpicyObj::Null)
}

fn list_subscribers(
    state: &EngineState,
    _stack: &mut Stack,
    _args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let d = state.list_topic_map()?;
    Ok(SpicyObj::DataFrame(d))
}

pub static BROKER_FN: LazyLock<HashMap<String, Func>> = LazyLock::new(|| {
    [
        (
            ".broker.validateSeq".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(validate_seq)),
                2,
                ".broker.validateSeq",
                &["file", "must_deserialize"],
            ),
        ),
        (
            ".broker.publish".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(publish)),
                3,
                ".broker.publish",
                &["upd_name", "table", "message"],
            ),
        ),
        (
            // call by subscriber
            ".broker.subscribe".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(subscribe)),
                2,
                ".broker.subscribe",
                &["handle", "topics"],
            ),
        ),
        (
            ".broker.eod".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eod)),
                1,
                ".broker.eod",
                &["eod_message"],
            ),
        ),
        (
            ".broker.unsubscribe".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(unsubscribe)),
                2,
                ".broker.unsubscribe",
                &["handle", "topics"],
            ),
        ),
        (
            ".broker.list".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(list_subscribers)),
                0,
                ".broker.list",
                &[],
            ),
        ),
    ]
    .into_iter()
    .collect()
});
