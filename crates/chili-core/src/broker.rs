use std::{
    collections::HashMap,
    fs::{self},
    io::Read,
    sync::LazyLock,
};

use crate::{
    ArgType, ConnType, EngineState, Func, SpicyError, SpicyObj, SpicyResult, Stack, serde9, utils,
    validate_args,
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

    let conn_type = utils::detect_conn_type(&mut file)?;
    match conn_type {
        ConnType::New => return Ok(SpicyObj::I64(0)),
        ConnType::Sequence => {
            // detect_conn_type read 4 bytes; skip the remaining 4 of the 8-byte magic header
            let mut pad = [0u8; 4];
            file.read_exact(&mut pad)
                .map_err(|e| SpicyError::Err(format!("failed to read header '{}': {}", path, e)))?;
        }
        _ => return Err(SpicyError::Err(format!("not a sequence file '{}'", path))),
    }
    let (count, valid_size) = utils::count_seq_messages(&mut file, must_deserialize)?;
    file.set_len(valid_size)
        .map_err(|e| SpicyError::Err(format!("failed to set valid size '{}': {}", path, e)))?;
    Ok(SpicyObj::I64(count))
}

fn validate_seq_strict(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Boolean])?;
    let path = args[0].str().unwrap();
    let must_deserialize = args[1].to_bool().unwrap();
    let mut file = fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(|e| SpicyError::Err(format!("failed to open file '{}': {}", path, e)))?;

    let conn_type = utils::detect_conn_type(&mut file)?;
    match conn_type {
        ConnType::New => return Ok(SpicyObj::I64(0)),
        ConnType::Sequence => {
            let mut pad = [0u8; 4];
            file.read_exact(&mut pad)
                .map_err(|e| SpicyError::Err(format!("failed to read header '{}': {}", path, e)))?;
        }
        _ => return Err(SpicyError::Err(format!("not a sequence file '{}'", path))),
    }
    let (count, _) = utils::count_seq_messages_strict(&mut file, must_deserialize)?;
    Ok(SpicyObj::I64(count))
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
            ".broker.validateSeqStrict".to_owned(),
            Func::new_built_in_fn(
                Some(Box::new(validate_seq_strict)),
                2,
                ".broker.validateSeqStrict",
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
