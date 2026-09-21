use std::{
    collections::HashMap,
    fs::{self},
    io::Read,
    sync::LazyLock,
};

use crate::{
    ArgType, ConnType, EngineState, Func, SpicyError, SpicyObj, SpicyResult, Stack, SubFilter,
    utils, validate_args,
};

// message broker functions

/// Topic list argument: str | sym | str/sym series | mixed list of str/sym.
fn topic_list(arg: &SpicyObj) -> SpicyResult<Vec<&str>> {
    arg.to_str_vec().map_err(|_| {
        SpicyError::MismatchedTypeErr("str(s) | sym(s)".to_owned(), arg.get_type_name())
    })
}

fn publish(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::StrOrSym, ArgType::Any])?;
    let table = args[1].str().unwrap();
    let message = args[2];
    state.publish(args[0], args[1], table, message)?;
    Ok(SpicyObj::Null)
}

/// Register `handle` on `topics` (pending until the sync Response is written,
/// see `activate_subscribers`) and return the replay bound `tick[0]` read in
/// the same `lpt_lock` section, so callers hand subscribers a bound that no
/// concurrent `lpt` can slip past.
/// Finish a `.broker.subscribe*` registration for `handle`.
///
/// A pending entry is normally activated by `handle`'s own connection thread
/// once its sync Response is written. When the call does not come from that
/// connection (REPL, embedding caller, a job, another connection) no such
/// Response exists, so the entry goes live at once instead of buffering frames
/// forever. If promotion fails, the pending entry is rolled back.
fn finish_subscribe(state: &EngineState, stack: &Stack, handle: i64) -> SpicyResult<()> {
    if let Err(e) = state.handle_subscriber(&handle) {
        state.drop_pending_subscribers(handle);
        return Err(e);
    }
    if stack.h != handle {
        state.activate_subscribers(handle);
    }
    Ok(())
}

fn subscribe(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::Any])?;
    let handle = args[0].to_i64().unwrap();
    // str | sym | str/sym series | mixed list of str/sym (what `.tick.subscribe` passes).
    let topics = topic_list(args[1])?;
    let bound = state.subscribe_pending(&topics, handle, None)?;
    // update connection type to publishing
    finish_subscribe(state, stack, handle)?;
    Ok(SpicyObj::I64(bound))
}

/// Register a subscriber on one topic with an optional row filter and return
/// the replay bound `tick[0]` (same contract as `.broker.subscribe`).
/// Empty `values` means no filter.
fn subscribe_filtered(
    state: &EngineState,
    stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::Int,
            ArgType::StrOrSym,
            ArgType::StrOrSym,
            ArgType::Any,
        ],
    )?;
    let handle = args[0].to_i64().unwrap();
    let topic = args[1].str().unwrap();
    let column = args[2].str().unwrap().to_owned();
    let values = args[3].to_str_vec().map_err(|e| {
        SpicyError::Err(format!(
            "expect symbol/string list for 4th argument, got '{}'",
            e
        ))
    })?;
    let filter = if values.is_empty() {
        None
    } else {
        Some(SubFilter::new(
            column,
            values.into_iter().map(|s| s.to_owned()).collect(),
        ))
    };
    let bound = state.subscribe_pending(&[topic], handle, filter)?;
    finish_subscribe(state, stack, handle)?;
    Ok(SpicyObj::I64(bound))
}

fn unsubscribe(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::Any])?;
    let handle = args[0].to_i64().unwrap();
    let topics = topic_list(args[1])?;
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
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .len()
        == 0
    {
        return Ok(SpicyObj::I64(0));
    }

    if utils::file_starts_with_gzip(&mut file)? {
        let (count, _, _) = utils::count_sequence_file_messages(path, must_deserialize, false)?;
        return Ok(SpicyObj::I64(count));
    }

    let conn_type = utils::detect_conn_type(&mut file)?;
    match conn_type {
        ConnType::New => Ok(SpicyObj::I64(0)),
        ConnType::Sequence => {
            let mut pad = [0u8; 4];
            file.read_exact(&mut pad)
                .map_err(|e| SpicyError::Err(format!("failed to read header '{}': {}", path, e)))?;
            let (count, valid_size) = utils::count_seq_messages(&mut file, must_deserialize)?;
            file.set_len(valid_size)
                .map_err(|e| SpicyError::Err(format!("failed to set valid size '{}': {}", path, e)))?;
            Ok(SpicyObj::I64(count))
        }
        _ => Err(SpicyError::Err(format!("not a sequence file '{}'", path))),
    }
}

fn validate_seq_strict(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Boolean])?;
    let path = args[0].str().unwrap();
    let must_deserialize = args[1].to_bool().unwrap();
    let (count, _, _) = utils::count_sequence_file_messages(path, must_deserialize, true)?;
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
            ".broker.subscribeFiltered".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(subscribe_filtered)),
                4,
                ".broker.subscribeFiltered",
                &["handle", "topic", "column", "values"],
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
