use chrono::Local;
use polars::prelude::{IntoLazy, col, lit};

use crate::{ArgType, EngineState, SpicyError, SpicyObj, SpicyResult, Stack, validate_args};

#[derive(Debug, Clone)]
pub struct Job {
    pub fn_name: String,
    pub start_time: i64,
    pub end_time: i64,
    // duration
    pub interval: i64,
    pub last_run_time: Option<i64>,
    pub next_run_time: i64,
    pub is_active: bool,
    pub description: String,
}

pub fn list(state: &EngineState, _stack: &mut Stack, _args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let df = state.list_job()?;
    Ok(SpicyObj::DataFrame(df))
}

pub fn add(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::StrOrSym,
            ArgType::Timestamp,
            ArgType::Timestamp,
            ArgType::Duration,
            ArgType::StrOrSym,
        ],
    )?;
    let fn_name = args[0].str().unwrap();
    let start_time = args[1].to_i64().unwrap();
    let end_time = args[2].to_i64().unwrap();
    let interval = args[3].to_i64().unwrap();
    let description = args[4].str().unwrap();

    if interval < 0 {
        return Err(SpicyError::Err("Interval must be positive".to_owned()));
    }

    let job = Job {
        fn_name: fn_name.to_owned(),
        start_time,
        end_time,
        interval,
        last_run_time: None,
        next_run_time: start_time,
        is_active: true,
        description: description.to_owned(),
    };
    let id = state.add_job(job);
    Ok(SpicyObj::I64(id))
}

pub fn get_local_now_ns() -> i64 {
    let now = Local::now();
    let local_offset = now.offset().local_minus_utc() as i64;
    now.timestamp_nanos_opt().unwrap() + local_offset * 1_000_000_000
}

pub fn add_after(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[ArgType::StrOrSym, ArgType::Duration, ArgType::StrOrSym],
    )?;
    let fn_name = args[0].str().unwrap();
    let interval = args[1].to_i64().unwrap();
    let description = args[2].str().unwrap();

    if interval < 0 {
        return Err(SpicyError::Err("Interval must be positive".to_owned()));
    }

    let job = Job {
        fn_name: fn_name.to_owned(),
        start_time: get_local_now_ns() + interval,
        end_time: 0,
        interval,
        last_run_time: None,
        next_run_time: 0,
        is_active: true,
        description: description.to_owned(),
    };
    let id = state.add_job(job);
    Ok(SpicyObj::I64(id))
}

pub fn add_at_time(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[ArgType::StrOrSym, ArgType::Timestamp, ArgType::StrOrSym],
    )?;
    let fn_name = args[0].str().unwrap();
    let start_time = args[1].to_i64().unwrap();
    let description = args[2].str().unwrap();

    let job = Job {
        fn_name: fn_name.to_owned(),
        start_time,
        end_time: start_time,
        interval: 0,
        last_run_time: None,
        next_run_time: 0,
        is_active: true,
        description: description.to_owned(),
    };
    let id = state.add_job(job);
    Ok(SpicyObj::I64(id))
}

pub fn get(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let any = list(state, _stack, args)?;
    let df = any.df().unwrap();
    if arg0.is_integer() {
        let id = args[0].to_i64().unwrap();
        Ok(SpicyObj::DataFrame(
            df.clone()
                .lazy()
                .filter(col("id").eq(id))
                .collect()
                .unwrap(),
        ))
    } else if arg0.str().is_ok() {
        let pattern = args[0].str().unwrap();
        Ok(SpicyObj::DataFrame(
            df.clone()
                .lazy()
                .filter(col("description").str().contains(lit(pattern), true))
                .collect()
                .unwrap(),
        ))
    } else {
        Err(SpicyError::Err(
            "Invalid argument, requires id or pattern".to_owned(),
        ))
    }
}

pub fn activate(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_integer() {
        let id = state.set_job_status(arg0.to_i64().unwrap(), true)?;
        Ok(SpicyObj::I64(id))
    } else if arg0.str().is_ok() {
        let ids = state.set_job_status_by_pattern(arg0.str().unwrap(), true)?;
        Ok(SpicyObj::Series(ids))
    } else {
        Err(SpicyError::Err(
            "Invalid argument, requires id or pattern".to_owned(),
        ))
    }
}

pub fn deactivate(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_integer() {
        let id = state.set_job_status(arg0.to_i64().unwrap(), false)?;
        Ok(SpicyObj::I64(id))
    } else if arg0.str().is_ok() {
        let ids = state.set_job_status_by_pattern(arg0.str().unwrap(), false)?;
        Ok(SpicyObj::Series(ids))
    } else {
        Err(SpicyError::Err(
            "Invalid argument, requires id or pattern".to_owned(),
        ))
    }
}

pub fn clear(
    state: &EngineState,
    _stack: &mut Stack,
    _args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    state.clear_job()?;
    Ok(SpicyObj::I64(0))
}
