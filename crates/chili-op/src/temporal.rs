use chili_core::{
    ArgType, SpicyError, SpicyObj, SpicyResult, constant::UNIX_EPOCH_DAY, validate_args,
};
use chrono::{DateTime, Datelike, Local, NaiveDateTime, TimeZone, Utc};
use chrono_tz::Tz;
use polars::{
    error::{PolarsResult, polars_err},
    prelude::{
        ChunkApply, ChunkedArray, Column, DataType, DatetimeChunked, DatetimeType, Int64Type,
        IntoColumn, Logical, StringChunked, TimeUnit, lit, time_zone::parse_time_zone,
    },
};

fn parse_tz(timezone_str: &str) -> SpicyResult<Tz> {
    match timezone_str.parse::<Tz>() {
        Ok(tz) => Ok(tz),
        Err(_) => Err(SpicyError::Err(format!(
            "not a valid timezone: {}",
            timezone_str
        ))),
    }
}

pub fn now(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let timezone_str = args[0].str().unwrap();
    if timezone_str.is_empty() {
        let now = Local::now();
        let local_offset = now.offset().local_minus_utc() as i64;
        return Ok(SpicyObj::Timestamp(
            now.timestamp_nanos_opt().unwrap() + local_offset * 1_000_000_000,
        ));
    }
    let tz = parse_tz(timezone_str)?;
    let dt = Utc::now().with_timezone(&tz);
    let tz_offset = dt.fixed_offset().offset().local_minus_utc() as i64;
    Ok(SpicyObj::Timestamp(
        dt.timestamp_nanos_opt().unwrap() + tz_offset * 1_000_000_000,
    ))
}

pub fn today(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let timezone_str = args[0].str().unwrap();
    if timezone_str.is_empty() {
        return Ok(SpicyObj::Date(
            Local::now().date_naive().num_days_from_ce() - UNIX_EPOCH_DAY,
        ));
    }
    let tz = parse_tz(timezone_str)?;

    let now = Utc::now().with_timezone(&tz);
    Ok(SpicyObj::Date(
        now.date_naive().num_days_from_ce() - UNIX_EPOCH_DAY,
    ))
}

pub fn utc(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let datetime = arg0.as_expr()?;
        let timezone = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(datetime.map_many(
            convert_tz_expr,
            &[timezone, lit("UTC")],
            |_, f| Ok(f[0].clone()),
        )));
    }
    validate_args(args, &[ArgType::TimestampLike, ArgType::StrOrSym])?;
    tz(&[arg0, arg1, &SpicyObj::String("UTC".to_owned())])
}

pub fn local(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let datetime = arg0.as_expr()?;
        let timezone = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(datetime.map_many(
            convert_tz_expr,
            &[lit("UTC"), timezone],
            |_, f| Ok(f[0].clone()),
        )));
    }
    validate_args(args, &[ArgType::TimestampLike, ArgType::StrOrSym])?;
    tz(&[arg0, &SpicyObj::String("UTC".to_owned()), arg1])
}

pub fn tz(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    let arg2 = args[2];
    if arg0.is_expr() || arg1.is_expr() || arg2.is_expr() {
        let datetime = arg0.as_expr()?;
        let from_tz = arg1.as_expr()?;
        let to_tz = arg2.as_expr()?;
        return Ok(SpicyObj::Expr(datetime.map_many(
            convert_tz_expr,
            &[from_tz, to_tz],
            |_, f| Ok(f[0].clone()),
        )));
    }

    validate_args(
        args,
        &[ArgType::TimestampLike, ArgType::StrOrSym, ArgType::StrOrSym],
    )?;

    let from_tz = parse_tz(arg1.str().unwrap())?;
    let to_tz = parse_tz(arg2.str().unwrap())?;
    match arg0 {
        SpicyObj::Datetime(dt) => {
            let ndt = DateTime::from_timestamp_millis(*dt)
                .unwrap_or(DateTime::<Utc>::MAX_UTC)
                .naive_utc();
            Ok(SpicyObj::Datetime(
                from_tz
                    .from_local_datetime(&ndt)
                    .unwrap()
                    .with_timezone(&to_tz)
                    .naive_local()
                    .and_utc()
                    .timestamp_millis(),
            ))
        }
        SpicyObj::Timestamp(dt) => {
            let ndt = DateTime::<Utc>::from_timestamp_nanos(*dt).naive_utc();
            Ok(SpicyObj::Timestamp(
                from_tz
                    .from_local_datetime(&ndt)
                    .unwrap()
                    .with_timezone(&to_tz)
                    .naive_local()
                    .and_utc()
                    .timestamp_nanos_opt()
                    .unwrap_or(0),
            ))
        }
        SpicyObj::Series(s) => {
            let s = s
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap();
            let s = convert_tz_column(
                s.clone().into_column(),
                arg1.as_series().unwrap().into_column(),
                SpicyObj::String(from_tz.to_string())
                    .as_series()
                    .unwrap()
                    .into_column(),
            )
            .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::Series(s.as_materialized_series().clone()))
        }
        _ => unreachable!(),
    }
}

fn convert_tz_expr(columns: &mut [Column]) -> PolarsResult<Column> {
    let dt = columns[0].clone();
    let from_tz = columns[1].clone();
    let to_tz = columns[2].clone();
    convert_tz_column(dt, from_tz, to_tz)
}

fn convert_tz_column(dt: Column, from_tz: Column, to_tz: Column) -> PolarsResult<Column> {
    if !dt.dtype().is_datetime() {
        return Err(
            polars_err!(InvalidOperation: format!("requires datetime/timestamp, got '{}'", dt.dtype())),
        );
    }
    if !(from_tz.dtype().is_string() || from_tz.dtype().is_categorical())
        || !(to_tz.dtype().is_string() || to_tz.dtype().is_categorical())
    {
        return Err(
            polars_err!(InvalidOperation: format!("requires string/categorical timezones, got '{}' and '{}'", from_tz.dtype(), to_tz.dtype())),
        );
    }

    let ca = dt.datetime().unwrap();
    let from_tz = from_tz.cast(&DataType::String).unwrap();
    let from_tz = from_tz.str().unwrap();
    let to_tz = to_tz.cast(&DataType::String).unwrap();
    let to_tz = to_tz.str().unwrap();
    elementwise_convert_tz(ca, from_tz, to_tz).map(|out| out.into_column())
}

pub fn elementwise_convert_tz(
    datetime: &Logical<DatetimeType, Int64Type>,
    from_tz: &StringChunked,
    to_tz: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    let ts_to_ndt: fn(i64) -> NaiveDateTime = match datetime.time_unit() {
        TimeUnit::Milliseconds => |t| {
            DateTime::<Utc>::from_timestamp_millis(t)
                .unwrap_or(DateTime::<Utc>::MAX_UTC)
                .naive_utc()
        },
        TimeUnit::Microseconds => |t| {
            DateTime::<Utc>::from_timestamp_micros(t)
                .unwrap_or(DateTime::<Utc>::MAX_UTC)
                .naive_utc()
        },
        TimeUnit::Nanoseconds => |t| DateTime::<Utc>::from_timestamp_nanos(t).naive_utc(),
    };
    let ndt_to_ts: fn(NaiveDateTime) -> i64 = match datetime.time_unit() {
        TimeUnit::Milliseconds => |ndt| ndt.and_utc().timestamp_millis(),
        TimeUnit::Microseconds => |ndt| ndt.and_utc().timestamp_micros(),
        TimeUnit::Nanoseconds => |ndt| ndt.and_utc().timestamp_nanos_opt().unwrap_or(0),
    };

    if from_tz.len() != 1 && from_tz.len() != datetime.len() {
        return Err(
            polars_err!(InvalidOperation: format!("mismatched lengths, datetime has length '{}', from_tz has length '{}'", datetime.len(), from_tz.len())),
        );
    } else if to_tz.len() != 1 && to_tz.len() != datetime.len() {
        return Err(
            polars_err!(InvalidOperation: format!("mismatched lengths, datetime has length '{}', to_tz has length '{}'", datetime.len(), to_tz.len())),
        );
    }

    if from_tz.len() == 1 && to_tz.len() == 1 {
        let from_tz = unsafe { from_tz.get_unchecked(0).unwrap() };
        let to_tz = unsafe { to_tz.get_unchecked(0).unwrap() };
        let from_tz = parse_time_zone(from_tz)?;
        let to_tz = parse_time_zone(to_tz)?;

        let out = datetime.phys.apply(|opt_time| {
            opt_time.map(|time| {
                let ndt = ts_to_ndt(time);
                ndt_to_ts(
                    from_tz
                        .from_local_datetime(&ndt)
                        .unwrap()
                        .with_timezone(&to_tz)
                        .naive_local(),
                )
            })
        });
        return Ok(out.into_datetime(datetime.time_unit(), None));
    }

    if from_tz.len() == 1 {
        let from_tz = unsafe { from_tz.get_unchecked(0).unwrap() };
        let from_tz = parse_time_zone(from_tz)?;
        let out = datetime
            .phys
            .iter()
            .zip(to_tz.iter())
            .map(|(dt, to_tz)| {
                if to_tz.is_none() {
                    return Ok(dt);
                }
                if dt.is_none() {
                    return Ok(None);
                }
                let dt = dt.unwrap();
                let to_tz = parse_time_zone(to_tz.unwrap())?;
                Ok(Some(convert_tz_single(
                    dt, &from_tz, &to_tz, ts_to_ndt, ndt_to_ts,
                )))
            })
            .collect::<PolarsResult<ChunkedArray<Int64Type>>>()?;
        return Ok(out.into_datetime(datetime.time_unit(), None));
    }

    if to_tz.len() == 1 {
        let to_tz = unsafe { to_tz.get_unchecked(0).unwrap() };
        let to_tz = parse_time_zone(to_tz)?;
        let out = datetime
            .phys
            .iter()
            .zip(from_tz.iter())
            .map(|(dt, from_tz)| {
                if from_tz.is_none() {
                    return Ok(dt);
                }
                if dt.is_none() {
                    return Ok(None);
                }
                let dt = dt.unwrap();
                let from_tz = parse_time_zone(from_tz.unwrap())?;
                Ok(Some(convert_tz_single(
                    dt, &from_tz, &to_tz, ts_to_ndt, ndt_to_ts,
                )))
            })
            .collect::<PolarsResult<ChunkedArray<Int64Type>>>()?;
        return Ok(out.into_datetime(datetime.time_unit(), None));
    }

    let out = datetime
        .phys
        .iter()
        .zip(from_tz.iter())
        .zip(to_tz.iter())
        .map(|((dt, from_tz), to_tz)| {
            if from_tz.is_none() || to_tz.is_none() {
                return Ok(dt);
            } else if dt.is_none() {
                return Ok(None);
            }

            let from_tz = from_tz.unwrap();
            let to_tz = to_tz.unwrap();
            let from_tz = parse_time_zone(from_tz)?;
            let to_tz = parse_time_zone(to_tz)?;
            Ok(Some(convert_tz_single(
                dt.unwrap(),
                &from_tz,
                &to_tz,
                ts_to_ndt,
                ndt_to_ts,
            )))
        })
        .collect::<PolarsResult<ChunkedArray<Int64Type>>>()?;

    Ok(out.into_datetime(datetime.time_unit(), None))
}

fn convert_tz_single(
    dt: i64,
    from_tz: &Tz,
    to_tz: &Tz,
    ts_to_ndt: fn(i64) -> NaiveDateTime,
    ndt_to_ts: fn(NaiveDateTime) -> i64,
) -> i64 {
    let ndt = ts_to_ndt(dt);
    ndt_to_ts(
        from_tz
            .from_local_datetime(&ndt)
            .unwrap()
            .with_timezone(to_tz)
            .naive_local(),
    )
}
