use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};
use polars_compute::rolling::RollingQuantileParams;

use crate::{
    math,
    operator::{eq, rand},
    random::get_global_random_u64,
};
use indexmap::IndexMap;
use ndarray::{Axis, s};
use polars::prelude::{Categories, Expr, QuantileMethod, lit};
use polars::{
    chunked_array::{ChunkedArray, ops::ChunkCast},
    datatypes::{LogicalType, PolarsFloatType},
    error::{PolarsResult, polars_bail},
    prelude::{
        ChunkApply, FillNullStrategy, ListNameSpaceImpl, NamedFrom, RollingFnParams,
        RollingOptionsFixedWindow, SortOptions, concat_str,
    },
    series::{IntoSeries, ops::NullBehavior},
    time::chunkedarray::SeriesOpsTime,
};

use polars::{
    datatypes::{DataType, TimeUnit::Milliseconds as ms, TimeUnit::Nanoseconds as ns},
    series::Series,
};
use polars_ops::{
    chunked_array::StringNameSpaceImpl,
    series::{InterpolationMethod, MomentSeries, clip as series_clip},
};

pub fn asc(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.sort(SortOptions::default())));
    }
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    s.sort(SortOptions::default())
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("asc".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn backward_fill(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(
            left.fill_null_with_strategy(FillNullStrategy::Backward(None)),
        ));
    }
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    s.fill_null(FillNullStrategy::Backward(None))
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("bfill".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn forward_fill(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(
            left.fill_null_with_strategy(FillNullStrategy::Forward(None)),
        ));
    }
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    s.fill_null(FillNullStrategy::Forward(None))
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("fill".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn diff(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.diff(lit(1), NullBehavior::Ignore)));
    }
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    polars_ops::series::diff(s, 1, NullBehavior::Ignore)
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("diff".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn desc(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(
            left.sort(SortOptions::default().with_order_descending(true)),
        ));
    }
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    let options = SortOptions::default();
    options.with_order_descending(true);
    s.sort(options)
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("asc".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn sum(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.sum()));
    }
    match arg0 {
        SpicyObj::Series(series) => match series.dtype() {
            DataType::Boolean
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64 => series
                .sum::<i64>()
                .map_err(|e| SpicyError::EvalErr(e.to_string()))
                .map(SpicyObj::I64),
            DataType::Float32 | DataType::Float64 => series
                .sum::<f64>()
                .map_err(|e| SpicyError::EvalErr(e.to_string()))
                .map(SpicyObj::F64),
            DataType::Duration(ns) => series
                .sum::<i64>()
                .map_err(|e| SpicyError::EvalErr(e.to_string()))
                .map(SpicyObj::Duration),
            DataType::List(d) if d.is_primitive_numeric() || d.is_bool() => series
                .list()
                .unwrap()
                .lst_sum()
                .map_err(|e| SpicyError::EvalErr(e.to_string()))
                .map(SpicyObj::Series),
            _ => Err(SpicyError::UnsupportedUnaryOpErr(
                "sum".to_owned(),
                arg0.get_type_name(),
            )),
        },
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(arg0.clone()),
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.sum())),
        SpicyObj::Dict(d) => sum(&d.values().collect::<Vec<_>>()),
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "sum".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn count(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        if let Expr::Alias(_, name) = &left
            && name == "i"
        {
            return Ok(SpicyObj::Expr(Expr::Len.alias(name.clone())));
        } else {
            return Ok(SpicyObj::Expr(left.len()));
        }
    }
    let arg0 = args[0];
    Ok(SpicyObj::I64(arg0.size() as i64))
}

pub fn cum_op(
    args: &[&SpicyObj],
    f: fn(s: &Series, reverse: bool) -> PolarsResult<Series>,
) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Series])?;
    let s = args[0].series().unwrap();
    f(s, false)
        .map_err(|_| SpicyError::UnsupportedUnaryOpErr("fill".to_owned(), args[0].get_type_name()))
        .map(SpicyObj::Series)
}

pub fn cum_count(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.cum_count(false)));
    }
    cum_op(args, polars_ops::series::cum_count)
}

pub fn cum_max(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.cum_max(false)));
    }
    cum_op(args, polars_ops::series::cum_max)
}

pub fn cum_min(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.cum_min(false)));
    }
    cum_op(args, polars_ops::series::cum_min)
}

pub fn cum_prod(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.cum_prod(false)));
    }
    cum_op(args, polars_ops::series::cum_prod)
}

pub fn cum_sum(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.cum_sum(false)));
    }
    cum_op(args, polars_ops::series::cum_sum)
}

pub fn first(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.first()));
    }
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => match s.get(0) {
            Ok(a) => Ok(SpicyObj::from_any_value(a)),
            Err(_) => Ok(SpicyObj::Null),
        },
        SpicyObj::Matrix(m) => {
            if m.nrows() == 0 {
                Ok(SpicyObj::Matrix(m.clone()))
            } else {
                Ok(SpicyObj::Matrix(m.slice(s![0..1, ..]).to_shared()))
            }
        }
        SpicyObj::MixedList(l) => Ok(l.first().unwrap_or(&SpicyObj::Null).clone()),
        SpicyObj::Dict(d) => match d.first() {
            Some((_, v)) => Ok(v.clone()),
            None => Ok(SpicyObj::Null),
        },
        SpicyObj::DataFrame(df) => {
            let values = df.get(0);
            let mut res: IndexMap<String, SpicyObj> = IndexMap::new();
            let columns = df.get_column_names();
            match values {
                Some(values) => {
                    for (column, args) in columns.into_iter().zip(values) {
                        res.insert(column.to_string(), SpicyObj::from_any_value(args));
                    }
                }
                None => {
                    for column in columns {
                        res.insert(column.to_string(), SpicyObj::Null);
                    }
                }
            }
            Ok(SpicyObj::Dict(res))
        }
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "first".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn range(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let op = "range";

    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_) => {
            let i = arg0.to_i64().unwrap();
            if i < 0 {
                Err(SpicyError::Err(format!(
                    "Requires a positive int for '{}'",
                    op
                )))
            } else {
                Ok(SpicyObj::Series((0..i).collect()))
            }
        }
        SpicyObj::Series(s)
            if (s.dtype().is_integer() || s.dtype().eq(&DataType::Date)) && s.len() == 2 =>
        {
            let i64s = s.cast(&DataType::Int64).unwrap();
            let chunk = i64s.i64().unwrap();
            let start = chunk.get(0).unwrap_or(0);
            let end = chunk.get(1).unwrap_or(0);

            if s.dtype().eq(&DataType::Date) {
                let start = start as i32;
                let end = end as i32;
                if start > end {
                    Err(SpicyError::Err(format!(
                        "Requires start date '{}' >= end date '{}' for '{}'",
                        SpicyObj::Date(start),
                        SpicyObj::Date(end),
                        op
                    )))
                } else {
                    let s: Series = (start..end + 1).collect();
                    let s = s.cast(&DataType::Date).unwrap();
                    Ok(SpicyObj::Series(s))
                }
            } else if start > end {
                Err(SpicyError::Err(format!(
                    "Requires stop '{}' >= '{}' for '{}'",
                    start, end, op
                )))
            } else {
                Ok(SpicyObj::Series((start..end).collect()))
            }
        }
        _ => Err(SpicyError::Err(
            "Requires i64 or i64s(size 2) or dates(size 2) for 'range'".to_owned(),
        )),
    }
}

// num | temporal, min, max
pub fn clip(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    let j2 = args[2];

    if arg0.is_expr() || arg1.is_expr() || j2.is_expr() {
        let min = arg1.as_expr()?;
        let max = j2.as_expr()?;
        return Ok(SpicyObj::Expr(arg0.as_expr()?.clip(min, max)));
    }

    validate_args(args, &[ArgType::NumericNative, ArgType::Any, ArgType::Any])?;

    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let c2 = j2.get_type_code();

    let err = || {
        SpicyError::Err(format!(
            "Unsupported '{}', '{}' and '{}'",
            arg0.get_type_name(),
            arg1.get_type_name(),
            j2.get_type_name()
        ))
    };

    if c0 < 0 && c1 <= 0 && c2 <= 0 {
        if c0 >= -10 && c1 >= -10 && c2 >= -10 {
            let res = if c1 == 0 {
                arg0.to_i64().unwrap()
            } else {
                arg0.to_i64().unwrap().max(arg1.to_i64().unwrap())
            };
            let res = if c2 == 0 {
                res
            } else {
                res.min(j2.to_i64().unwrap())
            };
            Ok(arg0.new_same_int_atom(res).unwrap())
        } else if c0 >= -12 && c1 >= -12 && c2 >= -12 {
            let res = if c1 == 0 {
                arg0.to_f64().unwrap()
            } else {
                arg0.to_f64().unwrap().max(arg1.to_f64().unwrap())
            };
            let res = if c2 == 2 {
                res
            } else {
                res.min(j2.to_f64().unwrap())
            };
            Ok(SpicyObj::F64(res))
        } else {
            Err(err())
        }
    } else if c0 > 0 {
        let s0 = arg0.series().unwrap();
        if c0 <= 12 && (-12..=12).contains(&c1) && (-12..=12).contains(&c2) {
            let min_series = if c1 == 0 {
                s0.clone()
            } else if (-14..0).contains(&c1) {
                arg1.into_series().unwrap().cast(s0.dtype()).unwrap()
            } else {
                arg1.series().unwrap().cast(s0.dtype()).unwrap()
            };
            let max_series = if c2 == 0 {
                s0.clone()
            } else if (-14..0).contains(&c2) {
                j2.into_series().unwrap().cast(s0.dtype()).unwrap()
            } else {
                j2.series().unwrap().cast(s0.dtype()).unwrap()
            };
            let res = series_clip(s0, &min_series, &max_series).unwrap();
            Ok(SpicyObj::Series(res))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

pub fn concat(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let sep = args[0].str().unwrap();
    let arg1 = args[1];
    let j2 = args[2];
    if arg1.is_expr() || j2.is_expr() {
        let s1 = arg1.as_expr()?;
        let s2 = j2.as_expr()?;
        return Ok(SpicyObj::Expr(concat_str([s1, s2], sep, false)));
    }
    validate_args(args, &[ArgType::Str, ArgType::StrLike, ArgType::StrLike])?;
    let c1 = arg1.get_type_code();
    let c2 = j2.get_type_code();

    if (c1 == -13 || c1 == -14) & (c2 == -13 || c2 == -14) {
        let res = [arg1.str().unwrap(), j2.str().unwrap()].join(sep);
        if c1 == -13 {
            Ok(SpicyObj::String(res))
        } else {
            Ok(SpicyObj::Symbol(res))
        }
    } else if (c1 == 13 || c1 == 14) & (c2 == 13 || c2 == 14) {
        let s1 = if c1 == 13 {
            arg1.series().unwrap().clone()
        } else {
            arg1.series().unwrap().cast(&DataType::String).unwrap()
        };

        let s1 = s1.str().unwrap();

        let s2 = if c2 == 13 {
            j2.series().unwrap().clone()
        } else {
            j2.series().unwrap().cast(&DataType::String).unwrap()
        };

        let s2 = s2.str().unwrap();

        let res = s1.concat(&ChunkedArray::new("".into(), [sep])).concat(s2);

        if c1 == 13 {
            Ok(SpicyObj::Series(res.into()))
        } else {
            Ok(SpicyObj::Series(
                res.cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap(),
            ))
        }
    } else {
        match arg1 {
            SpicyObj::String(s1) => match j2 {
                SpicyObj::Series(s2) => match s2.dtype() {
                    DataType::String => Ok(SpicyObj::Series(
                        ChunkedArray::new("".into(), [format!("{}{}", s1, sep)])
                            .concat(s2.str().unwrap())
                            .into(),
                    )),
                    DataType::Categorical(_, _) => Ok(SpicyObj::Series(
                        ChunkedArray::new("".into(), [format!("{}{}", s1, sep)])
                            .concat(s2.cast(&DataType::String).unwrap().str().unwrap())
                            .into(),
                    )),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            },
            SpicyObj::Symbol(s1) => match j2 {
                SpicyObj::Series(s2) => {
                    let res: Series = match s2.dtype() {
                        DataType::String => {
                            ChunkedArray::new("".into(), [format!("{}{}", s1, sep)])
                                .concat(s2.str().unwrap())
                                .into()
                        }
                        DataType::Categorical(_, _) => {
                            ChunkedArray::new("".into(), [format!("{}{}", s1, sep)])
                                .concat(s2.cast(&DataType::String).unwrap().str().unwrap())
                                .into()
                        }
                        _ => unreachable!(),
                    };
                    Ok(SpicyObj::Series(
                        res.cast(&DataType::Categorical(
                            Categories::global(),
                            Categories::global().mapping(),
                        ))
                        .unwrap(),
                    ))
                }
                _ => unreachable!(),
            },
            SpicyObj::Series(s1) => match s1.dtype() {
                DataType::String => match j2 {
                    SpicyObj::String(s2) | SpicyObj::Symbol(s2) => Ok(SpicyObj::Series(
                        s1.str()
                            .unwrap()
                            .concat(&ChunkedArray::new("".into(), [format!("{}{}", sep, s2)]))
                            .into(),
                    )),
                    _ => unreachable!(),
                },
                DataType::Categorical(_, _) => match j2 {
                    SpicyObj::String(s2) | SpicyObj::Symbol(s2) => {
                        let s1 = s1.cat32().unwrap().cast(&DataType::String).unwrap();
                        Ok(SpicyObj::Series(
                            s1.str()
                                .unwrap()
                                .concat(&ChunkedArray::new("".into(), [format!("{}{}", sep, s2)]))
                                .cast(&DataType::Categorical(
                                    Categories::global(),
                                    Categories::global().mapping(),
                                ))
                                .unwrap(),
                        ))
                    }
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

pub fn last(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.last()));
    }
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => match s.get(s.len() - 1) {
            Ok(a) => Ok(SpicyObj::from_any_value(a)),
            Err(_) => Ok(SpicyObj::Null),
        },
        SpicyObj::Matrix(m) => {
            if m.nrows() == 0 {
                Ok(SpicyObj::Matrix(m.clone()))
            } else {
                Ok(SpicyObj::Matrix(m.slice(s![-1.., ..]).to_shared()))
            }
        }
        SpicyObj::MixedList(l) => Ok(l.last().unwrap_or(&SpicyObj::Null).clone()),
        SpicyObj::Dict(d) => match d.last() {
            Some((_, v)) => Ok(v.clone()),
            None => Ok(SpicyObj::Null),
        },
        SpicyObj::DataFrame(df) => {
            let values = df.get(df.height() - 1);
            let mut res: IndexMap<String, SpicyObj> = IndexMap::new();
            let columns = df.get_column_names();
            match values {
                Some(values) => {
                    for (column, args) in columns.into_iter().zip(values) {
                        res.insert(column.to_string(), SpicyObj::from_any_value(args));
                    }
                }
                None => {
                    for column in columns {
                        res.insert(column.to_string(), SpicyObj::Null);
                    }
                }
            }
            Ok(SpicyObj::Dict(res))
        }
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "last".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn max(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.max()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("max".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::Boolean => math::any(args),
            DataType::UInt8 => Ok(s
                .max::<u8>()
                .unwrap()
                .map(SpicyObj::U8)
                .unwrap_or(SpicyObj::Null)),
            DataType::UInt16 | DataType::UInt32 | DataType::Int8 => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::I64)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int16 => Ok(s
                .max::<i16>()
                .unwrap()
                .map(SpicyObj::I16)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int32 => Ok(s
                .max::<i32>()
                .unwrap()
                .map(SpicyObj::I32)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int64 => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::I64)
                .unwrap_or(SpicyObj::Null)),
            DataType::Float32 => Ok(s
                .max::<f32>()
                .unwrap()
                .map(SpicyObj::F32)
                .unwrap_or(SpicyObj::F32(f32::NAN))),
            DataType::Float64 | DataType::Decimal(_, _) => Ok(s
                .max::<f64>()
                .unwrap()
                .map(SpicyObj::F64)
                .unwrap_or(SpicyObj::F64(f64::NAN))),
            DataType::Date => Ok(s
                .max::<i32>()
                .unwrap()
                .map(SpicyObj::Date)
                .unwrap_or(SpicyObj::Null)),
            DataType::Datetime(ms, _) => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::Datetime)
                .unwrap_or(SpicyObj::Null)),
            DataType::Datetime(ns, _) => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::Timestamp)
                .unwrap_or(SpicyObj::Null)),
            DataType::Duration(_) => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::Duration)
                .unwrap_or(SpicyObj::Null)),
            DataType::Time => Ok(s
                .max::<i64>()
                .unwrap()
                .map(SpicyObj::Time)
                .unwrap_or(SpicyObj::Null)),
            DataType::Null => Ok(SpicyObj::Null),
            _ => Err(err()),
        },
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(
            *m.iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&f64::NAN),
        )),
        _ => Err(err()),
    }
}

pub fn min(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.min()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("min".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::Boolean => math::all(args),
            DataType::UInt8 => Ok(s
                .min::<u8>()
                .unwrap()
                .map(SpicyObj::U8)
                .unwrap_or(SpicyObj::Null)),
            DataType::UInt16 | DataType::UInt32 | DataType::Int8 => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::I64)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int16 => Ok(s
                .min::<i16>()
                .unwrap()
                .map(SpicyObj::I16)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int32 => Ok(s
                .min::<i32>()
                .unwrap()
                .map(SpicyObj::I32)
                .unwrap_or(SpicyObj::Null)),
            DataType::Int64 => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::I64)
                .unwrap_or(SpicyObj::Null)),
            DataType::Float32 => Ok(s
                .min::<f32>()
                .unwrap()
                .map(SpicyObj::F32)
                .unwrap_or(SpicyObj::F32(f32::NAN))),
            DataType::Float64 | DataType::Decimal(_, _) => Ok(s
                .min::<f64>()
                .unwrap()
                .map(SpicyObj::F64)
                .unwrap_or(SpicyObj::F64(f64::NAN))),
            DataType::Date => Ok(s
                .min::<i32>()
                .unwrap()
                .map(SpicyObj::Date)
                .unwrap_or(SpicyObj::Null)),
            DataType::Datetime(ms, _) => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::Datetime)
                .unwrap_or(SpicyObj::Null)),
            DataType::Datetime(ns, _) => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::Timestamp)
                .unwrap_or(SpicyObj::Null)),
            DataType::Duration(_) => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::Duration)
                .unwrap_or(SpicyObj::Null)),
            DataType::Time => Ok(s
                .min::<i64>()
                .unwrap()
                .map(SpicyObj::Time)
                .unwrap_or(SpicyObj::Null)),
            DataType::Null => Ok(SpicyObj::Null),
            _ => Err(err()),
        },
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(
            *m.iter().min_by(|a, b| a.total_cmp(b)).unwrap_or(&f64::NAN),
        )),
        _ => Err(err()),
    }
}

pub fn mean(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.mean()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("mean".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => Ok(SpicyObj::F64(s.mean().unwrap_or(f64::NAN))),
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.mean().unwrap_or(f64::NAN))),
        _ => Err(err()),
    }
}

pub fn median(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.median()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("median".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::String(_)
        | SpicyObj::Symbol(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Series(s) => Ok(SpicyObj::F64(s.median().unwrap_or(f64::NAN))),
        _ => Err(err()),
    }
}

pub fn rolling_quantile(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[2].is_expr() {
        let series = args[2].as_expr()?;
        let percentile = args[0].to_f64()?;
        let size = args[1].to_i64()?;
        let options = RollingOptionsFixedWindow {
            window_size: size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(series.rolling_quantile(
            QuantileMethod::Midpoint,
            percentile,
            options,
        )));
    }
    validate_args(args, &[ArgType::Float, ArgType::Int, ArgType::Series])?;

    let quantile = args[0].to_f64().unwrap();
    let size = args[1].to_i64().unwrap();
    let series = args[2].series().unwrap();

    let params = RollingQuantileParams {
        prob: quantile,
        method: QuantileMethod::Midpoint,
    };

    let options = RollingOptionsFixedWindow {
        window_size: size as usize,
        fn_params: Some(RollingFnParams::Quantile(params)),
        ..Default::default()
    };
    series
        .rolling_quantile(options)
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(SpicyObj::Series)
}
// df,

pub fn interpolate(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(
            left.interpolate(InterpolationMethod::Linear),
        ));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("interp".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => Ok(SpicyObj::Series(
            polars_ops::series::interpolate(s, InterpolationMethod::Linear),
        )),
        _ => Err(err()),
    }
}

pub fn kurtosis(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.kurtosis(true, true)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("kurtosis".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => s
            .kurtosis(true, true)
            .map(|v| {
                if let Some(v) = v {
                    SpicyObj::F64(v)
                } else {
                    SpicyObj::Null
                }
            })
            .map_err(|e| SpicyError::Err(e.to_string())),
        _ => Err(err()),
    }
}

pub fn skew(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.skew(true)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("skew".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => s
            .skew(true)
            .map(|v| {
                if let Some(v) = v {
                    SpicyObj::F64(v)
                } else {
                    SpicyObj::Null
                }
            })
            .map_err(|e| SpicyError::Err(e.to_string())),
        _ => Err(err()),
    }
}

pub fn var0(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.var(0)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("var0".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => {
            Ok(s.var(0).map(SpicyObj::F64).unwrap_or(SpicyObj::Null))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.var(0.0))),
        _ => Err(err()),
    }
}

pub fn var1(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.var(1)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("var1".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => {
            Ok(s.var(1).map(SpicyObj::F64).unwrap_or(SpicyObj::Null))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.var(1.0))),
        _ => Err(err()),
    }
}

pub fn std0(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.std(0)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("std0".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => {
            Ok(s.std(0).map(SpicyObj::F64).unwrap_or(SpicyObj::Null))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.std(0.0))),
        _ => Err(err()),
    }
}

pub fn std1(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.std(1)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("std1".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => {
            Ok(s.std(1).map(SpicyObj::F64).unwrap_or(SpicyObj::Null))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.std(1.0))),
        _ => Err(err()),
    }
}

pub fn next(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.shift(polars::prelude::lit(-1))));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("next".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.shift(-1))),
        SpicyObj::MixedList(l) => {
            if l.is_empty() {
                Ok(arg0.clone())
            } else {
                let l = [&l[1..], &[SpicyObj::Null]].concat();
                Ok(SpicyObj::MixedList(l))
            }
        }
        _ => Err(err()),
    }
}

pub fn prev(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.shift(polars::prelude::lit(1))));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("prev".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.shift(1))),
        SpicyObj::MixedList(l) => {
            if l.is_empty() {
                Ok(arg0.clone())
            } else {
                let l = [&[SpicyObj::Null], &l[..l.len() - 1]].concat();
                Ok(SpicyObj::MixedList(l))
            }
        }
        _ => Err(err()),
    }
}

pub fn shift(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.shift(right)));
    }
    validate_args(args, &[ArgType::Int, ArgType::Any])?;
    let period = args[0].to_i64().unwrap();
    let arg1 = args[1];
    let err = || SpicyError::UnsupportedUnaryOpErr("shift".to_owned(), arg1.get_type_name());
    match arg1 {
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.shift(period))),
        SpicyObj::MixedList(l) => {
            if period.unsigned_abs() as usize >= l.len() {
                Ok(SpicyObj::MixedList(vec![SpicyObj::Null; l.len()]))
            } else if period > 0 {
                let period = period as usize;
                Ok(SpicyObj::MixedList(
                    [&vec![SpicyObj::Null; period], &l[..l.len() - period]].concat(),
                ))
            } else {
                let period = period as usize;
                Ok(SpicyObj::MixedList(
                    [&l[period..], &vec![SpicyObj::Null; period]].concat(),
                ))
            }
        }
        _ => Err(err()),
    }
}

pub fn percent_change(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.pct_change(polars::prelude::lit(1))));
    }
    validate_args(args, &[ArgType::Series])?;
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => {
            let s = match s.dtype() {
                DataType::Float64 | DataType::Float32 => s.clone(),
                _ => s.cast(&DataType::Float64).unwrap(),
            };

            let fill_null_s = s
                .fill_null(FillNullStrategy::Forward(None))
                .map_err(|e| SpicyError::Err(e.to_string()))?;

            let res = polars_ops::series::diff(&fill_null_s, 1, NullBehavior::Ignore)
                .map_err(|e| SpicyError::Err(e.to_string()))?
                .divide(&fill_null_s.shift(1))
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::Series(res))
        }
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "pc".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn product(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.product()));
    }
    match arg0 {
        SpicyObj::Series(series) if series.dtype().is_primitive_numeric() => {
            let res = series
                .product()
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::from_any_value(res.as_any_value()))
        }
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(arg0.clone()),
        SpicyObj::Matrix(m) => Ok(SpicyObj::F64(m.product())),
        SpicyObj::Dict(d) => product(&d.values().collect::<Vec<_>>()),
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "product".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn reverse(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.reverse()));
    }
    match arg0 {
        SpicyObj::Series(series) => Ok(SpicyObj::Series(series.reverse())),
        SpicyObj::MixedList(l) => Ok(SpicyObj::MixedList(
            l.iter().cloned().rev().collect::<Vec<_>>(),
        )),
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(arg0.clone()),
        SpicyObj::Matrix(m) => {
            if m.nrows() == 0 {
                Ok(arg0.clone())
            } else {
                Ok(SpicyObj::Matrix(
                    m.select(Axis(0), &(0..m.nrows()).rev().collect::<Vec<_>>())
                        .to_shared(),
                ))
            }
        }
        SpicyObj::Dict(d) => Ok(SpicyObj::Dict(
            d.iter()
                .rev()
                .map(|(k, v)| (k.to_owned(), v.clone()))
                .collect(),
        )),
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "reverse".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn shuffle(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.shuffle(Some(get_global_random_u64()))));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("shuffle".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(series) => Ok(SpicyObj::Series(
            series.shuffle(Some(get_global_random_u64())),
        )),
        SpicyObj::MixedList(_) | SpicyObj::DataFrame(_) | SpicyObj::Matrix(_) => {
            rand(&[&SpicyObj::I64(arg0.size() as i64), arg0]).map_err(|_| err())
        }
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(arg0.clone()),
        _ => Err(err()),
    }
}

pub(super) fn sign_op(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Float32 => {
            let ca = s.f32().unwrap();
            sign_float(ca)
        }
        DataType::Float64 => {
            let ca = s.f64().unwrap();
            sign_float(ca)
        }
        dt if dt.is_primitive_numeric() => {
            let s = s.cast(&DataType::Float64)?;
            sign_op(&s)
        }
        dt => polars_bail!(opq = sign, dt),
    }
}

fn sign_float<T>(ca: &ChunkedArray<T>) -> PolarsResult<Series>
where
    T: PolarsFloatType,
    T::Native: num::Float,
    ChunkedArray<T>: IntoSeries,
{
    ca.apply_values(signum_improved)
        .into_series()
        .cast(&DataType::Int64)
}

// Wrapper for the signum function that handles +/-0.0 inputs differently
fn signum_improved<F: num::Float>(v: F) -> F {
    if v.is_zero() { v } else { v.signum() }
}

pub fn sign(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.sign()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("sign".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_primitive_numeric() => Ok(SpicyObj::Series(
            sign_op(s).map_err(|e| SpicyError::Err(e.to_string()))?,
        )),
        SpicyObj::MixedList(l) => {
            let res = l
                .iter()
                .map(|a| sign(&[a]))
                .collect::<SpicyResult<Vec<_>>>()?;
            Ok(SpicyObj::MixedList(res))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::Matrix(m.mapv(|a| a.signum()).to_shared())),
        SpicyObj::Dict(d) => {
            let mut res = IndexMap::new();
            for (k, v) in d {
                res.insert(k.to_owned(), sign(&[v])?);
            }
            Ok(SpicyObj::Dict(res))
        }
        SpicyObj::Boolean(_) | SpicyObj::U8(_) => {
            Ok(SpicyObj::I16(arg0.to_i64().unwrap().signum() as i16))
        }
        SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_) => arg0.new_same_int_atom(arg0.to_i64().unwrap().signum()),
        SpicyObj::F32(v) => Ok(SpicyObj::F32(signum_improved(*v))),
        SpicyObj::F64(v) => Ok(SpicyObj::F64(signum_improved(*v))),
        SpicyObj::Null => Ok(SpicyObj::I64(0)),
        _ => Err(err()),
    }
}

pub fn unique(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.unique_stable()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("unique".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.unique_stable().map_err(|_| err())?)),
        SpicyObj::MixedList(l) => {
            let mut res = Vec::new();
            for obj in l {
                let exists = res.iter().any(|a| {
                    eq(&[obj, a])
                        .map(|b| b.bool().copied().unwrap_or(false))
                        .unwrap_or(false)
                });
                if exists {
                    continue;
                } else {
                    res.push(obj.clone())
                }
            }
            Ok(SpicyObj::MixedList(res))
        }
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(arg0.clone()),
        _ => Err(err()),
    }
}

pub fn unique_count(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.unique_stable()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("uc".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s) => Ok(SpicyObj::I64(s.n_unique().map_err(|_| err())? as i64)),
        SpicyObj::MixedList(l) => {
            let mut res = Vec::new();
            for obj in l {
                let exists = res.iter().any(|a| {
                    eq(&[obj, a])
                        .map(|b| b.bool().copied().unwrap_or(false))
                        .unwrap_or(false)
                });
                if exists {
                    continue;
                } else {
                    res.push(obj.clone())
                }
            }
            Ok(SpicyObj::I64(res.len() as i64))
        }
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_)
        | SpicyObj::F32(_)
        | SpicyObj::F64(_)
        | SpicyObj::Null => Ok(SpicyObj::I64(1)),
        _ => Err(err()),
    }
}

// op: in
//  Lbchij | efdtzpnMDS
// LSSSSSS | SSSSSSS-SS
// bbbbbbb | bbbbbbb-bb
// cbbbbbb | bbbbbbb-bb
// hbbbbbb | bbbbbbb-bb
// ibbbbbb | bbbbbbb-bb
// jbbbbbb | bbbbbbb-bb
// ebbbbbb | bbbbbbb-bb
// fbbbbbb | bbbbbbb-bb
// dbbbbbb | bbbbbbb-bb
// tbbbbbb | bbbbbbb-bb
// zbbbbbb | bbbbbbb-bb
// pbbbbbb | bbbbbbb-bb
// nbbbbbb | bbbbbbb-bb
// M------ | ----------
// DDDDDDD | DDDDDDD-DD
// SSSSSSS | SSSSSSS-SS
pub fn in_op(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.is_in(right, true)));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let err = || SpicyError::UnsupportedUnaryOpErr("in".to_owned(), arg0.get_type_name());

    if c0 <= 0 && c1 <= 0 {
        eq(args)
    } else if (arg0.is_series() && c1 <= 0) || (c0 <= 0 && arg1.is_series()) {
        let mut s0 = arg0.as_series().unwrap();
        let mut s1 = arg1.as_series().unwrap();
        if c1 == 0 {
            Ok(SpicyObj::Series(s0.is_null().into()))
        } else if c0 == 0 {
            Ok(SpicyObj::Boolean(s1.null_count() > 0))
        } else {
            if (s0.dtype().is_primitive_numeric() || s0.dtype().is_temporal())
                && (s1.dtype().is_primitive_numeric() || s1.dtype().is_bool())
            {
                s1 = s1.cast(s0.dtype()).map_err(|_| err())?;
            } else if s0.dtype().is_temporal() && s1.dtype().is_temporal() {
                match s0.dtype() {
                    DataType::Date => match s1.dtype() {
                        DataType::Date => {}
                        DataType::Datetime(_, _) => s0 = s0.cast(s1.dtype()).unwrap(),
                        _ => return Err(err()),
                    },
                    DataType::Datetime(_, _) => match s1.dtype() {
                        DataType::Date => s1 = s1.cast(s0.dtype()).unwrap(),
                        DataType::Datetime(_, _) => s0 = s0.cast(s1.dtype()).unwrap(),
                        DataType::Duration(_) => {
                            s0 = s0.cast(&DataType::Time).unwrap();
                            s1 = s1.cast(&DataType::Time).unwrap();
                        }
                        DataType::Time => s0 = s0.cast(s1.dtype()).unwrap(),
                        _ => return Err(err()),
                    },
                    DataType::Duration(_) => match s1.dtype() {
                        DataType::Datetime(_, _) => {
                            s0 = s0.cast(&DataType::Time).unwrap();
                            s1 = s1.cast(&DataType::Time).unwrap();
                        }
                        DataType::Duration(_) => {}
                        DataType::Time => s0 = s0.cast(s1.dtype()).unwrap(),
                        _ => return Err(err()),
                    },
                    DataType::Time => match s1.dtype() {
                        DataType::Datetime(_, _) | DataType::Duration(_) => {
                            s1 = s1.cast(s0.dtype()).unwrap();
                        }
                        DataType::Time => {}
                        _ => return Err(err()),
                    },
                    _ => unreachable!(),
                }
            }
            polars_ops::series::is_in(&s0, &s1, true)
                .map(|s| SpicyObj::Series(s.into()))
                .map_err(|_| err())
        }
    } else if (-14..=0).contains(&c0) && c1 > 0 {
        let v1;
        let v1: Vec<&SpicyObj> = match arg1 {
            SpicyObj::Series(_) => {
                v1 = arg1.as_vec().map_err(|_| err())?;
                v1.iter().collect()
            }
            SpicyObj::MixedList(l) => l.iter().collect(),
            SpicyObj::Dict(d) => d.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            _ => return Err(err()),
        };
        let res = v1.into_iter().any(|a| {
            eq(&[arg0, a])
                .map(|b| b.bool().copied().unwrap_or(false))
                .unwrap_or(false)
        });
        Ok(SpicyObj::Boolean(res))
    } else if c0 > 0 && (-14..=0).contains(&c1) {
        let v0;
        let v0 = match arg0 {
            SpicyObj::Series(_) => {
                v0 = arg1.as_vec().map_err(|_| err())?;
                v0.iter().collect()
            }
            SpicyObj::MixedList(l) => l.iter().collect(),
            SpicyObj::Dict(d) => d.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            _ => return Err(err()),
        };
        let res = v0
            .into_iter()
            .map(|a| {
                eq(&[a, arg1])
                    .map(|b| b.bool().copied().unwrap_or(false))
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();
        if arg0.is_dict() {
            let mut res_dict = IndexMap::new();
            for (i, k) in arg0.dict().unwrap().keys().enumerate() {
                res_dict.insert(k.to_owned(), SpicyObj::Boolean(res[i]));
            }
            Ok(SpicyObj::Dict(res_dict))
        } else {
            Ok(SpicyObj::Series(Series::new("".into(), res)))
        }
    } else if c0 > 0 && c1 > 0 {
        let v0;
        let v0 = match arg0 {
            SpicyObj::Series(_) => {
                v0 = arg1.as_vec().map_err(|_| err())?;
                v0.iter().collect()
            }
            SpicyObj::MixedList(l) => l.iter().collect(),
            SpicyObj::Dict(d) => d.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            _ => return Err(err()),
        };

        let v1;
        let v1: Vec<&SpicyObj> = match arg1 {
            SpicyObj::Series(_) => {
                v1 = arg1.as_vec().map_err(|_| err())?;
                v1.iter().collect()
            }
            SpicyObj::MixedList(l) => l.iter().collect(),
            SpicyObj::Dict(d) => d.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            _ => return Err(err()),
        };
        let res = v0
            .into_iter()
            .map(|a0| {
                v1.iter().any(|a1| {
                    eq(&[a0, a1])
                        .map(|b| b.bool().copied().unwrap_or(false))
                        .unwrap_or(false)
                })
            })
            .collect::<Vec<_>>();
        if arg0.is_dict() {
            let mut res_dict = IndexMap::new();
            for (i, k) in arg0.dict().unwrap().keys().enumerate() {
                res_dict.insert(k.to_owned(), SpicyObj::Boolean(res[i]));
            }
            Ok(SpicyObj::Dict(res_dict))
        } else {
            Ok(SpicyObj::Series(Series::new("".into(), res)))
        }
    } else {
        Err(err())
    }
}

pub fn keys(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Dict])?;
    let df = args[0].dict().unwrap();
    Ok(SpicyObj::Series(
        Series::new("".into(), df.keys().map(|k| k.as_str()).collect::<Vec<_>>())
            .cast(&DataType::Categorical(
                Categories::global(),
                Categories::global().mapping(),
            ))
            .unwrap(),
    ))
}

pub fn flag(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrameOrSeries])?;
    match args[0] {
        SpicyObj::Series(s) => match s.get_flags().is_sorted() {
            polars::series::IsSorted::Ascending => Ok(SpicyObj::Symbol("asc".to_owned())),
            polars::series::IsSorted::Descending => Ok(SpicyObj::Symbol("desc".to_owned())),
            polars::series::IsSorted::Not => Ok(SpicyObj::Symbol("".to_owned())),
        },
        SpicyObj::DataFrame(df) => {
            let mut res = IndexMap::new();
            for c in df.get_columns() {
                let flag = match c.get_flags().is_sorted() {
                    polars::series::IsSorted::Ascending => "asc",
                    polars::series::IsSorted::Descending => "desc",
                    polars::series::IsSorted::Not => "",
                };
                res.insert(c.name().to_string(), SpicyObj::Symbol(flag.to_owned()));
            }
            Ok(SpicyObj::Dict(res))
        }
        _ => unreachable!(),
    }
}

pub fn transpose(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrameOrMatrix])?;
    let arg0 = args[0];
    match arg0 {
        SpicyObj::Matrix(m) => {
            let mut m = m.clone();
            m.swap_axes(0, 1);
            Ok(SpicyObj::Matrix(m))
        }
        SpicyObj::DataFrame(df) => Ok(SpicyObj::DataFrame(
            df.clone()
                .transpose(None, None)
                .map_err(|e| SpicyError::Err(e.to_string()))?,
        )),
        _ => unreachable!(),
    }
}

pub fn flatten(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        return Ok(SpicyObj::Expr(left.flatten()));
    }
    Err(SpicyError::NotYetImplemented(format!(
        "flatten for {:?}",
        args[0]
    )))
}
