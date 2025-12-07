use indexmap::IndexMap;
use polars::{
    datatypes::{Categories, DataType},
    lazy::dsl,
    prelude::{
        Expr, IntoLazy, NamedFrom, NewChunkedArray, ReshapeDimension, StringChunked,
        col as polars_col,
    },
    series::Series,
};
use polars_ops::{
    chunked_array::{ListNameSpaceImpl, StringNameSpaceImpl},
    series::{ClosedInterval, RankMethod, RankOptions, SearchSortedSide, SeriesRank},
};
use regex::bytes::Regex;

use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};

use crate::{collection::in_op, operator::match_op, series_op};

pub fn null(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.is_null()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("null".to_owned(), arg0.get_type_name());
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
        | SpicyObj::Fn(_) => Ok(SpicyObj::Boolean(false)),
        SpicyObj::F32(v) => {
            if v.is_nan() {
                Ok(SpicyObj::Boolean(true))
            } else {
                Ok(SpicyObj::Boolean(false))
            }
        }
        SpicyObj::F64(v) => {
            if v.is_nan() {
                Ok(SpicyObj::Boolean(true))
            } else {
                Ok(SpicyObj::Boolean(false))
            }
        }
        SpicyObj::String(s) | SpicyObj::Symbol(s) => Ok(SpicyObj::Boolean(s.is_empty())),
        SpicyObj::Null => Ok(SpicyObj::Boolean(true)),
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.is_null().into())),
        SpicyObj::MixedList(l) => {
            let res = l
                .iter()
                .map(|args| null(&[args]))
                .collect::<SpicyResult<Vec<_>>>();
            Ok(SpicyObj::MixedList(res?))
        }
        SpicyObj::Dict(d) => {
            let mut res = IndexMap::new();
            for (k, v) in d {
                res.insert(k.to_owned(), null(&[v])?);
            }
            Ok(SpicyObj::Dict(res))
        }
        SpicyObj::DataFrame(df) => Ok(SpicyObj::DataFrame(
            df.get_columns()
                .iter()
                .map(|s| Into::<Series>::into(s.is_null()))
                .collect(),
        )),
        _ => Err(err()),
    }
}

pub fn mode(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.mode()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("mode".to_owned(), arg0.get_type_name());
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
        | SpicyObj::Fn(_) => Ok(arg0.clone()),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(s) => Ok(SpicyObj::from_any_value(
            polars_ops::chunked_array::mode::mode(s)
                .map_err(|e| SpicyError::Err(e.to_string()))?
                .first()
                .as_any_value(),
        )),
        _ => Err(err()),
    }
}

pub fn rank(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.rank(
            RankOptions {
                method: RankMethod::Ordinal,
                descending: false,
            },
            None,
        )));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("rank".to_owned(), arg0.get_type_name());
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
        | SpicyObj::Fn(_)
        | SpicyObj::Null => Ok(SpicyObj::I64(1)),
        SpicyObj::Series(s) => Ok(SpicyObj::Series(
            s.rank(
                RankOptions {
                    method: RankMethod::Ordinal,
                    descending: false,
                },
                None,
            )
            .cast(&DataType::Int64)
            .unwrap(),
        )),
        _ => Err(err()),
    }
}

pub fn type_op(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    Ok(SpicyObj::Symbol(arg0.get_type_name()))
}

pub fn enlist(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let err = || SpicyError::UnsupportedUnaryOpErr("enlist".to_owned(), arg0.get_type_name());
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
        | SpicyObj::Null => Ok(SpicyObj::Series(arg0.into_series().unwrap())),
        SpicyObj::Series(s) => Ok(SpicyObj::DataFrame(s.clone().into_frame())),
        SpicyObj::Matrix(_)
        | SpicyObj::MixedList(_)
        | SpicyObj::Dict(_)
        | SpicyObj::DataFrame(_)
        | SpicyObj::Fn(_)
        | SpicyObj::Expr(_) => Ok(SpicyObj::MixedList(vec![arg0.clone()])),
        _ => Err(err()),
    }
}

pub fn filter(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DictOrSeries])?;
    let arg0 = args[0];
    match arg0 {
        SpicyObj::Series(s) if s.dtype().is_bool() || s.dtype().is_primitive_numeric() => {
            let s = if s.dtype().is_bool() {
                s.clone()
            } else {
                s.cast(&DataType::Boolean).unwrap()
            };
            let indices = s
                .bool()
                .unwrap()
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    if b.unwrap_or(false) {
                        Some(i as i64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<i64>>();
            let s = Series::new("".into(), indices);
            Ok(SpicyObj::Series(s))
        }
        SpicyObj::Dict(d) => {
            let mut keys = Vec::new();
            d.iter().for_each(|(k, v)| {
                if v.is_truthy().unwrap_or(false) {
                    keys.push(k.as_str())
                }
            });
            let s = Series::new("".into(), keys)
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap();
            Ok(SpicyObj::Series(s))
        }
        _ => unreachable!(),
    }
}

pub fn within(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        let range = arg1.as_vec()?;
        if range.len() != 2 {
            return Err(SpicyError::Err(format!(
                "Expect two atoms as a range, got {}",
                range.len()
            )));
        } else {
            return Ok(SpicyObj::Expr(left.is_between(
                range[0].as_expr()?,
                range[1].as_expr()?,
                ClosedInterval::Both,
            )));
        }
    }
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            "within".to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };
    if arg1.size() != 2 || !(arg1.is_mixed_list() || arg1.is_series()) {
        Err(SpicyError::Err(format!(
            "Requires size 2 mixed list or series, got '{}' with size '{}'",
            arg1.get_type_name(),
            arg1.size()
        )))
    } else if arg0.is_atom() {
        let v = arg1.as_vec()?;
        let left = &v[0];
        let right = &v[1];

        if arg0.is_integer() || arg0.is_bool() {
            let left = left.to_i64()?;
            let right = right.to_i64()?;
            let value = arg0.to_i64().unwrap();
            Ok(SpicyObj::Boolean((value >= left) && (value <= right)))
        } else if arg0.is_float() {
            let left = left.to_f64()?;
            let right = right.to_f64()?;
            let value = arg0.to_f64().unwrap();
            Ok(SpicyObj::Boolean((value >= left) && (value <= right)))
        } else {
            Err(err())
        }
    } else if arg0.is_series() {
        let value = arg0.series().unwrap();
        let v = arg1.as_vec()?;
        let left = v[0].as_series()?;
        let right = v[1].as_series()?;
        polars_ops::series::is_between(value, &left, &right, ClosedInterval::Both)
            .map_err(|e| SpicyError::Err(e.to_string()))
            .map(|s| SpicyObj::Series(s.into()))
    } else {
        Err(err())
    }
}

pub fn bottom(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(right.bottom_k(left)));
    }
    validate_args(args, &[ArgType::Int, ArgType::Series])?;
    let series = args[1].series().unwrap();
    Ok(SpicyObj::Series(
        polars_ops::chunked_array::top_k(
            &[series.clone().into(), args[0].into_series().unwrap().into()],
            true,
        )
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .take_materialized_series(),
    ))
}

pub fn corr(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(dsl::functions::pearson_corr(left, right)));
    }
    validate_args(args, &[ArgType::Series, ArgType::Series])?;
    let s0 = args[0].series().unwrap();
    let s1 = args[1].series().unwrap();
    if s0.len() != s1.len() {
        return Err(SpicyError::MismatchedLengthErr(s0.len(), s1.len()));
    }
    use polars_ops::chunked_array::cov::pearson_corr;
    let ret = match s0.dtype() {
        DataType::Float32 => {
            let ret = pearson_corr(s0.f32().unwrap(), s1.f32().unwrap()).map(|v| v as f32);
            return Ok(SpicyObj::F32(ret.unwrap_or(f32::NAN)));
        }
        DataType::Float64 => pearson_corr(s0.f64().unwrap(), s1.f64().unwrap()),
        DataType::Int32 => pearson_corr(s0.i32().unwrap(), s1.i32().unwrap()),
        DataType::Int64 => pearson_corr(s0.i64().unwrap(), s1.i64().unwrap()),
        DataType::UInt32 => pearson_corr(s0.u32().unwrap(), s1.u32().unwrap()),
        _ => {
            let s0 = s0
                .cast(&DataType::Float64)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            let s1 = s1
                .cast(&DataType::Float64)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            pearson_corr(s0.f64().unwrap(), s1.f64().unwrap())
        }
    };
    Ok(SpicyObj::F64(ret.unwrap_or(f64::NAN)))
}

pub fn cov(args: &[&SpicyObj], ddof: u8) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(dsl::functions::cov(left, right, ddof)));
    }
    validate_args(args, &[ArgType::Series, ArgType::Series])?;
    let a = args[0].series().unwrap();
    let b = args[1].series().unwrap();
    if a.len() != b.len() {
        return Err(SpicyError::MismatchedLengthErr(a.len(), b.len()));
    }

    use polars_ops::chunked_array::cov::cov;
    let ret = match a.dtype() {
        DataType::Float32 => {
            let ret = cov(a.f32().unwrap(), b.f32().unwrap(), ddof).map(|v| v as f32);
            return Ok(SpicyObj::F32(ret.unwrap_or(f32::NAN)));
        }
        DataType::Float64 => cov(a.f64().unwrap(), b.f64().unwrap(), ddof),
        DataType::Int32 => cov(a.i32().unwrap(), b.i32().unwrap(), ddof),
        DataType::Int64 => cov(a.i64().unwrap(), b.i64().unwrap(), ddof),
        DataType::UInt32 => cov(a.u32().unwrap(), b.u32().unwrap(), ddof),
        DataType::UInt64 => cov(a.u64().unwrap(), b.u64().unwrap(), ddof),
        _ => {
            let a = a
                .cast(&DataType::Float64)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            let b = b
                .cast(&DataType::Float64)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            cov(a.f64().unwrap(), b.f64().unwrap(), ddof)
        }
    };
    Ok(SpicyObj::F64(ret.unwrap_or(f64::NAN)))
}

pub fn cov0(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    cov(args, 0)
}

pub fn cov1(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    cov(args, 1)
}

pub fn top(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(right.top_k(left)));
    }
    validate_args(args, &[ArgType::Int, ArgType::Series])?;
    let series = args[1].series().unwrap();
    Ok(SpicyObj::Series(
        polars_ops::chunked_array::top_k(
            &[series.clone().into(), args[0].into_series().unwrap().into()],
            false,
        )
        .map_err(|e| SpicyError::Err(e.to_string()))?
        .take_materialized_series(),
    ))
}

// atom + atom => list
// atom + series => list
// series + series => DataFrame
// atom | series | list + list => list
pub fn cross(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let err = || SpicyError::UnsupportedUnaryOpErr("cross".to_owned(), arg0.get_type_name());
    if c0 <= 0 && c1 <= 0 {
        Ok(SpicyObj::MixedList(vec![arg0.clone(), arg1.clone()]))
    } else if arg0.is_series() && arg1.is_series() {
        let lf0 = arg0.as_series().unwrap().into_frame().lazy();
        let lf1 = arg1.as_series().unwrap().into_frame().lazy();
        let df = lf0
            .cross_join(lf1, Some("_right".into()))
            .collect()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        Ok(SpicyObj::DataFrame(df))
    } else {
        let l0 = arg0.as_vec().map_err(|_| err())?;
        let l1 = arg1.as_vec().map_err(|_| err())?;
        let mut res = Vec::with_capacity(l0.len() * l1.len());
        for a0 in l0.iter() {
            for a1 in l1.iter() {
                res.push(SpicyObj::MixedList(vec![a0.clone(), a1.clone()]));
            }
        }
        Ok(SpicyObj::MixedList(res))
    }
}

// series differ (atom | series)
// (dict | list) differ (dict | list | atom)
pub fn differ(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.list().set_difference(right)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("differ".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s0) => {
            let s1 = arg1.as_series()?;
            let s1 = if s0.dtype().is_bool() && s1.dtype().is_bool() {
                s1
            } else if (s0.dtype().is_integer() && s1.dtype().is_integer())
                || (s0.dtype().is_float() && s1.dtype().is_primitive_numeric())
            {
                s1.cast(s0.dtype()).unwrap()
            } else if s0.dtype().eq(s1.dtype()) {
                s1
            } else {
                return Err(err());
            };
            let bools = !polars_ops::series::is_in(s0, &s1, true).unwrap();
            Ok(SpicyObj::Series(s0.filter(&bools).unwrap()))
        }
        SpicyObj::Dict(d0) => {
            let l1 = arg1.as_vec()?;
            let arg1 = SpicyObj::MixedList(l1);
            let mut res = IndexMap::new();
            for (k, v) in d0 {
                if !*in_op(&[v, &arg1]).unwrap().bool().unwrap() {
                    res.insert(k.clone(), v.clone());
                }
            }
            Ok(SpicyObj::Dict(res))
        }
        SpicyObj::MixedList(l0) => {
            let mut res = Vec::new();
            let l1 = arg1.as_vec()?;
            let arg1 = SpicyObj::MixedList(l1);
            for obj in l0 {
                if !*in_op(&[obj, &arg1]).unwrap().bool().unwrap() {
                    res.push(obj.clone());
                }
            }
            Ok(SpicyObj::MixedList(res))
        }
        _ => Err(err()),
    }
}

// series intersect (atom | series)
// (dict | list) intersect (dict | list | atom)
pub fn intersect(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.list().set_intersection(right)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("intersect".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Series(s0) => {
            let s1 = arg1.as_series()?;
            let s1 = if s0.dtype().is_bool() && s1.dtype().is_bool() {
                s1
            } else if (s0.dtype().is_integer() && s1.dtype().is_integer())
                || (s0.dtype().is_float() && s1.dtype().is_primitive_numeric())
            {
                s1.cast(s0.dtype()).unwrap()
            } else if s0.dtype().eq(s1.dtype()) {
                s1
            } else {
                return Err(err());
            };
            let bools = polars_ops::series::is_in(s0, &s1, true).unwrap();
            Ok(SpicyObj::Series(s0.filter(&bools).unwrap()))
        }
        SpicyObj::Dict(d0) => {
            let l1 = arg1.as_vec()?;
            let arg1 = SpicyObj::MixedList(l1);
            let mut res = IndexMap::new();
            for (k, v) in d0 {
                if *in_op(&[v, &arg1]).unwrap().bool().unwrap() {
                    res.insert(k.clone(), v.clone());
                }
            }
            Ok(SpicyObj::Dict(res))
        }
        SpicyObj::MixedList(l0) => {
            let mut res = Vec::new();
            let l1 = arg1.as_vec()?;
            let arg1 = SpicyObj::MixedList(l1);
            for obj in l0 {
                if *in_op(&[obj, &arg1]).unwrap().bool().unwrap() {
                    res.push(obj.clone());
                }
            }
            Ok(SpicyObj::MixedList(res))
        }
        _ => Err(err()),
    }
}

pub fn like(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.str().contains(right, true)));
    }
    validate_args(args, &[ArgType::StrLike, ArgType::StrOrStrs])?;
    let err = || SpicyError::UnsupportedUnaryOpErr("like".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Symbol(s0) | SpicyObj::String(s0) => {
            let p1 = arg1.str().map_err(|_| err())?;
            let p1 = Regex::new(p1).map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::Boolean(p1.is_match(s0.as_bytes())))
        }
        SpicyObj::Series(s0) => {
            let s0 = if s0.dtype().eq(&DataType::String) {
                s0.clone()
            } else {
                s0.cast(&DataType::String).unwrap()
            };
            let c0 = s0.str().unwrap();
            let s1 = arg1.as_series().unwrap();
            let c1 = s1.str().unwrap();
            Ok(SpicyObj::Series(
                c0.contains_chunked(c1, false, true)
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into(),
            ))
        }
        _ => unreachable!(),
    }
}

pub fn matches(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.str().count_matches(right, true)));
    }
    validate_args(args, &[ArgType::StrLike, ArgType::StrOrStrs])?;
    let err = || SpicyError::UnsupportedUnaryOpErr("matches".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Symbol(s0) | SpicyObj::String(s0) => {
            let p1 = arg1.str().map_err(|_| err())?;
            let p1 = Regex::new(p1).map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::I64(p1.find_iter(s0.as_bytes()).count() as i64))
        }
        SpicyObj::Series(s0) => {
            let s0 = if s0.dtype().eq(&DataType::String) {
                s0.clone()
            } else {
                s0.cast(&DataType::String).unwrap()
            };
            let str_chunks = s0.str().unwrap();
            if arg1.is_atom() {
                let p1 = arg1.str().unwrap();
                Ok(SpicyObj::Series(
                    str_chunks
                        .count_matches(p1, false)
                        .map_err(|_| err())?
                        .into(),
                ))
            } else {
                let s1 = arg1.as_series().unwrap();
                let pat_chunks = s1.str().unwrap();
                Ok(SpicyObj::Series(
                    str_chunks
                        .count_matches_many(pat_chunks, false)
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
        _ => unreachable!(),
    }
}

pub fn join(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];

    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.list().join(right, true)));
    }

    validate_args(args, &[ArgType::StrOrSym, ArgType::Any])?;
    let sep = arg0.str().unwrap();

    let c1 = arg1.get_type_code();
    let err = || SpicyError::UnsupportedUnaryOpErr("join".to_owned(), arg0.get_type_name());
    match arg1 {
        SpicyObj::Symbol(_) | SpicyObj::String(_) => Ok(arg1.clone()),
        SpicyObj::Series(s1) => match s1.dtype() {
            DataType::String | DataType::Categorical(_, _) => {
                let s1 = if s1.dtype().eq(&DataType::String) {
                    s1.clone()
                } else {
                    s1.cast(&DataType::String).unwrap()
                };
                let str_chunks = s1.str().unwrap();
                let strs = str_chunks.iter().flatten().collect::<Vec<_>>();
                if c1 == 13 {
                    Ok(SpicyObj::String(strs.join(sep)))
                } else {
                    Ok(SpicyObj::Symbol(strs.join(sep)))
                }
            }
            DataType::List(dtype) => match dtype.as_ref() {
                DataType::String | DataType::Categorical(_, _) => {
                    let lists = s1.list().unwrap();
                    Ok(SpicyObj::Series(
                        lists
                            .join_literal(sep, true)
                            .map_err(|e| SpicyError::Err(e.to_string()))?
                            .into(),
                    ))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        },
        _ => Err(err()),
    }
}

pub fn reshape(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        let shape = arg1.as_vec()?;
        let shape = shape
            .iter()
            .map(|i| i.to_i64())
            .collect::<SpicyResult<Vec<_>>>();
        let shape = shape.map_err(|_| SpicyError::Err("Expect 2 integers, got {}".to_owned()))?;
        return Ok(SpicyObj::Expr(left.reshape(&shape)));
    }
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            "reshape".to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };
    let invalid_shape = || {
        SpicyError::Err(format!(
            "Expect 2 integers, got '{}' with size '{}'",
            arg0.get_type_name(),
            arg0.size()
        ))
    };
    if arg0.size() != 2 {
        return Err(invalid_shape());
    }
    let shape = arg0.as_vec().map_err(|_| invalid_shape())?;
    let shape = shape
        .into_iter()
        .map(|i| i.to_i64())
        .collect::<SpicyResult<Vec<i64>>>();
    let shape = shape.map_err(|_| invalid_shape())?;
    match arg1 {
        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
            s1.reshape_list(
                &shape
                    .iter()
                    .map(|d| ReshapeDimension::new_dimension(*d as u64))
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| SpicyError::Err(e.to_string()))?,
        )),
        SpicyObj::MixedList(l1) => {
            let d0 = shape[0] as usize;
            let d1 = shape[1] as usize;
            if l1.len() != d0 * d1 {
                Err(SpicyError::Err(format!(
                    "cannot reshape len {} into shape {:?}",
                    l1.len(),
                    shape
                )))
            } else if l1.is_empty() {
                Ok(arg1.clone())
            } else {
                let mut l = Vec::with_capacity(d0);
                for i in 0..d0 {
                    l.push(SpicyObj::MixedList(l1[(i * d1)..((i + 1) * d1)].to_vec()));
                }
                Ok(SpicyObj::MixedList(l))
            }
        }
        _ => Err(err()),
    }
}

pub fn rotate(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::Any])?;
    let arg1 = args[1];
    let offset = args[0].to_i64().unwrap();

    if arg1.is_expr() {
        if offset >= 0 {
            let right = arg1.as_expr()?;
            return Ok(SpicyObj::Expr(
                dsl::functions::concat_expr(
                    &[
                        right
                            .clone()
                            .slice(offset, Expr::Len - polars::prelude::lit(offset)),
                        right.head(Some(offset as usize)),
                    ],
                    false,
                )
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?,
            ));
        } else {
            let right = arg1.as_expr()?;
            return Ok(SpicyObj::Expr(
                dsl::functions::concat_expr(
                    &[
                        right
                            .clone()
                            .slice(offset, Expr::Len - polars::prelude::lit(offset.abs())),
                        right.slice(0, Expr::Len - polars::prelude::lit(offset.abs())),
                    ],
                    false,
                )
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?,
            ));
        }
    }

    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            "rotate".to_owned(),
            args[0].get_type_name(),
            arg1.get_type_name(),
        )
    };
    if arg1.size() == 0 {
        return Ok(arg1.clone());
    }
    let offset = offset % arg1.size() as i64;
    if offset == 0 {
        return Ok(arg1.clone());
    }
    let offset = if offset > 0 {
        offset
    } else {
        offset + arg1.size() as i64
    };
    match arg1 {
        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
            s1.slice(offset, arg1.size() - offset as usize)
                .extend(&s1.slice(0, offset as usize))
                .unwrap()
                .clone(),
        )),
        SpicyObj::MixedList(l1) => {
            let offset = offset as usize;
            Ok(SpicyObj::MixedList([&l1[offset..], &l1[..offset]].concat()))
        }
        SpicyObj::DataFrame(df1) => Ok(SpicyObj::DataFrame(
            df1.slice(offset, arg1.size() - offset as usize)
                .vstack(&df1.slice(0, offset as usize))
                .unwrap()
                .clone(),
        )),
        _ => Err(err()),
    }
}

pub fn split(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.str().split(right)));
    }
    validate_args(args, &[ArgType::Str, ArgType::StrLike])?;
    let arg0 = args[0];
    let arg1 = args[1];
    let sep = arg0.str().unwrap();
    match arg1 {
        SpicyObj::String(s0) => {
            let strs = s0.split(sep).collect::<Vec<_>>();
            Ok(SpicyObj::Series(Series::new("".into(), strs)))
        }
        SpicyObj::Symbol(s0) => {
            let strs = s0.split(sep).collect::<Vec<_>>();
            Ok(SpicyObj::Series(
                Series::new("".into(), strs)
                    .cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
            ))
        }
        SpicyObj::Series(s0) => {
            let s0 = if s0.dtype().eq(&DataType::String) {
                s0.clone()
            } else {
                s0.cast(&DataType::String).unwrap()
            };
            let chunk = s0
                .str()
                .unwrap()
                .split(&StringChunked::from_slice("".into(), &[sep]))
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            Ok(SpicyObj::Series(chunk.into()))
        }
        _ => unreachable!(),
    }
}

pub fn xbar(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let bar = args[0].as_expr()?;
        let s = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(s.map_many(
            series_op::xbar_expr,
            &[bar],
            |_, f| Ok(f[0].clone()),
        )));
    }
    validate_args(args, &[ArgType::NumericNative, ArgType::NumericNative])?;
    let arg0 = args[0];
    let arg1 = args[1];

    if arg0.size() > 1 && arg1.size() > 1 && arg0.size() != arg1.size() {
        return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
    }

    if arg0.is_atom() && arg1.is_atom() {
        if arg0.is_float() || arg1.is_float() {
            let bar_size = arg0.to_f64().unwrap();
            Ok(SpicyObj::F64(
                (arg1.to_f64().unwrap() / bar_size).round() * bar_size,
            ))
        } else {
            let mut bar_size = arg0.to_i64().unwrap();
            let atom = arg1.to_i64().unwrap();
            if arg1.datetime().is_ok() && (arg0.duration().is_ok() || arg0.time().is_ok()) {
                bar_size /= 1000000;
            }
            Ok(arg1.new_same_int_atom(bar_size * atom / bar_size).unwrap())
        }
    } else {
        let s0 = arg0.as_series().unwrap();
        let s1 = arg1.as_series().unwrap();
        Ok(SpicyObj::Series(
            series_op::xbar(s1.into(), s0.into())
                .map_err(|e| SpicyError::Err(e.to_string()))?
                .take_materialized_series(),
        ))
    }
}

pub fn ss(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.search_sorted(
            right,
            SearchSortedSide::Left,
            false,
        )));
    }
    validate_args(
        args,
        &[ArgType::NumericNativeSeries, ArgType::NumericNative],
    )?;
    let s0 = args[0].series().unwrap();
    let arg1 = args[1];
    let s1 = arg1.as_series().unwrap();
    let out = polars_ops::series::search_sorted(s0, &s1, SearchSortedSide::Left, false)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    if arg1.is_atom() {
        Ok(SpicyObj::I64(out.get(0).unwrap() as i64))
    } else {
        let out: Series = out.into();
        Ok(SpicyObj::Series(out.cast(&DataType::Int64).unwrap()))
    }
}

pub fn ssr(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.search_sorted(
            right,
            SearchSortedSide::Right,
            false,
        )));
    }
    validate_args(
        args,
        &[ArgType::NumericNativeSeries, ArgType::NumericNative],
    )?;
    let s0 = args[0].series().unwrap();
    let arg1 = args[1];
    let s1 = arg1.as_series().unwrap();
    let out = polars_ops::series::search_sorted(s0, &s1, SearchSortedSide::Right, false)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    if arg1.is_atom() {
        Ok(SpicyObj::I64(out.get(0).unwrap() as i64))
    } else {
        let out: Series = out.into();
        Ok(SpicyObj::Series(out.cast(&DataType::Int64).unwrap()))
    }
}

pub fn weighted_mean(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.clone().dot(right) / left.sum()));
    }
    validate_args(args, &[ArgType::NumericLike, ArgType::NumericLike])?;
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_atom() && arg1.is_atom() {
        Ok(SpicyObj::F64(arg1.to_f64().unwrap()))
    } else if arg0.size() != arg1.size() {
        Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()))
    } else {
        let s0 = arg0.as_series().unwrap().to_float().unwrap();
        let s1 = arg1.as_series().unwrap().to_float().unwrap();
        Ok(SpicyObj::F64(
            (s0.clone() * s1).unwrap().sum::<f64>().unwrap() / s0.sum::<f64>().unwrap(),
        ))
    }
}

pub fn weighted_sum(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(left.clone().dot(right)));
    }
    validate_args(args, &[ArgType::NumericLike, ArgType::NumericLike])?;
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_atom() && arg1.is_atom() {
        if arg0.is_integer() && arg1.is_integer() {
            Ok(SpicyObj::I64(
                arg0.to_i64().unwrap() * arg1.to_i64().unwrap(),
            ))
        } else {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap() * arg1.to_f64().unwrap(),
            ))
        }
    } else if arg0.size() != arg1.size() {
        Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()))
    } else {
        let s0 = arg0.as_series().unwrap();
        let s1 = arg1.as_series().unwrap();
        if s0.dtype().is_integer() && s1.dtype().is_integer() {
            Ok(SpicyObj::I64(
                (s0.cast(&DataType::Int64).unwrap() * s1.cast(&DataType::Int64).unwrap())
                    .unwrap()
                    .sum::<i64>()
                    .unwrap(),
            ))
        } else {
            let s0 = s0.to_float().unwrap();
            let s1 = s1.to_float().unwrap();
            Ok(SpicyObj::F64(
                (s0.clone() * s1).unwrap().sum::<f64>().unwrap(),
            ))
        }
    }
}

pub fn equal(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if *match_op(args).unwrap().bool().unwrap() {
        Ok(SpicyObj::Null)
    } else {
        Err(SpicyError::Err(format!(
            "assertion equal failed:\n  Left: {}\n Right: {}\n",
            arg0, arg1
        )))
    }
}

pub fn assert(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let err = || SpicyError::Err(format!("assertion failed: {}", arg0));
    match arg0.is_truthy() {
        Ok(b) => {
            if b {
                Ok(SpicyObj::Null)
            } else {
                Err(err())
            }
        }
        Err(_) => Err(err()),
    }
}

pub fn lit(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    Ok(SpicyObj::Expr(arg0.as_expr()?))
}

pub fn col(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym])?;
    let arg0 = args[0];
    Ok(SpicyObj::Expr(polars_col(arg0.str().unwrap())))
}

pub fn fby(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Expr, ArgType::Any])?;
    let arg0 = args[0];
    let arg1 = args[1];
    if arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = vec![arg1.as_expr()?];
        Ok(SpicyObj::Expr(left.over(right)))
    } else {
        let by = arg1.as_vec()?;
        let by = by
            .iter()
            .map(|args| args.as_expr())
            .collect::<SpicyResult<Vec<_>>>();
        let by = by.map_err(|_| SpicyError::Err("Expect column names, got {}".to_owned()))?;
        let left = arg0.as_expr()?;
        Ok(SpicyObj::Expr(left.over(by)))
    }
}

pub fn union(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        Ok(SpicyObj::Expr(left.list().union(right)))
    } else {
        Err(SpicyError::NotYetImplemented(format!(
            "union for {:?} and {:?}",
            args[0], args[1]
        )))
    }
}

pub fn when(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0].as_expr()?;
    let arg1 = args[1].as_expr()?;
    let j2 = args[2].as_expr()?;
    Ok(SpicyObj::Expr(
        polars::prelude::when(arg0).then(arg1).otherwise(j2),
    ))
}

pub fn as_op(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Any, ArgType::StrOrSym])?;
    let arg0 = args[0];
    let arg1 = args[1];
    Ok(SpicyObj::Expr(arg0.as_expr()?.alias(arg1.str().unwrap())))
}

pub fn show(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    println!("{}", args[0]);
    Ok(args[0].clone())
}
