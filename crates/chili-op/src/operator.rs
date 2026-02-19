use indexmap::IndexMap;
use ndarray::{Array2, Axis, s};
use polars::chunked_array::ops::ChunkFillNullValue;
use polars::datatypes::{DataType, TimeUnit::Milliseconds as ms, TimeUnit::Nanoseconds as ns};
use polars::prelude::{
    Categories, ChunkCompareIneq, Expr, FunctionExpr, NamedFrom, Operator, concat_list,
    floor_div_series,
};
use polars::series::{ChunkCompareEq, Series};
use polars_ops::series::{max_horizontal, min_horizontal};
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::seq::index::IndexVec;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::random::get_global_random_u64;
use crate::util::{
    atom_op_dict, atom_op_list, dict_op_atom, dict_op_list, list_op_atom, list_op_dict,
    list_op_list,
};
use crate::{io::map_str_to_polars_dtype, math};
use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};

pub const NS_IN_DAY: i64 = 86_400_000_000_000;
pub const MS_IN_DAY: i64 = 86_400_000;
pub const NS_IN_MS: i64 = 1_000_000;

#[cfg(feature = "vintage")]
pub const TRUE_DIV_OP: &str = "%";
#[cfg(not(feature = "vintage"))]
pub const TRUE_DIV_OP: &str = "/";

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|  u32|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |   u8|   u8|   u8|  u16|  u32|  u64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |  u16|  u16|  u16|  u16|  u32|  u64|  i32|  i32|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |  u32|  u32|  u32|  u32|  u32|  u64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|  str|    -|
// |  u64|  u64|  u64|  u64|  u64|  u64|  f64|  f64|  f64|  f64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|  str|    -|
// |   i8|   i8|  i16|  i32|  i64|  f64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |  i16|  i16|  i16|  i32|  i64|  f64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |  i32|  i32|  i32|  i32|  i64|  f64|  i32|  i32|  i32|  i64| i128|  f64|  f64|  i32|  i64|  i64|  i64|  i64|  str|    -|
// |  i64|  i64|  i64|  i64|  i64|  f64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|  i64|  i64|  i64|  i64|  str|    -|
// | i128| i128| i128| i128| i128| i128| i128| i128| i128| i128| i128|  f64|  f64|    -|    -|    -|    -|    -|  str|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|  f32|  f64|  f64|  f64|  f64|  str|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  str|    -|
// | date|    -|    -|    -|  i64|  i64|    -|    -|  i32|  i64|    -|  f32|  f64|    -|    -|    -|    -| date|  str|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    -|    -|    -|    -|  str|    -|
// |   ms|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    -|    -|    -|   ms|  str|    -|
// |   ns|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    -|    -|    -|   ns|  str|    -|
// |    d|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64| date|    -|   ms|   ns|    d|    -|    -|
// |  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|  str|    -|  str|  str|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|  str|    -|
pub fn add(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(arg0.as_expr()? + arg1.as_expr()?));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "+";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.is_atom() && arg1.is_atom() && (arg0.str().is_ok() || arg1.str().is_ok()) {
        let s0 = if let Ok(s) = arg0.str() {
            s
        } else {
            &arg0.to_string()
        };
        let s1 = if let Ok(s) = arg1.str() {
            s
        } else {
            &arg1.to_string()
        };
        let s = format!("{}{}", s0, s1);
        if arg0.sym().is_ok() || arg1.sym().is_ok() {
            return Ok(SpicyObj::Symbol(s));
        } else {
            return Ok(SpicyObj::String(s));
        }
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        match arg0 {
            SpicyObj::Date(date) => match arg1 {
                SpicyObj::Time(_) | SpicyObj::Duration(_) => Ok(SpicyObj::Date(
                    *date + (arg1.to_i64().unwrap() / NS_IN_DAY) as i32,
                )),
                _ => Err(err()),
            },
            SpicyObj::Time(t0) => match arg1 {
                SpicyObj::Date(d1) => Ok(SpicyObj::Timestamp(*t0 + (*d1 as i64) * NS_IN_DAY)),
                SpicyObj::Time(v1) => Ok(SpicyObj::Duration(*t0 + *v1)),
                SpicyObj::Datetime(v1) => Ok(SpicyObj::Datetime(*t0 / NS_IN_MS + *v1)),
                SpicyObj::Timestamp(v1) => Ok(SpicyObj::Timestamp(*t0 + *v1)),
                SpicyObj::Duration(v1) => Ok(SpicyObj::Duration(*t0 + *v1)),
                _ => Err(err()),
            },
            SpicyObj::Datetime(t0) => match arg1 {
                SpicyObj::Time(t1) | SpicyObj::Duration(t1) => {
                    Ok(SpicyObj::Datetime(*t0 + *t1 / NS_IN_MS))
                }
                _ => Err(err()),
            },
            SpicyObj::Timestamp(t0) => match arg1 {
                SpicyObj::Time(t1) | SpicyObj::Duration(t1) => Ok(SpicyObj::Timestamp(*t0 + *t1)),
                _ => Err(err()),
            },
            SpicyObj::Duration(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Timestamp(*t0 + (*t1 as i64) * NS_IN_DAY)),
                SpicyObj::Time(t1) => Ok(SpicyObj::Duration(*t0 + *t1)),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Datetime(*t0 / NS_IN_MS + *t1)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Timestamp(*t0 + *t1)),
                SpicyObj::Duration(t1) => Ok(SpicyObj::Duration(*t0 + *t1)),
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            let i = arg0.to_i64().unwrap() + arg1.to_i64().unwrap();
            if c0 == -1 && c1 == -1 {
                Ok(SpicyObj::I64(i))
            } else if c0 < c1 {
                arg0.new_same_int_atom(i)
            } else {
                arg1.new_same_int_atom(i)
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap() + arg1.to_f32().unwrap(),
            ))
        } else if c0 == -12 || c1 == -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap() + arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, add)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, add)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, add)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, add)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, add)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, add)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0 + s1.clone()).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, add),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, add),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, add),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, add),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, add),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), add(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), v.clone());
                            }
                        }
                    }
                    for (k, v) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), v.clone());
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        add(&[arg1, arg0]).map_err(|e| match e {
            SpicyError::UnsupportedBinaryOpErr(_, _, _) => err(),
            _ => e,
        })
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            if m0.dim() == m1.dim() {
                Ok(SpicyObj::Matrix((m0 + m1).to_shared()))
            } else {
                Err(SpicyError::Err(format!(
                    "Matrix dim are not matched, '{:?}' vs '{:?}'",
                    m0.dim(),
                    m1.dim()
                )))
            }
        } else if arg0.is_matrix() && arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix((m0 + arg1.to_f64().unwrap()).to_shared()))
        } else if arg0.to_f64().is_ok() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix((m1 + arg0.to_f64().unwrap()).to_shared()))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    -|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   u8|   u8|   u8|  u16|  u32|  u64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u16|  u16|  u16|  u16|  u32|  u64|  i32|  i32|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u32|  u32|  u32|  u32|  u32|  u64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|    -|    -|
// |  u64|  u64|  u64|  u64|  u64|  u64|  f64|  f64|  f64|  f64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|    -|    -|
// |   i8|   i8|  i16|  i32|  i64|  f64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i16|  i16|  i16|  i32|  i64|  f64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i32|  i32|  i32|  i32|  i64|  f64|  i32|  i32|  i32|  i64| i128|  f64|  f64|  i32|  i64|  i64|  i64|  i64|    -|    -|
// |  i64|  i64|  i64|  i64|  i64|  f64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|  i64|  i64|  i64|  i64|    -|    -|
// | i128| i128| i128| i128| i128| i128| i128| i128| i128| i128| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|  f32|  f64|  f64|  f64|  f64|    -|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|
// | date|    -|    -|    -|  i64|  i64|    -|    -|  i32|  i64|    -|  f32|  f64|    -|    -|    -|    d| date|    -|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    d|    -|    -|    -|    -|    -|
// |   ms|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    -|    -|    -|   ms|    -|    -|
// |   ns|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|    d|    -|    -|    d|   ns|    -|    -|
// |    d|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  str|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
pub fn minus(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(arg0.as_expr()? - arg1.as_expr()?));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "-";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        return Err(err());
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        match arg0 {
            SpicyObj::Date(t0) => match arg1 {
                SpicyObj::Date(_) => Ok(SpicyObj::I64(
                    arg0.to_i64().unwrap() - arg1.to_i64().unwrap(),
                )),
                SpicyObj::Time(_) | SpicyObj::Duration(_) => Ok(SpicyObj::Date(
                    *t0 - (arg1.to_i64().unwrap() / NS_IN_DAY) as i32,
                )),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Duration(
                    arg0.to_i64().unwrap() * NS_IN_DAY - *t1 * NS_IN_MS,
                )),
                SpicyObj::Timestamp(t1) => {
                    Ok(SpicyObj::Duration(arg0.to_i64().unwrap() * NS_IN_DAY - *t1))
                }
                _ => Err(err()),
            },
            SpicyObj::Time(t0) => match arg1 {
                SpicyObj::Time(v1) | SpicyObj::Duration(v1) => Ok(SpicyObj::Duration(*t0 - *v1)),
                _ => Err(err()),
            },
            SpicyObj::Datetime(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Duration(
                    *t0 * NS_IN_MS - (*t1 as i64) * NS_IN_DAY,
                )),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Duration((*t0 - *t1) * NS_IN_MS)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Duration(*t0 * NS_IN_MS - *t1)),
                SpicyObj::Time(t1) | SpicyObj::Duration(t1) => {
                    Ok(SpicyObj::Datetime(*t0 - *t1 / NS_IN_MS))
                }
                _ => Err(err()),
            },
            SpicyObj::Timestamp(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Duration(*t0 - (*t1 as i64) * NS_IN_DAY)),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Duration(*t0 - *t1 * NS_IN_MS)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Duration(*t0 - *t1)),
                SpicyObj::Time(t1) | SpicyObj::Duration(t1) => Ok(SpicyObj::Timestamp(*t0 - *t1)),
                _ => Err(err()),
            },
            SpicyObj::Duration(t0) => match arg1 {
                SpicyObj::Time(t1) | SpicyObj::Duration(t1) => Ok(SpicyObj::Duration(*t0 - *t1)),
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            let i = arg0.to_i64().unwrap() - arg1.to_i64().unwrap();
            if c0 == -1 && c1 == -1 {
                Ok(SpicyObj::I64(i))
            } else if c0 < c1 {
                arg0.new_same_int_atom(i)
            } else {
                arg1.new_same_int_atom(i)
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap() - arg1.to_f32().unwrap(),
            ))
        } else if c0 == -12 || c1 == -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap() - arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, minus)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, minus)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, minus)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, minus)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, minus)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, minus)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0 - s1.clone()).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, minus),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, minus),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, minus),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, minus),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, minus),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), minus(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), v.clone());
                            }
                        }
                    }
                    for (k, v) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), math::neg(&[v])?);
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), l1.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, add)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, add)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.clone() - s1).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg0.is_matrix() {
        if arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            if m0.dim() == m1.dim() {
                Ok(SpicyObj::Matrix((m0 - m1).to_shared()))
            } else {
                Err(SpicyError::Err(format!(
                    "Matrix dim are not matched, '{:?}' vs '{:?}'",
                    m0.dim(),
                    m1.dim()
                )))
            }
        } else if arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix((m0 - arg1.to_f64().unwrap()).to_shared()))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    -|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   u8|   u8|   u8|  u16|  u32|  u64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  u16|  u16|  u16|  u16|  u32|  u64|  i32|  i32|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  u32|  u32|  u32|  u32|  u32|  u64|  i64|  i64|  i64|  i64| i128|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  u64|  u64|  u64|  u64|  u64|  u64|  f64|  f64|  f64|  f64| i128|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// |   i8|   i8|  i16|  i32|  i64|  f64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  i16|  i16|  i16|  i32|  i64|  f64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  i32|  i32|  i32|  i32|  i64|  f64|  i32|  i32|  i32|  i64| i128|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  i64|  i64|  i64|  i64|  i64|  f64|  i64|  i64|  i64|  i64| i128|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// | i128| i128| i128| i128| i128| i128| i128| i128| i128| i128| i128|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|    -|    -|    -|    -|    d|    -|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    d|    -|    -|
// | date|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ms|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ns|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |    d|    -|    d|    d|    d|    d|    d|    d|    d|    d|    d|    d|    d|    -|    -|    -|    -|    -|    -|    -|
// |  str|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
pub fn mul(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(arg0.as_expr()? * arg1.as_expr()?));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "*";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        return Err(err());
    }

    if c0 < 0 && c1 < 0 {
        if arg0.is_temporal() || arg1.is_temporal() {
            if c0 == -10 && !arg1.is_temporal() {
                if let Ok(f) = arg1.to_f64() {
                    Ok(SpicyObj::Duration(
                        (arg0.to_i64().unwrap() as f64 * f) as i64,
                    ))
                } else {
                    Ok(SpicyObj::Duration(
                        arg0.to_i64().unwrap() * arg1.to_i64().unwrap(),
                    ))
                }
            } else if c1 == -10 && !arg0.is_temporal() {
                if let Ok(f) = arg0.to_f64() {
                    Ok(SpicyObj::Duration(
                        (arg1.to_i64().unwrap() as f64 * f) as i64,
                    ))
                } else {
                    Ok(SpicyObj::Duration(
                        arg1.to_i64().unwrap() * arg0.to_i64().unwrap(),
                    ))
                }
            } else {
                Err(err())
            }
        } else if c0 == -1 && c1 == -1 {
            Ok(SpicyObj::I64(
                arg0.to_i64().unwrap() * arg1.to_i64().unwrap(),
            ))
        } else if c0 >= -5 && c1 >= -5 {
            if c0 < c1 {
                arg0.new_same_int_atom(arg0.to_i64().unwrap() * arg1.to_i64().unwrap())
            } else {
                arg1.new_same_int_atom(arg0.to_i64().unwrap() * arg1.to_i64().unwrap())
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap() * arg1.to_f32().unwrap(),
            ))
        } else if c0 == -12 || c1 == -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap() * arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, mul)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, mul)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, mul)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, mul)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, mul)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, mul)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0 * s1.clone()).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, mul),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, mul),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, mul),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, mul),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, mul),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), mul(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), v.clone());
                            }
                        }
                    }
                    for (k, v) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), v.clone());
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        mul(&[arg1, arg0]).map_err(|e| match e {
            SpicyError::UnsupportedBinaryOpErr(_, _, _) => err(),
            _ => e,
        })
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            if m0.ncols() == m1.nrows() {
                Ok(SpicyObj::Matrix(m0.dot(m1).to_shared()))
            } else {
                Err(SpicyError::Err(format!(
                    "Incompatible shape, left rows '{}' doesn't equal to right columns '{}'",
                    m0.ncols(),
                    m1.nrows()
                )))
            }
        } else if arg0.is_matrix() && arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix((m0 * arg1.to_f64().unwrap()).to_shared()))
        } else if arg0.to_f64().is_ok() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix((m1 * arg0.to_f64().unwrap()).to_shared()))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

// !
pub fn dict(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let keys = args[0];
    let values = args[1];
    let mut m: IndexMap<String, SpicyObj> = IndexMap::new();
    let err = || SpicyError::EvalErr("Not sym keys".to_owned());
    if keys.size() == values.size() {
        let keys: Vec<&str> = match keys {
            SpicyObj::Series(k) => match k.dtype() {
                DataType::Categorical(_, _) => {
                    if k.cat32().is_ok() {
                        k.cat32().unwrap().iter_str().map(|s| s.unwrap()).collect()
                    } else if k.cat8().is_ok() {
                        k.cat8().unwrap().iter_str().map(|s| s.unwrap()).collect()
                    } else {
                        k.cat16().unwrap().iter_str().map(|s| s.unwrap()).collect()
                    }
                }
                DataType::String => k.str().unwrap().iter().map(|s| s.unwrap_or("")).collect(),
                _ => return Err(err()),
            },
            SpicyObj::MixedList(k) => k
                .iter()
                .map(|k| {
                    if k.is_sym() {
                        Ok(k.str().unwrap())
                    } else {
                        Err(SpicyError::EvalErr("Not sym keys".to_owned()))
                    }
                })
                .collect::<Result<Vec<&str>, SpicyError>>()?,
            SpicyObj::Symbol(k) => {
                vec![k]
            }
            _ => return Err(err()),
        };
        if keys.len() == 1 {
            m.insert(keys[0].to_string(), values.clone());
            return Ok(SpicyObj::Dict(m));
        }
        let values = values.as_vec()?;
        for (key, value) in keys.into_iter().zip(values.into_iter()) {
            m.insert(key.to_owned(), value);
        }
        Ok(SpicyObj::Dict(m))
    } else {
        Err(SpicyError::MismatchedLengthErr(keys.size(), values.size()))
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   u8|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u16|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u32|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   i8|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i16|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i32|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// | i128|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// | date|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ms|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ns|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |    d|    -|    d|    d|    d|    d|    d|    d|    d|    d|    d|    d|    d|    -|    -|    -|    -|  f64|    -|    -|
// |  str|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
pub fn true_div(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(Expr::BinaryExpr {
            left: arg0.as_expr()?.into(),
            op: Operator::TrueDivide,
            right: arg1.as_expr()?.into(),
        }));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = TRUE_DIV_OP;
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        return Err(err());
    }

    if c0 < 0 && c1 < 0 {
        if arg0.is_temporal() || arg1.is_temporal() {
            if c0 == -10 && c1 == -10 {
                Ok(SpicyObj::F64(
                    arg0.to_f64().unwrap() / arg1.to_f64().unwrap(),
                ))
            } else {
                Err(err())
            }
        } else {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap() / arg1.to_f64().unwrap(),
            ))
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, true_div)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, true_div)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, true_div)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, true_div)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap().to_float().map_err(|_| err())?;
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, true_div)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, true_div)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0 / s1.clone()).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, true_div),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, true_div),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, true_div),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, true_div),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, true_div),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), true_div(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), v.clone());
                            }
                        }
                    }
                    for (k, v) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), v.clone());
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap().to_float().map_err(|_| err())?;
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(l1.len(), s0.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, true_div)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, true_div)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0 / s1).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            // matrix0 * inverse matrix1, pending
            Err(err())
        } else if arg0.is_matrix() && arg1.is_bool() || arg1.is_numeric() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix((m0 / arg1.to_f64().unwrap()).to_shared()))
        } else if arg0.is_bool() || arg0.is_numeric() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix((m1 / arg0.to_f64().unwrap()).to_shared()))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    -|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   u8|   u8|   u8|  u16|  u32|  u64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u16|  u16|  u16|  u16|  u32|  u64|  i32|  i32|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u32|  u32|  u32|  u32|  u32|  u64|  i64|  i64|  i64|  i64| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u64|  u64|  u64|  u64|  u64|  u64|  f64|  f64|  f64|  f64| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |   i8|   i8|  i16|  i32|  i64|  f64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i16|  i16|  i16|  i32|  i64|  f64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i32|  i32|  i32|  i32|  i64|  f64|  i32|  i32|  i32|  i64| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i64|  i64|  i64|  i64|  i64|  f64|  i64|  i64|  i64|  i64| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// | i128| i128| i128| i128| i128| i128| i128| i128| i128| i128| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// | date|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ms|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |   ns|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |    d|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |  str|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|
pub fn div(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(Expr::BinaryExpr {
            left: arg0.as_expr()?.into(),
            op: Operator::FloorDivide,
            right: arg1.as_expr()?.into(),
        }));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "div";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        return Err(err());
    }

    if c0 < 0 && c1 < 0 {
        if arg0.is_temporal() || arg1.is_temporal() {
            if c0 == -10 && c1 == -10 {
                Ok(SpicyObj::I64(
                    arg0.to_i64().unwrap() / arg1.to_i64().unwrap(),
                ))
            } else {
                Err(err())
            }
        } else if c0 >= -5 && c1 >= -5 {
            if c0 < c1 {
                arg0.new_same_int_atom(arg0.to_i64().unwrap() / arg1.to_i64().unwrap())
            } else {
                arg1.new_same_int_atom(arg0.to_i64().unwrap() / arg1.to_i64().unwrap())
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap().div_euclid(arg1.to_f32().unwrap()),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap().div_euclid(arg1.to_f64().unwrap()),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, div)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, div)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, div)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, div)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, div)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, div)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (floor_div_series(&s0, s1)).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, div),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, div),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, div),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, div),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, div),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), div(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), v.clone());
                            }
                        }
                    }
                    for (k, v) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), v.clone());
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(l1.len(), s0.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, div)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, div)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (floor_div_series(s0, &s1)).map_err(|e| SpicyError::Err(e.to_string()))?,
                ))
            }
        }
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            // matrix0 * inverse matrix1, pending
            Err(err())
        } else if arg0.is_matrix() && arg1.is_bool() || arg1.is_numeric() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix(
                m0.clone()
                    .mapv_into(|x| x.div_euclid(arg1.to_f64().unwrap())),
            ))
        } else if arg0.is_bool() || arg0.is_numeric() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix(
                m1.clone()
                    .mapv_into(|x| arg0.to_f64().unwrap().div_euclid(x)),
            ))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    b|    -|
// |   u8|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  u16|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  u32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    b|    b|    b|    -|    -|
// |  u64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    b|    b|    b|    -|    -|
// |   i8|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  i16|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  i32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// |  i64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// | i128|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  f32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// |  f64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// | date|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|panic|    b|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|    b|    b|    -|    b|    b|    -|    b|    -|    -|    -|    b|    -|
// |   ms|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|    b|    b|    -|
// |   ns|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|    b|    b|    -|
// |    d|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|panic|    -|    b|    b|    b|    -|    -|
// |  str|    b|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    b|    b|    b|    b|    -|    b|    b|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    b|    b|
pub fn gt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(
            arg0.as_expr().unwrap().gt(arg1.as_expr().unwrap()),
        ));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = ">";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        if arg0.str().is_ok() && arg1.str().is_ok() {
            return Ok(SpicyObj::Boolean(arg0.str().unwrap() > arg1.str().unwrap()));
        } else {
            return Err(err());
        }
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        match arg0 {
            SpicyObj::Date(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(*t0 > *t1)),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * MS_IN_DAY > *t1)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * NS_IN_DAY > *t1)),
                _ => Err(err()),
            },
            SpicyObj::Time(t0) => match arg1 {
                SpicyObj::Time(v1) => Ok(SpicyObj::Boolean(*t0 > *v1)),
                SpicyObj::Datetime(v1) => Ok(SpicyObj::Boolean(*t0 > *v1 * NS_IN_MS % NS_IN_DAY)),
                SpicyObj::Timestamp(v1) => Ok(SpicyObj::Boolean(*t0 > *v1 % NS_IN_DAY)),
                _ => Err(err()),
            },
            SpicyObj::Datetime(_) | SpicyObj::Timestamp(_) => {
                let t0 = match arg0 {
                    SpicyObj::Datetime(t0) => *t0 * NS_IN_MS,
                    SpicyObj::Timestamp(t0) => *t0,
                    _ => return Err(err()),
                };
                match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Boolean((t0 % NS_IN_DAY) > *t1)),
                    SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(t0 > (*t1 as i64) * NS_IN_DAY)),
                    SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean(t0 > *t1 * NS_IN_MS)),
                    SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean(t0 > *t1)),
                    _ => Err(err()),
                }
            }
            SpicyObj::Duration(t0) => match arg1 {
                SpicyObj::Duration(v1) => Ok(SpicyObj::Boolean(*t0 > *v1)),
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            Ok(SpicyObj::Boolean(
                arg0.to_i64().unwrap() > arg1.to_i64().unwrap(),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::Boolean(
                arg0.to_f64().unwrap() > arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && arg1.is_numeric() {
        match arg0 {
            SpicyObj::MixedList(l0) => {
                let res = l0
                    .iter()
                    .map(|args| gt(&[args, arg1]))
                    .collect::<SpicyResult<Vec<SpicyObj>>>();
                Ok(SpicyObj::MixedList(res?))
            }
            SpicyObj::Dict(d0) => {
                let mut res = d0.clone();
                for (k, v) in d0.iter() {
                    res.insert(k.to_string(), gt(&[v, arg1])?);
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if arg0.is_numeric() && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => {
                let l = l1
                    .iter()
                    .map(|arg1| gt(&[arg0, arg1]))
                    .collect::<SpicyResult<Vec<SpicyObj>>>();
                Ok(SpicyObj::MixedList(l?))
            }
            SpicyObj::Dict(d1) => {
                let mut res = d1.clone();
                for (k, v) in d1.iter() {
                    res.insert(k.to_string(), gt(&[arg0, v])?);
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, gt)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    Ok(dict_op_list(d0, &l1, gt)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.gt(s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, gt),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, gt),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, gt),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, gt),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, gt),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), gt(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), SpicyObj::Boolean(true));
                            }
                        }
                    }
                    for (k, _) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), SpicyObj::Boolean(false));
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(l1.len(), s0.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, gt)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, gt)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.gt(&s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            if m0.dim() == m1.dim() {
                Ok(SpicyObj::Matrix(
                    Array2::from_shape_vec(
                        m0.dim(),
                        m0.iter()
                            .zip(m1.iter())
                            .map(|(x, y)| if x > y { 1.0 } else { 0.0 })
                            .collect::<Vec<f64>>(),
                    )
                    .unwrap()
                    .to_shared(),
                ))
            } else {
                Err(SpicyError::Err(format!(
                    "Matrix dim are not matched, '{:?}' vs '{:?}'",
                    m0.dim(),
                    m1.dim()
                )))
            }
        } else if arg0.is_matrix() && arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix(m0.clone().mapv_into(|x| {
                if x > arg1.to_f64().unwrap() { 1.0 } else { 0.0 }
            })))
        } else if arg0.to_f64().is_ok() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix(m1.clone().mapv_into(|x| {
                if arg0.to_f64().unwrap() > x { 1.0 } else { 0.0 }
            })))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

pub fn lt_eq(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "<=";
    if args[0].is_expr() || args[1].is_expr() {
        return Ok(SpicyObj::Expr(
            args[0].as_expr().unwrap().lt_eq(args[1].as_expr().unwrap()),
        ));
    }
    let e = match gt(args) {
        Ok(args) => match not(&[&args]) {
            Ok(args) => return Ok(args),
            Err(e) => e,
        },
        Err(e) => e,
    };
    match e {
        SpicyError::UnsupportedBinaryOpErr(_, t0, t1) => {
            Err(SpicyError::UnsupportedBinaryOpErr(op.to_owned(), t0, t1))
        }
        e => Err(e),
    }
}

pub fn lt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(
            arg0.as_expr().unwrap().lt(arg1.as_expr().unwrap()),
        ));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "<";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        if arg0.str().is_ok() && arg1.str().is_ok() {
            return Ok(SpicyObj::Boolean(arg0.str().unwrap() < arg1.str().unwrap()));
        } else {
            return Err(err());
        }
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        match arg0 {
            SpicyObj::Date(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(*t0 < *t1)),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * MS_IN_DAY < *t1)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * NS_IN_DAY < *t1)),
                _ => Err(err()),
            },
            SpicyObj::Time(t0) => match arg1 {
                SpicyObj::Time(v1) => Ok(SpicyObj::Boolean(*t0 < *v1)),
                SpicyObj::Datetime(v1) => Ok(SpicyObj::Boolean(*t0 < *v1 * NS_IN_MS % NS_IN_DAY)),
                SpicyObj::Timestamp(v1) => Ok(SpicyObj::Boolean(*t0 < *v1 % NS_IN_DAY)),
                _ => Err(err()),
            },
            SpicyObj::Datetime(_) | SpicyObj::Timestamp(_) => {
                let t0 = match arg0 {
                    SpicyObj::Datetime(t0) => *t0 * NS_IN_MS,
                    SpicyObj::Timestamp(t0) => *t0,
                    _ => return Err(err()),
                };
                match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Boolean((t0 % NS_IN_DAY) < *t1)),
                    SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(t0 < (*t1 as i64) * NS_IN_DAY)),
                    SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean(t0 < *t1 * NS_IN_MS)),
                    SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean(t0 < *t1)),
                    _ => Err(err()),
                }
            }
            SpicyObj::Duration(t0) => match arg1 {
                SpicyObj::Duration(v1) => Ok(SpicyObj::Boolean(*t0 < *v1)),
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            Ok(SpicyObj::Boolean(
                arg0.to_i64().unwrap() < arg1.to_i64().unwrap(),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::Boolean(
                arg0.to_f64().unwrap() < arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && arg1.is_numeric() {
        match arg0 {
            SpicyObj::MixedList(l0) => {
                let res = l0
                    .iter()
                    .map(|args| lt(&[args, arg1]))
                    .collect::<SpicyResult<Vec<SpicyObj>>>();
                Ok(SpicyObj::MixedList(res?))
            }
            SpicyObj::Dict(d0) => {
                let mut res = d0.clone();
                for (k, v) in d0.iter() {
                    res.insert(k.to_string(), lt(&[v, arg1])?);
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if arg0.is_numeric() && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => {
                let l = l1
                    .iter()
                    .map(|arg1| lt(&[arg0, arg1]))
                    .collect::<SpicyResult<Vec<SpicyObj>>>();
                Ok(SpicyObj::MixedList(l?))
            }
            SpicyObj::Dict(d1) => {
                let mut res = d1.clone();
                for (k, v) in d1.iter() {
                    res.insert(k.to_string(), lt(&[arg0, v])?);
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, lt)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    Ok(dict_op_list(d0, &l1, lt)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.lt(s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg1.is_mixed_collection() {
        if arg0.size() != arg1.size() {
            return Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()));
        }
        match arg0 {
            SpicyObj::MixedList(l0) => match arg1 {
                SpicyObj::MixedList(l1) => list_op_list(l0, l1, lt),
                SpicyObj::Dict(d1) => list_op_dict(l0, d1, lt),
                _ => Err(err()),
            },
            SpicyObj::Series(_) => {
                let l0 = arg0.as_vec()?;
                match arg1 {
                    SpicyObj::MixedList(l1) => list_op_list(&l0, l1, lt),
                    SpicyObj::Dict(d1) => list_op_dict(&l0, d1, lt),
                    _ => Err(err()),
                }
            }
            SpicyObj::Dict(d0) => match arg1 {
                SpicyObj::MixedList(l1) => dict_op_list(d0, l1, lt),
                SpicyObj::Dict(d1) => {
                    let mut res = IndexMap::new();
                    for (k, v) in d0.iter() {
                        match d1.get(k) {
                            Some(obj) => {
                                res.insert(k.to_string(), lt(&[v, obj])?);
                            }
                            None => {
                                res.insert(k.to_string(), SpicyObj::Boolean(true));
                            }
                        }
                    }
                    for (k, _) in d1.iter() {
                        if !d0.contains_key(k) {
                            res.insert(k.to_string(), SpicyObj::Boolean(false));
                        }
                    }
                    Ok(SpicyObj::Dict(res))
                }
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(l1.len(), s0.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, lt)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, lt)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.lt(&s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            if m0.dim() == m1.dim() {
                Ok(SpicyObj::Matrix(
                    Array2::from_shape_vec(
                        m0.dim(),
                        m0.iter()
                            .zip(m1.iter())
                            .map(|(x, y)| if x < y { 1.0 } else { 0.0 })
                            .collect::<Vec<f64>>(),
                    )
                    .unwrap()
                    .to_shared(),
                ))
            } else {
                Err(SpicyError::Err(format!(
                    "Matrix dim are not matched, '{:?}' vs '{:?}'",
                    m0.dim(),
                    m1.dim()
                )))
            }
        } else if arg0.is_matrix() && arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix(m0.clone().mapv_into(|x| {
                if x < arg1.to_f64().unwrap() { 1.0 } else { 0.0 }
            })))
        } else if arg0.to_f64().is_ok() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix(m1.clone().mapv_into(|x| {
                if arg0.to_f64().unwrap() < x { 1.0 } else { 0.0 }
            })))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

pub fn gt_eq(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = ">=";
    if args[0].is_expr() || args[1].is_expr() {
        return Ok(SpicyObj::Expr(
            args[0].as_expr().unwrap().gt_eq(args[1].as_expr().unwrap()),
        ));
    }
    let e = match lt(args) {
        Ok(args) => match not(&[&args]) {
            Ok(args) => return Ok(args),
            Err(e) => e,
        },
        Err(e) => e,
    };
    match e {
        SpicyError::UnsupportedBinaryOpErr(_, t0, t1) => {
            Err(SpicyError::UnsupportedBinaryOpErr(op.to_owned(), t0, t1))
        }
        e => Err(e),
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    b|    -|
// |   u8|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  u16|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  u32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    b|    b|    b|    -|    -|
// |  u64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    b|    b|    b|    -|    -|
// |   i8|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  i16|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  i32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// |  i64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// | i128|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|    -|    -|    -|    -|    -|
// |  f32|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// |  f64|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    b|    -|    -|
// | date|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|panic|    b|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|    b|    b|    -|    b|    b|    -|    b|    -|    -|    -|    b|    -|
// |   ms|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|    b|    b|    -|
// |   ns|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|    b|    -|    b|    b|    b|    b|    -|
// |    d|    -|    -|    -|    b|    b|    -|    -|    b|    b|    -|    b|    b|panic|    -|    b|    b|    b|    -|    -|
// |  str|    b|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    b|    b|    b|    b|    -|    b|    b|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    b|    b|
pub fn eq(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(
            arg0.as_expr().unwrap().eq_missing(arg1.as_expr().unwrap()),
        ));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "=";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() || arg1.str().is_ok() {
        if arg0.str().is_ok() && arg1.str().is_ok() {
            return Ok(SpicyObj::Boolean(
                arg0.str().unwrap() == arg1.str().unwrap(),
            ));
        } else {
            return Err(err());
        }
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        match arg0 {
            SpicyObj::Date(t0) => match arg1 {
                SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(*t0 == *t1)),
                SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * MS_IN_DAY == *t1)),
                SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean((*t0 as i64) * NS_IN_DAY == *t1)),
                _ => Err(err()),
            },
            SpicyObj::Time(t0) => match arg1 {
                SpicyObj::Time(v1) => Ok(SpicyObj::Boolean(*t0 == *v1)),
                SpicyObj::Datetime(v1) => Ok(SpicyObj::Boolean(*t0 == *v1 * NS_IN_MS % NS_IN_DAY)),
                SpicyObj::Timestamp(v1) => Ok(SpicyObj::Boolean(*t0 == *v1 % NS_IN_DAY)),
                _ => Err(err()),
            },
            SpicyObj::Datetime(_) | SpicyObj::Timestamp(_) => {
                let t0 = match arg0 {
                    SpicyObj::Datetime(t0) => *t0 * NS_IN_MS,
                    SpicyObj::Timestamp(t0) => *t0,
                    _ => return Err(err()),
                };
                match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Boolean((t0 % NS_IN_DAY) == *t1)),
                    SpicyObj::Date(t1) => Ok(SpicyObj::Boolean(t0 == (*t1 as i64) * NS_IN_DAY)),
                    SpicyObj::Datetime(t1) => Ok(SpicyObj::Boolean(t0 == *t1 * NS_IN_MS)),
                    SpicyObj::Timestamp(t1) => Ok(SpicyObj::Boolean(t0 == *t1)),
                    _ => Err(err()),
                }
            }
            SpicyObj::Duration(t0) => match arg1 {
                SpicyObj::Duration(v1) => Ok(SpicyObj::Boolean(*t0 == *v1)),
                _ => Err(err()),
            },
            _ => Err(err()),
        }
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            Ok(SpicyObj::Boolean(
                arg0.to_i64().unwrap() == arg1.to_i64().unwrap(),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::Boolean(
                arg0.to_f64().unwrap() == arg1.to_f64().unwrap(),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, eq)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, eq)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, eq)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, eq)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    list_op_list(l0, &l1, eq)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    Ok(dict_op_list(d0, &l1, eq)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.equal(s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(l1.len(), s0.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    list_op_list(&l0, l1, eq)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, eq)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                Ok(SpicyObj::Series(
                    (s0.equal(&s1))
                        .map_err(|e| SpicyError::Err(e.to_string()))?
                        .into(),
                ))
            }
        }
    } else if arg0.is_matrix() || arg1.is_matrix() {
        if arg0.is_matrix() && arg1.is_matrix() {
            let m0 = arg0.matrix().unwrap();
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Boolean(m0 == m1))
        } else if arg0.is_matrix() && arg1.to_f64().is_ok() {
            let m0 = arg0.matrix().unwrap();
            Ok(SpicyObj::Matrix(m0.clone().mapv_into(|x| {
                if x == arg1.to_f64().unwrap() {
                    1.0
                } else {
                    0.0
                }
            })))
        } else if arg0.to_f64().is_ok() && arg1.is_matrix() {
            let m1 = arg1.matrix().unwrap();
            Ok(SpicyObj::Matrix(m1.clone().mapv_into(|x| {
                if arg0.to_f64().unwrap() == x {
                    1.0
                } else {
                    0.0
                }
            })))
        } else {
            Err(err())
        }
    } else {
        Err(err())
    }
}

pub fn ne(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "!=";
    if args[0].is_expr() || args[1].is_expr() {
        return Ok(SpicyObj::Expr(
            args[0].as_expr().unwrap().neq(args[1].as_expr().unwrap()),
        ));
    }
    let e = match eq(args) {
        Ok(args) => match not(&[&args]) {
            Ok(args) => return Ok(args),
            Err(e) => e,
        },
        Err(e) => e,
    };
    match e {
        SpicyError::UnsupportedBinaryOpErr(_, t0, t1) => {
            Err(SpicyError::UnsupportedBinaryOpErr(op.to_owned(), t0, t1))
        }
        e => Err(e),
    }
}

pub fn match_op(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(
            args[0]
                .as_expr()
                .unwrap()
                .eq_missing(args[1].as_expr().unwrap()),
        ));
    }
    match arg0 {
        SpicyObj::Series(s0) => match arg1.series() {
            Ok(s1) => Ok(SpicyObj::Boolean(
                s0.clone().rename("".into()) == s1.clone().rename("".into()),
            )),
            Err(_) => Ok(SpicyObj::Boolean(false)),
        },
        _ => Ok(SpicyObj::Boolean(arg0 == arg1)),
    }
}

pub fn not(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.not()));
    }
    let op = "not";
    match arg0 {
        SpicyObj::MixedList(l) => Ok(SpicyObj::MixedList(
            l.iter()
                .map(|a| not(&[a]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?,
        )),
        SpicyObj::Boolean(b) => Ok(SpicyObj::Boolean(!b)),
        SpicyObj::U8(_)
        | SpicyObj::I16(_)
        | SpicyObj::I32(_)
        | SpicyObj::I64(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_)
        | SpicyObj::Duration(_) => Ok(SpicyObj::Boolean(arg0.to_i64().unwrap() == 0)),
        SpicyObj::F32(f) => Ok(SpicyObj::Boolean(*f == 0.0)),
        SpicyObj::F64(f) => Ok(SpicyObj::Boolean(*f == 0.0)),
        SpicyObj::Null => Ok(SpicyObj::Boolean(false)),
        SpicyObj::Series(s) => {
            if s.dtype().is_bool() {
                Ok(SpicyObj::Series((!s.bool().unwrap()).into()))
            } else if s.dtype().is_integer() {
                Ok(SpicyObj::Series(s.equal(0).unwrap().into()))
            } else if s.dtype().is_float() {
                Ok(SpicyObj::Series(s.equal(0.0).unwrap().into()))
            } else if s.dtype().is_temporal() {
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Int64).unwrap().equal(0).unwrap().into(),
                ))
            } else {
                Err(SpicyError::UnsupportedUnaryOpErr(
                    op.to_owned(),
                    s.dtype().to_string(),
                ))
            }
        }
        SpicyObj::Dict(d) => {
            let mut res = d.clone();
            for (k, v) in d.iter() {
                res.insert(k.to_string(), not(&[v])?);
            }
            Ok(SpicyObj::Dict(res))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::Matrix(
            m.clone().mapv_into(|x| if x == 0.0 { 1.0 } else { 0.0 }),
        )),
        // J::Matrix(_) => todo!(),
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub fn append(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = ",";
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || (arg1.is_expr() && !arg0.is_mixed_list()) {
        return concat_list(&[arg0.as_expr()?, arg1.as_expr()?])
            .map(SpicyObj::Expr)
            .map_err(|e| SpicyError::EvalErr(e.to_string()));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() && arg1.is_null() {
        return Ok(SpicyObj::MixedList(vec![SpicyObj::Null, SpicyObj::Null]));
    }

    if (arg0.is_null() || arg1.is_null() || c0.abs() == c1.abs()) && c0 > -20 && c0 < 20 {
        let s0 = arg0.as_series();
        let s1 = arg1.as_series();

        if let (Ok(s0), Ok(s1)) = (s0, s1) {
            let mut s0 = if s0.dtype().is_null() {
                s0.cast(s1.dtype()).unwrap()
            } else {
                s0
            };
            return Ok(SpicyObj::Series(s0.extend(&s1).unwrap().clone()));
        } else {
            return Err(err());
        }
    }

    if arg0.series().is_ok() && arg1.series().is_ok() {
        let mut s0 = arg0.series().unwrap().clone();
        let s1 = arg1.series().unwrap();
        return Ok(SpicyObj::Series(s0.extend(s1).unwrap().clone()));
    }

    if c0 == c1 && (90..=92).contains(&c0) {
        if c0 == 90 {
            // list
            let l0 = arg0.list().unwrap().clone();
            let l1 = arg1.list().unwrap();
            l0.clone().extend(l1.clone());
            return Ok(SpicyObj::MixedList(l0));
        } else if c0 == 91 {
            // dict
            let d0 = arg0.dict().unwrap();
            let d1 = arg1.dict().unwrap();
            let mut res = d0.clone();
            for (k, v) in d1.iter() {
                res.insert(k.to_string(), v.clone());
            }
            return Ok(SpicyObj::Dict(res));
        } else if c0 == 92 {
            // df
            return arg0
                .df()
                .unwrap()
                .vstack(arg1.df().unwrap())
                .map(SpicyObj::DataFrame)
                .map_err(|e| SpicyError::EvalErr(e.to_string()));
        }
    }

    if arg0.size() == 0 {
        if (-14..0).contains(&c1) {
            Ok(SpicyObj::Series(arg1.into_series().unwrap()))
        } else {
            Ok(arg1.clone())
        }
    } else if arg1.size() == 0 {
        if (-14..0).contains(&c0) {
            Ok(SpicyObj::Series(arg0.into_series().unwrap()))
        } else {
            Ok(arg0.clone())
        }
    } else {
        // mixed list
        let v0 = arg0.as_vec();
        let v1 = arg1.as_vec();
        if let (Ok(v0), Ok(v1)) = (v0, v1) {
            Ok(SpicyObj::MixedList([v0, v1].concat()))
        } else {
            Err(err())
        }
    }
}

pub fn take(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "#";
    let arg0 = args[0];
    let arg1 = args[1];

    if arg1.is_expr() {
        let n = arg0
            .to_i64()
            .map_err(|_| SpicyError::new_arg_type_err(arg0, 0, &ArgType::Int))?;
        let expr = arg1.as_expr().unwrap();
        if n >= 0 {
            return Ok(SpicyObj::Expr(expr.head(Some(n as usize))));
        } else {
            return Ok(SpicyObj::Expr(expr.tail(Some(n.unsigned_abs() as usize))));
        }
    }

    let c0 = arg0.get_type_code();
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if !(-5..=5).contains(&c0) && c0 != -14 && c0 != 14 {
        return Err(SpicyError::new_arg_type_err(arg0, 0, &ArgType::IntLike));
    }

    if (-5..0).contains(&c0) {
        let n = arg0.to_i64().unwrap();
        let take_size = n.unsigned_abs() as usize;
        match arg1 {
            SpicyObj::Boolean(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::U8(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::I16(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::I32(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::I64(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::Date(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![*a; take_size])
                    .cast(&DataType::Date)
                    .unwrap(),
            )),
            SpicyObj::Time(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![*a; take_size])
                    .cast(&DataType::Time)
                    .unwrap(),
            )),
            SpicyObj::Datetime(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![*a; take_size])
                    .cast(&DataType::Datetime(ms, None))
                    .unwrap(),
            )),
            SpicyObj::Timestamp(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![*a; take_size])
                    .cast(&DataType::Datetime(ns, None))
                    .unwrap(),
            )),
            SpicyObj::Duration(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![*a; take_size])
                    .cast(&DataType::Duration(ns))
                    .unwrap(),
            )),
            SpicyObj::F32(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::F64(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![*a; take_size],
            ))),
            SpicyObj::String(a) => Ok(SpicyObj::Series(Series::new(
                "".into(),
                vec![a.to_owned(); take_size],
            ))),
            SpicyObj::Symbol(a) => Ok(SpicyObj::Series(
                Series::new("".into(), vec![a.to_owned(); take_size])
                    .cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
            )),
            SpicyObj::Null => Ok(SpicyObj::MixedList(vec![SpicyObj::Null; take_size])),
            SpicyObj::Series(s) => {
                let mut s = s.clone();
                if n == 0 {
                    return Ok(SpicyObj::Series(s.slice(0, 0)));
                }

                if s.is_empty() {
                    s.extend(&SpicyObj::Null.as_series().unwrap()).unwrap();
                }
                while s.len() < take_size {
                    s.extend(&s.clone()).unwrap();
                }
                if n > 0 {
                    Ok(SpicyObj::Series(s.slice(0, take_size)))
                } else {
                    Ok(SpicyObj::Series(s.slice(n, take_size)))
                }
            }
            SpicyObj::Matrix(m) => {
                let d = m.nrows();
                if take_size >= d {
                    Ok(SpicyObj::Matrix(m.clone()))
                } else if n > 0 {
                    Ok(SpicyObj::Matrix(m.slice(s![..n as isize, ..]).to_shared()))
                } else {
                    Ok(SpicyObj::Matrix(
                        m.slice(s![(n as isize).., ..]).to_shared(),
                    ))
                }
            }
            SpicyObj::DataFrame(df) => {
                if n == 0 {
                    return Ok(SpicyObj::DataFrame(df.slice(0, 0)));
                }
                let mut df = df.clone();
                while df.height() < take_size {
                    df = df.vstack(&df).unwrap();
                }
                if n > 0 {
                    Ok(SpicyObj::DataFrame(df.slice(0, take_size)))
                } else {
                    Ok(SpicyObj::DataFrame(df.slice(n, take_size)))
                }
            }
            SpicyObj::MixedList(l) => {
                let skip = if n < 0 {
                    let r = n % (l.len() as i64);
                    if r < 0 { r + l.len() as i64 } else { 0 }
                } else if n == 0 {
                    return Ok(SpicyObj::MixedList(vec![]));
                } else {
                    0
                };
                Ok(SpicyObj::MixedList(
                    l.iter()
                        .cycle()
                        .skip(skip as usize)
                        .take(take_size)
                        .cloned()
                        .collect(),
                ))
            }
            SpicyObj::Dict(d) => {
                let mut res = IndexMap::new();
                let length = d.len().min(take_size);

                if n > 0 {
                    for (k, v) in d.iter().take(length) {
                        res.insert(k.clone(), v.clone());
                    }
                } else {
                    for (k, v) in d.iter().skip(d.len() - length) {
                        res.insert(k.clone(), v.clone());
                    }
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if c0 > 0 && c0 <= 5 && arg0.size() == 2 {
        let s0 = arg0.series().unwrap().cast(&DataType::Int64).unwrap();
        let s0 = s0.i64().unwrap();
        let d0 = s0.get(0).unwrap_or(0);
        let d1 = s0.get(1).unwrap_or(0);
        let take_size0 = d0.unsigned_abs() as usize;
        let take_size1 = d1.unsigned_abs() as usize;

        if arg1.is_matrix() {
            match arg1 {
                SpicyObj::Matrix(m) => {
                    let nrows = m.nrows();
                    let ncols = m.ncols();
                    if take_size0 >= nrows && take_size1 >= ncols {
                        Ok(SpicyObj::Matrix(m.clone()))
                    } else {
                        let (s0, e0) = if d0 >= 0 {
                            (0, take_size0.min(nrows) as isize)
                        } else {
                            (-(take_size0.min(nrows) as isize), nrows as isize)
                        };

                        let (s1, e1) = if d1 >= 0 {
                            (0, take_size1.min(ncols) as isize)
                        } else {
                            (-(take_size1.min(ncols) as isize), ncols as isize)
                        };

                        Ok(SpicyObj::Matrix(m.slice(s![s0..e0, s1..e1]).to_shared()))
                    }
                }
                _ => Err(err()),
            }
        } else {
            if d0 < 0 || d1 < 0 {
                return Err(SpicyError::Err(format!(
                    "Requires positive dimensions, got '{}, {}'",
                    d0, d1
                )));
            }
            if arg1.is_bool() || arg1.is_integer() || arg1.is_float() {
                Ok(SpicyObj::Matrix(
                    Array2::from_elem([take_size0, take_size1], arg1.to_f64().unwrap()).to_shared(),
                ))
            } else if arg1.is_series() {
                let s = arg1.series().unwrap();
                if s.is_empty() || !(s.dtype().is_bool() || s.dtype().is_primitive_numeric()) {
                    Err(SpicyError::Err(format!(
                        "Requires non-empty bool or numeric series to generate matrix, got '{}' with len '{}'",
                        arg1.get_type_name(),
                        s.len()
                    )))
                } else {
                    let mut s = s.clone();
                    while s.len() < take_size0 * take_size1 {
                        s.extend(&s.clone()).unwrap();
                    }
                    let s = s
                        .slice(0, take_size0 * take_size1)
                        .cast(&DataType::Float64)
                        .unwrap();
                    let s = s.f64().unwrap();
                    let v = s
                        .fill_null_with_values(f64::NAN)
                        .unwrap()
                        .to_vec_null_aware()
                        .unwrap_left();
                    Ok(SpicyObj::Matrix(
                        Array2::from_shape_vec([take_size0, take_size1], v)
                            .unwrap()
                            .into_shared(),
                    ))
                }
            } else {
                Err(err())
            }
        }
    } else if (c0 == -14 || c0 == 14) && arg1.is_dict() {
        let keys = if c0 == -14 {
            vec![arg0.str().unwrap()]
        } else {
            arg0.to_str_vec().unwrap()
        };
        let d1 = arg1.dict().unwrap();
        let mut res = IndexMap::new();
        for (k, v) in d1 {
            if keys.contains(&k.as_str()) {
                res.insert(k.to_owned(), v.clone());
            } else {
                res.insert(k.to_owned(), SpicyObj::Null);
            }
        }
        Ok(SpicyObj::Dict(res))
    } else {
        Err(err())
    }
}

pub fn apply(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = ".";
    let arg0 = args[0];
    // mixed list or series
    let arg1 = args[1];
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };
    if c0 < 0 || (!arg1.is_series() && c1 != 90) {
        return Err(err());
    }
    if arg1.size() == 0 {
        return Err(SpicyError::EvalErr("Unsupported 0-D indices".to_owned()));
    }
    let indices = arg1.as_vec().map_err(|_| {
        SpicyError::EvalErr("Unable to use this kind of series for indices".to_owned())
    })?;
    let mut i_iter = indices.iter();
    let mut res = chili_core::at(&[arg0, i_iter.next().unwrap()])
        .map_err(|e| SpicyError::Err(format!("Failed to apply first index, '{}'", e)))?;
    let mut i: usize = 2;
    for idx in i_iter {
        res = chili_core::at(&[&res, idx])
            .map_err(|e| SpicyError::Err(format!("Failed to apply {} index, '{}'", i, e)))?;
        i += 1;
    }
    Ok(res)
}

pub const CAST_DATA_TYPES: [&str; 20] = [
    "bool",
    "u8",
    "u16",
    "u32",
    "u64",
    "i8",
    "i16",
    "i32",
    "i64",
    "i128",
    "f32",
    "f64",
    "date",
    "timestamp",
    "datetime",
    "time",
    "duration",
    "sym",
    "cat",
    "str",
];

pub static POLARS_DATA_TYPES: LazyLock<HashMap<&str, DataType>> = LazyLock::new(|| {
    HashMap::from_iter([
        ("bool", DataType::Boolean),
        ("u8", DataType::UInt8),
        ("u16", DataType::UInt16),
        ("u32", DataType::UInt32),
        ("u64", DataType::UInt64),
        ("i8", DataType::Int8),
        ("i16", DataType::Int16),
        ("i32", DataType::Int32),
        ("i64", DataType::Int64),
        ("i128", DataType::Int128),
        ("f32", DataType::Float32),
        ("f64", DataType::Float64),
        ("date", DataType::Date),
        ("timestamp", DataType::Datetime(ns, None)),
        ("datetime", DataType::Datetime(ms, None)),
        ("time", DataType::Time),
        ("duration", DataType::Duration(ns)),
        (
            "sym",
            DataType::Categorical(Categories::global(), Categories::global().mapping()),
        ),
        (
            "cat",
            DataType::Categorical(Categories::global(), Categories::global().mapping()),
        ),
        ("str", DataType::String),
    ])
});

pub const CAST_TEMPORAL_DATA_TYPES: [&str; 12] = [
    "year",
    "quarter",
    "month",
    "month_start",
    "month_end",
    "weekday",
    "day",
    "hour",
    "minute",
    "second",
    "ms",
    "ns",
];

pub fn cast(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::StrOrSym, ArgType::Any])?;
    let arg0 = args[0];
    let arg1 = args[1];
    let cast_err = |t: &str, v: &SpicyObj| {
        SpicyError::Err(format!("Failed to cast($) '{}' to type '{}'", v, t))
    };
    if arg1.is_expr() {
        let data_type = arg0.str().unwrap();
        let expr = arg1.as_expr()?;
        if CAST_DATA_TYPES.contains(&data_type) {
            return Ok(SpicyObj::Expr(
                expr.cast(map_str_to_polars_dtype(data_type)?),
            ));
        } else if CAST_TEMPORAL_DATA_TYPES.contains(&data_type) {
            return match data_type {
                "year" => Ok(SpicyObj::Expr(expr.dt().year())),
                "quarter" => Ok(SpicyObj::Expr(expr.dt().quarter())),
                "month" => Ok(SpicyObj::Expr(expr.dt().month())),
                "month_start" => Ok(SpicyObj::Expr(expr.dt().month_start().cast(DataType::Date))),
                "month_end" => Ok(SpicyObj::Expr(expr.dt().month_end().cast(DataType::Date))),
                "weekday" => Ok(SpicyObj::Expr(expr.dt().weekday())),
                "day" => Ok(SpicyObj::Expr(expr.dt().day())),
                "hour" => Ok(SpicyObj::Expr(expr.dt().hour())),
                "minute" => Ok(SpicyObj::Expr(expr.dt().minute())),
                "second" => Ok(SpicyObj::Expr(expr.dt().second())),
                "ms" => Ok(SpicyObj::Expr(expr.dt().millisecond())),
                "ns" => Ok(SpicyObj::Expr(expr.dt().nanosecond())),
                _ => return Err(cast_err(data_type, arg1)),
            };
        } else {
            return Err(SpicyError::EvalErr(format!(
                "Unrecognized data type '{}'",
                data_type
            )));
        }
    }

    let datatype = arg0.str().unwrap();
    if !CAST_DATA_TYPES.contains(&datatype) {
        return Err(SpicyError::EvalErr(format!(
            "Unrecognized data type '{}'",
            datatype
        )));
    }

    if arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(_) if arg1.size() == 0 => match POLARS_DATA_TYPES.get(datatype) {
                Some(dtype) => Ok(SpicyObj::Series(Series::new_empty(datatype.into(), dtype))),
                None => Ok(arg1.clone()),
            },
            SpicyObj::MixedList(l1) => {
                let res = l1
                    .iter()
                    .map(|args| cast(&[arg0, args]))
                    .collect::<SpicyResult<Vec<SpicyObj>>>()?;
                Ok(SpicyObj::MixedList(res))
            }
            SpicyObj::Dict(d1) => {
                let mut res = IndexMap::new();
                for (k, v) in d1 {
                    res.insert(k.to_owned(), cast(&[arg0, v])?);
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(cast_err(datatype, arg1)),
        }
    } else {
        match datatype {
            "bool" => {
                if arg1.is_integer() || arg1.is_float() || arg1.is_temporal() {
                    Ok(SpicyObj::Boolean(arg1.is_truthy().unwrap()))
                } else {
                    match arg1 {
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Boolean).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::Boolean(false)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "u8" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::U8(arg1.to_i64().unwrap() as u8))
                } else if arg1.is_float() {
                    Ok(SpicyObj::U8(arg1.to_f64().unwrap() as u8))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_u8(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::UInt8).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::U8(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "i16" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::I16(arg1.to_i64().unwrap() as i16))
                } else if arg1.is_float() {
                    Ok(SpicyObj::I16(arg1.to_f64().unwrap() as i16))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_i16(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Int16).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::I16(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "i32" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::I32(arg1.to_i64().unwrap() as i32))
                } else if arg1.is_float() {
                    Ok(SpicyObj::I32(arg1.to_f64().unwrap() as i32))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_i32(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Int32).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::I32(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "i64" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::I64(arg1.to_i64().unwrap()))
                } else if arg1.is_float() {
                    Ok(SpicyObj::I64(arg1.to_f64().unwrap() as i64))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_i64(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Int64).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::I64(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "f32" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::F32(arg1.to_i64().unwrap() as f32))
                } else if arg1.is_float() {
                    Ok(SpicyObj::F32(arg1.to_f64().unwrap() as f32))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_f32(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Float32).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::F32(0.0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "f64" => {
                if arg1.is_integer() || arg1.is_temporal() {
                    Ok(SpicyObj::F64(arg1.to_i64().unwrap() as f64))
                } else if arg1.is_float() {
                    Ok(SpicyObj::F64(arg1.to_f64().unwrap()))
                } else {
                    match arg1 {
                        SpicyObj::String(s) => Ok(SpicyObj::parse_f64(s)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Float64).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        // J::Matrix(_) => todo!(),
                        SpicyObj::Null => Ok(SpicyObj::F64(0.0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "date" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::Date(arg1.to_i64().unwrap() as i32))
                } else if arg1.is_float() {
                    Ok(SpicyObj::Date(arg1.to_f64().unwrap() as i32))
                } else {
                    match arg1 {
                        SpicyObj::Date(_) => Ok(arg1.clone()),
                        SpicyObj::Datetime(v) => Ok(SpicyObj::Date((v / MS_IN_DAY) as i32)),
                        SpicyObj::Timestamp(v) => Ok(SpicyObj::Date((v / NS_IN_DAY) as i32)),
                        SpicyObj::Series(s1) => {
                            Ok(SpicyObj::Series(s1.cast(&DataType::Date).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?))
                        }
                        SpicyObj::Null => Ok(SpicyObj::Date(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "timestamp" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::Timestamp(arg1.to_i64().unwrap()))
                } else if arg1.is_float() {
                    Ok(SpicyObj::Timestamp(arg1.to_f64().unwrap() as i64))
                } else {
                    match arg1 {
                        SpicyObj::Date(v) => Ok(SpicyObj::Timestamp((*v as i64) * NS_IN_DAY)),
                        SpicyObj::Datetime(v) => Ok(SpicyObj::Timestamp(v * NS_IN_MS)),
                        SpicyObj::Timestamp(_) => Ok(arg1.clone()),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Datetime(ns, None)).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        SpicyObj::Null => Ok(SpicyObj::Timestamp(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "datetime" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::Datetime(arg1.to_i64().unwrap()))
                } else if arg1.is_float() {
                    Ok(SpicyObj::Datetime(arg1.to_f64().unwrap() as i64))
                } else {
                    match arg1 {
                        SpicyObj::Date(v) => Ok(SpicyObj::Datetime((*v as i64) * MS_IN_DAY)),
                        SpicyObj::Datetime(_) => Ok(arg1.clone()),
                        SpicyObj::Timestamp(v) => Ok(SpicyObj::Datetime(v / NS_IN_MS)),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Datetime(ms, None)).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        SpicyObj::Null => Ok(SpicyObj::Timestamp(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "time" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::Time(arg1.to_i64().unwrap() % NS_IN_DAY))
                } else if arg1.is_float() {
                    Ok(SpicyObj::Time(arg1.to_f64().unwrap() as i64 % NS_IN_DAY))
                } else {
                    match arg1 {
                        SpicyObj::Time(_) => Ok(arg1.clone()),
                        SpicyObj::Duration(v) => Ok(SpicyObj::Time(*v % NS_IN_DAY)),
                        SpicyObj::Series(s1) => {
                            Ok(SpicyObj::Series(s1.cast(&DataType::Time).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?))
                        }
                        SpicyObj::Null => Ok(SpicyObj::Time(0)),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "duration" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::Duration(arg1.to_i64().unwrap()))
                } else if arg1.is_float() {
                    Ok(SpicyObj::Duration(arg1.to_f64().unwrap() as i64))
                } else {
                    match arg1 {
                        SpicyObj::Time(v) => Ok(SpicyObj::Duration(*v)),
                        SpicyObj::Duration(_) => Ok(arg1.clone()),
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Duration(ns)).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        SpicyObj::Null => Ok(SpicyObj::Null),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "sym" | "cat" => {
                if arg1.is_str() {
                    Ok(SpicyObj::Symbol(arg1.str().unwrap().to_owned()))
                } else if arg1.is_atom() {
                    Ok(SpicyObj::Symbol(arg1.to_string()))
                } else {
                    match arg1 {
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::Categorical(
                                Categories::global(),
                                Categories::global().mapping(),
                            ))
                            .map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            "str" => {
                if arg1.is_integer() {
                    Ok(SpicyObj::String(format!("{}", arg1.to_i64().unwrap())))
                } else if arg1.is_temporal() || arg1.is_null() {
                    Ok(SpicyObj::String(format!("{}", arg1)))
                } else if arg1.is_float() {
                    Ok(SpicyObj::String(format!("{}", arg1.to_f64().unwrap())))
                } else {
                    match arg1 {
                        SpicyObj::Series(s1) => Ok(SpicyObj::Series(
                            s1.cast(&DataType::String).map_err(|e| {
                                SpicyError::Err(format!(
                                    "Failed to cast series to '{}', {}",
                                    datatype, e
                                ))
                            })?,
                        )),
                        _ => Err(cast_err(datatype, arg1)),
                    }
                }
            }
            _ => Err(SpicyError::EvalErr(format!(
                "Unrecognized data type '{}'",
                datatype
            ))),
        }
    }
}

// arg0 int: rand, deal
// arg1 null: permute
// arg0 series, mixedList, dict
pub fn rand(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "?";
    let arg0 = args[0];
    // mixed list or series
    let arg1 = args[1];
    let c1 = arg1.get_type_code();
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };
    if !(arg0.is_integer() || arg0.is_series() || arg0.is_mixed_collection() || arg0.is_null()) {
        return Err(err());
    }

    if arg0.is_integer() {
        let mut small_rng = SmallRng::seed_from_u64(get_global_random_u64());
        let i = arg0.to_i64().unwrap();
        if arg1.is_integer() && arg1.to_i64().unwrap() < 0 {
            Err(SpicyError::Err(format!(
                "Requires the upper of the range >= 0 for repeat rand(?), got '{}'",
                arg1.to_i64().unwrap()
            )))
        } else if i >= 0 {
            let i = i as usize;
            match arg1 {
                SpicyObj::Boolean(_) => {
                    let arr: Vec<bool> = (0..i).map(|_| small_rng.random_bool(0.5)).collect();
                    Ok(SpicyObj::Series(Series::new("".into(), arr)))
                }
                SpicyObj::U8(v) => {
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<u8> = if *v == 0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    Ok(SpicyObj::Series(Series::new("".into(), arr)))
                }
                SpicyObj::I16(v) => {
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<i16> = if *v == 0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    Ok(SpicyObj::Series(Series::new("".into(), arr)))
                }
                SpicyObj::I32(v) | SpicyObj::Date(v) => {
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<i32> = if *v == 0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    let series = Series::new("".into(), arr);
                    if c1 == -4 {
                        Ok(SpicyObj::Series(series))
                    } else {
                        Ok(SpicyObj::Series(
                            series
                                .cast(&DataType::Date)
                                .map_err(|e| SpicyError::Err(e.to_string()))?,
                        ))
                    }
                }
                SpicyObj::I64(v)
                | SpicyObj::Time(v)
                | SpicyObj::Datetime(v)
                | SpicyObj::Timestamp(v)
                | SpicyObj::Duration(v) => {
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<i64> = if *v == 0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    let series = Series::new("".into(), arr);
                    Ok(SpicyObj::Series(
                        series
                            .cast(&arg1.get_series_data_type())
                            .map_err(|e| SpicyError::Err(e.to_string()))?,
                    ))
                }
                SpicyObj::F32(v) => {
                    let dist = Uniform::new(0.0, v).unwrap();
                    let arr: Vec<f32> = if *v == 0.0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    let series = Series::new("".into(), arr);
                    Ok(SpicyObj::Series(series))
                }
                SpicyObj::F64(v) => {
                    let dist = Uniform::new(0.0, v).unwrap();
                    let arr: Vec<f64> = if *v == 0.0 {
                        (0..i).map(|_| small_rng.random()).collect()
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    let series = Series::new("".into(), arr);
                    Ok(SpicyObj::Series(series))
                }
                SpicyObj::Series(s) => Ok(SpicyObj::Series(
                    s.sample_n(i, true, false, Some(get_global_random_u64()))
                        .map_err(|e| SpicyError::Err(e.to_string()))?,
                )),
                SpicyObj::MixedList(l) => {
                    let v = l.len();
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<usize> = if v == 0 {
                        vec![]
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    Ok(SpicyObj::MixedList(
                        arr.iter().map(|i| l[*i].clone()).collect(),
                    ))
                }
                SpicyObj::Matrix(m) => {
                    let v = m.nrows();
                    let dist = Uniform::new(0, v).unwrap();
                    let arr: Vec<usize> = if v == 0 {
                        vec![]
                    } else {
                        (0..i).map(move |_| dist.sample(&mut small_rng)).collect()
                    };
                    Ok(SpicyObj::Matrix(m.select(Axis(0), &arr).to_shared()))
                }
                SpicyObj::DataFrame(df) => Ok(SpicyObj::DataFrame(
                    df.sample_n_literal(i, true, false, Some(get_global_random_u64()))
                        .map_err(|e| SpicyError::Err(e.to_string()))?,
                )),
                _ => Err(SpicyError::Err(format!(
                    "Unsupported '{}' for repeat rand(?)",
                    arg1.get_type_name()
                ))),
            }
        } else {
            let i0 = i.unsigned_abs() as usize;
            let i1 = if arg1.is_integer() {
                let i1 = arg1.to_i64().unwrap();
                if i1 > 0 {
                    i1 as usize
                } else {
                    return Err(SpicyError::Err(format!(
                        "Requires the upper of the range > 0 for no-repeat rand(?), got '{}'",
                        arg1.to_i64().unwrap()
                    )));
                }
            } else if arg1.is_float() {
                return Err(SpicyError::Err(format!(
                    "Unsupported float type for no-repeat rand(?), got '{}'",
                    arg1.to_f64().unwrap()
                )));
            } else {
                arg1.size()
            };
            if i0 > i1 {
                Err(SpicyError::Err(format!(
                    "Requires num less than and equals to collection length '{}' for no repeats rand(?), got '{}'",
                    i1, i0
                )))
            } else {
                match arg1 {
                    SpicyObj::Boolean(_) => {
                        Ok(SpicyObj::Series(Series::new("".into(), vec![false])))
                    }
                    SpicyObj::U8(_)
                    | SpicyObj::I16(_)
                    | SpicyObj::I32(_)
                    | SpicyObj::I64(_)
                    | SpicyObj::Date(_)
                    | SpicyObj::Datetime(_)
                    | SpicyObj::Timestamp(_)
                    | SpicyObj::Duration(_)
                    | SpicyObj::Time(_) => {
                        let seq = match rand::seq::index::sample(&mut small_rng, i1, i0) {
                            IndexVec::U32(v) => v,
                            IndexVec::U64(v) => v.into_iter().map(|x| x as u32).collect(),
                        };
                        Ok(SpicyObj::Series(
                            Series::new("".into(), seq)
                                .cast(&arg1.get_series_data_type())
                                .map_err(|e| SpicyError::Err(e.to_string()))?,
                        ))
                    }
                    SpicyObj::Series(s) => Ok(SpicyObj::Series(
                        s.sample_n(i0, false, true, Some(get_global_random_u64()))
                            .map_err(|e| SpicyError::Err(e.to_string()))?,
                    )),
                    SpicyObj::MixedList(l) => {
                        let indices: Vec<usize> =
                            match rand::seq::index::sample(&mut small_rng, i1, i0) {
                                IndexVec::U32(v) => v.into_iter().map(|x| x as usize).collect(),
                                IndexVec::U64(v) => v.into_iter().map(|x| x as usize).collect(),
                            };
                        Ok(SpicyObj::MixedList(
                            indices.iter().map(|i| l[*i].clone()).collect(),
                        ))
                    }
                    SpicyObj::Matrix(m) => {
                        let indices: Vec<usize> =
                            match rand::seq::index::sample(&mut small_rng, i1, i0) {
                                IndexVec::U32(v) => v.into_iter().map(|x| x as usize).collect(),
                                IndexVec::U64(v) => v.into_iter().map(|x| x as usize).collect(),
                            };
                        Ok(SpicyObj::Matrix(m.select(Axis(0), &indices).to_shared()))
                    }
                    SpicyObj::DataFrame(df) => Ok(SpicyObj::DataFrame(
                        df.sample_n_literal(i0, false, true, Some(get_global_random_u64()))
                            .map_err(|e| SpicyError::Err(e.to_string()))?,
                    )),
                    _ => Err(SpicyError::Err(format!(
                        "Unsupported '{}' for no-repeat rand(?)",
                        arg1.get_type_name()
                    ))),
                }
            }
        }
    } else if arg0.is_null() {
        let mut small_rng = SmallRng::seed_from_u64(get_global_random_u64());
        match arg1 {
            SpicyObj::Series(s) => Ok(SpicyObj::Series(s.shuffle(Some(get_global_random_u64())))),
            SpicyObj::MixedList(l) => {
                let indices: Vec<usize> =
                    match rand::seq::index::sample(&mut small_rng, l.len(), l.len()) {
                        IndexVec::U32(v) => v.into_iter().map(|x| x as usize).collect(),
                        IndexVec::U64(v) => v.into_iter().map(|x| x as usize).collect(),
                    };
                Ok(SpicyObj::MixedList(
                    indices.iter().map(|i| l[*i].clone()).collect(),
                ))
            }
            SpicyObj::Matrix(m) => {
                let indices: Vec<usize> =
                    match rand::seq::index::sample(&mut small_rng, m.nrows(), m.nrows()) {
                        IndexVec::U32(v) => v.into_iter().map(|x| x as usize).collect(),
                        IndexVec::U64(v) => v.into_iter().map(|x| x as usize).collect(),
                    };
                Ok(SpicyObj::Matrix(m.select(Axis(0), &indices).to_shared()))
            }
            SpicyObj::DataFrame(df) => Ok(SpicyObj::DataFrame(
                df.sample_n_literal(df.height(), false, true, Some(get_global_random_u64()))
                    .map_err(|e| SpicyError::Err(e.to_string()))?,
            )),
            _ => Err(SpicyError::Err(format!(
                "Unsupported permute rand(?) for '{}'",
                arg1.get_type_name()
            ))),
        }
    } else {
        // find
        match arg0 {
            SpicyObj::Dict(d0) => {
                let item = d0
                    .into_iter()
                    .find(|(_, v)| *match_op(&[*v, arg1]).unwrap().bool().unwrap());
                if let Some((k, _)) = item {
                    Ok(SpicyObj::Symbol(k.to_owned()))
                } else {
                    Ok(SpicyObj::Symbol("".to_owned()))
                }
            }
            SpicyObj::MixedList(l0) => {
                for (i, obj) in l0.iter().enumerate() {
                    if *match_op(&[obj, arg1]).unwrap().bool().unwrap() {
                        return Ok(SpicyObj::I64(i as i64));
                    }
                }
                Ok(SpicyObj::I64(l0.len() as i64))
            }
            _ => Err(err()),
        }
    }
}

// |     |    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64| date| time|   ms|   ns|    d|  str|  cat|
// |    b|    b|   u8|  u16|  u32|  u64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|  str|    -|
// |   u8|   u8|   u8|  u16|  u32|  u64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u16|  u16|  u16|  u16|  u32|  u64|  i32|  i32|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  u32|  u32|  u32|  u32|  u32|  u64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|    -|    -|
// |  u64|  u64|  u64|  u64|  u64|  u64|  f64|  f64|  f64|  f64| i128|  f64|  f64|  i64|    -|  i64|  i64|  i64|    -|    -|
// |   i8|   i8|  i16|  i32|  i64|  f64|   i8|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i16|  i16|  i16|  i32|  i64|  f64|  i16|  i16|  i32|  i64| i128|  f32|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  i32|  i32|  i32|  i32|  i64|  f64|  i32|  i32|  i32|  i64| i128|  f64|  f64|  i32|  i64|  i64|  i64|  i64|    -|    -|
// |  i64|  i64|  i64|  i64|  i64|  f64|  i64|  i64|  i64|  i64| i128|  f64|  f64|  i64|  i64|  i64|  i64|  i64|    -|    -|
// | i128| i128| i128| i128| i128| i128| i128| i128| i128| i128| i128|  f64|  f64|    -|    -|    -|    -|    -|    -|    -|
// |  f32|  f32|  f32|  f32|  f64|  f64|  f32|  f32|  f64|  f64|  f64|  f32|  f64|  f32|  f64|  f64|  f64|  f64|    -|    -|
// |  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|  f64|    -|    -|
// | date|    -|    -|    -|  i64|  i64|    -|    -|  i32|  i64|    -|  f32|  f64| date|    -|   ms|   ns|panic|  str|    -|
// | time|    -|    -|    -|    -|    -|    -|    -|  i64|  i64|    -|  f64|  f64|    -| time|    -|    -|    -|  str|    -|
// |   ms|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|   ms|    -|   ms|   ms|   ms|  str|    -|
// |   ns|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|   ns|    -|   ms|   ns|   ns|  str|    -|
// |    d|    -|    -|    -|  i64|  i64|    -|    -|  i64|  i64|    -|  f64|  f64|panic|    -|    -|    d|    d|    -|    -|
// |  str|  str|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|  str|  str|  str|  str|    -|  str|  str|
// |  cat|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|    -|  str|  cat|
pub fn or(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(Expr::n_ary(
            FunctionExpr::MaxHorizontal,
            vec![arg0.as_expr()?, arg1.as_expr()?],
        )));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "|";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() && arg1.str().is_ok() {
        let s0 = arg0.str().unwrap();
        let s1 = arg1.str().unwrap();
        let res = if s0 > s1 { s0 } else { s1 };
        return Ok(SpicyObj::String(res.to_owned()));
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        if c0 == c1 {
            if arg0.to_i64().unwrap() > arg1.to_i64().unwrap() {
                Ok(arg0.clone())
            } else {
                Ok(arg1.clone())
            }
        } else {
            match arg0 {
                SpicyObj::Date(t0) => match arg1 {
                    SpicyObj::Datetime(t1) => {
                        Ok(SpicyObj::Datetime(((*t0 as i64) * MS_IN_DAY).max(*t1)))
                    }
                    SpicyObj::Timestamp(t1) => {
                        Ok(SpicyObj::Timestamp(((*t0 as i64) * NS_IN_DAY).max(*t1)))
                    }
                    _ => Err(err()),
                },
                SpicyObj::Time(t0) => match arg1 {
                    SpicyObj::Datetime(v1) => {
                        Ok(SpicyObj::Time(*t0.max(&(*v1 * NS_IN_MS % NS_IN_DAY))))
                    }
                    SpicyObj::Timestamp(v1) => Ok(SpicyObj::Time(*t0.max(&(*v1 % NS_IN_DAY)))),
                    _ => Err(err()),
                },
                SpicyObj::Datetime(t0) => match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Time((*t0 * NS_IN_MS % NS_IN_DAY).max(*t1))),
                    SpicyObj::Date(t1) => {
                        Ok(SpicyObj::Datetime(*t0.max(&((*t1 as i64) * MS_IN_DAY))))
                    }
                    SpicyObj::Timestamp(t1) => Ok(SpicyObj::Timestamp((*t0 * NS_IN_MS).max(*t1))),
                    _ => Err(err()),
                },
                SpicyObj::Timestamp(t0) => match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Time((*t0 % NS_IN_DAY).max(*t1))),
                    SpicyObj::Date(t1) => {
                        Ok(SpicyObj::Datetime(*t0.max(&((*t1 as i64) * NS_IN_DAY))))
                    }
                    SpicyObj::Datetime(t1) => Ok(SpicyObj::Timestamp((*t0).max(*t1 * NS_IN_MS))),
                    _ => Err(err()),
                },
                _ => Err(err()),
            }
        }
        // bool, int | bool, int, temporal
        // bool, int, temporal | bool, int
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            let i = arg0.to_i64().unwrap().max(arg1.to_i64().unwrap());
            if c0 < c1 {
                arg0.new_same_int_atom(i)
            } else {
                arg1.new_same_int_atom(i)
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap().max(arg1.to_f32().unwrap()),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap().max(arg1.to_f64().unwrap()),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, or)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, or)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, or)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, or)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    Ok(list_op_list(l0, &l1, or)?)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, or)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                let res = max_horizontal(&[s0.into(), s1.clone().into()]).map_err(|_| err())?;
                if let Some(col) = res {
                    Ok(SpicyObj::Series(col.as_materialized_series().clone()))
                } else {
                    Err(err())
                }
            }
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, or)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, or)?),
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), l1.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    Ok(list_op_list(&l0, l1, or)?)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, or)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                let res = max_horizontal(&[s1.into(), s0.clone().into()]).map_err(|_| err())?;
                if let Some(col) = res {
                    Ok(SpicyObj::Series(col.as_materialized_series().clone()))
                } else {
                    Err(err())
                }
            }
        }
    } else {
        Err(err())
    }
}

pub fn and(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(Expr::n_ary(
            FunctionExpr::MinHorizontal,
            vec![arg0.as_expr()?, arg1.as_expr()?],
        )));
    }
    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();
    let op = "&";
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if arg0.is_null() || arg1.is_null() {
        return Ok(SpicyObj::Null);
    }

    if arg0.str().is_ok() && arg1.str().is_ok() {
        let s0 = arg0.str().unwrap();
        let s1 = arg1.str().unwrap();
        let res = if s0 < s1 { s0 } else { s1 };
        return Ok(SpicyObj::String(res.to_owned()));
    }

    if arg0.is_temporal() && arg1.is_temporal() {
        if c0 == c1 {
            if arg0.to_i64().unwrap() > arg1.to_i64().unwrap() {
                Ok(arg0.clone())
            } else {
                Ok(arg1.clone())
            }
        } else {
            match arg0 {
                SpicyObj::Date(t0) => match arg1 {
                    SpicyObj::Datetime(t1) => {
                        Ok(SpicyObj::Datetime(((*t0 as i64) * MS_IN_DAY).min(*t1)))
                    }
                    SpicyObj::Timestamp(t1) => {
                        Ok(SpicyObj::Timestamp(((*t0 as i64) * NS_IN_DAY).min(*t1)))
                    }
                    _ => Err(err()),
                },
                SpicyObj::Time(t0) => match arg1 {
                    SpicyObj::Datetime(v1) => {
                        Ok(SpicyObj::Time(*t0.min(&(*v1 * NS_IN_MS % NS_IN_DAY))))
                    }
                    SpicyObj::Timestamp(v1) => Ok(SpicyObj::Time(*t0.min(&(*v1 % NS_IN_DAY)))),
                    _ => Err(err()),
                },
                SpicyObj::Datetime(t0) => match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Time((*t0 * NS_IN_MS % NS_IN_DAY).min(*t1))),
                    SpicyObj::Date(t1) => {
                        Ok(SpicyObj::Datetime(*t0.min(&((*t1 as i64) * MS_IN_DAY))))
                    }
                    SpicyObj::Timestamp(t1) => Ok(SpicyObj::Timestamp((*t0 * NS_IN_MS).min(*t1))),
                    _ => Err(err()),
                },
                SpicyObj::Timestamp(t0) => match arg1 {
                    SpicyObj::Time(t1) => Ok(SpicyObj::Time((*t0 % NS_IN_DAY).min(*t1))),
                    SpicyObj::Date(t1) => {
                        Ok(SpicyObj::Datetime(*t0.min(&((*t1 as i64) * NS_IN_DAY))))
                    }
                    SpicyObj::Datetime(t1) => Ok(SpicyObj::Timestamp((*t0).min(*t1 * NS_IN_MS))),
                    _ => Err(err()),
                },
                _ => Err(err()),
            }
        }
        // bool, int | bool, int, temporal
        // bool, int, temporal | bool, int
    } else if c0 < 0 && c1 < 0 {
        if c0 >= -10 && c1 >= -10 {
            let i = arg0.to_i64().unwrap().min(arg1.to_i64().unwrap());
            if c0 < c1 {
                arg0.new_same_int_atom(i)
            } else {
                arg1.new_same_int_atom(i)
            }
        } else if c0 >= -11 && c1 >= -11 {
            Ok(SpicyObj::F32(
                arg0.to_f32().unwrap().min(arg1.to_f32().unwrap()),
            ))
        } else if c0 >= -12 && c1 >= -12 {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap().min(arg1.to_f64().unwrap()),
            ))
        } else {
            Err(err())
        }
    } else if arg0.is_mixed_collection() && c1 < 0 {
        match arg0 {
            SpicyObj::MixedList(l0) => Ok(list_op_atom(l0, arg1, and)?),
            SpicyObj::Dict(d0) => Ok(dict_op_atom(d0, arg1, and)?),
            _ => Err(err()),
        }
    } else if c0 < 0 && arg1.is_mixed_collection() {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, and)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, and)?),
            _ => Err(err()),
        }
    } else if arg1.is_series() {
        let s1 = arg1.series().unwrap();
        match arg0 {
            SpicyObj::MixedList(l0) => {
                if l0.len() != s1.len() {
                    Err(SpicyError::MismatchedLengthErr(l0.len(), s1.len()))
                } else {
                    let l1 = arg1.as_vec()?;
                    Ok(list_op_list(l0, &l1, and)?)
                }
            }
            SpicyObj::Dict(d0) => {
                if s1.len() != d0.len() {
                    Err(SpicyError::MismatchedLengthErr(s1.len(), d0.len()))
                } else {
                    Ok(dict_op_list(d0, &arg1.as_vec()?, and)?)
                }
            }
            _ => {
                let s0 = arg0.as_series().map_err(|_| err())?;
                let res = min_horizontal(&[s0.into(), s1.clone().into()]).map_err(|_| err())?;
                if let Some(col) = res {
                    Ok(SpicyObj::Series(col.as_materialized_series().clone()))
                } else {
                    Err(err())
                }
            }
        }
    } else if arg0.is_mixed_collection() && c0 < 0 {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(atom_op_list(arg0, l1, and)?),
            SpicyObj::Dict(d1) => Ok(atom_op_dict(arg0, d1, and)?),
            _ => Err(err()),
        }
    } else if arg0.is_series() {
        let s0 = arg0.series().unwrap();
        match arg1 {
            SpicyObj::MixedList(l1) => {
                if l1.len() != s0.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), l1.len()))
                } else {
                    let l0 = arg0.as_vec()?;
                    Ok(list_op_list(&l0, l1, and)?)
                }
            }
            SpicyObj::Dict(d1) => {
                if s0.len() != d1.len() {
                    Err(SpicyError::MismatchedLengthErr(s0.len(), d1.len()))
                } else {
                    Ok(list_op_dict(&arg0.as_vec()?, d1, and)?)
                }
            }
            _ => {
                let s1 = arg1.as_series().map_err(|_| err())?;
                let res = min_horizontal(&[s1.into(), s0.clone().into()]).map_err(|_| err())?;
                if let Some(col) = res {
                    Ok(SpicyObj::Series(col.as_materialized_series().clone()))
                } else {
                    Err(err())
                }
            }
        }
    } else {
        Err(err())
    }
}

pub fn remove(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "_";
    let arg0 = args[0];
    let arg1 = args[1];
    let c0 = arg0.get_type_code();
    let err = || {
        SpicyError::UnsupportedBinaryOpErr(
            op.to_owned(),
            arg0.get_type_name(),
            arg1.get_type_name(),
        )
    };

    if !(-5..=5).contains(&c0) && c0 != -14 && c0 != 14 {
        return Err(SpicyError::new_arg_type_err(arg0, 0, &ArgType::IntLike));
    }

    if (-5..0).contains(&c0) {
        let n = arg0.to_i64().unwrap();
        let drop_size = n.unsigned_abs() as usize;
        if n == 0 {
            return Ok(arg1.clone());
        }
        match arg1 {
            SpicyObj::Series(s) => {
                if drop_size >= s.len() {
                    return Ok(SpicyObj::Series(s.clear()));
                }
                if n > 0 {
                    Ok(SpicyObj::Series(s.slice(n, s.len() - drop_size)))
                } else {
                    Ok(SpicyObj::Series(s.slice(0, s.len() - drop_size)))
                }
            }
            SpicyObj::Matrix(m) => {
                let d = m.nrows();

                if drop_size >= d {
                    Ok(SpicyObj::Matrix(m.slice(s![d.., ..]).to_shared()))
                } else if n >= 0 {
                    Ok(SpicyObj::Matrix(m.slice(s![drop_size.., ..]).to_shared()))
                } else {
                    Ok(SpicyObj::Matrix(m.slice(s![0..n as isize, ..]).to_shared()))
                }
            }
            SpicyObj::DataFrame(df) => {
                if df.height() <= drop_size {
                    return Ok(SpicyObj::DataFrame(df.clear()));
                }
                if n > 0 {
                    Ok(SpicyObj::DataFrame(df.slice(n, df.height() - drop_size)))
                } else {
                    Ok(SpicyObj::DataFrame(df.slice(0, df.height() - drop_size)))
                }
            }
            SpicyObj::MixedList(l) => {
                if drop_size >= l.len() {
                    Ok(SpicyObj::MixedList(vec![]))
                } else if n >= 0 {
                    Ok(SpicyObj::MixedList(l[drop_size..].to_vec()))
                } else {
                    Ok(SpicyObj::MixedList(l[..(l.len() - drop_size)].to_vec()))
                }
            }
            SpicyObj::Dict(d) => {
                let mut res = IndexMap::new();
                if drop_size >= d.len() {
                    return Ok(SpicyObj::Dict(IndexMap::new()));
                };
                if n > 0 {
                    for (k, v) in d.iter().skip(drop_size) {
                        res.insert(k.clone(), v.clone());
                    }
                } else {
                    for (k, v) in d.iter().take(d.len() - drop_size) {
                        res.insert(k.clone(), v.clone());
                    }
                }
                Ok(SpicyObj::Dict(res))
            }
            _ => Err(err()),
        }
    } else if c0 > 0 && c0 <= 5 && arg0.size() == 2 {
        let s0 = arg0.series().unwrap().cast(&DataType::Int64).unwrap();
        let s0 = s0.i64().unwrap();
        let d0 = s0.get(0).unwrap_or(0);
        let d1 = s0.get(1).unwrap_or(0);

        let drop_size0 = d0.unsigned_abs() as usize;
        let drop_size1 = d1.unsigned_abs() as usize;
        if arg1.is_matrix() {
            match arg1 {
                SpicyObj::Matrix(m) => {
                    let nrows = m.nrows();
                    let ncols = m.ncols();
                    if drop_size0 >= nrows && drop_size1 >= ncols {
                        Ok(SpicyObj::Matrix(m.slice(s![0..0, 0..0]).to_shared()))
                    } else {
                        let (s0, e0) = if d0 >= 0 {
                            (drop_size0.min(nrows) as isize, nrows as isize)
                        } else {
                            (0isize, -(drop_size0.min(nrows) as isize))
                        };

                        let (s1, e1) = if d1 >= 0 {
                            (drop_size1.min(ncols) as isize, ncols as isize)
                        } else {
                            (0isize, -(drop_size1.min(ncols) as isize))
                        };

                        Ok(SpicyObj::Matrix(m.slice(s![s0..e0, s1..e1]).to_shared()))
                    }
                }
                _ => Err(err()),
            }
        } else {
            Err(err())
        }
    } else if (c0 == -14 || c0 == 14) && arg1.is_dict() {
        let keys = arg0.to_str_vec().unwrap();
        let d1 = arg1.dict().unwrap();
        let mut res = IndexMap::new();
        for (k, v) in d1 {
            if !keys.contains(&k.as_str()) {
                res.insert(k.to_owned(), v.clone());
            }
        }
        Ok(SpicyObj::Dict(res))
    } else {
        Err(err())
    }
}

// atom ^ (series | list | dict)
// series ^ series
// list ^ (list | dict)
// dict & dict
// float ^ matrix
pub fn fill(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(arg0.as_expr()?.fill_null(arg1.as_expr()?)));
    }
    let c0 = arg0.get_type_code();
    let err = || SpicyError::UnsupportedUnaryOpErr("fill".to_owned(), arg0.get_type_name());
    if c0 < 0 {
        match arg1 {
            SpicyObj::MixedList(l1) => Ok(SpicyObj::MixedList(
                l1.iter()
                    .map(|a| if a.is_null() { arg0.clone() } else { a.clone() })
                    .collect(),
            )),
            SpicyObj::Dict(d1) => {
                let mut res = IndexMap::new();
                for (k, v) in d1 {
                    if v.is_null() {
                        res.insert(k.clone(), arg0.clone());
                    } else {
                        res.insert(k.clone(), v.clone());
                    }
                }
                Ok(SpicyObj::Dict(res))
            }
            SpicyObj::Series(s1) if c0 >= -14 => {
                if s1.null_count() == 0 {
                    Ok(arg1.clone())
                } else {
                    let s0 = arg0.into_series().unwrap();
                    let s0 = s0.cast(s1.dtype()).map_err(|_| err())?;
                    let res = s1
                        .zip_with_same_type(&s1.is_not_null(), &s0)
                        .map_err(|_| err())?;
                    Ok(SpicyObj::Series(res))
                }
            }
            SpicyObj::Matrix(m1) if c0 >= -12 => {
                let f0 = arg0.to_f64().unwrap();
                Ok(SpicyObj::Matrix(
                    m1.clone().mapv_into(|v| if v.is_nan() { f0 } else { v }),
                ))
            }
            _ => Err(err()),
        }
    } else if arg0.is_series() && arg1.is_series() {
        let s1 = arg1.as_series().unwrap();
        if s1.null_count() == 0 {
            Ok(arg1.clone())
        } else {
            let s0 = arg0.as_series().unwrap();
            let s0 = s0.cast(s1.dtype()).map_err(|_| err())?;
            let res = s1
                .zip_with_same_type(&s1.is_not_null(), &s0)
                .map_err(|_| err())?;
            Ok(SpicyObj::Series(res))
        }
    } else if arg0.is_mixed_list() && arg1.is_mixed_collection() {
        if arg0.size() == arg1.size() {
            if arg1.is_dict() {
                let mut res = IndexMap::new();
                let l0 = arg0.list().unwrap();
                let d1 = arg1.dict().unwrap();
                for (i, (k, v)) in d1.iter().enumerate() {
                    if v.is_null() {
                        res.insert(k.clone(), l0[i].clone());
                    } else {
                        res.insert(k.clone(), v.clone());
                    }
                }
                Ok(SpicyObj::Dict(res))
            } else {
                let l0 = arg0.list().unwrap();
                let l1 = arg1.list().unwrap();
                Ok(SpicyObj::MixedList(
                    l1.iter()
                        .zip(l0.iter())
                        .map(
                            |(a1, a0)| {
                                if a1.is_null() { a0.clone() } else { a1.clone() }
                            },
                        )
                        .collect(),
                ))
            }
        } else {
            Err(SpicyError::MismatchedLengthErr(arg0.size(), arg1.size()))
        }
    } else if arg0.is_dict() && arg1.is_dict() {
        let d0 = arg0.dict().unwrap();
        let d1 = arg1.dict().unwrap();
        let mut res = IndexMap::new();
        for (k, v) in d1 {
            if v.is_null() && d0.contains_key(k) {
                res.insert(k.clone(), d0.get(k).unwrap().clone());
            } else {
                res.insert(k.clone(), v.clone());
            }
        }
        Ok(SpicyObj::Dict(res))
    } else {
        Err(err())
    }
}
