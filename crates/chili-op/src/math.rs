use indexmap::IndexMap;
use polars::{
    chunked_array::ops::ChunkApply,
    datatypes::DataType,
    lazy::dsl::col,
    prelude::{
        EWMOptions, IntoLazy, QuantileMethod, RollingFnParams, RollingOptionsFixedWindow,
        RollingVarParams, RoundMode,
    },
    series::IntoSeries,
    time::chunkedarray::SeriesOpsTime,
};
use polars_compute::rolling::RollingQuantileParams;
use polars_ops::series::{LogSeries, RoundSeries, negate};

use crate::util::cast_to_int;
use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};

pub fn abs(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.abs()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("abs".to_owned(), arg0.get_type_name());
    match arg0 {
        SpicyObj::Boolean(_)
        | SpicyObj::U8(_)
        | SpicyObj::Date(_)
        | SpicyObj::Time(_)
        | SpicyObj::Datetime(_)
        | SpicyObj::Timestamp(_) => Ok(arg0.clone()),
        SpicyObj::I16(v) => Ok(SpicyObj::I16(v.abs())),
        SpicyObj::I32(v) => Ok(SpicyObj::I32(v.abs())),
        SpicyObj::I64(v) => Ok(SpicyObj::I64(v.abs())),
        SpicyObj::Duration(v) => Ok(SpicyObj::Duration(v.abs())),
        SpicyObj::F32(v) => Ok(SpicyObj::F32(v.abs())),
        SpicyObj::F64(v) => Ok(SpicyObj::F64(v.abs())),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(s) => match polars_ops::series::abs(s) {
            Ok(s) => Ok(SpicyObj::Series(s)),
            Err(_) => Err(err()),
        },
        SpicyObj::Matrix(m) => Ok(SpicyObj::Matrix(m.mapv(f64::abs).to_shared())),
        SpicyObj::MixedList(l) => {
            let l = l
                .iter()
                .map(|args| abs(&[args]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::MixedList(l))
        }
        SpicyObj::Dict(d) => {
            let mut res = IndexMap::new();
            for (k, v) in d {
                res.insert(k.clone(), abs(&[v])?);
            }
            Ok(SpicyObj::Dict(res))
        }
        _ => Err(err()),
    }
}

pub fn all(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.all(true)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("all".to_owned(), arg0.get_type_name());
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
        | SpicyObj::F64(_) => arg0.is_truthy().map(SpicyObj::Boolean),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(s) => {
            let s = s.cast(&DataType::Boolean).map_err(|_| err())?;
            Ok(SpicyObj::Boolean(s.bool().unwrap().all()))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::Boolean(m.iter().all(|a| *a != 0.0))),
        SpicyObj::MixedList(l) => {
            let res = l
                .iter()
                .map(|a| all(&[a]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::Boolean(
                res.into_iter().all(|a| *a.bool().unwrap()),
            ))
        }
        SpicyObj::Dict(d) => {
            let res = d
                .values()
                .map(|a| all(&[a]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::Boolean(
                res.into_iter().all(|a| *a.bool().unwrap()),
            ))
        }
        _ => Err(err()),
    }
}

pub fn any(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.any(true)));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("any".to_owned(), arg0.get_type_name());
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
        | SpicyObj::F64(_) => arg0.is_truthy().map(SpicyObj::Boolean),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(s) => {
            let s = s.cast(&DataType::Boolean).map_err(|_| err())?;
            Ok(SpicyObj::Boolean(s.bool().unwrap().any()))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::Boolean(m.iter().any(|a| *a != 0.0))),
        SpicyObj::MixedList(l) => {
            let res = l
                .iter()
                .map(|a| any(&[a]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::Boolean(
                res.into_iter().any(|a| *a.bool().unwrap()),
            ))
        }
        SpicyObj::Dict(d) => {
            let res = d
                .values()
                .map(|a| any(&[a]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::Boolean(
                res.into_iter().any(|a| *a.bool().unwrap()),
            ))
        }
        _ => Err(err()),
    }
}

pub(crate) fn float_op(
    args: &[&SpicyObj],
    f1: fn(f32) -> f32,
    f2: fn(f64) -> f64,
    err: &dyn Fn() -> SpicyError,
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
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
        | SpicyObj::F64(_) => Ok(SpicyObj::F64(f2(arg0.to_f64().unwrap()))),
        SpicyObj::F32(v) => Ok(SpicyObj::F32(f1(*v))),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(s) => {
            let s = s.to_float().map_err(|_| err())?;
            let s = match s.dtype() {
                DataType::Float32 => s.f32().unwrap().apply_values(&f1).into_series(),
                DataType::Float64 => s.f64().unwrap().apply_values(&f2).into_series(),
                _ => unreachable!(),
            };
            Ok(SpicyObj::Series(s))
        }
        SpicyObj::Matrix(m) => Ok(SpicyObj::Matrix(m.mapv(&f2).to_shared())),
        SpicyObj::MixedList(l) => {
            let res = l
                .iter()
                .map(|a| float_op(&[a], f1, f2, err))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?;
            Ok(SpicyObj::MixedList(res))
        }
        SpicyObj::Dict(d) => {
            let mut res = IndexMap::new();
            for (k, v) in d {
                res.insert(k.clone(), float_op(&[v], f1, f2, err)?);
            }
            Ok(SpicyObj::Dict(res))
        }
        _ => Err(err()),
    }
}

pub(crate) fn arccos(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arccos()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::acos, f64::acos, &err)
}

pub(crate) fn arccosh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arccosh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::acosh, f64::acosh, &err)
}

pub(crate) fn arcsin(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arcsin()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::asin, f64::asin, &err)
}

pub(crate) fn arcsinh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arcsinh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::asinh, f64::asinh, &err)
}

pub(crate) fn arctan(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arctan()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::atan, f64::atan, &err)
}

pub(crate) fn arctanh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.arctanh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("arccos".to_owned(), arg0.get_type_name());
    float_op(args, f32::atanh, f64::atanh, &err)
}

pub(crate) fn cbrt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.cbrt()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("cbrt".to_owned(), arg0.get_type_name());
    float_op(args, f32::cbrt, f64::cbrt, &err)
}

pub(crate) fn ceil(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.ceil()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("ceil".to_owned(), arg0.get_type_name());
    float_op(args, f32::ceil, f64::ceil, &err)
}

pub(crate) fn cos(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.cos()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("cos".to_owned(), arg0.get_type_name());
    float_op(args, f32::cos, f64::cos, &err)
}

pub(crate) fn cosh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.cosh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("cosh".to_owned(), arg0.get_type_name());
    float_op(args, f32::cosh, f64::cosh, &err)
}

pub(crate) fn cot(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.cot()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("cot".to_owned(), arg0.get_type_name());
    float_op(args, cot_f32, cot_f64, &err)
}

fn cot_f32(ang: f32) -> f32 {
    1.0 / ang.tan()
}

fn cot_f64(ang: f64) -> f64 {
    1.0 / ang.tan()
}

pub(crate) fn exp(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.exp()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("exp".to_owned(), arg0.get_type_name());
    float_op(args, f32::exp, f64::exp, &err)
}

pub(crate) fn floor(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.floor()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("floor".to_owned(), arg0.get_type_name());
    float_op(args, f32::floor, f64::floor, &err)
}

pub(crate) fn ln(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.log(1.0f64.exp().into())));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("ln".to_owned(), arg0.get_type_name());
    float_op(args, f32::ln, f64::ln, &err)
}

pub(crate) fn log10(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.log(10.0f64.into())));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("log10".to_owned(), arg0.get_type_name());
    float_op(args, f32::log10, f64::log10, &err)
}

pub(crate) fn log1p(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.log1p()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("log1p".to_owned(), arg0.get_type_name());
    float_op(args, f32::ln_1p, f64::ln_1p, &err)
}

pub fn neg(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(-left));
    }
    match arg0 {
        SpicyObj::MixedList(v) => Ok(SpicyObj::MixedList(
            v.iter()
                .map(|args| neg(&[args]))
                .collect::<SpicyResult<Vec<SpicyObj>>>()?,
        )),
        SpicyObj::Boolean(v) => Ok(SpicyObj::I64(if *v { -1 } else { 0 })),
        SpicyObj::U8(v) => Ok(SpicyObj::I64(-(*v as i64))),
        SpicyObj::I16(v) => Ok(SpicyObj::I16(-*v)),
        SpicyObj::I32(v) => Ok(SpicyObj::I32(-*v)),
        SpicyObj::I64(v) => Ok(SpicyObj::I64(-*v)),
        SpicyObj::Time(v) => Ok(SpicyObj::Duration(-*v)),
        SpicyObj::Duration(v) => Ok(SpicyObj::Duration(-*v)),
        SpicyObj::F32(v) => Ok(SpicyObj::F32(-*v)),
        SpicyObj::F64(v) => Ok(SpicyObj::F64(-*v)),
        SpicyObj::Null => Ok(SpicyObj::Null),
        SpicyObj::Series(v) => Ok(SpicyObj::Series(
            negate(v).map_err(|e| SpicyError::EvalErr(e.to_string()))?,
        )),
        SpicyObj::Dict(v) => {
            let mut d = IndexMap::new();
            for (k, v) in v.iter() {
                d.insert(k.to_owned(), neg(&[v])?);
            }
            Ok(SpicyObj::Dict(d))
        }
        // J::Matrix(_) => todo!(),
        _ => Err(SpicyError::UnsupportedUnaryOpErr(
            "neg".to_owned(),
            arg0.get_type_name(),
        )),
    }
}

pub(crate) fn sin(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.sin()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("sin".to_owned(), arg0.get_type_name());
    float_op(args, f32::sin, f64::sin, &err)
}

pub(crate) fn sinh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.sinh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("sinh".to_owned(), arg0.get_type_name());
    float_op(args, f32::sinh, f64::sinh, &err)
}

pub(crate) fn sqrt(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.sqrt()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("sqrt".to_owned(), arg0.get_type_name());
    float_op(args, f32::sqrt, f64::sqrt, &err)
}

pub(crate) fn tan(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.tan()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("tan".to_owned(), arg0.get_type_name());
    float_op(args, f32::tan, f64::tan, &err)
}

pub(crate) fn tanh(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.tanh()));
    }
    let err = || SpicyError::UnsupportedUnaryOpErr("tanh".to_owned(), arg0.get_type_name());
    float_op(args, f32::tanh, f64::tanh, &err)
}

pub fn ewm_mean(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let alpha = args[0].to_f64()?;
        let s1 = args[1].as_expr()?;
        let options = EWMOptions {
            alpha,
            adjust: true,
            bias: false,
            min_periods: 1,
            ignore_nulls: true,
        };
        return Ok(SpicyObj::Expr(s1.ewm_mean(options)));
    }
    validate_args(args, &[ArgType::Float, ArgType::Series])?;
    let alpha = args[0].to_f64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = EWMOptions {
        alpha,
        adjust: true,
        bias: false,
        min_periods: 1,
        ignore_nulls: true,
    };
    Ok(SpicyObj::Series(
        polars_ops::series::ewm_mean(s1, options).map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn ewm_std(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let alpha = args[0].to_f64()?;
        let s1 = args[1].as_expr()?;
        let options = EWMOptions {
            alpha,
            adjust: true,
            bias: false,
            min_periods: 1,
            ignore_nulls: true,
        };
        return Ok(SpicyObj::Expr(s1.ewm_std(options)));
    }
    validate_args(args, &[ArgType::Float, ArgType::Series])?;
    let alpha = args[0].to_f64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = EWMOptions {
        alpha,
        adjust: true,
        bias: false,
        min_periods: 1,
        ignore_nulls: true,
    };
    Ok(SpicyObj::Series(
        polars_ops::series::ewm_std(s1, options).map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn ewm_var(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let alpha = args[0].to_f64()?;
        let s1 = args[1].as_expr()?;
        let options = EWMOptions {
            alpha,
            adjust: true,
            bias: false,
            min_periods: 1,
            ignore_nulls: true,
        };
        return Ok(SpicyObj::Expr(s1.ewm_var(options)));
    }
    validate_args(args, &[ArgType::Float, ArgType::Series])?;
    let alpha = args[0].to_f64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = EWMOptions {
        alpha,
        adjust: true,
        bias: false,
        min_periods: 1,
        ignore_nulls: true,
    };
    Ok(SpicyObj::Series(
        polars_ops::series::ewm_var(s1, options).map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

// value log base
pub fn log(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].to_f64()?;
        return Ok(SpicyObj::Expr(left.log(right.into())));
    }
    validate_args(args, &[ArgType::NumericLike, ArgType::Float])?;
    let arg0 = args[0];
    let f1 = args[1].to_f64().unwrap();

    if arg0.is_atom() {
        let f0 = arg0.to_f64().unwrap();
        Ok(SpicyObj::F64(f0.log(f1)))
    } else {
        let s0 = arg0.series().unwrap();
        let s0 = if s0.dtype().is_float() {
            s0.clone()
        } else {
            s0.cast(&DataType::Float64).unwrap()
        };
        Ok(SpicyObj::Series(
            s0.log(&SpicyObj::F64(f1).as_series().unwrap()),
        ))
    }
}

pub fn mod_op(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];

    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left % right));
    }

    validate_args(args, &[ArgType::NumericNative, ArgType::NumericNative])?;

    let c0 = arg0.get_type_code();
    let c1 = arg1.get_type_code();

    if arg1.is_float() || arg0.is_float_like() {
        if c0 == -11 {
            let f0 = arg0.f32().unwrap();
            let f1 = arg1.to_f64().unwrap() as f32;
            Ok(SpicyObj::F32((*f0) % f1))
        } else if c0 < 0 {
            let f0 = arg0.to_f64().unwrap();
            let f1 = arg1.to_f64().unwrap();
            Ok(SpicyObj::F64(f0 % f1))
        } else if c0 == 11 {
            let s0 = arg0.series().unwrap();
            let f1 = arg1.to_f64().unwrap() as f32;
            Ok(SpicyObj::Series(
                s0.f32().unwrap().apply_values(|f| f % f1).into(),
            ))
        } else {
            let s0 = arg0.series().unwrap();
            let s0 = s0.cast(&DataType::Float64).unwrap();
            let f1 = arg1.to_f64().unwrap();
            Ok(SpicyObj::Series(
                s0.f64().unwrap().apply_values(|f| f % f1).into(),
            ))
        }
    } else if c0 < 0 && c1 < 0 {
        Ok(SpicyObj::I64(
            arg0.to_i64().unwrap() % arg1.to_i64().unwrap(),
        ))
    } else {
        let s0 = arg0.as_series().unwrap();
        let s0 = cast_to_int(&s0)?;
        let s1 = arg1.as_series().unwrap();
        let s1 = cast_to_int(&s1)?.cast(s0.dtype()).unwrap();
        Ok(SpicyObj::Series(
            s0.remainder(&s1)
                .map_err(|e| SpicyError::Err(e.to_string()))?,
        ))
    }
}

pub fn rolling_max(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_max(options)));
    }

    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_max(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_mean(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_mean(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_mean(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_median(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_mean(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        fn_params: Some(RollingFnParams::Quantile(RollingQuantileParams {
            prob: 0.5,
            method: polars::prelude::QuantileMethod::Midpoint,
        })),
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_quantile(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_min(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_min(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_min(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_skew(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let windows_size = args[0].to_i64()?;
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    if args[1].is_expr() {
        if windows_size < 0 {
            return Err(SpicyError::Err(format!(
                "Requires a positive window size, got {}",
                windows_size
            )));
        }
        let s1 = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(s1.rolling_skew(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    if windows_size < 0 {
        return Err(SpicyError::Err(format!(
            "Requires a positive window size, got {}",
            args[0]
        )));
    }
    let s1 = args[1].series().unwrap();
    Ok(SpicyObj::Series(
        polars_ops::series::rolling_skew(s1, options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_std0(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 0u8 })),
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_std(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_std(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_std1(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 1u8 })),
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_std(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 1u8 })),
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_std(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_sum(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_sum(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_sum(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_var0(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 0u8 })),
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_var(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_var(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn rolling_var1(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[1].is_expr() {
        let windows_size = args[0].to_i64()?;
        let s1 = args[1].as_expr()?;
        let options = RollingOptionsFixedWindow {
            window_size: windows_size as usize,
            fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 1u8 })),
            ..Default::default()
        };
        return Ok(SpicyObj::Expr(s1.rolling_var(options)));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let windows_size = args[0].to_i64().unwrap();
    let s1 = args[1].series().unwrap();
    let options = RollingOptionsFixedWindow {
        window_size: windows_size as usize,
        fn_params: Some(RollingFnParams::Var(RollingVarParams { ddof: 1u8 })),
        ..Default::default()
    };
    Ok(SpicyObj::Series(
        s1.rolling_var(options)
            .map_err(|e| SpicyError::Err(e.to_string()))?,
    ))
}

pub fn pow(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        let left = arg0.as_expr()?;
        let right = arg1.as_expr()?;
        return Ok(SpicyObj::Expr(left.pow(right)));
    }
    validate_args(args, &[ArgType::NumericLike, ArgType::NumericLike])?;

    if arg0.is_atom() && arg1.is_atom() {
        if arg0.is_float() || arg1.is_float() {
            Ok(SpicyObj::F64(
                arg0.to_f64().unwrap().powf(arg1.to_f64().unwrap()),
            ))
        } else {
            Ok(SpicyObj::I64(
                arg0.to_i64().unwrap().pow(arg1.to_i64().unwrap() as u32),
            ))
        }
    } else {
        let s0 = arg0.as_series().unwrap();
        let s0_name = s0.name();
        let df = s0.clone().into_frame();
        let res = df
            .lazy()
            .select([col(s0_name.clone()).pow(arg1.as_expr().unwrap())])
            .collect()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        Ok(SpicyObj::Series(
            res.columns()
                .first()
                .unwrap()
                .clone()
                .take_materialized_series(),
        ))
    }
}

pub fn quantile(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() || args[1].is_expr() {
        let left = args[0].as_expr()?;
        let right = args[1].as_expr()?;
        return Ok(SpicyObj::Expr(
            left.quantile(right, QuantileMethod::Midpoint),
        ));
    }
    validate_args(args, &[ArgType::NumericLike, ArgType::Float])?;
    let arg0 = args[0];
    let quantile = args[1].to_f64().unwrap();
    let s0 = arg0.as_series().unwrap();
    let res = s0
        .quantile_reduce(quantile, QuantileMethod::Midpoint)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::from_any_value(res.as_any_value()))
}

pub fn round(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    if args[0].is_expr() {
        let left = args[0].as_expr()?;
        let decimals = args[1].to_i64()?;
        return Ok(SpicyObj::Expr(
            left.round(decimals as u32, RoundMode::default()),
        ));
    }
    validate_args(args, &[ArgType::Int, ArgType::NumericLike])?;
    let arg0 = args[0];
    let arg1 = args[1];
    let decimals = arg0.to_i64().unwrap();
    if decimals == 0 {
        return Ok(arg1.clone());
    }

    if arg1.is_atom() {
        match arg1.f32() {
            Ok(f1) => {
                let multiplier = 10.0f32.powf(decimals as f32);
                Ok(SpicyObj::F32((f1 * multiplier).round() / multiplier))
            }
            Err(_) => {
                let multiplier = 10.0f64.powf(decimals as f64);
                Ok(SpicyObj::F64(
                    (arg1.to_f64().unwrap() * multiplier).round() / multiplier,
                ))
            }
        }
    } else {
        let s1 = arg1.series().unwrap();
        Ok(SpicyObj::Series(
            s1.round(decimals as u32, RoundMode::default())
                .map_err(|e| SpicyError::Err(e.to_string()))?,
        ))
    }
}
