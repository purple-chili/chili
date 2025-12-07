use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};

use polars::prelude::{Categories, UInt64Chunked, lit};
use polars::{datatypes::DataType, series::Series};
use polars_ops::chunked_array::StringNameSpaceImpl;
use regex::Regex;
use std::str::FromStr;

// str_like, pat, val
pub fn replace(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let s = arg0.as_expr()?;
        let pat = args[1].as_expr()?;
        let val = args[2].as_expr()?;
        return Ok(SpicyObj::Expr(s.str().replace(pat, val, false)));
    }
    validate_args(args, &[ArgType::StrLike, ArgType::Str, ArgType::Str])?;

    let pat = args[1].str().unwrap();
    let val = args[2].str().unwrap();
    let re = Regex::from_str(pat).map_err(|e| SpicyError::Err(e.to_string()))?;

    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(re.replace_all(s, val).to_string())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(re.replace_all(s, val).to_string())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(
                s.str()
                    .unwrap()
                    .replace(pat, val)
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into(),
            )),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s
                    .str()
                    .unwrap()
                    .replace(pat, val)
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

// str_like
pub fn lowercase(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.str().to_lowercase()));
    }
    validate_args(args, &[ArgType::StrLike])?;
    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(s.to_lowercase())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(s.to_lowercase())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(s.str().unwrap().to_lowercase().into())),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s.str().unwrap().to_lowercase().into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

// str_like
pub fn uppercase(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];

    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(left.str().to_uppercase()));
    }
    validate_args(args, &[ArgType::StrLike])?;

    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(s.to_uppercase())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(s.to_uppercase())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(s.str().unwrap().to_uppercase().into())),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s.str().unwrap().to_uppercase().into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

// str_like
pub fn trim_end(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        return Ok(SpicyObj::Expr(
            arg0.as_expr()?
                .str()
                .strip_chars_end(polars::prelude::lit("")),
        ));
    }
    validate_args(args, &[ArgType::StrLike])?;
    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(s.trim_end().to_owned())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(s.trim_end().to_owned())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(
                s.str()
                    .unwrap()
                    .strip_chars_end(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into(),
            )),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s
                    .str()
                    .unwrap()
                    .strip_chars_end(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

pub fn trim_start(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(
            left.str().strip_chars_start(polars::prelude::lit("")),
        ));
    }
    validate_args(args, &[ArgType::StrLike])?;
    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(s.trim_start().to_owned())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(s.trim_start().to_owned())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(
                s.str()
                    .unwrap()
                    .strip_chars_start(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into(),
            )),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s
                    .str()
                    .unwrap()
                    .strip_chars_start(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

pub fn trim(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];

    if arg0.is_expr() {
        let left = arg0.as_expr()?;
        return Ok(SpicyObj::Expr(
            left.str().strip_chars(polars::prelude::lit("")),
        ));
    }
    validate_args(args, &[ArgType::StrLike])?;

    match arg0 {
        SpicyObj::String(s) => Ok(SpicyObj::String(s.trim().to_owned())),
        SpicyObj::Symbol(s) => Ok(SpicyObj::Symbol(s.trim().to_owned())),
        SpicyObj::Series(s) => match s.dtype() {
            DataType::String => Ok(SpicyObj::Series(
                s.str()
                    .unwrap()
                    .strip_chars(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into(),
            )),
            DataType::Categorical(_, _) => {
                let s = s.cast(&DataType::String).unwrap();
                let s: Series = s
                    .str()
                    .unwrap()
                    .strip_chars(&Series::new_null("".into(), 0).into())
                    .map_err(|e| SpicyError::Err(e.to_string()))?
                    .into();
                Ok(SpicyObj::Series(
                    s.cast(&DataType::Categorical(
                        Categories::global(),
                        Categories::global().mapping(),
                    ))
                    .unwrap(),
                ))
            }
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}

pub fn pad(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Int, ArgType::Any])?;
    let arg0 = args[0];
    let arg1 = args[1];
    let length = arg0.to_i64().unwrap();
    if arg1.is_expr() {
        let s = arg1.as_expr()?;
        if length > 0 {
            return Ok(SpicyObj::Expr(s.str().pad_start(lit(length), ' ')));
        } else {
            return Ok(SpicyObj::Expr(s.str().pad_end(lit(-length), ' ')));
        }
    }

    validate_args(args, &[ArgType::Int, ArgType::StrOrStrs])?;
    match arg1 {
        SpicyObj::String(s) => Ok(SpicyObj::String(if length > 0 {
            format!("{:<length$}", s, length = length as usize)
        } else {
            format!("{:>length$}", s, length = length.unsigned_abs() as usize)
        })),
        SpicyObj::Series(s) => {
            let pad_length = UInt64Chunked::from_vec("".into(), vec![length.unsigned_abs()]);
            if length > 0 {
                Ok(SpicyObj::Series(
                    s.str().unwrap().pad_end(&pad_length, ' ').into(),
                ))
            } else {
                Ok(SpicyObj::Series(
                    s.str().unwrap().pad_start(&pad_length, ' ').into(),
                ))
            }
        }
        _ => unreachable!(),
    }
}
