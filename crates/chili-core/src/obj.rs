use crate::constant::{NS_IN_DAY, NS_IN_MS, UNIX_EPOCH_DAY};
use crate::errors::SpicyResult;
use crate::par_df::PartitionedDataFrame;
use crate::{errors::SpicyError, func::Func};
use chrono::{DateTime, Datelike, NaiveDate};
use indexmap::IndexMap;
use ndarray::ArcArray2;
use polars::datatypes::{AnyValue, DataType, PolarsNumericType, TimeUnit};
use polars::lazy::dsl::{Expr, lit};
use polars::prelude::{
    Categories, Column, DataType as PolarsDataType, IntoSeries, LargeListArray, LazyFrame,
    LiteralValue, NamedFrom, NewChunkedArray, Scalar,
};
use polars::{chunked_array::ChunkedArray, frame::DataFrame, series::Series};
use polars_arrow::array::{FixedSizeListArray, ValueSize};
use rayon::iter::ParallelIterator;
use std::fmt;
use std::fmt::Debug;
use std::{fmt::Display, str::FromStr};

#[derive(Clone)]
pub enum SpicyObj {
    Boolean(bool),  // -1
    U8(u8),         // -2
    I16(i16),       // -3
    I32(i32),       // -4
    I64(i64),       // -5
    Date(i32),      // -6 start from 1970.01.01
    Time(i64),      // -7 00:00:00.0 - 23:59:59.999999999
    Datetime(i64),  // -8 start from 1970.01.01T00:00:00.0
    Timestamp(i64), // -9 start from 1970.01.01D00:00:00.0
    Duration(i64),  // -10
    F32(f32),       // -11
    F64(f64),       // -12
    String(String), // -13
    Symbol(String), // -14
    Expr(Expr),     // -15

    Null, // 0

    Series(Series), // 1-14 -> Arrow IPC

    Matrix(ArcArray2<f64>), // 21

    MixedList(Vec<SpicyObj>),         // 90
    Dict(IndexMap<String, SpicyObj>), // 91 -> skip Dataframe
    DataFrame(DataFrame),             // 92 -> Arrow IPC
    LazyFrame(LazyFrame),

    Fn(Func), // -102 => string

    Err(String), // 128 => string
    Return(Box<SpicyObj>),
    DelayedArg, // projection null - internal use
    ParDataFrame(PartitionedDataFrame),
}

impl PartialEq for SpicyObj {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SpicyObj::Boolean(a), SpicyObj::Boolean(b)) => a == b,
            (SpicyObj::U8(a), SpicyObj::U8(b)) => a == b,
            (SpicyObj::I16(a), SpicyObj::I16(b)) => a == b,
            (SpicyObj::I32(a), SpicyObj::I32(b)) => a == b,
            (SpicyObj::I64(a), SpicyObj::I64(b)) => a == b,
            (SpicyObj::Date(a), SpicyObj::Date(b)) => a == b,
            (SpicyObj::Time(a), SpicyObj::Time(b)) => a == b,
            (SpicyObj::Datetime(a), SpicyObj::Datetime(b)) => a == b,
            (SpicyObj::Timestamp(a), SpicyObj::Timestamp(b)) => a == b,
            (SpicyObj::Duration(a), SpicyObj::Duration(b)) => a == b,
            (SpicyObj::F32(a), SpicyObj::F32(b)) => a == b,
            (SpicyObj::F64(a), SpicyObj::F64(b)) => a == b,
            (SpicyObj::String(a), SpicyObj::String(b)) => a == b,
            (SpicyObj::Symbol(a), SpicyObj::Symbol(b)) => a == b,
            (SpicyObj::Expr(a), SpicyObj::Expr(b)) => a == b,
            (SpicyObj::Null, SpicyObj::Null) => true,
            (SpicyObj::Series(a), SpicyObj::Series(b)) => a.eq(b),
            (SpicyObj::Matrix(a), SpicyObj::Matrix(b)) => a == b,
            (SpicyObj::MixedList(a), SpicyObj::MixedList(b)) => a == b,
            (SpicyObj::Dict(a), SpicyObj::Dict(b)) => a == b,
            (SpicyObj::DataFrame(a), SpicyObj::DataFrame(b)) => a == b,
            (SpicyObj::Fn(a), SpicyObj::Fn(b)) => a == b,
            (SpicyObj::Err(a), SpicyObj::Err(b)) => a == b,
            (SpicyObj::Return(a), SpicyObj::Return(b)) => a == b,
            (SpicyObj::DelayedArg, SpicyObj::DelayedArg) => true,
            (SpicyObj::ParDataFrame(a), SpicyObj::ParDataFrame(b)) => a == b,
            _ => false,
        }
    }
}

impl Debug for SpicyObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpicyObj::LazyFrame(_) => write!(f, "LazyFrame(...)"),
            _ => write!(f, "{:?}", self),
        }
    }
}

impl SpicyObj {
    pub fn q6_len(&self) -> Result<usize, SpicyError> {
        // k type + value
        match self {
            SpicyObj::Boolean(_) => Ok(2),
            SpicyObj::U8(_) => Ok(2),
            SpicyObj::I16(_) => Ok(3),
            SpicyObj::I32(_) => Ok(5),
            SpicyObj::I64(_) => Ok(9),
            SpicyObj::Date(_) => Ok(5),
            SpicyObj::F32(_) => Ok(5),
            SpicyObj::F64(_) => Ok(9),
            SpicyObj::Symbol(k) => Ok(k.len() + 2),
            SpicyObj::String(k) => Ok(k.len() + 6),
            SpicyObj::Datetime(_) => Ok(9),
            SpicyObj::Timestamp(_) => Ok(9),
            SpicyObj::Time(_) => Ok(5),
            SpicyObj::Duration(_) => Ok(9),
            SpicyObj::MixedList(l) => {
                let lens = l
                    .iter()
                    .map(|k| k.q6_len())
                    .collect::<Result<Vec<_>, SpicyError>>();
                Ok(lens?.into_iter().sum::<usize>() + 6)
            }
            SpicyObj::Series(series) => get_series_len(series),
            SpicyObj::DataFrame(df) => {
                // 98 0 99 + symbol list(6) + values(6)
                let mut length: usize = 15;
                for column in df.get_columns().iter() {
                    length += column.name().len() + 1;
                    length += get_series_len(column.as_materialized_series())?
                }
                Ok(length)
            }
            SpicyObj::Null => Ok(2),
            SpicyObj::Dict(dict) => {
                let mut length = 13;
                for (k, v) in dict.iter() {
                    length += k.len() + 1;
                    length += v.q6_len()?;
                }
                Ok(length)
            }
            SpicyObj::Err(e) => Ok(e.to_string().len() + 2),
            _ => Err(SpicyError::NotAbleToSerializeErr(self.get_type_name())),
        }
    }

    pub fn is_atom(&self) -> bool {
        matches!(
            self,
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
        )
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, SpicyObj::Boolean(_))
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            SpicyObj::U8(_) | SpicyObj::I16(_) | SpicyObj::I32(_) | SpicyObj::I64(_)
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, SpicyObj::F32(_) | SpicyObj::F64(_))
    }

    pub fn is_float_like(&self) -> bool {
        match self {
            SpicyObj::F32(_) | SpicyObj::F64(_) => true,
            SpicyObj::Series(s) => matches!(s.dtype(), DataType::Float32 | DataType::Float64),
            _ => false,
        }
    }

    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            SpicyObj::Date(_)
                | SpicyObj::Timestamp(_)
                | SpicyObj::Datetime(_)
                | SpicyObj::Duration(_)
                | SpicyObj::Time(_)
        )
    }

    pub fn is_sym(&self) -> bool {
        matches!(self, SpicyObj::Symbol(_))
    }

    pub fn is_sym_or_syms(&self) -> bool {
        match self {
            SpicyObj::Symbol(_) => true,
            SpicyObj::Series(s) => matches!(s.dtype(), DataType::Categorical(_, _)),
            _ => false,
        }
    }

    pub fn is_str_like(&self) -> bool {
        match self {
            SpicyObj::Symbol(_) | SpicyObj::String(_) => true,
            SpicyObj::Series(s) => {
                matches!(s.dtype(), DataType::String | DataType::Categorical(_, _))
            }
            _ => false,
        }
    }

    pub fn is_str(&self) -> bool {
        matches!(self, SpicyObj::String(_))
    }

    pub fn is_str_or_strs(&self) -> bool {
        match self {
            SpicyObj::String(_) => true,
            SpicyObj::Series(s) => matches!(s.dtype(), DataType::String),
            _ => false,
        }
    }

    pub fn is_syms(&self) -> bool {
        matches!(self, SpicyObj::Series(s) if matches!(s.dtype(), DataType::Categorical(_, _)))
    }

    pub fn is_return(&self) -> bool {
        matches!(self, SpicyObj::Return(_))
    }

    pub fn is_collection(&self) -> bool {
        matches!(
            self,
            SpicyObj::MixedList(_)
                | SpicyObj::Series(_)
                | SpicyObj::DataFrame(_)
                | SpicyObj::Dict(_)
                | SpicyObj::Matrix(_)
        )
    }

    pub fn is_mixed_collection(&self) -> bool {
        matches!(self, SpicyObj::MixedList(_) | SpicyObj::Dict(_))
    }

    pub fn is_mixed_list(&self) -> bool {
        matches!(self, SpicyObj::MixedList(_))
    }

    pub fn is_matrix(&self) -> bool {
        matches!(self, SpicyObj::Matrix(_))
    }

    pub fn is_df(&self) -> bool {
        matches!(self, SpicyObj::DataFrame(_))
    }

    pub fn is_lf(&self) -> bool {
        matches!(self, SpicyObj::LazyFrame(_))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            SpicyObj::U8(_)
                | SpicyObj::I16(_)
                | SpicyObj::I32(_)
                | SpicyObj::I64(_)
                | SpicyObj::F32(_)
                | SpicyObj::F64(_)
        )
    }

    pub fn is_numeric_like(&self) -> bool {
        match self {
            SpicyObj::U8(_)
            | SpicyObj::I16(_)
            | SpicyObj::I32(_)
            | SpicyObj::I64(_)
            | SpicyObj::F32(_)
            | SpicyObj::F64(_) => true,
            SpicyObj::Series(s) => matches!(
                s.dtype(),
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Float32
                    | DataType::Float64
            ),
            _ => false,
        }
    }

    pub fn is_datetime(&self) -> bool {
        matches!(self, SpicyObj::Datetime(_))
    }

    pub fn is_timestamp(&self) -> bool {
        matches!(self, SpicyObj::Timestamp(_))
    }

    pub fn is_duration(&self) -> bool {
        matches!(self, SpicyObj::Duration(_))
    }

    pub fn is_truthy(&self) -> SpicyResult<bool> {
        match self {
            SpicyObj::Boolean(_)
            | SpicyObj::U8(_)
            | SpicyObj::I16(_)
            | SpicyObj::I32(_)
            | SpicyObj::I64(_)
            | SpicyObj::Date(_)
            | SpicyObj::Time(_)
            | SpicyObj::Datetime(_)
            | SpicyObj::Timestamp(_)
            | SpicyObj::Duration(_) => {
                if self.to_i64().unwrap() == 0 {
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            SpicyObj::F32(_) | SpicyObj::F64(_) => {
                if self.to_f64().unwrap() == 0.0 {
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            _ => Err(SpicyError::MismatchedTypeErr(
                "bool".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn is_fn(&self) -> bool {
        matches!(self, SpicyObj::Fn(_))
    }

    pub fn size(&self) -> usize {
        match self {
            SpicyObj::Series(s) => s.len(),
            SpicyObj::DataFrame(df) => df.height(),
            SpicyObj::MixedList(m) => m.len(),
            SpicyObj::Dict(d) => d.len(),
            SpicyObj::Matrix(m) => m.nrows(),
            _ => 1,
        }
    }

    pub fn str(&self) -> SpicyResult<&str> {
        match self {
            SpicyObj::String(s) | SpicyObj::Symbol(s) => Ok(s.as_str()),
            _ => Err(SpicyError::MismatchedTypeErr(
                "str".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_bool(&self) -> SpicyResult<bool> {
        match self {
            SpicyObj::Boolean(v) => Ok(*v),
            _ => Ok(self.to_i64()? != 0),
        }
    }

    pub fn to_i64(&self) -> Result<i64, SpicyError> {
        match self {
            SpicyObj::Boolean(v) => Ok(*v as i64),
            SpicyObj::U8(v) => Ok(*v as i64),
            SpicyObj::I16(v) => Ok(*v as i64),
            SpicyObj::I32(v) => Ok(*v as i64),
            SpicyObj::I64(v) => Ok(*v),
            SpicyObj::Date(v) => Ok(*v as i64),
            SpicyObj::Time(v) => Ok(*v),
            SpicyObj::Datetime(v) => Ok(*v),
            SpicyObj::Timestamp(v) => Ok(*v),
            SpicyObj::Duration(v) => Ok(*v),
            _ => Err(SpicyError::MismatchedTypeErr(
                "i64".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_handles(&self) -> Result<Vec<i64>, SpicyError> {
        match self {
            SpicyObj::I64(v) => Ok(vec![*v]),
            SpicyObj::Series(s) if s.dtype().eq(&DataType::Int64) => Ok(s
                .i64()
                .unwrap()
                .into_iter()
                .map(|i| i.unwrap_or(0))
                .collect()),
            _ => Err(SpicyError::MismatchedTypeErr(
                "requires i64 or series of i64 for handles".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_par_num(&self) -> Result<i32, SpicyError> {
        match self {
            SpicyObj::I32(v) => Ok(*v),
            SpicyObj::I64(v) => Ok(*v as i32),
            SpicyObj::Date(v) => Ok(*v),
            _ => Err(SpicyError::Err(format!(
                "'{}' is not valid partition",
                self
            ))),
        }
    }

    pub fn to_par_nums(&self) -> Result<Vec<i32>, SpicyError> {
        match self {
            SpicyObj::I32(v) => Ok(vec![(*v)]),
            SpicyObj::I64(v) => Ok(vec![*v as i32]),
            SpicyObj::Date(v) => Ok(vec![(*v)]),
            SpicyObj::Series(s) => match s.dtype() {
                DataType::Int32 => Ok(s
                    .i32()
                    .unwrap()
                    .into_iter()
                    .map(|i| i.unwrap_or(i32::MIN))
                    .collect()),
                DataType::Int64 => Ok(s
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| i.unwrap_or(i32::MIN as i64) as i32)
                    .collect()),
                DataType::Date => Ok(s
                    .to_physical_repr()
                    .i32()
                    .unwrap()
                    .into_iter()
                    .map(|i| i.unwrap_or(i32::MIN))
                    .collect()),
                _ => Err(SpicyError::Err(format!("{} is not valid partition", self))),
            },
            _ => Err(SpicyError::Err(format!("{} is not valid partition", self))),
        }
    }

    pub fn to_f64(&self) -> Result<f64, SpicyError> {
        match self {
            SpicyObj::Boolean(v) => Ok(*v as u8 as f64),
            SpicyObj::U8(v) => Ok(*v as f64),
            SpicyObj::I16(v) => Ok(*v as f64),
            SpicyObj::I32(v) => Ok(*v as f64),
            SpicyObj::I64(v) => Ok(*v as f64),
            SpicyObj::F32(v) => Ok(*v as f64),
            SpicyObj::F64(v) => Ok(*v),
            SpicyObj::Date(v) => Ok(*v as f64),
            SpicyObj::Timestamp(v) => Ok(*v as f64),
            SpicyObj::Datetime(v) => Ok(*v as f64),
            SpicyObj::Time(v) => Ok(*v as f64),
            SpicyObj::Duration(v) => Ok(*v as f64),
            _ => Err(SpicyError::MismatchedTypeErr(
                "f64".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_f32(&self) -> Result<f32, SpicyError> {
        match self {
            SpicyObj::Boolean(v) => Ok(*v as u8 as f32),
            SpicyObj::U8(v) => Ok(*v as f32),
            SpicyObj::I16(v) => Ok(*v as f32),
            SpicyObj::I32(v) => Ok(*v as f32),
            SpicyObj::I64(v) => Ok(*v as f32),
            SpicyObj::F32(v) => Ok(*v),
            SpicyObj::Date(v) => Ok(*v as f32),
            SpicyObj::Timestamp(v) => Ok(*v as f32),
            SpicyObj::Datetime(v) => Ok(*v as f32),
            SpicyObj::Time(v) => Ok(*v as f32),
            SpicyObj::Duration(v) => Ok(*v as f32),
            _ => Err(SpicyError::MismatchedTypeErr(
                "f32".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_str_vec(&self) -> SpicyResult<Vec<&str>> {
        match self {
            SpicyObj::String(s) => Ok(vec![s.as_str()]),
            SpicyObj::Symbol(s) => Ok(vec![s]),
            SpicyObj::Series(s) => match s.dtype() {
                DataType::String => Ok(s
                    .str()
                    .unwrap()
                    .into_iter()
                    .map(|s| s.unwrap_or(""))
                    .collect()),
                DataType::Categorical(_, _) => Ok(s
                    .cat32()
                    .unwrap()
                    .iter_str()
                    .map(|s| s.unwrap_or(""))
                    .collect()),
                _ => Err(SpicyError::MismatchedTypeErr(
                    "sym | syms".to_owned(),
                    self.get_type_name(),
                )),
            },
            SpicyObj::MixedList(l) if l.is_empty() => Ok(vec![]),
            _ => Err(SpicyError::MismatchedTypeErr(
                "sym | syms".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn into_series(&self) -> SpicyResult<Series> {
        match self {
            SpicyObj::Boolean(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::U8(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::I16(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::I32(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::I64(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::F32(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::F64(s) => Ok(Series::new("".into(), vec![*s])),
            SpicyObj::Date(s) => Ok(Series::new("".into(), vec![*s])
                .cast(&DataType::Date)
                .unwrap()),
            SpicyObj::Timestamp(s) => Ok(Series::new("".into(), vec![*s])
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap()),
            SpicyObj::Datetime(s) => Ok(Series::new("".into(), vec![*s])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()),
            SpicyObj::Time(s) => Ok(Series::new("".into(), vec![*s])
                .cast(&DataType::Time)
                .unwrap()),
            SpicyObj::Duration(s) => Ok(Series::new("".into(), vec![*s])
                .cast(&DataType::Duration(TimeUnit::Nanoseconds))
                .unwrap()),
            SpicyObj::Symbol(s) => Ok(Series::new("".into(), vec![s.to_owned()])
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap()),
            SpicyObj::String(s) => Ok(Series::new("".into(), vec![s.to_owned()])),
            SpicyObj::Null => Ok(Series::new_null("".into(), 1)),
            _ => Err(SpicyError::MismatchedTypeErr(
                "series".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn as_series(&self) -> Result<Series, SpicyError> {
        match self.into_series() {
            Ok(s) => Ok(s),
            Err(_) => match self.series() {
                Ok(s) => Ok(s.clone()),
                Err(_) => Err(SpicyError::MismatchedTypeErr(
                    "series".to_owned(),
                    self.get_type_name(),
                )),
            },
        }
    }

    pub fn as_vec(&self) -> SpicyResult<Vec<SpicyObj>> {
        let vec = match self {
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
            | SpicyObj::Symbol(_)
            | SpicyObj::String(_)
            | SpicyObj::Null
            | SpicyObj::Expr(_)
            | SpicyObj::Fn(_) => vec![self.clone()],
            SpicyObj::MixedList(l) => l.clone(),
            SpicyObj::Series(s) => match s.dtype() {
                DataType::Boolean => s
                    .bool()
                    .unwrap()
                    .into_iter()
                    .map(|b| SpicyObj::Boolean(b.unwrap_or(false)))
                    .collect(),
                DataType::UInt8 => s
                    .u8()
                    .unwrap()
                    .into_iter()
                    .map(|i| SpicyObj::U8(i.unwrap_or(0)))
                    .collect(),
                DataType::Int16 => s
                    .i16()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::I16(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Int32 => s
                    .i32()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::I32(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Int64 => s
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::I64(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Float32 => s
                    .f32()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::F32(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Float64 => s
                    .f64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::F64(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::String => s
                    .str()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::String(i.to_owned())
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Date => s
                    .to_physical_repr()
                    .i32()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Date(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Datetime(TimeUnit::Milliseconds, _) => s
                    .to_physical_repr()
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Datetime(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Datetime(TimeUnit::Nanoseconds, _) => s
                    .to_physical_repr()
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Timestamp(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Duration(TimeUnit::Nanoseconds) => s
                    .to_physical_repr()
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Duration(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::Time => s
                    .to_physical_repr()
                    .i64()
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Time(i)
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                DataType::List(_) => s
                    .list()
                    .unwrap()
                    .into_iter()
                    .map(|s| {
                        SpicyObj::Series(s.unwrap_or(Series::new_empty("".into(), &DataType::Null)))
                    })
                    .collect(),
                DataType::Categorical(_, _) => s
                    .cat32()
                    .unwrap()
                    .iter_str()
                    .map(|i| {
                        if let Some(i) = i {
                            SpicyObj::Symbol(i.to_owned())
                        } else {
                            SpicyObj::Null
                        }
                    })
                    .collect(),
                _ => {
                    return Err(SpicyError::EvalErr(format!(
                        "Not support series type '{}' as J list",
                        s.dtype(),
                    )));
                }
            },
            _ => {
                return Err(SpicyError::MismatchedTypeErr(
                    "mixed list".to_owned(),
                    self.get_type_name(),
                ));
            }
        };
        Ok(vec)
    }

    pub fn new_same_int_atom(&self, i: i64) -> SpicyResult<SpicyObj> {
        match self {
            SpicyObj::Boolean(_) => Ok(SpicyObj::Boolean(i != 0)),
            SpicyObj::U8(_) => Ok(SpicyObj::U8(i as u8)),
            SpicyObj::I16(_) => Ok(SpicyObj::I16(i as i16)),
            SpicyObj::I32(_) => Ok(SpicyObj::I32(i as i32)),
            SpicyObj::I64(_) => Ok(SpicyObj::I64(i)),
            SpicyObj::Date(_) => Ok(SpicyObj::Date(i as i32)),
            SpicyObj::Time(_) => Ok(if i >= NS_IN_DAY {
                SpicyObj::Time(NS_IN_DAY - 1)
            } else if i < 0 {
                SpicyObj::Time(0)
            } else {
                SpicyObj::Time(i)
            }),
            SpicyObj::Datetime(_) => Ok(SpicyObj::Datetime(i)),
            SpicyObj::Timestamp(_) => Ok(SpicyObj::Timestamp(i)),
            SpicyObj::Duration(_) => Ok(SpicyObj::Duration(i)),
            _ => Err(SpicyError::Err(format!(
                "Unsupported creating atom int type for '{}'",
                self.get_type_name()
            ))),
        }
    }

    pub fn get_series_data_type(&self) -> DataType {
        match self {
            SpicyObj::Boolean(_) => DataType::Boolean,
            SpicyObj::U8(_) => DataType::UInt8,
            SpicyObj::I16(_) => DataType::Int16,
            SpicyObj::I32(_) => DataType::Int32,
            SpicyObj::I64(_) => DataType::Int64,
            SpicyObj::Date(_) => DataType::Date,
            SpicyObj::Time(_) => DataType::Time,
            SpicyObj::Datetime(_) => DataType::Datetime(TimeUnit::Milliseconds, None),
            SpicyObj::Timestamp(_) => DataType::Datetime(TimeUnit::Nanoseconds, None),
            SpicyObj::Duration(_) => DataType::Duration(TimeUnit::Nanoseconds),
            SpicyObj::F32(_) => DataType::Float32,
            SpicyObj::F64(_) => DataType::Float64,
            SpicyObj::String(_) => DataType::String,
            SpicyObj::Symbol(_) => {
                DataType::Categorical(Categories::global(), Categories::global().mapping())
            }
            _ => DataType::Null,
        }
    }

    pub fn get_type_code(&self) -> i16 {
        match self {
            SpicyObj::Series(s) => match s.dtype() {
                DataType::Boolean => 1,
                DataType::UInt8 => 2,
                DataType::Int16 => 3,
                DataType::Int32 => 4,
                DataType::Int64 => 5,
                DataType::Date => 6,
                DataType::Time => 7,
                DataType::Datetime(TimeUnit::Milliseconds, _) => 8,
                DataType::Datetime(TimeUnit::Nanoseconds, _) => 9,
                DataType::Duration(TimeUnit::Nanoseconds) => 10,
                DataType::Float32 => 11,
                DataType::Float64 => 12,
                DataType::String => 13,
                DataType::Categorical(_, _) => 14,
                DataType::Int8 => 15,
                DataType::UInt16 => 16,
                DataType::UInt32 => 17,
                DataType::UInt64 => 18,
                _ => 93,
            },
            // atom
            SpicyObj::Boolean(_) => -1,
            SpicyObj::U8(_) => -2,
            SpicyObj::I16(_) => -3,
            SpicyObj::I32(_) => -4,
            SpicyObj::I64(_) => -5,
            SpicyObj::Date(_) => -6,
            SpicyObj::Time(_) => -7,
            SpicyObj::Datetime(_) => -8,
            SpicyObj::Timestamp(_) => -9,
            SpicyObj::Duration(_) => -10,
            SpicyObj::F32(_) => -11,
            SpicyObj::F64(_) => -12,
            SpicyObj::String(_) => -13,
            SpicyObj::Symbol(_) => -14,
            // other
            SpicyObj::MixedList(_) => 90,
            SpicyObj::Dict(_) => 91,
            SpicyObj::DataFrame(_) => 92,
            SpicyObj::Matrix(_) => 94,
            SpicyObj::Null => 0,
            SpicyObj::Fn(_) => -102,
            SpicyObj::Err(_) => 128,
            // 0xFF -> 255 as sequence messages
            _ => 100,
        }
    }

    pub fn unify_series(&self) -> SpicyResult<SpicyObj> {
        if let SpicyObj::MixedList(l) = self {
            let codes: Vec<i16> = l
                .iter()
                .filter(|args| !args.is_null())
                .map(|args| args.get_type_code())
                .collect();
            if !codes.is_empty() {
                let min_code = codes.iter().min().unwrap();
                if *min_code < 0 && codes.iter().all(|c| c == min_code) {
                    if *min_code <= -13 {
                        let v: Vec<Option<String>> = l
                            .iter()
                            .map(|args| {
                                if args.is_null() {
                                    None
                                } else {
                                    Some(args.str().unwrap().to_owned())
                                }
                            })
                            .collect();
                        let s = Series::new("".into(), v);
                        if *min_code == -14 {
                            return Ok(SpicyObj::Series(
                                s.cast(&DataType::Categorical(
                                    Categories::global(),
                                    Categories::global().mapping(),
                                ))
                                .unwrap(),
                            ));
                        } else {
                            return Ok(SpicyObj::Series(s));
                        }
                    } else if *min_code <= -11 {
                        let v: Vec<Option<f64>> = l
                            .iter()
                            .map(|args| {
                                if args.is_null() {
                                    None
                                } else {
                                    Some(args.to_f64().unwrap())
                                }
                            })
                            .collect();
                        let s = Series::new("".into(), v);
                        if *min_code == 21 {
                            return Ok(SpicyObj::Series(s.cast(&DataType::Float32).unwrap()));
                        } else {
                            return Ok(SpicyObj::Series(s));
                        }
                    } else {
                        let v: Vec<Option<i64>> = l
                            .iter()
                            .map(|args| {
                                if args.is_null() {
                                    None
                                } else {
                                    Some(args.to_i64().unwrap())
                                }
                            })
                            .collect();
                        let s = Series::new("".into(), v);
                        let s = match *min_code {
                            -1 => s.cast(&DataType::Boolean).unwrap(),
                            -2 => s.cast(&DataType::UInt8).unwrap(),
                            -3 => s.cast(&DataType::Int16).unwrap(),
                            -4 => s.cast(&DataType::Int32).unwrap(),
                            -5 => s,
                            -6 => s.cast(&DataType::Date).unwrap(),
                            -7 => s.cast(&DataType::Time).unwrap(),
                            -8 => s
                                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                                .unwrap(),
                            -9 => s
                                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                                .unwrap(),
                            -10 => s.cast(&DataType::Duration(TimeUnit::Nanoseconds)).unwrap(),
                            _ => unreachable!(),
                        };
                        return Ok(SpicyObj::Series(s));
                    }
                }
            }
        }

        Err(SpicyError::EvalErr(format!(
            "Not a unified data type mixed list, '{}'",
            self,
        )))
    }

    pub fn is_delayed_arg(&self) -> bool {
        matches!(self, SpicyObj::DelayedArg)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, SpicyObj::Null)
    }

    pub fn is_series(&self) -> bool {
        matches!(self, SpicyObj::Series(_))
    }

    pub fn is_expr(&self) -> bool {
        matches!(self, SpicyObj::Expr(_))
    }

    pub fn is_dict(&self) -> bool {
        matches!(self, SpicyObj::Dict(_))
    }

    pub fn parse_numeric_series<T, U>(s: &str) -> Result<Self, T::Err>
    where
        T: FromStr,
        U: PolarsNumericType<Native = T>,
        ChunkedArray<U>: IntoSeries,
    {
        let iter = s.split_whitespace().map(|s| s.parse::<T>().ok());
        Ok(SpicyObj::Series(
            ChunkedArray::<U>::from_iter_options("".into(), iter).into_series(),
        ))
    }

    pub fn parse_numeric_series_f32(s: &str) -> Result<Self, SpicyError> {
        let iter = s.split_whitespace().map(|s| {
            if s == "-0w" {
                Some(f32::NEG_INFINITY)
            } else if s == "0w" {
                Some(f32::INFINITY)
            } else {
                s.parse::<f32>().ok()
            }
        });
        Ok(SpicyObj::Series(Series::new(
            "".into(),
            iter.collect::<Vec<_>>(),
        )))
    }

    pub fn parse_numeric_series_f64(s: &str) -> Result<Self, SpicyError> {
        let iter = s.split_whitespace().map(|s| {
            if s == "-0w" {
                Some(f64::NEG_INFINITY)
            } else if s == "0w" {
                Some(f64::INFINITY)
            } else {
                s.parse::<f64>().ok()
            }
        });
        Ok(SpicyObj::Series(Series::new(
            "".into(),
            iter.collect::<Vec<_>>(),
        )))
    }

    pub fn as_expr(&self) -> Result<Expr, SpicyError> {
        match self {
            SpicyObj::Boolean(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Boolean,
                AnyValue::Boolean(*v),
            )))),
            SpicyObj::U8(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::UInt8,
                AnyValue::UInt8(*v),
            )))),
            SpicyObj::I16(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Int16,
                AnyValue::Int16(*v),
            )))),
            SpicyObj::I32(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Int32,
                AnyValue::Int32(*v),
            )))),
            SpicyObj::I64(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Int64,
                AnyValue::Int64(*v),
            )))),
            SpicyObj::F32(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Float32,
                AnyValue::Float32(*v),
            )))),
            SpicyObj::F64(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Float64,
                AnyValue::Float64(*v),
            )))),
            SpicyObj::Date(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Date,
                AnyValue::Date(*v),
            )))),
            SpicyObj::Timestamp(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Datetime(TimeUnit::Nanoseconds, None),
                AnyValue::Datetime(*v, TimeUnit::Nanoseconds, None),
            )))),
            SpicyObj::Datetime(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Datetime(TimeUnit::Milliseconds, None),
                AnyValue::Datetime(*v, TimeUnit::Milliseconds, None),
            )))),
            SpicyObj::Time(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Time,
                AnyValue::Time(*v),
            )))),
            SpicyObj::Duration(v) => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Duration(TimeUnit::Nanoseconds),
                AnyValue::Duration(*v, TimeUnit::Nanoseconds),
            )))),
            SpicyObj::Symbol(v) => Ok(lit(v.as_str())),
            SpicyObj::String(v) => Ok(lit(v.as_str())),
            SpicyObj::Expr(v) => Ok(v.clone()),
            SpicyObj::Series(v) => Ok(lit(v.clone())),
            SpicyObj::Null => Ok(lit(LiteralValue::Scalar(Scalar::new(
                DataType::Null,
                AnyValue::Null,
            )))),
            _ => Err(SpicyError::UnsupportedQueryJTypeErr(self.get_type_name())),
        }
    }

    pub fn as_exprs(&self) -> Result<Vec<Expr>, SpicyError> {
        match self {
            SpicyObj::MixedList(l) => Ok(l.iter().map(|args| args.as_expr().unwrap()).collect()),
            _ => Ok(vec![self.as_expr()?]),
        }
    }
    pub fn get_type_name(&self) -> String {
        match self {
            SpicyObj::Boolean(_) => "bool".to_owned(),
            SpicyObj::U8(_) => "u8".to_owned(),
            SpicyObj::I16(_) => "i16".to_owned(),
            SpicyObj::I32(_) => "i32".to_owned(),
            SpicyObj::I64(_) => "i64".to_owned(),
            SpicyObj::F32(_) => "f32".to_owned(),
            SpicyObj::F64(_) => "f64".to_owned(),
            SpicyObj::Date(_) => "date".to_owned(),
            SpicyObj::Timestamp(_) => "timestamp".to_owned(),
            SpicyObj::Datetime(_) => "datetime".to_owned(),
            SpicyObj::Time(_) => "time".to_owned(),
            SpicyObj::Duration(_) => "duration".to_owned(),
            SpicyObj::Symbol(_) => "sym".to_owned(),
            SpicyObj::String(_) => "str".to_owned(),
            SpicyObj::Expr(_) => "expr".to_owned(),

            SpicyObj::MixedList(_) => "list".to_owned(),
            SpicyObj::Series(s) => match s.dtype() {
                DataType::Boolean => "bools".to_owned(),
                DataType::UInt8 => "u8s".to_owned(),
                DataType::Int16 => "i16s".to_owned(),
                DataType::Int32 => "i32s".to_owned(),
                DataType::Int64 => "i64s".to_owned(),
                DataType::Float32 => "f32s".to_owned(),
                DataType::Float64 => "f64s".to_owned(),
                DataType::String => "strs".to_owned(),
                DataType::Date => "dates".to_owned(),
                DataType::Datetime(TimeUnit::Milliseconds, _) => "datetimes".to_owned(),
                DataType::Datetime(TimeUnit::Nanoseconds, _) => "timestamps".to_owned(),
                DataType::Duration(_) => "durations".to_owned(),
                DataType::Time => "times".to_owned(),
                DataType::Categorical(_, _) => "syms".to_owned(),
                DataType::Datetime(TimeUnit::Microseconds, _) => {
                    "datetimes(microseconds)".to_owned()
                }
                DataType::UInt16 => "u16s".to_owned(),
                DataType::UInt32 => "u32s".to_owned(),
                DataType::UInt64 => "u64s".to_owned(),
                DataType::Int8 => "i8s".to_owned(),
                DataType::Decimal(_, _) => "decimals".to_owned(),
                DataType::Binary => "binaries".to_owned(),
                DataType::BinaryOffset => "binary_offsets".to_owned(),
                DataType::List(_) => "lists".to_owned(),
                DataType::Null => "nulls".to_owned(),
                DataType::Enum(_, _) => "enums".to_owned(),
                DataType::Unknown(_) => "unknowns".to_owned(),
                DataType::Struct(_) => "structs".to_owned(),
                _ => "series".to_owned(),
            },
            SpicyObj::Matrix(_) => "matrix".to_owned(),
            SpicyObj::Dict(_) => "dict".to_owned(),
            SpicyObj::DataFrame(_) => "df".to_owned(),
            SpicyObj::LazyFrame(_) => "lf".to_owned(),
            SpicyObj::Fn(_) => "fn".to_owned(),
            SpicyObj::Err(_) => "err".to_owned(),
            SpicyObj::Return(_) => "return".to_owned(),
            SpicyObj::Null => "null".to_owned(),
            SpicyObj::DelayedArg => "delayed_arg".to_owned(),
            SpicyObj::ParDataFrame { .. } => "par_df".to_owned(),
        }
    }

    pub fn parse_date(date: &str) -> SpicyResult<SpicyObj> {
        match chrono::NaiveDate::parse_from_str(date, "%Y.%m.%d") {
            Ok(d) => Ok(SpicyObj::Date(d.num_days_from_ce() - UNIX_EPOCH_DAY)),
            Err(_) => Err(SpicyError::ParserErr(format!("Not a valid date, {}", date))),
        }
    }

    pub fn parse_time(time: &str) -> SpicyResult<SpicyObj> {
        let err = || SpicyError::ParserErr(format!("Not a valid time, {}", time));
        let mut nano = "";
        let time = if time.len() > 8 {
            let v: Vec<&str> = time.split(".").collect();
            nano = v[1];
            v[0]
        } else {
            time
        };
        let v: Vec<&str> = time.split(":").collect();
        let hh = v[0].parse::<i64>().map_err(|_| err())?;
        if hh > 23 {
            return Err(err());
        }
        let mm = v[1].parse::<i64>().map_err(|_| err())?;
        if mm > 59 {
            return Err(err());
        }
        let ss = v[2].parse::<i64>().map_err(|_| err())?;
        if ss > 59 {
            return Err(err());
        }
        let nano = format!("{:0<9}", nano);
        let nano = nano.parse::<i64>().map_err(|_| err())?;
        if nano > 999_999_999 {
            return Err(err());
        }
        Ok(SpicyObj::Time(
            (hh * 3600 + mm * 60 + ss) * 1_000_000_000 + nano,
        ))
    }

    pub fn parse_duration(duration: &str) -> SpicyResult<SpicyObj> {
        let err = || SpicyError::ParserErr(format!("Not a valid duration, {}", duration));
        let is_neg = duration.starts_with("-");
        let v: Vec<&str> = duration.split("D").collect();
        let time = v[1];
        let day = v[0].parse::<i64>().map_err(|_| err())?;
        let nano = if !time.is_empty() {
            SpicyObj::parse_time(time)
                .map_err(|_| err())?
                .to_i64()
                .unwrap()
        } else {
            0
        };
        Ok(SpicyObj::Duration(if is_neg {
            day * NS_IN_DAY - nano
        } else {
            day * NS_IN_DAY + nano
        }))
    }

    pub fn parse_datetime(datetime: &str) -> SpicyResult<SpicyObj> {
        match chrono::NaiveDateTime::parse_from_str(datetime, "%Y.%m.%dT%H:%M:%S%.f") {
            Ok(d) => Ok(SpicyObj::Datetime(d.and_utc().timestamp_millis())),
            Err(_) => Err(SpicyError::ParserErr(format!(
                "Not a valid datetime, {}",
                datetime
            ))),
        }
    }

    pub fn parse_timestamp(datetime: &str) -> SpicyResult<SpicyObj> {
        match chrono::NaiveDateTime::parse_from_str(datetime, "%Y.%m.%dD%H:%M:%S%.f") {
            Ok(d) => Ok(SpicyObj::Timestamp(
                d.and_utc().timestamp_nanos_opt().unwrap_or(0),
            )),
            Err(_) => Err(SpicyError::ParserErr(format!(
                "Not a valid datetime, {}",
                datetime
            ))),
        }
    }

    pub fn from_any_value(a: AnyValue) -> SpicyObj {
        match a {
            AnyValue::Boolean(b) => SpicyObj::Boolean(b),
            AnyValue::String(s) => SpicyObj::String(s.to_owned()),
            AnyValue::UInt8(v) => SpicyObj::U8(v),
            AnyValue::Int16(v) => SpicyObj::I16(v),
            AnyValue::Int32(v) => SpicyObj::I32(v),
            AnyValue::Int64(v) => SpicyObj::I64(v),
            AnyValue::Float32(v) => SpicyObj::F32(v),
            AnyValue::Float64(v) => SpicyObj::F64(v),
            AnyValue::Date(v) => SpicyObj::Date(v),
            AnyValue::Datetime(v, TimeUnit::Milliseconds, _) => SpicyObj::Datetime(v),
            AnyValue::Datetime(v, TimeUnit::Nanoseconds, _) => SpicyObj::Timestamp(v),
            AnyValue::Duration(v, TimeUnit::Nanoseconds) => SpicyObj::Duration(v),
            AnyValue::Time(v) => SpicyObj::Time(v),
            AnyValue::Categorical(i, g) => {
                let sym = g.cat_to_str(i);
                SpicyObj::Symbol(sym.unwrap_or("").to_owned())
            }
            AnyValue::List(s) => SpicyObj::Series(s),
            AnyValue::StringOwned(s) => SpicyObj::String(s.to_string()),
            _ => SpicyObj::Null,
        }
    }

    pub fn mut_df(&mut self) -> SpicyResult<&mut DataFrame> {
        match self {
            SpicyObj::DataFrame(df) => Ok(df),
            _ => Err(SpicyError::MismatchedTypeErr(
                "df".to_owned(),
                self.get_type_name(),
            )),
        }
    }

    pub fn to_short_string(&self) -> String {
        match self {
            SpicyObj::MixedList(l) => format!("( `list: {})", l.len()),
            SpicyObj::Series(series) => {
                format!(
                    "( `{} [{}]: {})",
                    series.name(),
                    series.dtype(),
                    series.len(),
                )
            }
            SpicyObj::Matrix(array_base) => {
                format!(
                    "[[ `matrix : {}, {} ]]",
                    array_base.nrows(),
                    array_base.ncols()
                )
            }
            SpicyObj::Dict(index_map) => format!("{{ `dict: {} }}", index_map.len()),
            SpicyObj::DataFrame(data_frame) => format!(
                "([]{})",
                data_frame
                    .get_column_names()
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            SpicyObj::Fn(jfn) => format!("{{[{}] `fn }}", jfn.params.join("; ")),
            _ => format!("{}", self),
        }
    }
}

macro_rules! impl_cast {
    ($fn_name:ident, $enum:ident, $ty:ty, $ty_str:literal) => {
        impl SpicyObj {
            pub fn $fn_name(&self) -> SpicyResult<&$ty> {
                if let SpicyObj::$enum(v) = self {
                    Ok(v)
                } else {
                    Err(SpicyError::MismatchedTypeErr(
                        $ty_str.to_owned(),
                        self.get_type_name(),
                    ))
                }
            }
        }
    };
}

impl_cast!(bool, Boolean, bool, "bool");
impl_cast!(u8, U8, u8, "u8");
impl_cast!(i16, I16, i16, "i16");
impl_cast!(i32, I32, i32, "i32");
impl_cast!(i64, I64, i64, "i64");
impl_cast!(date, Date, i32, "date");
impl_cast!(time, Time, i64, "time");
impl_cast!(datetime, Datetime, i64, "datetime");
impl_cast!(timestamp, Timestamp, i64, "timestamp");
impl_cast!(duration, Duration, i64, "duration");
impl_cast!(f32, F32, f32, "f32");
impl_cast!(f64, F64, f64, "f64");

impl_cast!(df, DataFrame, DataFrame, "df");
impl_cast!(lf, LazyFrame, LazyFrame, "lf");
impl_cast!(sym, Symbol, String, "str");
impl_cast!(fn_, Fn, Func, "func");
impl_cast!(series, Series, Series, "series");
impl_cast!(dict, Dict, IndexMap<String, SpicyObj>, "dict");
impl_cast!(list, MixedList, Vec<SpicyObj>, "list");
impl_cast!(matrix, Matrix, ArcArray2<f64>, "matrix");

macro_rules! impl_parse_str {
    ($fn_name:ident, $enum:ident, $ty:ty) => {
        impl SpicyObj {
            pub fn $fn_name(s: &str) -> SpicyObj {
                match s.parse::<$ty>() {
                    Ok(i) => SpicyObj::$enum(i),
                    Err(_) => SpicyObj::Null,
                }
            }
        }
    };
}

impl_parse_str!(parse_u8, U8, u8);
impl_parse_str!(parse_i16, I16, i16);
impl_parse_str!(parse_i32, I32, i32);
impl_parse_str!(parse_i64, I64, i64);
impl_parse_str!(parse_f32, F32, f32);
impl_parse_str!(parse_f64, F64, f64);

impl Display for SpicyObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s: String = match self {
            SpicyObj::Boolean(v) => {
                if *v {
                    "1b".to_owned()
                } else {
                    "0b".to_owned()
                }
            }
            SpicyObj::U8(v) => format!("{:#04x}", v),
            SpicyObj::I16(v) => format!("{}h", v),
            SpicyObj::I32(v) => format!("{}i", v),
            SpicyObj::I64(v) => format!("{}", v),
            SpicyObj::F32(v) => format!("{}e", v),
            SpicyObj::F64(v) => format!("{}", v),
            SpicyObj::Date(v) => {
                let date = NaiveDate::from_num_days_from_ce_opt(v + UNIX_EPOCH_DAY);

                match date {
                    Some(dt) => dt.format("%Y.%m.%d").to_string(),
                    None => "XXXX.XX.XX".to_owned(),
                }
            }
            SpicyObj::Timestamp(v) => {
                match DateTime::from_timestamp(v / 1_000_000_000, (v % 1_000_000_000) as u32) {
                    Some(t) => t.format("%Y.%m.%dD%H:%M:%S%.9f").to_string(),
                    None => "****.**.**D**:**:**.*********".to_string(),
                }
            }
            SpicyObj::Datetime(v) => {
                match DateTime::from_timestamp(v / 1000, (v % 1000 * NS_IN_MS) as u32) {
                    Some(t) => t.format("%Y.%m.%dT%H:%M:%S%.3f").to_string(),
                    None => "****.**.**D**:**:**.***".to_string(),
                }
            }
            SpicyObj::Time(v) => {
                if *v >= 0 && *v < 86400000000000 {
                    let s = v / 1_000_000_000;
                    let ns = v % 1_000_000_000;
                    let m = s / 60;
                    let s = s % 60;
                    let h = m / 60;
                    let m = m % 60;
                    format!("{:02}:{:02}:{:02}.{:09}", h, m, s, ns)
                } else {
                    "*D**:**:**.*********".to_owned()
                }
            }
            SpicyObj::Expr(e) => format!("expr: {:?}", e),
            SpicyObj::Duration(v) => {
                let sign = if v.is_negative() { "-" } else { "" };
                let ns = v.abs();
                let s = ns / 1_000_000_000;
                let ns = ns % 1_000_000_000;
                let m = s / 60;
                let s = s % 60;
                let h = m / 60;
                let m = m % 60;
                let d = h / 24;
                let h = h % 24;
                format!("{}{}D{:02}:{:02}:{:02}.{:09}", sign, d, h, m, s, ns)
            }
            SpicyObj::MixedList(l) => {
                let mut output = "".to_owned();
                if l.is_empty() {
                    output.push_str("()");
                } else if l.len() <= 10 && !l.is_empty() {
                    output.push_str("(\n");
                    for obj in &l[0..(l.len() - 1)] {
                        output.push_str(&format!("  {}; \n", obj.to_short_string()))
                    }
                    if let Some(obj) = l.last() {
                        output.push_str(&format!("  {}\n", obj.to_short_string()));
                    }
                    output.push(')');
                } else {
                    output.push_str("(\n");
                    for obj in l.iter().take(5) {
                        output.push_str(&format!("  {}; \n", obj.to_short_string()))
                    }
                    output.push_str(" ...; \n");
                    for obj in l.iter().skip(l.len() - 5).take(4) {
                        output.push_str(&format!("  {}; \n", obj.to_short_string()))
                    }
                    output.push_str(&format!("  {}\n", l.last().unwrap().to_short_string()));
                    output.push(')');
                }
                output
            }
            SpicyObj::Symbol(v) => format!("`{}", v),
            SpicyObj::String(v) => format!("\"{}\"", v),
            SpicyObj::Series(v) => format!("{}", v),
            SpicyObj::Matrix(m) => format!("{}\nshape: ({}, {})", m, m.nrows(), m.ncols()),
            SpicyObj::Dict(d) => {
                let width = d.keys().map(|d| d.len()).max().unwrap_or(8) + 1;
                let mut output = "".to_owned();
                if d.is_empty() {
                    output.push_str("()!()");
                } else if d.len() <= 10 {
                    for (i, (key, value)) in d.iter().enumerate() {
                        if i == d.len() - 1 {
                            output.push_str(&format!(
                                "{:width$} | {}",
                                key,
                                value.to_short_string()
                            ))
                        } else {
                            output.push_str(&format!(
                                "{:width$} | {}\n",
                                key,
                                value.to_short_string()
                            ))
                        }
                    }
                } else {
                    for (key, value) in d.iter().take(5) {
                        output.push_str(&format!("{:width$} | {}\n", key, value.to_short_string()))
                    }
                    output.push_str("...\n");
                    for (i, (key, value)) in d.iter().skip(d.len() - 5).enumerate() {
                        if i == 4 {
                            output.push_str(&format!(
                                "{:width$} | {}",
                                key,
                                value.to_short_string()
                            ))
                        } else {
                            output.push_str(&format!(
                                "{:width$} | {}\n",
                                key,
                                value.to_short_string()
                            ))
                        }
                    }
                }
                output
            }
            SpicyObj::LazyFrame(lf) => format!(
                "LazyFrame: {}",
                lf.describe_plan()
                    .unwrap_or("failed to describe plan".to_owned())
            ),
            SpicyObj::DataFrame(df) => format!("{}", df),
            SpicyObj::Fn(fn_) => {
                format!("{}", fn_)
            }
            SpicyObj::Err(err) => format!("Error: {}", err),
            SpicyObj::Return(obj) => format!("Return: {}", obj),
            SpicyObj::Null => "0n".to_owned(),
            SpicyObj::DelayedArg => "::".to_owned(),
            SpicyObj::ParDataFrame(par_df) => format!("{}", par_df),
        };
        write!(f, "{}", s)
    }
}

impl TryFrom<SpicyObj> for Series {
    type Error = SpicyError;

    fn try_from(other: SpicyObj) -> Result<Self, Self::Error> {
        match other {
            SpicyObj::Series(series) => Ok(series),
            obj => Err(SpicyError::Err(format!("Not Series - {:?}", obj))),
        }
    }
}

impl TryFrom<SpicyObj> for Column {
    type Error = SpicyError;

    fn try_from(other: SpicyObj) -> Result<Self, Self::Error> {
        match other {
            SpicyObj::Series(series) => Ok(series.into()),
            obj => Err(SpicyError::Err(format!("Not Series - {:?}", obj))),
        }
    }
}

impl TryFrom<SpicyObj> for DataFrame {
    type Error = SpicyError;

    fn try_from(other: SpicyObj) -> Result<Self, Self::Error> {
        match other {
            SpicyObj::DataFrame(df) => Ok(df),
            obj => Err(SpicyError::Err(format!("Not DataFrame - {:?}", obj))),
        }
    }
}

pub fn get_series_len(series: &Series) -> Result<usize, SpicyError> {
    let length = series.len();
    let data_type = series.dtype();
    match data_type {
        PolarsDataType::Null => Ok(length * 2 + 6),
        PolarsDataType::Boolean => Ok(length + 6),
        PolarsDataType::Int16 => Ok(length * 2 + 6),
        PolarsDataType::Int32 => Ok(length * 4 + 6),
        PolarsDataType::Int64 => Ok(length * 8 + 6),
        PolarsDataType::UInt8 => Ok(length * 2 + 6),
        PolarsDataType::UInt16 => Ok(length * 4 + 6),
        PolarsDataType::UInt32 => Ok(length * 8 + 6),
        PolarsDataType::Float32 => Ok(length * 4 + 6),
        PolarsDataType::Float64 => Ok(length * 8 + 6),
        // to k datetime
        PolarsDataType::Datetime(_, _) => Ok(length * 8 + 6),
        PolarsDataType::Date => Ok(length * 8 + 6),
        // to time
        // to timespan
        PolarsDataType::Time => Ok(length * 8 + 6),
        // to timespan
        PolarsDataType::Duration(_) => Ok(length * 8 + 6),
        // to string
        PolarsDataType::String => {
            let ptr = series.to_physical_repr();
            let array = ptr.str().unwrap();
            let str_size: usize = array.par_iter().map(|s| s.unwrap_or("").len()).sum();
            Ok(array.get_values_size() * 6 + str_size)
        }
        PolarsDataType::List(data_type) => {
            let array = series.chunks()[0]
                .as_any()
                .downcast_ref::<LargeListArray>()
                .unwrap();
            let length = array.offsets().len();
            let values_length = array.len();
            match data_type.as_ref() {
                PolarsDataType::Boolean => Ok(values_length + 6 * length + 6),
                PolarsDataType::UInt8 => Ok(values_length + 6 * length + 6),
                PolarsDataType::Int16 => Ok(2 * values_length + 6 * length + 6),
                PolarsDataType::Int32 => Ok(4 * values_length + 6 * length + 6),
                PolarsDataType::Int64 => Ok(8 * values_length + 6 * length + 6),
                PolarsDataType::Float32 => Ok(4 * values_length + 6 * length + 6),
                PolarsDataType::Float64 => Ok(8 * values_length + 6 * length + 6),
                _ => Err(SpicyError::NotSupportedSeriesTypeErr(
                    data_type.as_ref().clone(),
                )),
            }
        }
        PolarsDataType::Array(data_type, size) => {
            let array = series.chunks()[0]
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            let length = array.len();
            match data_type.as_ref() {
                PolarsDataType::Boolean => Ok((size + 6) * length + 6),
                PolarsDataType::UInt8 => Ok((size + 6) * length + 6),
                PolarsDataType::Int16 => Ok((2 * size + 6) * length + 6),
                PolarsDataType::Int32 => Ok((4 * size + 6) * length + 6),
                PolarsDataType::Int64 => Ok((8 * size + 6) * length + 6),
                PolarsDataType::Float32 => Ok((4 * size + 6) * length + 6),
                PolarsDataType::Float64 => Ok((8 * size + 6) * length + 6),
                _ => Err(SpicyError::NotSupportedSeriesTypeErr(
                    data_type.as_ref().clone(),
                )),
            }
        }
        PolarsDataType::Binary => {
            let array = series.binary().unwrap();
            let is_16_fixed_binary = array.into_iter().any(|v| 16 == v.unwrap_or(&[]).len());
            if is_16_fixed_binary {
                Ok(16 * length + 6)
            } else {
                Err(SpicyError::Err(
                    "Only support 16 fixed size binary as guid".to_string(),
                ))
            }
        }
        // to symbol
        PolarsDataType::Categorical(_, _) => {
            let cat = series.cat32().unwrap();
            let mut length: usize = 6;
            for s in cat.iter_str() {
                length += s.unwrap_or("").len() + 1;
            }
            Ok(length)
        }
        _ => Err(SpicyError::NotSupportedSeriesTypeErr(data_type.clone())),
    }
}

#[cfg(test)]
mod tests {
    use crate::obj::SpicyObj;
    #[test]
    fn parse_time() {
        assert_eq!(
            SpicyObj::parse_time("23:59:59").unwrap(),
            SpicyObj::Time(86399000000000)
        );
        assert_eq!(
            SpicyObj::parse_time("07:59:59").unwrap(),
            SpicyObj::Time(28799000000000)
        );
        assert_eq!(
            SpicyObj::parse_time("23:59:59.").unwrap(),
            SpicyObj::Time(86399000000000)
        );
        assert_eq!(
            SpicyObj::parse_time("23:59:59.123456789").unwrap(),
            SpicyObj::Time(86399123456789)
        );
        assert_eq!(
            SpicyObj::parse_time("23:59:59.123").unwrap(),
            SpicyObj::Time(86399123000000)
        );
        assert_eq!(
            SpicyObj::parse_time("23:59:59.000123").unwrap(),
            SpicyObj::Time(86399000123000)
        );
        assert!(SpicyObj::parse_time("24:59:59.123456789").is_err())
    }

    #[test]
    fn parse_duration() {
        assert_eq!(
            SpicyObj::parse_duration("0D23:59:59").unwrap(),
            SpicyObj::Duration(86399000000000)
        );
        assert_eq!(
            SpicyObj::parse_duration("1D23:59:59").unwrap(),
            SpicyObj::Duration(172799000000000)
        );
        assert_eq!(
            SpicyObj::parse_duration("100D23:59:59").unwrap(),
            SpicyObj::Duration(8726399000000000)
        );
        assert!(SpicyObj::parse_duration("100D23:60:59.123456789").is_err())
    }
}
