use std::fs::File;

use indexmap::IndexMap;
use polars::datatypes::{DataType, TimeUnit};
use polars::frame::DataFrame;
use polars::io::parquet::write::{ParquetCompression, ParquetWriter};
use polars::series::Series;

use chili_core::{FuncType, SpicyError, SpicyObj, SpicyResult};

// use crate::operator::{MS_IN_DAY, NS_IN_DAY, NS_IN_MS};

pub(crate) fn dict_op_atom(
    d: &IndexMap<String, SpicyObj>,
    atom: &SpicyObj,
    op: FuncType,
) -> SpicyResult<SpicyObj> {
    let mut res = d.clone();
    for (k, v) in d.iter() {
        res.insert(k.to_string(), op(&[v, atom])?);
    }
    Ok(SpicyObj::Dict(res))
}

pub(crate) fn atom_op_dict(
    atom: &SpicyObj,
    d: &IndexMap<String, SpicyObj>,
    op: FuncType,
) -> SpicyResult<SpicyObj> {
    let mut res = d.clone();
    for (k, v) in d.iter() {
        res.insert(k.to_string(), op(&[atom, v])?);
    }
    Ok(SpicyObj::Dict(res))
}

pub(crate) fn dict_op_list(
    d: &IndexMap<String, SpicyObj>,
    list: &[SpicyObj],
    op: FuncType,
) -> SpicyResult<SpicyObj> {
    let mut res = IndexMap::with_capacity(d.len());
    for (i, (k, v)) in d.iter().enumerate() {
        res.insert(k.to_string(), op(&[v, &list[i]])?);
    }
    Ok(SpicyObj::Dict(res))
}

pub(crate) fn list_op_dict(
    list: &[SpicyObj],
    d: &IndexMap<String, SpicyObj>,
    op: FuncType,
) -> SpicyResult<SpicyObj> {
    let mut res = IndexMap::with_capacity(d.len());
    for (i, (k, v)) in d.iter().enumerate() {
        res.insert(k.to_string(), op(&[&list[i], v])?);
    }
    Ok(SpicyObj::Dict(res))
}

pub(crate) fn list_op_atom(l: &[SpicyObj], atom: &SpicyObj, op: FuncType) -> SpicyResult<SpicyObj> {
    Ok(SpicyObj::MixedList(
        l.iter()
            .map(|args| op(&[args, atom]))
            .collect::<SpicyResult<Vec<SpicyObj>>>()?,
    ))
}

pub(crate) fn atom_op_list(atom: &SpicyObj, l: &[SpicyObj], op: FuncType) -> SpicyResult<SpicyObj> {
    Ok(SpicyObj::MixedList(
        l.iter()
            .map(|args| op(&[atom, args]))
            .collect::<SpicyResult<Vec<SpicyObj>>>()?,
    ))
}

pub(crate) fn list_op_list(
    l0: &[SpicyObj],
    l1: &[SpicyObj],
    op: FuncType,
) -> SpicyResult<SpicyObj> {
    Ok(SpicyObj::MixedList(
        l0.iter()
            .zip(l1.iter())
            .map(|(a0, a1)| op(&[a0, a1]))
            .collect::<SpicyResult<Vec<SpicyObj>>>()?,
    ))
}

// pub(crate) fn duration_series_to_time(s: &Series) -> Series {
//     (s.cast(&DataType::Int64).unwrap() % NS_IN_MS)
//         .cast(&DataType::Time)
//         .unwrap()
// }

// pub(crate) fn date_series_to_datetime_i64(s: &Series) -> Series {
//     s.cast(&DataType::Int64).unwrap() * MS_IN_DAY
// }

// pub(crate) fn date_series_to_timestamp_i64(s: &Series) -> Series {
//     s.cast(&DataType::Int64).unwrap() * NS_IN_DAY
// }

// pub(crate) fn datetime_series_to_time_i64(s: &Series) -> Series {
//     s.cast(&DataType::Int64).unwrap() * NS_IN_MS % NS_IN_DAY
// }

// pub(crate) fn timestamp_series_to_time_i64(s: &Series) -> Series {
//     s.cast(&DataType::Int64).unwrap() % NS_IN_DAY
// }

// pub(crate) fn datetime_series_to_timestamp_i64(s: &Series) -> Series {
//     s.cast(&DataType::Int64).unwrap() * NS_IN_MS
// }

// pub(crate) fn cast_series(s: &Series, dtype: &DataType) -> SpicyResult<Series> {
//     match dtype {
//         DataType::Date if s.dtype().is_integer() => {
//             let array = s.cast(&DataType::Int32).unwrap();
//             let array = array.i32().unwrap();
//             let res: Series = array
//                 .iter()
//                 .map(|d| d.filter(|&d| (-96465658..=95026601).contains(&d)))
//                 .collect();
//             Ok(res.cast(dtype).unwrap())
//         }
//         DataType::Time if s.dtype().is_integer() => {
//             let array = s.cast(&DataType::Int64).unwrap();
//             let array = array.i64().unwrap();
//             let res: Series = array
//                 .iter()
//                 .map(|t| t.filter(|&t| (0..=86399999999999).contains(&t)))
//                 .collect();
//             Ok(res.cast(dtype).unwrap())
//         }
//         _ => s.cast(dtype).map_err(|e| SpicyError::Err(e.to_string())),
//     }
// }

pub(crate) fn write_parquet_to_filepath(filepath: &str, df: &DataFrame) -> SpicyResult<u64> {
    let mut file = match File::create(filepath) {
        Ok(f) => f,
        Err(e) => {
            return Err(SpicyError::Err(format!(
                "failed to create file, dose the directory exist?: {}",
                e
            )));
        }
    };

    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::default())
        .finish(&mut df.clone())
        .map_err(|e| SpicyError::Err(e.to_string()))
}

pub(crate) fn get_data_type_name(datatype: &DataType) -> &str {
    match datatype {
        DataType::Boolean => "bool",
        DataType::UInt8 => "u8",
        DataType::UInt16 => "u16",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Int8 => "i8",
        DataType::Int16 => "i16",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Decimal(_, _) => "decimal",
        DataType::String => "str",
        DataType::Date => "date",
        DataType::Datetime(TimeUnit::Milliseconds, _) => "datetime",
        DataType::Datetime(TimeUnit::Nanoseconds, _) => "timestamp",
        DataType::Duration(_) => "duration",
        DataType::Time => "time",
        DataType::List(_) => "list",
        DataType::Null => "null",
        DataType::Categorical(_, _) => "sym",
        _ => "unknown",
    }
}

pub(crate) fn cast_to_int(s: &Series) -> SpicyResult<Series> {
    match s.dtype() {
        DataType::Boolean
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64 => Ok(s.clone()),
        DataType::Date => Ok(s.cast(&DataType::Int32).unwrap()),
        DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            Ok(s.cast(&DataType::Int64).unwrap())
        }
        _ => Err(SpicyError::Err(format!(
            "Not an integer series: {}",
            s.dtype()
        ))),
    }
}
