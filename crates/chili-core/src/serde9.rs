use std::{
    env,
    io::{Cursor, Write},
    sync::LazyLock,
};

use indexmap::IndexMap;
// use ndarray::ArcArray2;
use polars::{
    io::{
        SerReader, SerWriter,
        ipc::{IpcCompression, IpcStreamReader, IpcStreamWriter},
    },
    prelude::{ArrowDataType, ArrowTimeUnit, Categories, DataType, StringNameSpaceImpl, TimeUnit},
    series::Series,
};
use polars_arrow::{
    array::{BooleanArray, PrimitiveArray, Utf8Array},
    bitmap::Bitmap,
    offset::OffsetsBuffer,
    types::NativeType,
};

use crate::{
    Func, SpicyObj,
    errors::{SpicyError, SpicyResult},
};

const PADDING: [&[u8]; 8] = [
    &[],
    &[0, 0, 0, 0, 0, 0, 0],
    &[0, 0, 0, 0, 0, 0],
    &[0, 0, 0, 0, 0],
    &[0, 0, 0, 0],
    &[0, 0, 0],
    &[0, 0],
    &[0],
];

// 1MB
const IPC_COMPRESS_THRESHOLD: usize = 1048576;

static IPC_COMPRESS_ESTIMATE_RATIO: LazyLock<usize> = LazyLock::new(|| {
    let network_bandwidth = env::var("CHILI_NETWORK_BANDWIDTH")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(1000);
    if network_bandwidth > 2500 {
        1009
    } else if network_bandwidth > 1000 {
        647
    } else {
        487
    }
});

static IPC_COMPRESSION: LazyLock<Option<IpcCompression>> = LazyLock::new(|| {
    let network_bandwidth = env::var("CHILI_NETWORK_BANDWIDTH")
        .unwrap_or_default()
        .parse::<usize>()
        .unwrap_or(1000);
    if network_bandwidth > 2500 {
        None
    } else if network_bandwidth > 1000 {
        Some(IpcCompression::LZ4)
    } else {
        Some(IpcCompression::default())
    }
});

pub fn deserialize(vec: &[u8], pos: &mut usize) -> SpicyResult<SpicyObj> {
    let code = vec[*pos];
    *pos += 4;
    let obj;
    match code {
        255 => {
            obj = SpicyObj::Boolean(vec[*pos] == 1);
            *pos += 4
        }
        254 => {
            obj = SpicyObj::U8(vec[*pos]);
            *pos += 4
        }
        253 => {
            obj = SpicyObj::I16(i16::from_le_bytes(vec[*pos..*pos + 2].try_into().unwrap()));
            *pos += 4
        }
        252 | 250 => {
            let i = i32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap());
            if code == 252 {
                obj = SpicyObj::I32(i);
            } else {
                obj = SpicyObj::Date(i);
            }
            *pos += 4
        }
        251 | 249 | 248 | 247 | 246 => {
            *pos += 4;
            let i = i64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap());
            obj = match code {
                251 => SpicyObj::I64(i),
                249 => SpicyObj::Time(i),
                248 => SpicyObj::Datetime(i),
                247 => SpicyObj::Timestamp(i),
                246 => SpicyObj::Duration(i),
                _ => unreachable!(),
            };
            *pos += 8
        }
        245 => {
            let f = f32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap());
            obj = SpicyObj::F32(f);
            *pos += 4
        }
        244 => {
            *pos += 4;
            let f = f64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap());
            obj = SpicyObj::F64(f);
            *pos += 8
        }
        0 => {
            obj = SpicyObj::Null;
            *pos += 4
        }
        243 | 242 | 128 => {
            let byte_len = u32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let s = String::from_utf8(vec[*pos..*pos + byte_len].to_vec()).unwrap();
            obj = if code == 243 {
                SpicyObj::String(s)
            } else if code == 242 {
                SpicyObj::Symbol(s)
            } else {
                return Err(SpicyError::Err(s));
            };
            *pos += byte_len + PADDING[byte_len % 8].len();
        }
        1..=19 => match code {
            1 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let offset = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length];
                let bitmap = Bitmap::from_u8_slice(data, length + offset);
                let array = BooleanArray::try_new(ArrowDataType::Boolean, bitmap, None).unwrap();
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            2 | 15 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length];
                let data_type = if code == 2 {
                    ArrowDataType::UInt8
                } else {
                    ArrowDataType::Int8
                };
                let array = PrimitiveArray::new(data_type, data.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            3 | 16 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 2];
                let array = if code == 3 {
                    let (_, i16s, _) = unsafe { data.align_to::<i16>() };
                    PrimitiveArray::new(ArrowDataType::Int16, i16s.to_vec().into(), None).boxed()
                } else {
                    let (_, u16s, _) = unsafe { data.align_to::<u16>() };
                    PrimitiveArray::new(ArrowDataType::UInt16, u16s.to_vec().into(), None).boxed()
                };
                obj = SpicyObj::Series(Series::from_arrow("".into(), array).unwrap());
            }
            4 | 6 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 4];
                let i32s: &[i32] =
                    unsafe { core::slice::from_raw_parts(data.as_ptr().cast(), length) };
                let data_type = match code {
                    4 => ArrowDataType::Int32,
                    6 => ArrowDataType::Date32,
                    _ => unreachable!(),
                };
                let array = PrimitiveArray::new(data_type, i32s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            5 | 7..=10 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 8];
                let (_, i64s, _) = unsafe { data.align_to::<i64>() };
                let data_type = match code {
                    5 => ArrowDataType::Int64,
                    7 => ArrowDataType::Time64(ArrowTimeUnit::Nanosecond),
                    8 => ArrowDataType::Timestamp(ArrowTimeUnit::Millisecond, None),
                    9 => ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, None),
                    10 => ArrowDataType::Duration(ArrowTimeUnit::Nanosecond),
                    _ => unreachable!(),
                };
                let array = PrimitiveArray::new(data_type, i64s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            11 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 4];
                let (_, f32s, _) = unsafe { data.align_to::<f32>() };
                let array = PrimitiveArray::new(ArrowDataType::Float32, f32s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            12 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 8];
                let (_, f64s, _) = unsafe { data.align_to::<f64>() };
                let array = PrimitiveArray::new(ArrowDataType::Float64, f64s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            13 | 14 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let offsets = &data_bytes[i..i + (length + 1) * 8];
                i += (length + 1) * 8;
                let (_, offsets, _) = unsafe { offsets.align_to::<i64>() };
                let str_bytes = &data_bytes[i..i + offsets[length] as usize];
                let array = Utf8Array::new(
                    ArrowDataType::LargeUtf8,
                    OffsetsBuffer::try_from(offsets.to_vec()).unwrap(),
                    str_bytes.to_vec().into(),
                    None,
                );
                if code == 13 {
                    obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
                } else {
                    obj = SpicyObj::Series(
                        Series::from_arrow("".into(), array.boxed())
                            .unwrap()
                            .cast(&DataType::Categorical(
                                Categories::global(),
                                Categories::global().mapping(),
                            ))
                            .unwrap(),
                    );
                }
            }
            17 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 4];
                let (_, u32s, _) = unsafe { data.align_to::<u32>() };
                let array = PrimitiveArray::new(ArrowDataType::UInt32, u32s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            18 => {
                *pos += 4;
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let data_bytes = &vec[*pos..*pos + byte_len];
                *pos += byte_len;
                let mut i = 0;
                let length = u64::from_le_bytes(data_bytes[i..i + 8].try_into().unwrap()) as usize;
                i += 8;
                let data = &data_bytes[i..i + length * 8];
                let (_, u64s, _) = unsafe { data.align_to::<u64>() };
                let array = PrimitiveArray::new(ArrowDataType::UInt64, u64s.to_vec().into(), None);
                obj = SpicyObj::Series(Series::from_arrow("".into(), array.boxed()).unwrap());
            }
            _ => return Err(SpicyError::NotAbleToDeserializeErr(code)),
        },
        90 => {
            let list_len = u32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap()) as usize;
            let mut list = Vec::with_capacity(list_len);
            *pos += 4;

            if list_len > 0 {
                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let mut v_pos = 16;
                for _ in 0..list_len {
                    list.push(deserialize(vec, &mut v_pos)?)
                }
                *pos += byte_len + PADDING[byte_len % 8].len();
            }
            obj = SpicyObj::MixedList(list);
        }
        // dict
        91 => {
            let dict_len = u32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;

            obj = if dict_len == 0 {
                SpicyObj::Dict(IndexMap::new())
            } else {
                let mut dict: IndexMap<String, SpicyObj> = IndexMap::with_capacity(dict_len);

                let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;

                let keys = &vec[*pos..*pos + byte_len];
                let k_len = u64::from_le_bytes(keys[0..8].try_into().unwrap()) as usize;

                let array_vec = keys[8..4 * dict_len + 8].to_vec();
                let (_, offsets, _) = unsafe { array_vec.align_to::<u32>() };
                let keys_start = 4 * dict_len + 8;
                let keys_end = 8 + k_len;

                let keys = &keys[keys_start..keys_end];
                *pos += keys_end + PADDING[keys_end % 8].len();
                let v_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
                *pos += 8;
                let values = &vec[*pos..*pos + v_len];

                *pos += v_len + PADDING[v_len % 8].len();
                let mut v_pos = 0;
                let mut prev_offset = 0;
                for i in offsets.iter().take(dict_len) {
                    let offset = *i as usize;
                    dict.insert(
                        String::from_utf8(keys[prev_offset..offset].to_vec()).unwrap(),
                        deserialize(values, &mut v_pos)?,
                    );
                    prev_offset = offset;
                }
                SpicyObj::Dict(dict)
            };
        }
        // dataframe
        92 => {
            *pos += 4;
            let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
            *pos += 8;
            let df = IpcStreamReader::new(Cursor::new(&vec[*pos..*pos + byte_len]))
                .finish()
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            obj = SpicyObj::DataFrame(df);
            *pos += byte_len + PADDING[byte_len % 8].len();
        }
        // series
        93 => {
            *pos += 4;
            let byte_len = u64::from_le_bytes(vec[*pos..*pos + 8].try_into().unwrap()) as usize;
            *pos += 8;
            let df = IpcStreamReader::new(Cursor::new(&vec[*pos..*pos + byte_len]))
                .finish()
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            obj = SpicyObj::Series(
                df.select_at_idx(0)
                    .unwrap()
                    .clone()
                    .take_materialized_series(),
            );
            *pos += byte_len + PADDING[byte_len % 8].len();
        }
        // 94 => {
        //     let row = u32::from_le_bytes(v[..4].try_into().unwrap()) as usize;
        //     let col = u32::from_le_bytes(v[4..8].try_into().unwrap()) as usize;
        //     let v8 = v[8..].to_vec();
        //     let ptr = v8.as_ptr() as *const f64;
        //     let f64s = unsafe { core::slice::from_raw_parts(ptr, row * col) };
        //     J::Matrix(ArcArray2::from_shape_vec((row, col), f64s.into()).unwrap())
        // }
        // function
        154 => {
            let byte_len = u32::from_le_bytes(vec[*pos..*pos + 4].try_into().unwrap()) as usize;
            *pos += 4;
            let s = String::from_utf8(vec[*pos..*pos + byte_len].to_vec()).unwrap();
            obj = SpicyObj::Fn(Func::new_raw_fn(&s));
            *pos += byte_len + PADDING[byte_len % 8].len();
        }
        _ => return Err(SpicyError::NotAbleToDeserializeErr(code)),
    }
    Ok(obj)
}

pub fn serialize_err(err: &str) -> Vec<u8> {
    let len = 8 + err.len() + PADDING[err.len() % 8].len();
    let mut vec = Vec::with_capacity(len);
    vec.write_all(&[1, 2, 0, 0, 0, 0, 0, 0]).unwrap();
    vec.write_all(&(len as u64).to_le_bytes()).unwrap();
    vec.write_all(&[128, 0, 0, 0]).unwrap();
    vec.write_all(&(err.len() as u32).to_le_bytes()).unwrap();
    vec.write_all(err.as_bytes()).unwrap();
    vec.write_all(PADDING[err.len() % 8]).unwrap();
    vec
}

const TYPE_SIZE: [usize; 20] = [0, 0, 1, 2, 4, 8, 4, 8, 8, 8, 8, 4, 8, 0, 0, 1, 2, 4, 8, 0];

pub fn serialize(args: &SpicyObj, compress: bool) -> SpicyResult<Vec<Vec<u8>>> {
    // -1 => 255
    let code = args.get_type_code() as u8;
    match args {
        SpicyObj::Boolean(v) => Ok(vec![vec![code, 0, 0, 0, (*v as u8), 0, 0, 0]]),
        SpicyObj::U8(v) => Ok(vec![vec![code, 0, 0, 0, *v, 0, 0, 0]]),
        SpicyObj::I16(v) => {
            let mut buf = vec![code, 0, 0, 0];
            buf.extend_from_slice(&(*v as i32).to_le_bytes());
            Ok(vec![buf])
        }
        SpicyObj::I32(v) | SpicyObj::Date(v) => {
            let mut buf = vec![code, 0, 0, 0];
            buf.extend_from_slice(&v.to_le_bytes());
            Ok(vec![buf])
        }
        SpicyObj::I64(v)
        | SpicyObj::Time(v)
        | SpicyObj::Datetime(v)
        | SpicyObj::Timestamp(v)
        | SpicyObj::Duration(v) => {
            let mut buf = vec![code, 0, 0, 0, 0, 0, 0, 0];
            buf.extend_from_slice(&v.to_le_bytes());
            Ok(vec![buf])
        }
        SpicyObj::F32(v) => {
            let mut buf = vec![code, 0, 0, 0];
            buf.extend_from_slice(&v.to_le_bytes());
            Ok(vec![buf])
        }
        SpicyObj::F64(v) => {
            let mut buf = vec![code, 0, 0, 0, 0, 0, 0, 0];
            buf.extend_from_slice(&v.to_le_bytes());
            Ok(vec![buf])
        }
        SpicyObj::Symbol(s) | SpicyObj::String(s) => {
            let mut buf = Vec::with_capacity(s.len() + 8 + PADDING[s.len() % 8].len());
            buf.write_all(&[code, 0, 0, 0]).unwrap();
            buf.write_all(&(s.len() as u32).to_le_bytes()).unwrap();
            buf.write_all(s.as_bytes()).unwrap();
            buf.write_all(PADDING[s.len() % 8]).unwrap();
            Ok(vec![buf])
        }
        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
        SpicyObj::Series(s) => {
            if s.null_count() == 0 {
                match s.dtype() {
                    DataType::Boolean => {
                        let arr = s.rechunk();
                        let arr = arr.bool().unwrap().chunks().first().unwrap();
                        let arr = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                        let data = arr.values();
                        // byte_len, offset, length, data
                        let mut v8s = serialize_bitmap(data);
                        let header = vec![code, 0, 0, 0, 0, 0, 0, 0];
                        v8s.insert(0, header);
                        return Ok(v8s);
                    }
                    DataType::UInt8 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk.as_any().downcast_ref::<PrimitiveArray<u8>>().unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            vec.write_all(slice).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Int8 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk.as_any().downcast_ref::<PrimitiveArray<i8>>().unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::UInt16 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<u16>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Int16 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<i16>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::UInt32 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<u32>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::UInt64 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<u64>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Int64
                    | DataType::Time
                    | DataType::Datetime(TimeUnit::Milliseconds, _)
                    | DataType::Datetime(TimeUnit::Nanoseconds, _)
                    | DataType::Duration(TimeUnit::Nanoseconds) => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size;
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<i64>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        return Ok(vec![vec]);
                    }
                    DataType::Date | DataType::Int32 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<i32>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Float32 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<f32>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Float64 => {
                        let size = TYPE_SIZE[code as usize];
                        let byte_len = 8 + s.len() * size + PADDING[s.len() * size % 8].len();
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [data] + [padding]
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
                        let chunks = s.chunks();
                        for chunk in chunks {
                            let arr = chunk
                                .as_any()
                                .downcast_ref::<PrimitiveArray<f64>>()
                                .unwrap();
                            let buf = arr.values();
                            let slice = buf.as_slice();
                            let v8 = unsafe {
                                core::slice::from_raw_parts(slice.as_ptr().cast(), size * buf.len())
                            };
                            vec.write_all(v8).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::String => {
                        let str_lens = s.str().unwrap().str_len_bytes();
                        let mut offsets: Vec<i64> = vec![0; s.len() + 1];
                        for i in 0..str_lens.len() {
                            offsets[i + 1] = offsets[i] + str_lens.get(i).unwrap() as i64;
                        }
                        let total_str_len = *offsets.last().unwrap() as usize;
                        // [code, 0, 0, 0, 0, 0, 0, 0] + [byte_len] + [series len] + [offsets] + [data] + [padding]
                        let byte_len = 8
                            + (1 + s.len()) * 8
                            + total_str_len
                            + PADDING[total_str_len % 8].len();
                        let mut vec: Vec<u8> = Vec::with_capacity(16 + byte_len);
                        vec.write_all(&[code, 0, 0, 0, 0, 0, 0, 0]).unwrap();
                        vec.write_all(&(byte_len as u64).to_le_bytes()).unwrap();
                        vec.write_all(&(s.len() as u64).to_le_bytes()).unwrap();

                        let offsets_ptr = offsets.as_ptr().cast::<u8>();
                        let offsets_slice =
                            unsafe { core::slice::from_raw_parts(offsets_ptr, offsets.len() * 8) };
                        vec.write_all(offsets_slice).unwrap();

                        let arr = s.str().unwrap();
                        for str in arr {
                            vec.write_all(str.unwrap().as_bytes()).unwrap();
                        }
                        vec.write_all(PADDING[vec.len() % 8]).unwrap();
                        return Ok(vec![vec]);
                    }
                    DataType::Categorical(_, _) => {
                        let mut offsets = vec![0; s.len() + 1];
                        let str_bytes = if let Some(cat) = s.try_cat8() {
                            cat.iter_str().enumerate().for_each(|(i, s)| {
                                offsets[i + 1] = offsets[i] + s.unwrap().len();
                            });
                            let total_str_len = *offsets.last().unwrap() as usize;
                            let mut str_bytes = Vec::with_capacity(
                                total_str_len + PADDING[total_str_len % 8].len(),
                            );
                            cat.iter_str()
                                .for_each(|s| str_bytes.write_all(s.unwrap().as_bytes()).unwrap());
                            str_bytes.write_all(PADDING[str_bytes.len() % 8]).unwrap();
                            str_bytes
                        } else if let Some(cat) = s.try_cat16() {
                            cat.iter_str().enumerate().for_each(|(i, s)| {
                                offsets[i + 1] = offsets[i] + s.unwrap().len();
                            });
                            let total_str_len = *offsets.last().unwrap() as usize;
                            let mut str_bytes = Vec::with_capacity(
                                total_str_len + PADDING[total_str_len % 8].len(),
                            );
                            cat.iter_str()
                                .for_each(|s| str_bytes.write_all(s.unwrap().as_bytes()).unwrap());
                            str_bytes.write_all(PADDING[str_bytes.len() % 8]).unwrap();
                            str_bytes
                        } else {
                            let cat = s.cat32().unwrap();
                            cat.iter_str().enumerate().for_each(|(i, s)| {
                                offsets[i + 1] = offsets[i] + s.unwrap().len();
                            });
                            let total_str_len = *offsets.last().unwrap() as usize;
                            let mut str_bytes = Vec::with_capacity(
                                total_str_len + PADDING[total_str_len % 8].len(),
                            );
                            cat.iter_str()
                                .for_each(|s| str_bytes.write_all(s.unwrap().as_bytes()).unwrap());
                            str_bytes.write_all(PADDING[str_bytes.len() % 8]).unwrap();
                            str_bytes
                        };
                        let byte_len = 8 + (1 + s.len()) * 8 + str_bytes.len();
                        let mut header = vec![code, 0, 0, 0, 0, 0, 0, 0];
                        header.extend_from_slice(&(byte_len as u64).to_le_bytes());
                        header.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        let offsets_slice: Vec<u8> = unsafe {
                            let offsets_ptr = offsets.as_mut_ptr().cast::<u8>();
                            let length = offsets.len() * 8;
                            std::mem::forget(offsets);
                            Vec::from_raw_parts(offsets_ptr, length, length)
                        };
                        return Ok(vec![header, offsets_slice, str_bytes]);
                    }
                    _ => {}
                }
            };
            let mut df = s.clone().into_frame();
            let estimated_size = df.estimated_size();
            let (compression, ratio) = if compress && estimated_size > IPC_COMPRESS_THRESHOLD {
                (*IPC_COMPRESSION, *IPC_COMPRESS_ESTIMATE_RATIO)
            } else {
                (None, 1009)
            };
            let mut buf = Vec::with_capacity(1699 + (estimated_size * ratio / 1000));
            IpcStreamWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut df)
                .map_err(|e| SpicyError::NotAbleToSerializeErr(e.to_string()))?;
            buf.write_all(PADDING[buf.len() % 8]).unwrap();
            let mut header = vec![93, 0, 0, 0, 0, 0, 0, 0];
            header.extend_from_slice(&(buf.len() as u64).to_le_bytes());
            Ok(vec![header, buf])
        }
        SpicyObj::MixedList(l) => {
            let mut header = vec![code, 0, 0, 0];
            header.extend_from_slice(&(l.len() as u32).to_le_bytes());
            // length of list
            if !l.is_empty() {
                let v = l
                    .iter()
                    .map(|args| serialize(args, compress))
                    .collect::<SpicyResult<Vec<Vec<Vec<u8>>>>>()?;
                let mut v = v.into_iter().flatten().collect::<Vec<Vec<u8>>>();
                // length of list size
                let length: usize = v.iter().map(|b| b.len()).sum();
                header.extend_from_slice(&(length as u64).to_le_bytes());
                v.insert(0, header);
                Ok(v)
            } else {
                Ok(vec![header])
            }
        }
        // J::Matrix(m) => {
        //     let length = m.len() * 8 + 8;
        //     vec.write_all(&(length as u32).to_le_bytes()).unwrap();
        //     vec.write_all(&(m.nrows() as u32).to_le_bytes()).unwrap();
        //     vec.write_all(&(m.ncols() as u32).to_le_bytes()).unwrap();
        //     let ptr = m.as_slice().unwrap();
        //     let ptr = ptr.as_ptr() as *const u8;
        //     let v8 = unsafe { core::slice::from_raw_parts(ptr, m.len() * 8) };
        //     vec.write_all(v8).unwrap();
        // }
        SpicyObj::Dict(d) => {
            let mut header = vec![code, 0, 0, 0];
            header.extend_from_slice(&(d.len() as u32).to_le_bytes());

            if !d.is_empty() {
                let key_lengths = d.keys().map(|s| s.len()).collect::<Vec<usize>>();
                let key_length = d.len() * 4 + key_lengths.iter().sum::<usize>();

                let key_length_with_padding = key_length + PADDING[key_length % 8].len();
                let mut keys_bytes = Vec::with_capacity(key_length_with_padding);
                let mut offsets: Vec<u32> = vec![0; d.len()];
                offsets[0] = key_lengths[0] as u32;
                for i in 1..d.len() {
                    offsets[i] = key_lengths[i] as u32 + offsets[i - 1]
                }

                let v8: &[u8] =
                    unsafe { std::slice::from_raw_parts(offsets.as_ptr().cast(), d.len() * 4) };
                keys_bytes.write_all(v8).unwrap();

                let mut values_v8s = Vec::with_capacity(d.len());

                for (k, v) in d.iter() {
                    keys_bytes.write_all(k.as_bytes()).unwrap();
                    let v8 = serialize(v, compress)?;
                    values_v8s.push(v8);
                }

                keys_bytes.write_all(PADDING[key_length % 8]).unwrap();

                let values_v8 = values_v8s.into_iter().flatten().collect::<Vec<Vec<u8>>>();

                let value_length: usize = values_v8.iter().map(|v| v.len()).sum::<usize>();

                // reserve dict byte full len
                header.extend_from_slice(
                    &((key_length_with_padding + value_length + 16) as u64).to_le_bytes(),
                );

                // reserve keys byte full len
                header.extend_from_slice(&(key_length_with_padding as u64).to_le_bytes());

                let mut v8s = vec![
                    header,
                    keys_bytes,
                    (value_length as u64).to_le_bytes().to_vec(),
                ];
                v8s.extend_from_slice(&values_v8);
                Ok(v8s)
            } else {
                Ok(vec![header])
            }
        }
        SpicyObj::DataFrame(df) => {
            let mut df = df.clone();
            let estimated_size = df.estimated_size();
            let (compression, ratio) = if compress && estimated_size > IPC_COMPRESS_THRESHOLD {
                (*IPC_COMPRESSION, *IPC_COMPRESS_ESTIMATE_RATIO)
            } else {
                (None, 1009)
            };
            let mut buf = Vec::with_capacity(1699 + (estimated_size * ratio / 1000));
            IpcStreamWriter::new(&mut buf)
                .with_compression(compression)
                .finish(&mut df)
                .map_err(|e| SpicyError::NotAbleToSerializeErr(e.to_string()))?;
            buf.write_all(PADDING[buf.len() % 8]).unwrap();
            let mut header = vec![code, 0, 0, 0, 0, 0, 0, 0];
            header.extend_from_slice(&(buf.len() as u64).to_le_bytes());
            Ok(vec![header, buf])
        }
        SpicyObj::Null => Ok(vec![vec![code, 0, 0, 0, 0, 0, 0, 0]]),
        SpicyObj::Fn(f) if f.part_args.is_none() => {
            let fn_body = &f.fn_body;
            let mut buf = Vec::with_capacity(fn_body.len() + 8 + PADDING[fn_body.len() % 8].len());
            buf.write_all(&[code, 0, 0, 0]).unwrap();
            buf.write_all(&(fn_body.len() as u32).to_le_bytes())
                .unwrap();
            buf.write_all(fn_body.as_bytes()).unwrap();
            buf.write_all(PADDING[fn_body.len() % 8]).unwrap();
            Ok(vec![buf])
        }
        _ => Err(SpicyError::NotAbleToSerializeErr(format!(
            "not support spicy obj type {}",
            args.get_type_name()
        ))),
    }
}

// byte_len, offset, length, v8
fn serialize_bitmap(bitmap: &Bitmap) -> Vec<Vec<u8>> {
    let (v8, offset, length) = bitmap.as_slice();
    let mut v8 = v8.to_vec();
    v8.write_all(PADDING[v8.len() % 8]).unwrap();
    let mut header = Vec::with_capacity(24);
    header.write_all(&(v8.len() + 16).to_le_bytes()).unwrap();
    header.write_all(&offset.to_le_bytes()).unwrap();
    header.write_all(&length.to_le_bytes()).unwrap();
    vec![header, v8.to_vec()]
}

#[cfg(test)]
mod tests {
    use crate::SpicyObj;
    use crate::{serde9::deserialize, serde9::serialize, serde9::serialize_err};
    use indexmap::IndexMap;
    use polars::prelude::{Categories, DataType, TimeUnit};
    // use ndarray::ArcArray2;
    use polars::{df, prelude::NamedFrom, series::Series};

    fn serialize_as_v8(args: &SpicyObj) -> Vec<u8> {
        serialize(args, false)
            .unwrap()
            .into_iter()
            .flatten()
            .collect()
    }

    #[test]
    fn serde_bool() {
        let obj = SpicyObj::Boolean(true);
        let v8: &[u8] = &[255, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_u8() {
        let obj = SpicyObj::U8(1);
        let v8: &[u8] = &[254, 0, 0, 0, 1, 0, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_i16() {
        let obj = SpicyObj::I16(258);
        let v8: &[u8] = &[253, 0, 0, 0, 2, 1, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_i32() {
        let obj = SpicyObj::I32(16909060);
        let v8: &[u8] = &[252, 0, 0, 0, 4, 3, 2, 1];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_i64() {
        let obj = SpicyObj::I64(72623859790382856);
        let v8: &[u8] = &[251, 0, 0, 0, 0, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_date() {
        let obj = SpicyObj::Date(16909060);
        let v8: &[u8] = &[250, 0, 0, 0, 4, 3, 2, 1];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_time() {
        let obj = SpicyObj::Time(86399999999999);
        let v8: &[u8] = &[249, 0, 0, 0, 0, 0, 0, 0, 255, 255, 78, 145, 148, 78, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_datetime() {
        let obj = SpicyObj::Datetime(86399999999999);
        let v8: &[u8] = &[248, 0, 0, 0, 0, 0, 0, 0, 255, 255, 78, 145, 148, 78, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_timestamp() {
        let obj = SpicyObj::Timestamp(86399999999999);
        let v8: &[u8] = &[247, 0, 0, 0, 0, 0, 0, 0, 255, 255, 78, 145, 148, 78, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_duration() {
        let obj = SpicyObj::Duration(86399999999999);
        let v8: &[u8] = &[246, 0, 0, 0, 0, 0, 0, 0, 255, 255, 78, 145, 148, 78, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_f32() {
        let obj = SpicyObj::F32(9.9e10);
        let v8: &[u8] = &[245, 0, 0, 0, 225, 102, 184, 81];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_f64() {
        let obj = SpicyObj::F64(9.9e10);
        let v8: &[u8] = &[244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 220, 12, 55, 66];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_string() {
        let obj = SpicyObj::String("🍺".to_owned());
        let v8: &[u8] = &[243, 0, 0, 0, 4, 0, 0, 0, 240, 159, 141, 186, 0, 0, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_symbol() {
        let obj = SpicyObj::Symbol("".to_owned());
        let v8: &[u8] = &[242, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn deserialize_err() {
        let v8: &[u8] = &[
            1, 2, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 5, 0, 0, 0, 101, 114,
            114, 111, 114, 0, 0, 0,
        ];
        assert_eq!(serialize_err("error"), v8);
        let err = deserialize(v8, &mut 16);
        assert!(err.is_err());
        let err = err.unwrap_err();
        assert_eq!(err.to_string(), "error");
    }

    // #[test]
    // fn serde_matrix() {
    //     let obj = J::Matrix(ArcArray2::from_shape_vec((1, 2), [1.0, 2.0f64].into()).unwrap());
    //     let v8: &[u8] = &[
    //         21, 24, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0,
    //         0, 64,
    //     ];
    //     assert_eq!(serialize_as_v8(&obj), v8);
    //     assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    // }

    #[test]
    fn serde_series_bool() {
        let obj = SpicyObj::Series(Series::new("".into(), [true, false, true]));
        let v8: &[u8] = &vec![
            1, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_u8() {
        let obj = SpicyObj::Series(Series::new("".into(), [1u8, 2u8, 3u8]));
        let v8: &[u8] = &vec![
            2, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0,
            0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_i8() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1i8, 2i8, 3i8]));
        let v8: &[u8] = &vec![
            15, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 255, 2, 3, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
    }

    #[test]
    fn serde_series_i16() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1i16, 0i16, 3i16]));
        let v8: &[u8] = &vec![
            3, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0,
            0, 3, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_u16() {
        let obj = SpicyObj::Series(Series::new("".into(), [1u16, 2u16, 3u16]));
        let v8: &[u8] = &vec![
            16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0,
            3, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_i32() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1i32, 0i32, 3i32]));
        let v8: &[u8] = &vec![
            4, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
            255, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_u32() {
        let obj = SpicyObj::Series(Series::new("".into(), [1u32, 2u32, 3u32]));
        let v8: &[u8] = &vec![
            17, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_date() {
        let obj = SpicyObj::Series(
            Series::new("".into(), [19670, -96465658, 95026601])
                .cast(&DataType::Date)
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            6, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 214, 76, 0, 0,
            6, 13, 64, 250, 169, 253, 169, 5, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_i64() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1i64, 0i64, 3i64]));
        let v8: &[u8] = &vec![
            5, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255,
            255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_u64() {
        let obj = SpicyObj::Series(Series::new("".into(), [1u64, 2u64, 3u64]));
        let v8: &[u8] = &vec![
            18, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_time() {
        let obj = SpicyObj::Series(
            Series::new("".into(), [1000i64, 2000i64, 3000i64])
                .cast(&DataType::Time)
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            7, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0,
            0, 0, 0, 0, 208, 7, 0, 0, 0, 0, 0, 0, 184, 11, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_datetime_ms() {
        let obj = SpicyObj::Series(
            Series::new("".into(), [1000i64, 2000i64, 3000i64])
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            8, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0,
            0, 0, 0, 0, 208, 7, 0, 0, 0, 0, 0, 0, 184, 11, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_datetime_ns() {
        let obj = SpicyObj::Series(
            Series::new("".into(), [1000i64, 2000i64, 3000i64])
                .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            9, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0,
            0, 0, 0, 0, 208, 7, 0, 0, 0, 0, 0, 0, 184, 11, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_duration() {
        let obj = SpicyObj::Series(
            Series::new("".into(), [1000i64, 2000i64, 3000i64])
                .cast(&DataType::Duration(TimeUnit::Nanoseconds))
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            10, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0,
            0, 0, 0, 0, 208, 7, 0, 0, 0, 0, 0, 0, 184, 11, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_f32() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1f32, 0f32, 3f32]));
        let v8: &[u8] = &vec![
            11, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128,
            191, 0, 0, 0, 0, 0, 0, 64, 64, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_f64() {
        let obj = SpicyObj::Series(Series::new("".into(), [-1f64, 0f64, 3f64]));
        let v8: &[u8] = &vec![
            12, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 240, 191, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 64,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_string() {
        let obj = SpicyObj::Series(Series::new("".into(), ["Hello", "World", ""]));
        let v8: &[u8] = &vec![
            13, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0,
            72, 101, 108, 108, 111, 87, 111, 114, 108, 100, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series_categorical() {
        let obj = SpicyObj::Series(
            Series::new("".into(), ["Hello", "World", ""])
                .cast(&DataType::Categorical(
                    Categories::global(),
                    Categories::global().mapping(),
                ))
                .unwrap(),
        );
        let v8: &[u8] = &vec![
            14, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0,
            72, 101, 108, 108, 111, 87, 111, 114, 108, 100, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_series() {
        let obj = SpicyObj::Series(Series::new("".into(), [Some(0), Some(1), Some(2), None]));
        let v8: &[u8] = &vec![
            93, 0, 0, 0, 0, 0, 0, 0, 144, 1, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 120, 0, 0, 0, 4,
            0, 0, 0, 242, 255, 255, 255, 20, 0, 0, 0, 4, 0, 1, 0, 0, 0, 10, 0, 11, 0, 8, 0, 10, 0,
            4, 0, 248, 255, 255, 255, 12, 0, 0, 0, 8, 0, 8, 0, 0, 0, 4, 0, 1, 0, 0, 0, 4, 0, 0, 0,
            236, 255, 255, 255, 56, 0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 1, 2, 0, 0, 16, 0, 18, 0, 4,
            0, 16, 0, 17, 0, 8, 0, 0, 0, 12, 0, 0, 0, 0, 0, 244, 255, 255, 255, 32, 0, 0, 0, 1, 0,
            0, 0, 8, 0, 9, 0, 4, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 128, 0, 0, 0,
            4, 0, 0, 0, 236, 255, 255, 255, 128, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 4, 0, 3, 0, 12,
            0, 19, 0, 16, 0, 18, 0, 12, 0, 4, 0, 234, 255, 255, 255, 4, 0, 0, 0, 0, 0, 0, 0, 60, 0,
            0, 0, 16, 0, 0, 0, 0, 0, 10, 0, 20, 0, 4, 0, 12, 0, 16, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 247, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_dict() {
        let mut dict = IndexMap::new();
        dict.insert("byte".to_owned(), SpicyObj::U8(9));
        dict.insert("date".to_owned(), SpicyObj::Date(0));
        dict.insert("null".to_owned(), SpicyObj::Null);
        let obj = SpicyObj::Dict(dict);
        let v8: &[u8] = &vec![
            91, 0, 0, 0, 3, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
            8, 0, 0, 0, 12, 0, 0, 0, 98, 121, 116, 101, 100, 97, 116, 101, 110, 117, 108, 108, 24,
            0, 0, 0, 0, 0, 0, 0, 254, 0, 0, 0, 9, 0, 0, 0, 250, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }

    #[test]
    fn serde_df() {
        let df = df!("Element" => &["Copper", "Silver", "Gold"],
        "Melting Point (K)" => &[1357.77, 1234.93, 1337.33])
        .unwrap();
        let obj = SpicyObj::DataFrame(df);
        let v8: &[u8] = &vec![
            92, 0, 0, 0, 0, 0, 0, 0, 72, 2, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 176, 0, 0, 0, 4,
            0, 0, 0, 242, 255, 255, 255, 20, 0, 0, 0, 4, 0, 1, 0, 0, 0, 10, 0, 11, 0, 8, 0, 10, 0,
            4, 0, 248, 255, 255, 255, 12, 0, 0, 0, 8, 0, 8, 0, 0, 0, 4, 0, 2, 0, 0, 0, 68, 0, 0, 0,
            4, 0, 0, 0, 176, 255, 255, 255, 32, 0, 0, 0, 16, 0, 0, 0, 8, 0, 0, 0, 1, 3, 0, 0, 0, 0,
            0, 0, 250, 255, 255, 255, 2, 0, 6, 0, 6, 0, 4, 0, 17, 0, 0, 0, 77, 101, 108, 116, 105,
            110, 103, 32, 80, 111, 105, 110, 116, 32, 40, 75, 41, 0, 0, 0, 236, 255, 255, 255, 44,
            0, 0, 0, 32, 0, 0, 0, 24, 0, 0, 0, 1, 20, 0, 0, 16, 0, 18, 0, 4, 0, 16, 0, 17, 0, 8, 0,
            0, 0, 12, 0, 0, 0, 0, 0, 252, 255, 255, 255, 4, 0, 4, 0, 7, 0, 0, 0, 69, 108, 101, 109,
            101, 110, 116, 0, 255, 255, 255, 255, 192, 0, 0, 0, 4, 0, 0, 0, 236, 255, 255, 255,
            192, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 4, 0, 3, 0, 12, 0, 19, 0, 16, 0, 18, 0, 12, 0,
            4, 0, 234, 255, 255, 255, 3, 0, 0, 0, 0, 0, 0, 0, 108, 0, 0, 0, 16, 0, 0, 0, 0, 0, 10,
            0, 20, 0, 4, 0, 12, 0, 16, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0,
            0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0,
            0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,
            0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67,
            111, 112, 112, 101, 114, 83, 105, 108, 118, 101, 114, 71, 111, 108, 100, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 71, 225, 122, 20, 55, 149, 64, 31, 133,
            235, 81, 184, 75, 147, 64, 184, 30, 133, 235, 81, 229, 148, 64, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0,
        ];
        assert_eq!(serialize_as_v8(&obj), v8);
        assert_eq!(deserialize(v8, &mut 0).unwrap(), obj);
    }
}
