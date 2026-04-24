//! Python bindings for [`chili_core::EngineState`].

use std::sync::Arc;

use chili_core::constant::{NS_IN_DAY, UNIX_EPOCH_DAY};
use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveTime, Timelike, Utc};
use indexmap::IndexMap;
use polars::frame::DataFrame;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDateTime, PyDelta, PyDict, PyFloat, PyInt, PyList, PyString, PyTime,
    PyTuple, PyTzInfo,
};
use pyo3::{create_exception, intern};
use pyo3_polars::{PyDataFrame, PySeries};

create_exception!(engine, ChiliError, PyException);

fn map_spicy_error<T>(r: Result<T, chili_core::SpicyError>) -> PyResult<T> {
    r.map_err(|e| ChiliError::new_err(e.to_string()))
}

fn unwrap_return(mut o: SpicyObj) -> SpicyObj {
    while let SpicyObj::Return(inner) = o {
        o = *inner;
    }
    o
}

fn spicy_from_py_bound(any: &Bound<'_, PyAny>) -> PyResult<SpicyObj> {
    if any.is_instance_of::<PyBool>() {
        Ok(SpicyObj::Boolean(any.extract::<bool>()?))
        // TODO: this heap allocs on failure
    } else if any.is_instance_of::<PyInt>() {
        match any.extract::<i64>() {
            Ok(v) => Ok(SpicyObj::I64(v)),
            Err(e) => Err(e),
        }
    } else if any.is_instance_of::<PyFloat>() {
        Ok(SpicyObj::F64(any.extract::<f64>()?))
    } else if any.is_instance_of::<PyString>() {
        let value = any.extract::<&str>()?;
        Ok(SpicyObj::Symbol(value.to_string()))
    } else if any.is_instance_of::<PyBytes>() {
        let value = any.cast::<PyBytes>()?;
        Ok(SpicyObj::String(String::from_utf8(
            value.as_bytes().to_vec(),
        )?))
    } else if any.hasattr(intern!(any.py(), "_s"))? {
        let series = any.extract::<PySeries>()?.into();
        Ok(SpicyObj::Series(series))
    } else if any.hasattr(intern!(any.py(), "_df"))? {
        let df = any.extract::<PyDataFrame>()?.into();
        Ok(SpicyObj::DataFrame(df))
    } else if any.is_none() {
        Ok(SpicyObj::Null)
    } else if any.is_instance_of::<PyDateTime>() {
        let datetime: DateTime<Utc> = any.extract()?;
        Ok(SpicyObj::Timestamp(
            datetime.timestamp_nanos_opt().unwrap_or(0),
        ))
    } else if any.is_instance_of::<PyDate>() {
        let date = any.cast::<PyDate>()?;
        let dt: NaiveDate = date.extract()?;
        Ok(SpicyObj::Date(dt.num_days_from_ce() - UNIX_EPOCH_DAY))
    } else if any.is_instance_of::<PyTime>() {
        let time = any.cast::<PyTime>()?;
        let dt: NaiveTime = time.extract()?;
        Ok(SpicyObj::Time(
            dt.nanosecond() as i64 + dt.num_seconds_from_midnight() as i64 * 1_000_000_000,
        ))
    } else if any.is_instance_of::<PyDelta>() {
        let delta: Duration = any.extract()?;
        Ok(SpicyObj::Duration(delta.num_nanoseconds().unwrap_or(0)))
    } else if any.is_instance_of::<PyDict>() {
        let py_dict = any.cast::<PyDict>()?;
        let mut dict = IndexMap::with_capacity(py_dict.len());
        for (k, v) in py_dict.into_iter() {
            let k = match k.extract::<&str>() {
                Ok(s) => s.to_string(),
                Err(_) => {
                    return Err(ChiliError::new_err(format!(
                        "Requires str as key, got {:?}",
                        k.get_type()
                    )));
                }
            };
            let v = spicy_from_py_bound(&v)?;
            dict.insert(k, v);
        }
        Ok(SpicyObj::Dict(dict))
    } else if any.is_instance_of::<PyList>() {
        let py_list = any.cast::<PyList>()?;
        let mut k_list = Vec::with_capacity(py_list.len());
        for py_any in py_list {
            k_list.push(spicy_from_py_bound(&py_any)?);
        }
        Ok(SpicyObj::MixedList(k_list))
    } else if any.is_instance_of::<PyTuple>() {
        let py_tuple = any.cast::<PyTuple>()?;
        let mut k_tuple = Vec::with_capacity(py_tuple.len());
        for py_any in py_tuple {
            k_tuple.push(spicy_from_py_bound(&py_any)?);
        }
        Ok(SpicyObj::MixedList(k_tuple))
    } else {
        Err(ChiliError::new_err(format!(
            "Unsupported Python type for chili conversion: {}",
            any.get_type().name()?
        )))
    }
}

fn spicy_to_py(py: Python<'_>, obj: SpicyObj) -> PyResult<Py<PyAny>> {
    let obj = unwrap_return(obj);
    match obj {
        SpicyObj::Null => Ok(py.None()),
        SpicyObj::Boolean(v) => Ok(v.into_pyobject(py)?.to_owned().into_any().unbind()),
        SpicyObj::U8(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        SpicyObj::I16(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        SpicyObj::I32(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        SpicyObj::I64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        SpicyObj::F32(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        SpicyObj::F64(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        // String => Python Bytes
        SpicyObj::String(v) => Ok(v.as_bytes().into_pyobject(py)?.into_any().unbind()),
        SpicyObj::Symbol(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        // Date => Python datetime.date
        SpicyObj::Date(v) => Ok(PyDate::from_timestamp(py, v as i64 * 86400)?
            .into_any()
            .unbind()),
        // Time => Python datetime.time
        SpicyObj::Time(v) => {
            let seconds = v / 1000000000;
            let microseconds = v % 1000000000 / 1000;
            let hour = seconds / 3600 % 24;
            let minute = seconds / 60 % 60;
            let second = seconds % 60;
            Ok(PyTime::new(
                py,
                hour as u8,
                minute as u8,
                second as u8,
                microseconds as u32,
                Some(&PyTzInfo::utc(py).unwrap()),
            )?
            .into_any()
            .unbind())
        }
        // Datetime => Python datetime.datetime
        SpicyObj::Datetime(v) => Ok(PyDateTime::from_timestamp(
            py,
            v as f64 / 1000.0,
            Some(&PyTzInfo::utc(py).unwrap()),
        )?
        .into_any()
        .unbind()),
        // Timestamp => Python datetime.datetime
        SpicyObj::Timestamp(v) => Ok(PyDateTime::from_timestamp(
            py,
            v as f64 / 1000000000.0,
            Some(&PyTzInfo::utc(py).unwrap()),
        )?
        .into_any()
        .unbind()),
        // Duration => Python timedelta
        SpicyObj::Duration(v) => {
            let abs_v = v.abs();
            let sign = if v < 0 { -1 } else { 1 };
            let days = abs_v / NS_IN_DAY;
            let seconds = abs_v / 1000000000 % 86400;
            let microseconds = v % 1000000000 / 1000;
            Ok(PyDelta::new(
                py,
                days as i32 * sign,
                seconds as i32,
                microseconds as i32,
                false,
            )?
            .into_any()
            .unbind())
        }
        // MixedList => Python list
        SpicyObj::MixedList(items) => {
            let mut list = Vec::with_capacity(items.len());
            for it in items {
                list.push(spicy_to_py(py, it)?);
            }
            Ok(PyList::new(py, &list)?.into_any().unbind())
        }
        SpicyObj::Dict(map) => {
            let d = PyDict::new(py);
            for (k, v) in map {
                d.set_item(k, spicy_to_py(py, v)?)?;
            }
            Ok(d.into_any().unbind())
        }
        SpicyObj::DataFrame(df) => Ok(PyDataFrame(df).into_pyobject(py)?.into_any().unbind()),
        SpicyObj::Series(s) => Ok(PySeries(s).into_pyobject(py)?.into_any().unbind()),
        SpicyObj::Err(msg) => Err(ChiliError::new_err(msg)),
        other => Ok(other.to_string().into_pyobject(py)?.into_any().unbind()),
    }
}

/// Chili evaluation engine; mirrors Rust ``chili_core::EngineState``.
#[pyclass(name = "EngineState")]
struct PyEngineState {
    inner: Arc<EngineState>,
}

#[pymethods]
impl PyEngineState {
    #[new]
    #[pyo3(signature = (debug=false, lazy=false, pepper=false))]
    fn new(debug: bool, lazy: bool, pepper: bool) -> PyResult<Self> {
        let state = EngineState::new(debug, lazy, pepper);
        state.register_fn(&BUILT_IN_FN);
        let arc = Arc::new(state);
        map_spicy_error(arc.set_arc_self(Arc::clone(&arc)))?;
        Ok(Self { inner: arc })
    }

    /// Evaluate a Chili or Pepper expression string (same as the REPL).
    fn eval(&self, py: Python<'_>, source: &str) -> PyResult<Py<PyAny>> {
        let mut stack = Stack::new(None, 0, 0, "");
        let args = SpicyObj::String(source.to_string());
        let src_path = if self.inner.is_repl_use_chili_syntax() {
            "repl.chi"
        } else {
            "repl.pep"
        };
        let obj = map_spicy_error(self.inner.eval(&mut stack, &args, src_path))?;
        spicy_to_py(py, obj)
    }

    fn get_var(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        let obj = map_spicy_error(self.inner.get_var(id))?;
        spicy_to_py(py, obj)
    }

    fn set_var(&self, id: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        let obj = spicy_from_py_bound(&value)?;
        map_spicy_error(self.inner.set_var(id, obj))
    }

    fn has_var(&self, id: &str) -> PyResult<bool> {
        map_spicy_error(self.inner.has_var(id))
    }

    fn del_var(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        let obj = map_spicy_error(self.inner.del_var(id))?;
        spicy_to_py(py, obj)
    }

    fn import_source_path(
        &self,
        py: Python<'_>,
        relative: &str,
        path: &str,
    ) -> PyResult<Py<PyAny>> {
        let obj = map_spicy_error(self.inner.import_source_path(relative, path))?;
        spicy_to_py(py, obj)
    }

    fn set_source(&self, path: &str, src: &str) -> PyResult<usize> {
        map_spicy_error(self.inner.set_source(path, src))
    }

    fn get_source(&self, index: usize) -> PyResult<(String, String)> {
        map_spicy_error(self.inner.get_source(index))
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }

    fn get_displayed_vars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = map_spicy_error(self.inner.get_displayed_vars())?;
        let d = PyDict::new(py);
        for (k, v) in m {
            d.set_item(k, v)?;
        }
        Ok(d.into_any().unbind())
    }

    fn list_vars(&self, py: Python<'_>, pattern: &str) -> PyResult<Py<PyAny>> {
        let df: DataFrame = map_spicy_error(self.inner.list_vars(pattern))?;
        Ok(PyDataFrame(df).into_pyobject(py)?.into_any().unbind())
    }

    fn parse_cache_len(&self) -> usize {
        self.inner.parse_cache_len()
    }

    fn get_tick_count(&self) -> i64 {
        self.inner.get_tick_count()
    }

    fn tick(&self, py: Python<'_>, inc: i64) -> PyResult<Py<PyAny>> {
        let obj = map_spicy_error(self.inner.tick(inc))?;
        spicy_to_py(py, obj)
    }

    fn is_lazy_mode(&self) -> bool {
        self.inner.is_lazy_mode()
    }

    fn is_repl_use_chili_syntax(&self) -> bool {
        self.inner.is_repl_use_chili_syntax()
    }

    fn fn_call(&self, py: Python<'_>, func: &str, args: Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
        let args = args
            .iter()
            .map(|a| spicy_from_py_bound(&a))
            .collect::<Result<Vec<SpicyObj>, PyErr>>()?;
        let args = args.iter().map(|a| a).collect::<Vec<&SpicyObj>>();
        let obj = map_spicy_error(self.inner.fn_call(func, &args))?;
        spicy_to_py(py, obj)
    }

    fn load_par_df(&self, hdb_path: &str) -> PyResult<()> {
        map_spicy_error(self.inner.load_par_df(hdb_path))?;
        Ok(())
    }

    fn clear_par_df(&self) -> PyResult<()> {
        map_spicy_error(self.inner.clear_par_df())?;
        Ok(())
    }
}

#[pymodule]
fn engine_state(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ChiliError", m.py().get_type::<ChiliError>())?;
    m.add_class::<PyEngineState>()?;
    Ok(())
}
