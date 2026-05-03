//! Python bindings for [`chili_core::EngineState`].
//!
//! This module exposes the Chili evaluation engine to Python via PyO3.
//! It handles bidirectional conversion between Python and Chili types:
//!
//! | Python type            | Chili type        |
//! |------------------------|-------------------|
//! | `bool`                 | `Boolean`         |
//! | `int`                  | `I64`             |
//! | `float`                | `F64`             |
//! | `str`                  | `Symbol`          |
//! | `bytes`                | `String`          |
//! | `datetime.date`        | `Date`            |
//! | `datetime.time`        | `Time`            |
//! | `datetime.datetime`    | `Timestamp`       |
//! | `datetime.timedelta`   | `Duration`        |
//! | `dict`                 | `Dict`            |
//! | `list` / `tuple`       | `MixedList`       |
//! | `polars.Series`        | `Series`          |
//! | `polars.DataFrame`     | `DataFrame`       |
//! | `polars.LazyFrame`     | `LazyFrame`       |
//! | `None`                 | `Null`            |

use std::process;
use std::sync::Arc;

use chili_core::constant::{NS_IN_DAY, UNIX_EPOCH_DAY};
use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use chrono::{DateTime, Datelike, Duration, NaiveDate, NaiveTime, Timelike, Utc};
use indexmap::IndexMap;
use polars::frame::DataFrame;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDate, PyDateTime, PyDelta, PyDict, PyFloat, PyInt, PyList, PyString, PyTime,
    PyTuple, PyTzInfo,
};
use pyo3::{create_exception, intern};
use pyo3_polars::{PyDataFrame, PyLazyFrame, PySeries};

// ---------------------------------------------------------------------------
// Exception hierarchy (all inherit from `ChiliError` → `RuntimeError`)
// ---------------------------------------------------------------------------
create_exception!(chili, ChiliError, PyRuntimeError);
create_exception!(chili, ChiliParseError, ChiliError);
create_exception!(chili, ChiliEvalError, ChiliError);
create_exception!(chili, PartitionError, ChiliError);
create_exception!(chili, TypeMismatchError, ChiliError);
create_exception!(chili, NameError, ChiliError);
create_exception!(chili, SerializationError, ChiliError);

/// Convert a [`chili_core::SpicyError`] into the most specific Python exception.
fn spicy_error_to_pyerr(err: &chili_core::SpicyError) -> PyErr {
    use chili_core::SpicyError;
    match err {
        SpicyError::ParserErr(_) => ChiliParseError::new_err(err.to_string()),
        SpicyError::MissingParCondErr(_) => PartitionError::new_err(err.to_string()),
        SpicyError::EvalErr(_) => ChiliEvalError::new_err(err.to_string()),
        SpicyError::NameErr(_) => NameError::new_err(err.to_string()),
        SpicyError::MismatchedTypeErr(..)
        | SpicyError::MismatchedArgTypeErr(..)
        | SpicyError::MismatchedArgNumErr(..)
        | SpicyError::MismatchedArgNumFnErr(..) => TypeMismatchError::new_err(err.to_string()),
        SpicyError::NotAbleToSerializeErr(_)
        | SpicyError::DeserializationErr(_)
        | SpicyError::NotAbleToDeserializeErr(_) => SerializationError::new_err(err.to_string()),
        _ => ChiliError::new_err(err.to_string()),
    }
}

/// Convenience wrapper: map a `SpicyResult<T>` to a `PyResult<T>`.
fn map_spicy_error<T>(r: Result<T, chili_core::SpicyError>) -> PyResult<T> {
    r.map_err(|e| spicy_error_to_pyerr(&e))
}

/// Recursively strip `Return` wrappers so the inner value can be converted.
fn unwrap_return(mut o: SpicyObj) -> SpicyObj {
    while let SpicyObj::Return(inner) = o {
        o = *inner;
    }
    o
}

/// Convert a Python object into the equivalent [`SpicyObj`].
///
/// Order matters: `PyBool` must be checked before `PyInt` (bool is a
/// subclass of int in Python), and `PyDateTime` before `PyDate`.
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

/// Convert a [`SpicyObj`] into the equivalent Python object.
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
        SpicyObj::LazyFrame(lf) => Ok(PyLazyFrame(lf).into_pyobject(py)?.into_any().unbind()),
        other => Ok(other.to_string().into_pyobject(py)?.into_any().unbind()),
    }
}

/// Chili evaluation engine; mirrors Rust ``chili_core::EngineState``.
#[pyclass(name = "EngineState")]
struct PyEngineState {
    inner: Arc<EngineState>,
    init_pid: u32,
}

impl PyEngineState {
    /// Guard against use in a forked child process.
    ///
    /// Rust mutexes / rwlocks are **not** fork-safe.  If the PID has
    /// changed since construction we refuse to operate and suggest
    /// `multiprocessing.get_context('spawn')`.
    fn check_fork(&self) -> PyResult<()> {
        let current_pid = process::id();
        if self.init_pid != current_pid {
            return Err(PyRuntimeError::new_err(format!(
                "EngineState is not fork-safe. Created in PID {} but running in PID {}. \
                 Use multiprocessing.get_context('spawn') instead of 'fork', \
                 or create a new EngineState in each child process.",
                self.init_pid,
                std::process::id(),
            )));
        }
        Ok(())
    }
}

#[pymethods]
impl PyEngineState {
    /// Create a new engine.
    ///
    /// * `debug`  – enable debug-level logging.
    /// * `lazy`   – enable lazy evaluation mode.
    /// * `pepper` – use Pepper syntax instead of Chili.
    /// * `job_interval` – job scheduler polling interval in milliseconds (0 = disabled).
    /// * `memory_limit` – memory limit in MB (0 = unlimited).
    #[new]
    #[pyo3(signature = (debug=false, lazy=false, pepper=false, job_interval=0, memory_limit=0.0))]
    fn new(
        debug: bool,
        lazy: bool,
        pepper: bool,
        job_interval: u64,
        memory_limit: f64,
    ) -> PyResult<Self> {
        let mut state = EngineState::new(debug, lazy, pepper);
        if job_interval > 0 {
            state.set_interval(job_interval);
        }
        if memory_limit > 0.0 {
            state.set_memory_limit(memory_limit);
        }
        state.register_fn(&BUILT_IN_FN);
        let arc = Arc::new(state);
        map_spicy_error(arc.set_arc_self(Arc::clone(&arc)))?;
        arc.start_job_scheduler();
        arc.start_memory_monitor();
        Ok(Self {
            inner: arc,
            init_pid: process::id(),
        })
    }

    /// Evaluate a Chili or Pepper expression string (same as the REPL).
    fn eval(&self, py: Python<'_>, source: &str, src_path: &str) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let obj = py.detach(move || {
            let mut stack = Stack::new(None, 0, 0, "");
            let args = SpicyObj::String(source.to_string());
            map_spicy_error(self.inner.eval(&mut stack, &args, src_path))
        });
        spicy_to_py(py, obj?)
    }

    /// Retrieve a variable by name, converted to a Python object.
    fn get_var(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let obj = py.detach(move || map_spicy_error(self.inner.get_var(id)));
        spicy_to_py(py, obj?)
    }

    /// Set or overwrite a variable in the engine.
    fn set_var(&self, id: &str, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.check_fork()?;
        let obj = spicy_from_py_bound(&value)?;
        map_spicy_error(self.inner.set_var(id, obj))
    }

    /// Return `True` if the variable exists.
    fn has_var(&self, id: &str) -> PyResult<bool> {
        self.check_fork()?;
        map_spicy_error(self.inner.has_var(id))
    }

    /// Delete a variable and return its last value.
    fn del_var(&self, py: Python<'_>, id: &str) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let obj = map_spicy_error(self.inner.del_var(id))?;
        spicy_to_py(py, obj)
    }

    /// Append rows to an existing DataFrame variable, or create it.
    ///
    /// Returns the number of rows appended.
    fn upsert(&self, id: &str, value: Bound<'_, PyAny>) -> PyResult<i64> {
        self.check_fork()?;
        let obj = spicy_from_py_bound(&value)?;
        let obj = map_spicy_error(self.inner.upsert_var(id, &obj));
        Ok(obj.map(|o| *o.i64().unwrap_or(&0i64))?)
    }

    /// Insert rows into a DataFrame variable, deduplicating by `by` columns.
    ///
    /// Returns the net change in row count.
    fn insert(&self, id: &str, value: Bound<'_, PyAny>, by: Vec<String>) -> PyResult<i64> {
        self.check_fork()?;
        let obj = spicy_from_py_bound(&value)?;
        let by = by.iter().map(|s| s.as_str()).collect::<Vec<&str>>();
        let obj = map_spicy_error(self.inner.insert_var(id, &obj, &by));
        Ok(obj.map(|o| *o.i64().unwrap_or(&0i64))?)
    }

    /// Return engine statistics as a Python dict.
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let stats = map_spicy_error(self.inner.stats())?;
        spicy_to_py(py, stats)
    }

    /// Import and evaluate a source file from the filesystem.
    fn import_source_path(
        &self,
        py: Python<'_>,
        relative: &str,
        path: &str,
    ) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let obj = py.detach(move || map_spicy_error(self.inner.import_source_path(relative, path)));
        spicy_to_py(py, obj?)
    }

    /// Register an in-memory source string under the given logical path.
    fn set_source(&self, path: &str, src: &str) -> PyResult<usize> {
        self.check_fork()?;
        map_spicy_error(self.inner.set_source(path, src))
    }

    /// Retrieve a registered source by index as a `(path, source)` tuple.
    fn get_source(&self, index: usize) -> PyResult<(String, String)> {
        self.check_fork()?;
        map_spicy_error(self.inner.get_source(index))
    }

    /// Shut down the engine and release all IPC handles.
    fn shutdown(&self) {
        // Don't error on shutdown in forked children; just attempt cleanup.
        self.inner.shutdown();
    }

    /// Return a `{name: display_string}` dict for all variables.
    fn get_displayed_vars(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let m = map_spicy_error(self.inner.get_displayed_vars())?;
        let d = PyDict::new(py);
        for (k, v) in m {
            d.set_item(k, v)?;
        }
        Ok(d.into_any().unbind())
    }

    /// List variables as a `polars.DataFrame` filtered by name prefix.
    fn list_vars(&self, py: Python<'_>, pattern: &str) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let df: DataFrame = map_spicy_error(self.inner.list_vars(pattern))?;
        Ok(PyDataFrame(df).into_pyobject(py)?.into_any().unbind())
    }

    /// Return the number of entries in the LRU parse cache.
    fn parse_cache_len(&self) -> usize {
        // Safe to read, but keep behavior consistent.
        let _ = self.check_fork();
        self.inner.parse_cache_len()
    }

    /// Return the tick counter at the given index (default 0).
    #[pyo3(signature = (index=0))]
    fn get_tick_count(&self, index: usize) -> PyResult<i64> {
        self.check_fork()?;
        map_spicy_error(self.inner.get_tick_count(index))
    }

    /// Increment the tick counter at `index` by `inc` and return the updated value.
    #[pyo3(signature = (index=0, inc=1))]
    fn tick(&self, py: Python<'_>, index: usize, inc: i64) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let obj = map_spicy_error(self.inner.tick(index, inc))?;
        spicy_to_py(py, obj)
    }

    /// Return `True` if lazy evaluation mode is enabled.
    fn is_lazy_mode(&self) -> bool {
        let _ = self.check_fork();
        self.inner.is_lazy_mode()
    }

    /// Return `True` if the engine uses Chili syntax (not Pepper).
    fn is_repl_use_chili_syntax(&self) -> bool {
        let _ = self.check_fork();
        self.inner.is_repl_use_chili_syntax()
    }

    /// Call a registered engine function by name with positional arguments.
    fn fn_call(&self, py: Python<'_>, func: &str, args: Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
        self.check_fork()?;
        let args = args
            .iter()
            .map(|a| spicy_from_py_bound(&a))
            .collect::<Result<Vec<SpicyObj>, PyErr>>()?;
        let args = args.iter().collect::<Vec<&SpicyObj>>();
        let obj = py.detach(move || map_spicy_error(self.inner.fn_call(func, &args)));
        spicy_to_py(py, obj?)
    }

    /// Load a partitioned database from the given directory.
    fn load_par_df(&self, hdb_path: &str) -> PyResult<()> {
        self.check_fork()?;
        map_spicy_error(self.inner.load_par_df(hdb_path))?;
        Ok(())
    }

    /// Remove all loaded partitioned DataFrames from memory.
    fn clear_par_df(&self) -> PyResult<()> {
        self.check_fork()?;
        map_spicy_error(self.inner.clear_par_df())?;
        Ok(())
    }

    /// Start a TCP listener on the given port in a background thread.
    ///
    /// The listener runs until the process exits.  The GIL is released so
    /// the calling Python thread is not blocked.
    #[pyo3(signature = (port, remote=false, users=vec![]))]
    fn start_tcp_listener(&self, port: i32, remote: bool, users: Vec<String>) -> PyResult<()> {
        self.check_fork()?;
        let state = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            state.start_tcp_listener(port, remote, users);
        });
        Ok(())
    }

    /// Return a DataFrame listing all active handles.
    fn list_handle(&self) -> PyResult<PyDataFrame> {
        self.check_fork()?;
        let df = map_spicy_error(self.inner.list_handle())?;
        let pydf = PyDataFrame(df);
        Ok(pydf)
    }
}

/// PyO3 module entry point — registers the `EngineState` class and
/// all custom exception types.
#[pymodule]
fn engine_state(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ChiliError", m.py().get_type::<ChiliError>())?;
    m.add("ChiliParseError", m.py().get_type::<ChiliParseError>())?;
    m.add("ChiliEvalError", m.py().get_type::<ChiliEvalError>())?;
    m.add("PartitionError", m.py().get_type::<PartitionError>())?;
    m.add("TypeMismatchError", m.py().get_type::<TypeMismatchError>())?;
    m.add("NameError", m.py().get_type::<NameError>())?;
    m.add(
        "SerializationError",
        m.py().get_type::<SerializationError>(),
    )?;
    m.add_class::<PyEngineState>()?;
    Ok(())
}
