use log::{info, warn};
use polars::prelude::{IntoLazy, SortMultipleOptions, SortOptions, col};
use polars::series::ops::NullBehavior;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::fs;
use std::sync::LazyLock;
use std::time::Instant;

use crate::errors::{SpicyError, SpicyResult};
use crate::eval::{eval_call, eval_fn_call, eval_for_console, eval_for_ide, eval_op};
use crate::func::Func;
use crate::utils::convert_list_to_df;
use crate::{ArgType, EngineState, SpicyObj, Stack, eval_query, job, validate_args};

fn time_it(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let times = args.last().unwrap();
    let times = times.to_i64().unwrap_or(1);
    let start = Instant::now();
    for _ in 0..times {
        eval_op(state, stack, &args[0..1])?;
    }
    Ok(SpicyObj::F64(start.elapsed().as_secs_f64()))
}

fn parallel(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let f = args[0];
    let vec = args[1].as_vec()?;
    let result = vec
        .par_iter()
        .map(|args| eval_op(state, &mut stack.clone(), &[f, args]))
        .collect::<Result<Vec<SpicyObj>, SpicyError>>()?;
    Ok(SpicyObj::MixedList(result))
}

fn load(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let path = args[0].str()?;
    state.load_par_df(path).map(|_| SpicyObj::Null)
}

fn open_handle(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let socket = args[0].str()?;
    state.open_handle(socket, 0)
}

fn close_handle(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let handle_num = args[0].to_i64()?;
    state.close_handle(&handle_num)
}

fn exit(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let exit_code = args[0].to_i64()?;
    state.shutdown();
    std::process::exit(exit_code as i32)
}

fn upsert(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Any, ArgType::DataFrameOrList])?;
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_sym() {
        let id = arg0.str().unwrap();
        state.upsert_var(id, arg1)
    } else if arg0.is_df() {
        let mut df = args[0].df().unwrap().clone();
        match arg1 {
            SpicyObj::DataFrame(df1) => {
                df.clone()
                    .extend(&df1)
                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                Ok(SpicyObj::DataFrame(df))
            }
            SpicyObj::MixedList(list) => {
                let df1 = convert_list_to_df(&list, &df)?;
                df.extend(&df1)
                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                Ok(SpicyObj::DataFrame(df))
            }
            _ => unreachable!(),
        }
    } else {
        Err(SpicyError::EvalErr(format!(
            "Expect data type 'sym' or 'df' for '1' argument , got '{}'.",
            arg0.get_type_name()
        )))
    }
}

fn insert(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[ArgType::Any, ArgType::StrLike, ArgType::DataFrameOrList],
    )?;
    let arg0 = args[0];
    let by = args[1].to_str_vec().unwrap();
    let value = args[2];
    if arg0.is_sym() {
        let id = arg0.str().unwrap();
        state.insert_var(id, value, &by)
    } else if arg0.is_df() {
        let mut df = arg0.df().unwrap().clone();
        let df = match value {
            SpicyObj::DataFrame(df1) => {
                df.extend(&df1)
                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                df
            }
            SpicyObj::MixedList(list) => {
                let df1 = convert_list_to_df(&list, &df)?;
                df.extend(&df1)
                    .map_err(|e| SpicyError::Err(e.to_string()))?;
                df
            }
            _ => unreachable!(),
        };
        let df = df
            .lazy()
            .group_by(by)
            .agg([col("*").last()])
            .collect()
            .map_err(|e| SpicyError::Err(e.to_string()))?;
        Ok(SpicyObj::DataFrame(df))
    } else {
        Err(SpicyError::EvalErr(format!(
            "Expect data type 'sym' or 'df' for '1' argument , got '{}'.",
            arg0.get_type_name()
        )))
    }
}

fn replay_q(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let file = args[0].str()?;
    let start = match args[1] {
        SpicyObj::Timestamp(t) => {
            let time_file = file.replace(".data", ".time");
            let time_bytes = fs::read(time_file).map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            if time_bytes.is_empty() {
                0
            } else {
                let length = time_bytes.len() / 8;
                let ptr = time_bytes.as_ptr() as *const u64;
                let times = unsafe { core::slice::from_raw_parts(ptr, length) };
                match times.binary_search(&(*t as u64)) {
                    Ok(index) => index as i64,
                    Err(index) => {
                        if index == 0 {
                            0
                        } else {
                            index as i64 - 1
                        }
                    }
                }
            }
        }
        _ => args[1].to_i64()?,
    };
    let end = args[2].to_i64()?;
    let table_names = args[3].to_str_vec()?;
    state.replay_q_msgs_log(file, start, end, &table_names)
}

// replay9[file; start; end; table_names; eval]
fn replay_chili(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let path = args[0].str()?;
    let mut start_time = 0;
    let start = match args[1] {
        SpicyObj::Timestamp(t) => {
            start_time = *t;
            0
        }
        SpicyObj::I64(t) => *t,
        _ => {
            return Err(SpicyError::EvalErr(format!(
                "expect timestamp or i64 for 'start' parameter, got '{}'",
                args[1].get_type_name()
            )));
        }
    };
    let end = args[2].to_i64()?;
    let table_names = args[3].to_str_vec()?;
    let eval = args[4].to_bool()?;
    state.replay_chili_msgs_log(path, start, end, start_time, &table_names, eval)
}

fn sub_q(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let socket = args[0];
    let table_names = args[1];
    let is_sub_all = table_names.size() == 0;
    let h: i64 = match socket {
        SpicyObj::Symbol(s) | SpicyObj::String(s) => *state.open_handle(s, 0)?.i64()?,
        SpicyObj::I64(h) => *h,
        _ => {
            return Err(SpicyError::EvalErr(format!(
                "expect symbol, string or i64, got '{}'",
                socket.get_type_name()
            )));
        }
    };
    // logFile, logCount, schemas
    let log_info = state.sync(
        &h,
        &SpicyObj::MixedList(vec![
            SpicyObj::Symbol("sub".to_owned()),
            table_names.clone(),
        ]),
    )?;
    let log_info = log_info.as_vec()?;
    info!("log_file: {}", log_info[0]);
    let schemas = log_info[2].clone();
    let table_names: Vec<&str> = schemas.dict()?.keys().map(|k| k.as_str()).collect();
    info!("subscribe to tables: {:?}", table_names);
    // define schema as global variables
    for (k, v) in schemas.dict()? {
        if state.has_var(k)? {
            warn!("variable '{}' already exists", k);
        } else {
            state.set_var(k, v.clone())?;
        }
    }

    let tick_count = state.get_tick_count();

    if is_sub_all {
        // only replay the log file from the previous tick count
        state.replay_q_msgs_log(
            log_info[0].str()?,
            tick_count,
            log_info[1].to_i64()?,
            &table_names,
        )?;
    } else {
        // replay the log file from the beginning
        state.replay_q_msgs_log(log_info[0].str()?, 0, log_info[1].to_i64()?, &table_names)?;
    }
    state.handle_publisher(&h)?;
    // return schemas
    Ok(schemas)
}

fn subscribing(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let handle = args[0].to_i64()?;
    state.handle_publisher(&handle).map(|_| SpicyObj::Null)
}

fn del(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let id = args[0].str()?;
    state.del_var(id)
}

fn tick(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let inc = args[0].to_i64()?;
    state.tick(inc)
}

fn set(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let id = args[0].str()?;
    let value = args[1];
    state.set_var(id, value.clone())?;
    Ok(args[0].clone())
}

fn get(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let id = args[0].str()?;
    state.get_var(id)
}

fn tables(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let start_with = args[0].str()?;
    state.get_table_names(start_with)
}

fn list_handle(
    state: &EngineState,
    _stack: &mut Stack,
    _args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let df = state.list_handle()?;
    Ok(SpicyObj::DataFrame(df))
}

fn each(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let f = args[0];
    let collection = args[1];
    match collection {
        SpicyObj::Dict(dict) => {
            let mut result = vec![];
            for (k, v) in dict.iter() {
                result.push(eval_call(
                    state,
                    stack,
                    f,
                    &vec![&SpicyObj::Symbol(k.clone()), v],
                    &None,
                    "",
                )?);
            }
            Ok(SpicyObj::MixedList(result))
        }
        SpicyObj::Series(_) => {
            let mut result = vec![];
            let list = collection.as_vec()?;
            for obj in list {
                result.push(eval_call(state, stack, f, &vec![&obj], &None, "")?);
            }
            let result = SpicyObj::MixedList(result);
            match result.unify_series() {
                Ok(obj) => Ok(obj),
                Err(_) => Ok(result),
            }
        }
        SpicyObj::Expr(expr) => {
            if let SpicyObj::Fn(func) = f {
                let expr = expr.clone();
                // TODO: support more fn with args = 2, e.g. @, #, join, shirt, head, tail
                // contains, count_matches,
                match func.fn_body.as_str() {
                    "sum" => Ok(SpicyObj::Expr(expr.list().sum())),
                    "any" => Ok(SpicyObj::Expr(expr.list().any())),
                    "all" => Ok(SpicyObj::Expr(expr.list().all())),
                    "count" => Ok(SpicyObj::Expr(expr.list().len())),
                    "max" => Ok(SpicyObj::Expr(expr.list().max())),
                    "min" => Ok(SpicyObj::Expr(expr.list().min())),
                    "mean" => Ok(SpicyObj::Expr(expr.list().mean())),
                    "median" => Ok(SpicyObj::Expr(expr.list().median())),
                    "std0" => Ok(SpicyObj::Expr(expr.list().std(0))),
                    "std1" => Ok(SpicyObj::Expr(expr.list().std(1))),
                    "var0" => Ok(SpicyObj::Expr(expr.list().var(0))),
                    "var1" => Ok(SpicyObj::Expr(expr.list().var(1))),
                    "asc" => Ok(SpicyObj::Expr(expr.list().sort(SortOptions::default()))),
                    "desc" => Ok(SpicyObj::Expr(
                        expr.list()
                            .sort(SortOptions::default().with_order_descending(true)),
                    )),
                    "unique" => Ok(SpicyObj::Expr(expr.list().unique_stable())),
                    "uc" => Ok(SpicyObj::Expr(expr.list().n_unique())),
                    "first" => Ok(SpicyObj::Expr(expr.list().first())),
                    "last" => Ok(SpicyObj::Expr(expr.list().last())),
                    "diff" => Ok(SpicyObj::Expr(expr.list().diff(1, NullBehavior::Ignore))),
                    _ => Err(SpicyError::Err(format!(
                        "unsupported aggregation: {}",
                        func.fn_body
                    ))),
                }
            } else {
                Err(SpicyError::Err(
                    "only support limited aggregations fn for list".to_owned(),
                ))
            }
        }
        _ => Err(SpicyError::NotYetImplemented(format!(
            "each for spicy object type '{}'",
            collection.get_type_name()
        ))),
    }
}

fn over(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let f = args[0];
    let lhs = args[1];
    let rhs = args[2];
    let err = SpicyError::EvalErr(
        "Unexpected 'over' iter exp, requires
                    - over[f1; times; var]
                    - over[f2; init; list]
                    - over[f2; f1; list]"
            .to_string(),
    );
    match f {
        SpicyObj::Fn(func) => {
            if func.arg_num == 1 && lhs.is_integer() {
                let n = lhs.to_i64().unwrap();
                if n < 0 {
                    Err(SpicyError::EvalErr(format!(
                        "Requires non-negative loop times, got {}",
                        n
                    )))
                } else {
                    let mut any = rhs.clone();
                    for _ in 0..n {
                        any = eval_fn_call(state, stack, func, &vec![&any])?;
                    }
                    Ok(any)
                }
            } else if func.arg_num == 2 {
                Err(SpicyError::NotYetImplemented("over for f2".to_owned()))
            } else {
                Err(err)
            }
        }
        _ => Err(SpicyError::EvalErr(format!(
            "unexpected spicy object type for binary iter exp: {}",
            f.get_type_name()
        ))),
    }
}

fn scan(_state: &EngineState, _stack: &mut Stack, _args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    Err(SpicyError::NotYetImplemented("scan".to_owned()))
}

fn import(state: &EngineState, stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let path = args[0].str()?;
    let base_path = stack.get_base_path().unwrap_or_default();
    state.import_source_path(&base_path, path)
}

fn set_callback(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let handle = args[0].to_i64()?;
    let callback = args[1].str()?;
    state.set_callback(&handle, callback.to_owned())?;
    Ok(SpicyObj::Null)
}

fn connect_to_handle(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let handle = args[0].to_i64()?;
    state.open_handle("", handle)
}

fn list(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let pattern = args[0].str()?;
    let df = state.list_vars(pattern)?;
    let df = df.sort(["name"], SortMultipleOptions::default()).unwrap();
    Ok(SpicyObj::DataFrame(df))
}

fn partition(state: &EngineState, _stack: &mut Stack, args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let name = args[0].str()?;
    state.get_par_df(name).map(SpicyObj::ParDataFrame)
}

pub static SIDE_EFFECT_FN: LazyLock<HashMap<String, Func>> = LazyLock::new(|| {
    [
        (
            "eval".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(eval_op)), 1, "eval", &["fn_args"]),
        ),
        (
            "evalc".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eval_for_console)),
                1,
                "evalc",
                &["string"],
            ),
        ),
        (
            // eval for IDE
            "evali".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eval_for_ide)),
                2,
                "evali",
                &["string", "limit_num"],
            ),
        ),
        (
            "timeit".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(time_it)),
                2,
                "timeit",
                &["fn_args", "times"],
            ),
        ),
        (
            "parallel".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(parallel)),
                2,
                "parallel",
                &["f", "collection"],
            ),
        ),
        (
            "load".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(load)), 1, "load", &["hdb_path"]),
        ),
        (
            "exit".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(exit)), 1, "exit", &["exit_code"]),
        ),
        (
            "upsert".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(upsert)),
                2,
                "upsert",
                &["id", "value"],
            ),
        ),
        (
            "insert".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(insert)),
                3,
                "insert",
                &["id", "by", "df"],
            ),
        ),
        (
            "replay_q".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(replay_q)),
                4,
                "replay_q",
                &["file", "start", "end", "table_names"],
            ),
        ),
        (
            "replay".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(replay_chili)),
                5,
                "replay",
                &["file", "start", "end", "table_names", "eval"],
            ),
        ),
        (
            "sub_q".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(sub_q)),
                2,
                "sub_q",
                &["socket", "table_names"],
            ),
        ),
        (
            "del".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(del)), 1, "del", &["id"]),
        ),
        (
            "tick".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(tick)), 1, "tick", &["inc"]),
        ),
        (
            "set".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(set)), 2, "set", &["id", "value"]),
        ),
        (
            "get".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(get)), 1, "get", &["id"]),
        ),
        (
            "tables".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(tables)), 1, "tables", &["start_with"]),
        ),
        (
            "each".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(each)),
                2,
                "each",
                &["f", "collection"],
            ),
        ),
        (
            "over".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(over)),
                3,
                "over",
                &["f", "init_value", "collection"],
            ),
        ),
        (
            "scan".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(scan)),
                3,
                "scan",
                &["f", "init_value", "collection"],
            ),
        ),
        (
            "import".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(import)), 1, "import", &["path"]),
        ),
        (
            "list".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(list)), 1, "list", &["pattern"]),
        ),
        (
            ".handle.list".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(list_handle)), 0, ".handle.list", &[]),
        ),
        (
            ".handle.open".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(open_handle)),
                1,
                ".handle.open",
                &["socket"],
            ),
        ),
        (
            ".handle.close".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(close_handle)),
                1,
                ".handle.close",
                &["handle_num"],
            ),
        ),
        (
            ".handle.onDisconnected".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(set_callback)),
                2,
                ".handle.onDisconnected",
                &["handle_num", "callback"],
            ),
        ),
        (
            ".handle.connect".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(connect_to_handle)),
                1,
                ".handle.connect",
                &["handle_num"],
            ),
        ),
        (
            ".handle.subscribing".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(subscribing)),
                1,
                ".handle.subscribing",
                &["handle"],
            ),
        ),
        (
            ".fn.select".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eval_query::functional_select)),
                6,
                ".fn.select",
                &[
                    "table",
                    "partitions",
                    "where",
                    "group",
                    "operations",
                    "limit",
                ],
            ),
        ),
        (
            ".fn.update".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eval_query::functional_update)),
                4,
                ".fn.update",
                &["table", "where", "group", "operations"],
            ),
        ),
        (
            ".fn.delete".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(eval_query::functional_delete)),
                3,
                ".fn.delete",
                &["table", "where", "columns"],
            ),
        ),
        (
            ".job.list".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(job::list)), 0, ".job.list", &[]),
        ),
        (
            ".job.add".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::add)),
                5,
                ".job.add",
                &[
                    "fn_name",
                    "start_time",
                    "end_time",
                    "interval",
                    "description",
                ],
            ),
        ),
        (
            ".job.addAfter".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::add_after)),
                3,
                ".job.add_after",
                &["fn_name", "interval", "description"],
            ),
        ),
        (
            ".job.addAtTime".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::add_at_time)),
                3,
                ".job.addAtTime",
                &["fn_name", "start_time", "description"],
            ),
        ),
        (
            ".job.get".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::get)),
                1,
                ".job.get",
                &["idOrPattern"],
            ),
        ),
        (
            ".job.activate".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::activate)),
                1,
                ".job.activate",
                &["idOrPattern"],
            ),
        ),
        (
            ".job.deactivate".to_owned(),
            Func::new_side_effect_built_in_fn(
                Some(Box::new(job::deactivate)),
                1,
                ".job.deactivate",
                &["idOrPattern"],
            ),
        ),
        (
            ".job.clear".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(job::clear)), 0, ".job.clear", &[]),
        ),
        (
            "par".to_owned(),
            Func::new_side_effect_built_in_fn(Some(Box::new(partition)), 1, "par", &["name"]),
        ),
    ]
    .into_iter()
    .collect()
});
