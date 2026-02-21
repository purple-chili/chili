use indexmap::IndexMap;
use polars::datatypes::Float64Type;
use polars::frame::DataFrame;
use polars::prelude::{
    ChunkCompareIneq, Column, DataType, Expr, IdxCa, IndexOrder, IntoLazy, NamedFrom, int_range,
    lit,
};
use polars::series::Series;

use crate::ast_node::AstNode;
use crate::ast_node::SourcePos;
use crate::errors::{SpicyError, SpicyResult, trace};
use crate::eval_query::eval_query;
use crate::func::Func;
use crate::obj::SpicyObj;
use crate::{ArgType, validate_args};
use crate::{Stack, engine_state::EngineState};

pub fn eval_fn_call(
    state: &EngineState,
    stack: &mut Stack,
    func: &Func,
    args: &Vec<&SpicyObj>,
) -> SpicyResult<SpicyObj> {
    let param_num: usize = args
        .iter()
        .map(|a| if a.is_delayed_arg() { 0 } else { 1 })
        .sum();

    if param_num == func.arg_num {
        let all_args = if let Some(part_args) = &func.part_args {
            let mut all_args: Vec<&SpicyObj> = part_args.iter().collect();
            for (i, j) in func.missing_index.iter().enumerate() {
                all_args[*j] = args[i]
            }
            all_args
        } else {
            args.clone()
        };
        if func.is_side_effect() {
            func.f_with_side_effect.as_ref().unwrap()(state, stack, &all_args)
        } else if func.is_built_in_fn() {
            let f = func
                .f
                .as_ref()
                .ok_or_else(|| SpicyError::EvalErr("built-in fn not found".to_owned()))?;
            f(&all_args)
        } else {
            let mut new_stack = Stack::new(
                stack.src_path.clone(),
                stack.stack_layer + 1,
                stack.h,
                &stack.user,
            );
            // TODO: review this limit
            if new_stack.stack_layer >= 37 {
                return Err(SpicyError::EvalErr("Stack overflow reached".to_owned()));
            }

            for (i, param) in func.params.iter().enumerate() {
                new_stack.set_var(param, all_args[i].clone())
            }

            new_stack.set_f(func);
            let mut r = SpicyObj::Null;
            for node in func.nodes.iter() {
                r = eval_by_node(state, &mut new_stack, node, &func.fn_body, None)?;
                if r.is_return() {
                    break;
                }
            }
            new_stack.unset_f();
            if let SpicyObj::Return(obj) = r {
                Ok(*obj)
            } else {
                Ok(r)
            }
        }
    } else {
        Ok(SpicyObj::Fn(func.project(args)))
    }
}

pub fn eval_index_assignment(
    left: SpicyObj,
    right: SpicyObj,
    indices: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    match left {
        SpicyObj::Dict(mut d) => {
            d.insert(indices[0].str()?.to_owned(), right);
            Ok(SpicyObj::Dict(d))
        }
        // J::Series(mut s) => todo!(),
        // J::DataFrame(df) => {}
        _ => Err(SpicyError::EvalErr(format!(
            "Not support index assignment for '{}'",
            left
        ))),
    }
}

pub fn eval_by_node(
    state: &EngineState,
    stack: &mut Stack,
    node: &AstNode,
    src: &str,
    columns: Option<&Vec<String>>,
) -> SpicyResult<SpicyObj> {
    match node {
        AstNode::UnaryExp { op: f, exp } => {
            let obj = eval_by_node(state, stack, f, src, columns)?;
            let exp = eval_by_node(state, stack, exp, src, columns)?;
            eval_call(state, stack, &obj, &vec![&exp], &f.get_pos(), src)
        }

        AstNode::BinaryExp { op: f2, lhs, rhs } => {
            let lhs = eval_by_node(state, stack, lhs, src, columns)?;
            let rhs = eval_by_node(state, stack, rhs, src, columns)?;
            let obj = eval_by_node(state, stack, f2, src, columns)?;
            eval_call(state, stack, &obj, &vec![&lhs, &rhs], &f2.get_pos(), src)
        }
        AstNode::AssignmentExp { id, exp } => {
            let obj = eval_by_node(state, stack, exp, src, columns)?;
            if stack.is_in_fn() && !id.starts_with(".") {
                stack.set_var(id, obj.clone());
            } else {
                state.set_var(id, obj.clone())?;
            }
            Ok(obj)
        }
        AstNode::IndexAssignmentExp { id, indices, exp } => {
            let left = state.get_var(id)?;
            let right = eval_by_node(state, stack, exp, src, columns)?;
            let indices = indices
                .iter()
                .map(|i| eval_by_node(state, stack, i, src, columns))
                .collect::<Result<Vec<SpicyObj>, SpicyError>>()?;
            let indices = indices.iter().collect::<Vec<&SpicyObj>>();
            let obj = eval_index_assignment(left, right, &indices)?;
            if stack.is_in_fn() && !id.starts_with(".") {
                stack.set_var(id, obj);
            } else {
                state.set_var(id, obj)?;
            }
            Ok(SpicyObj::Null)
        }
        AstNode::SpicyObj(obj) => Ok(obj.clone()),
        AstNode::Id { name, .. } => {
            if let Some(columns) = columns {
                if columns.contains(name) {
                    return Ok(SpicyObj::Expr(polars::prelude::col(name)));
                } else if name == "i" {
                    return Ok(SpicyObj::Expr(
                        int_range(lit(0), Expr::Len, 1, DataType::Int64).alias("i"),
                    ));
                }
            }

            if stack.is_in_fn()
                && let Ok(obj) = stack.get_var(name)
            {
                return Ok(obj);
            }
            if let Ok(obj) = state.get_var(name) {
                return Ok(obj);
            }
            if let Ok(p) = state.get_par_df(name) {
                Ok(SpicyObj::ParDataFrame(p))
            } else {
                Err(SpicyError::NameErr(name.to_owned()))
            }
        }
        AstNode::FnCall {
            pos: _, f, args, ..
        } => {
            let obj = eval_by_node(state, stack, f, src, columns)?;
            let args: Result<Vec<SpicyObj>, SpicyError> = args
                .iter()
                .map(|a| eval_by_node(state, stack, a, src, columns))
                .collect();
            let args = args?;
            let args: Vec<&SpicyObj> = args.iter().collect();
            eval_call(state, stack, &obj, &args, &f.get_pos(), src)
        }
        AstNode::DelayedArg => Ok(SpicyObj::DelayedArg),
        AstNode::ColExp { name, exp } => {
            let obj = eval_by_node(state, stack, exp, src, columns)?;
            if obj.is_expr() || columns.is_some() {
                Ok(SpicyObj::Expr(obj.as_expr()?.alias(name)))
            } else {
                let s = obj.as_series()?;
                Ok(SpicyObj::Series(s.clone().rename(name.into()).clone()))
            }
        }
        AstNode::DataFrame(nodes) => {
            let mut cols: Vec<Column> = Vec::with_capacity(nodes.len());
            for (i, node) in nodes.iter().enumerate() {
                let obj = match node {
                    AstNode::SpicyObj(obj) => obj.clone(),
                    _ => eval_by_node(state, stack, node, src, columns)?,
                };
                let mut column: Column = obj
                    .series()
                    .map(|s| s.clone().into())
                    .map_err(|e| SpicyError::EvalErr(format!("column {} - {}", i, e)))?;
                if column.name().is_empty() {
                    column.rename(format!("col{:02}", i).into());
                } else if let AstNode::Id { name, .. } = node {
                    column.rename(name.into());
                }
                cols.push(column)
            }
            let height = cols.first().map(|c| c.len()).unwrap_or(0);
            let df =
                DataFrame::new(height, cols).map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            Ok(SpicyObj::DataFrame(df))
        }
        AstNode::Matrix(nodes) => {
            let mut cols: Vec<Column> = Vec::with_capacity(nodes.len());
            for (i, node) in nodes.iter().enumerate() {
                let obj = match node {
                    AstNode::SpicyObj(obj) => obj.clone(),
                    _ => eval_by_node(state, stack, node, src, columns)?,
                };
                let s = obj
                    .series()
                    .cloned()
                    .map_err(|e| SpicyError::EvalErr(format!("column {} - {}", i, e)))?;
                if !(s.dtype().is_primitive_numeric() || s.dtype().is_bool()) {
                    return Err(SpicyError::Err(format!(
                        "Requires numeric data type, got '{}'",
                        obj.get_type_name()
                    )));
                }
                cols.push(s.into())
            }
            let height = cols.first().map(|c| c.len()).unwrap_or(0);
            let df =
                DataFrame::new(height, cols).map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            let matrix = df
                .to_ndarray::<Float64Type>(IndexOrder::C)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::Matrix(matrix.reversed_axes().to_shared()))
        }
        AstNode::List(nodes) => {
            let mut list = Vec::with_capacity(nodes.len());
            for node in nodes.iter() {
                list.push(eval_by_node(state, stack, node, src, columns)?)
            }
            let obj = SpicyObj::MixedList(list);
            match obj.unify_series() {
                Ok(s) => Ok(s),
                Err(_) => Ok(obj),
            }
        }
        AstNode::Query {
            op,
            op_exp,
            by_exp,
            from_exp,
            where_exp,
            limited_exp,
        } => {
            let obj = eval_by_node(state, stack, from_exp, src, columns)?;
            eval_query(
                state,
                stack,
                op,
                op_exp,
                by_exp,
                obj,
                where_exp,
                limited_exp,
                src,
            )
        }
        AstNode::Dict { keys, values } => {
            let mut m: IndexMap<String, SpicyObj> = IndexMap::new();
            for (i, value) in values.iter().enumerate() {
                m.insert(
                    keys[i].to_owned(),
                    eval_by_node(state, stack, value, src, columns)?,
                );
            }
            Ok(SpicyObj::Dict(m))
        }
        AstNode::If {
            cond,
            nodes,
            else_nodes,
        } => {
            let obj = eval_by_node(state, stack, cond, src, columns)?;
            if obj.is_truthy()? {
                for node in nodes {
                    let obj = eval_by_node(state, stack, node, src, columns)?;
                    if obj.is_return() {
                        return Ok(obj);
                    }
                }
            } else {
                let obj = eval_by_node(state, stack, else_nodes, src, columns)?;
                if obj.is_return() {
                    return Ok(obj);
                }
            }
            Ok(SpicyObj::Null)
        }
        AstNode::While { cond, nodes } => {
            let mut condition = eval_by_node(state, stack, cond, src, columns)?;
            while condition.is_truthy()? {
                for node in nodes {
                    let obj = eval_by_node(state, stack, node, src, columns)?;
                    if obj.is_return() {
                        return Ok(obj);
                    }
                }
                condition = eval_by_node(state, stack, cond, src, columns)?;
            }
            Ok(SpicyObj::Null)
        }
        AstNode::IfElse { nodes } => {
            if nodes.len() >= 3 && nodes.len() % 2 != 1 {
                return Err(SpicyError::EvalErr(
                    "if else nodes must be odd number".to_owned(),
                ));
            }
            // last node is for returning value
            for i in (0..nodes.len()).take(nodes.len() - 1).step_by(2) {
                let obj = eval_by_node(state, stack, &nodes[i], src, columns)?;
                if obj.is_truthy()? {
                    return eval_by_node(state, stack, &nodes[i + 1], src, columns);
                }
            }
            if nodes.len() % 2 == 0 {
                return Ok(SpicyObj::Null);
            }
            // return last node if no condition is truthy
            eval_by_node(state, stack, &nodes[nodes.len() - 1], src, columns)
        }
        AstNode::Try {
            tries,
            err_id,
            catches,
        } => {
            let mut is_err = false;
            for node in tries {
                match eval_by_node(state, stack, node, src, columns) {
                    Ok(obj) => {
                        if obj.is_return() {
                            return Ok(obj);
                        }
                    }
                    Err(e) => {
                        is_err = true;
                        if stack.is_in_fn() {
                            stack.set_var(err_id, SpicyObj::String(e.to_string()));
                        } else {
                            state.set_var(err_id, SpicyObj::String(e.to_string()))?;
                        }
                        break;
                    }
                }
            }
            if is_err {
                for node in catches {
                    let obj = eval_by_node(state, stack, node, src, columns)?;
                    if obj.is_return() {
                        return Ok(obj);
                    }
                }
            }
            Ok(SpicyObj::Null)
        }
        AstNode::Return(node) => {
            let obj = eval_by_node(state, stack, node, src, columns)?;
            Ok(SpicyObj::Return(Box::new(obj)))
        }
        AstNode::Raise(node) => {
            let obj = eval_by_node(state, stack, node, src, columns)?;
            Err(SpicyError::RaiseErr(obj.to_string()))
        }
        AstNode::ShortCircuit {
            op,
            left_cond,
            right_cond,
        } => {
            let left_cond = eval_by_node(state, stack, left_cond, src, columns)?;
            if op == "||" {
                if left_cond.is_truthy()? {
                    return Ok(SpicyObj::Boolean(true));
                } else {
                    let right_cond = eval_by_node(state, stack, right_cond, src, columns)?;
                    return Ok(SpicyObj::Boolean(right_cond.is_truthy()?));
                }
            }
            if op == "&&" {
                if !left_cond.is_truthy()? {
                    return Ok(SpicyObj::Boolean(false));
                } else {
                    let right_cond = eval_by_node(state, stack, right_cond, src, columns)?;
                    return Ok(SpicyObj::Boolean(right_cond.is_truthy()?));
                }
            }
            if op == "??" {
                if left_cond.is_null() {
                    let right_cond = eval_by_node(state, stack, right_cond, src, columns)?;
                    return Ok(right_cond);
                } else {
                    return Ok(left_cond);
                }
            }
            Ok(SpicyObj::Null)
        }
    }
}

// fn, list
pub fn eval_op(
    state: &EngineState,
    stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let args = match arg0 {
        SpicyObj::MixedList(list) => list,
        SpicyObj::Symbol(s) | SpicyObj::String(s) => {
            let any = stack.get_var(s);
            if any.is_ok() {
                return any;
            } else {
                return state.get_var(s);
            }
        }
        _ => return Err(SpicyError::EvalErr(format!("Not able to eval '{}'", arg0))),
    };
    let f = &args[0];
    let f = match f {
        SpicyObj::Symbol(s) | SpicyObj::String(s) => {
            let any = stack.get_var(s);
            if let Ok(obj) = any {
                obj
            } else {
                state.get_var(s)?
            }
        }
        _ => f.clone(),
    };
    let args = &args[1..].iter().collect();
    match &f {
        SpicyObj::Fn(func) => eval_call(state, stack, &f, args, &Some(func.pos.clone()), ""),
        SpicyObj::I64(h) => state.sync(h, args[0]),
        _ => Err(SpicyError::EvalErr(format!(
            "Not able to eval a list with first item '{}'",
            f
        ))),
    }
}

pub fn eval_for_console(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Str])?;
    let src = args[0].str()?;
    let ast = state
        .parse("", src)
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
    let obj = state.eval_ast(ast, "", src)?;
    Ok(SpicyObj::String(obj.to_string()))
}

pub fn eval_for_ide(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Str, ArgType::Int])?;
    let src = args[0].str()?;
    let limit = args[1].to_i64()?;
    if limit < 0 {
        return Err(SpicyError::EvalErr(
            "limit(2nd arg) must be positive".to_owned(),
        ));
    }
    let limit = limit as usize;
    let ast = state
        .parse("", src)
        .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
    let obj = state.eval_ast(ast, "", src)?;
    match obj {
        SpicyObj::DataFrame(df) => {
            let df = df.slice(0, limit);
            Ok(SpicyObj::DataFrame(df))
        }
        SpicyObj::Series(s) => Ok(SpicyObj::Series(s.slice(0, limit))),
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
        | SpicyObj::Symbol(_) => Ok(SpicyObj::Series(obj.into_series().unwrap())),
        SpicyObj::Expr(_)
        | SpicyObj::Err(_)
        | SpicyObj::Return(_)
        | SpicyObj::Null
        | SpicyObj::Fn(_)
        | SpicyObj::DelayedArg
        | SpicyObj::Matrix(_)
        | SpicyObj::LazyFrame(_)
        | SpicyObj::ParDataFrame(_) => Ok(SpicyObj::String(obj.to_string())),
        SpicyObj::MixedList(l) => {
            let s = l.iter().map(|args| args.to_string()).collect::<Vec<_>>();
            let series = Series::new("list".into(), s);
            let df = series.into_frame().lazy().with_row_index("index", None);
            Ok(SpicyObj::DataFrame(df.collect().unwrap().slice(0, limit)))
        }
        SpicyObj::Dict(index_map) => {
            let keys = index_map.keys().map(|k| k.to_string()).collect::<Vec<_>>();
            let values = index_map
                .values()
                .map(|v| v.to_string())
                .collect::<Vec<_>>();
            let keys = Series::new("keys".into(), keys);
            let values = Series::new("values".into(), values);
            let df = DataFrame::new(keys.len(), vec![keys.into(), values.into()])
                .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
            Ok(SpicyObj::DataFrame(df.slice(0, limit)))
        }
    }
}

pub fn eval_call(
    state: &EngineState,
    stack: &mut Stack,
    f: &SpicyObj,
    args: &Vec<&SpicyObj>,
    pos: &Option<SourcePos>,
    src: &str,
) -> SpicyResult<SpicyObj> {
    match f {
        SpicyObj::Fn(f) => {
            let func = if f.is_raw {
                if f.fn_body.ends_with("}") {
                    let nodes = state
                        .parse_raw_fn(&f.fn_body, f.lang)
                        .map_err(|e| SpicyError::Err(format!("failed to parse raw fn: {}", e)))?;

                    state.eval_ast(nodes, "", "")?
                } else {
                    state.get_var(f.fn_body.as_str())?
                }
            } else {
                SpicyObj::Null
            };

            let f = if !func.is_null() { func.fn_()? } else { f };

            if state.is_repl_use_chili_syntax() {
                // chili syntax requires arguments number to be matched, however, allow calling with empty(delayed arg) to represent no argument, hence create a partially applied function
                if args.len() != f.arg_num
                    && !(f.arg_num == 0
                        && args.len() == 1
                        && (*args[0] == SpicyObj::DelayedArg || args[0].size() == 0))
                {
                    return Err(SpicyError::MismatchedArgNumErr(f.arg_num, args.len()));
                }
            } else {
                // pepper syntax allows arguments number to be less to create a partially applied function
                if args.len() > f.arg_num
                    && !(f.arg_num == 0
                        && args.len() == 1
                        && (*args[0] == SpicyObj::DelayedArg || args[0].size() == 0))
                {
                    return Err(SpicyError::MismatchedArgNumErr(f.arg_num, args.len()));
                }
            }

            let res = if f.arg_num == 0 && args.len() == 1 && args[0].size() == 0 {
                eval_fn_call(state, stack, f, &vec![&SpicyObj::DelayedArg])
            } else {
                eval_fn_call(state, stack, f, args)
            };

            match res {
                Ok(obj) => Ok(obj),
                Err(e) => {
                    if pos.is_some() {
                        let p = pos.as_ref().unwrap();
                        if stack.is_in_fn() {
                            let current_fn = stack.get_f().unwrap();
                            Err(SpicyError::Err(trace(
                                &current_fn.fn_body,
                                "",
                                p.pos - current_fn.pos.pos,
                                &e.to_string(),
                            )))
                        } else if p.source_id == 0 {
                            Err(SpicyError::Err(trace(src, "", p.pos, &e.to_string())))
                        } else {
                            let (path, src) = state.get_source(p.source_id).unwrap();
                            Err(SpicyError::Err(trace(&src, &path, p.pos, &e.to_string())))
                        }
                    } else {
                        Err(e)
                    }
                }
            }
        }
        SpicyObj::Dict(_) => {
            if args.len() == 1 {
                at(&[f, args[0]])
            } else {
                Err(SpicyError::NotYetImplemented(format!(
                    "Unsupported multidimensional indices for dict, got '{}D' indices",
                    args.len()
                )))
            }
        }
        SpicyObj::DataFrame(_) => {
            if args.len() == 1 {
                at(&[f, args[0]])
            } else {
                Err(SpicyError::NotYetImplemented(format!(
                    "Unsupported multidimensional indices for dataframe, got '{}D' indices",
                    args.len()
                )))
            }
        }
        SpicyObj::I64(h) => state.sync(h, args[0]),
        SpicyObj::MixedList(list) => {
            if args.len() == 1 {
                let arg0 = args[0];
                match arg0.to_i64() {
                    Ok(i) => Ok(list.get(i as usize).unwrap_or(&SpicyObj::Null).clone()),
                    Err(_) => {
                        let indices = arg0.into_series()?;
                        if indices.dtype().is_integer() {
                            let indices = indices.cast(&DataType::Int64).unwrap();
                            let res = indices
                                .i64()
                                .unwrap()
                                .into_iter()
                                .map(|i| match i {
                                    Some(i) => {
                                        list.get(i as usize).unwrap_or(&SpicyObj::Null).clone()
                                    }
                                    None => SpicyObj::Null,
                                })
                                .collect::<Vec<_>>();
                            Ok(SpicyObj::MixedList(res))
                        } else {
                            Err(SpicyError::EvalErr("indices must be integer".to_owned()))
                        }
                    }
                }
            } else {
                Err(SpicyError::NotYetImplemented(format!(
                    "Unsupported multidimensional indices for mixed list, got '{}D' indices",
                    args.len()
                )))
            }
        }
        _ => Err(SpicyError::NotYetImplemented(format!(
            "fn call for '{}': {}",
            f.get_type_name(),
            f
        ))),
    }
}

// #Lbchij | efdtzpnsMDS
// LLaaaaa | ---------DL
// b------ | -----------
// c------ | -----------
// h------ | -----------
// i------ | -----------
// j------ | -----------
// e------ | -----------
// f------ | -----------
// d------ | -----------
// t------ | -----------
// z------ | -----------
// p------ | -----------
// n------ | -----------
// MMSSSSS | ----------M
// D------ | -------a--D
// SSaaaaa | ----------S
pub fn at(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let op = "@";
    let arg0 = args[0];
    let arg1 = args[1];
    if arg0.is_expr() || arg1.is_expr() {
        return Ok(SpicyObj::Expr(
            arg0.as_expr().unwrap().gather(arg1.as_expr().unwrap()),
        ));
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
    if c0 < 0 || arg1.is_temporal() {
        return Err(err());
    }
    match arg0 {
        SpicyObj::Series(s0) => {
            let s_len = s0.len() as i64;
            if arg1.is_integer() {
                let i = arg1.to_i64().unwrap();
                let i = if i < 0 { i + s_len } else { i };
                match s0.get(i as usize) {
                    Ok(a) => Ok(SpicyObj::from_any_value(a)),
                    Err(_) => Ok(SpicyObj::Null),
                }
            } else if c1 > 0 && c1 <= 5 {
                let indices = arg1.series().unwrap().cast(&DataType::Int64).unwrap();
                let indices = if indices.lt(0).unwrap().any() || indices.gt_eq(s_len).unwrap().any()
                {
                    indices
                        .i64()
                        .unwrap()
                        .iter()
                        .map(|i| {
                            if let Some(i) = i {
                                let i = if i < 0 { i + s_len } else { i };
                                if i < 0 || i > s_len { None } else { Some(i) }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    indices
                };
                let indices = indices.cast(&DataType::UInt32).unwrap();
                Ok(SpicyObj::Series(s0.take(indices.u32().unwrap()).unwrap()))
            } else {
                Err(err())
            }
        }
        SpicyObj::MixedList(l0) => {
            let l_len = l0.len() as i64;
            if arg1.is_integer() {
                let i = arg1.to_i64().unwrap();
                let i = if i < 0 { i + l_len } else { i };
                match l0.get(i as usize) {
                    Some(obj) => Ok(obj.clone()),
                    None => Ok(SpicyObj::Null),
                }
            } else if c1 > 0 && c1 <= 5 {
                let s1 = arg1.series().unwrap();
                let mut res = Vec::with_capacity(s1.len());
                s1.cast(&DataType::Int64)
                    .unwrap()
                    .i64()
                    .unwrap()
                    .iter()
                    .for_each(|i| match i {
                        Some(i) => {
                            let i = if i < 0 { i + l_len } else { i };
                            if i < 0 || i >= l_len {
                                res.push(SpicyObj::Null)
                            } else {
                                res.push(l0[i as usize].clone())
                            }
                        }
                        None => res.push(SpicyObj::Null),
                    });
                Ok(SpicyObj::MixedList(res))
            } else {
                Err(err())
            }
        }
        // J::Matrix(_) => todo!(),
        SpicyObj::Dict(d0) => match c1 {
            -13 | -14 => {
                let key = arg1.str().unwrap();
                Ok(d0.get(key).unwrap_or(&SpicyObj::Null).clone())
            }
            13 | 14 => {
                let s1 = arg1.series().unwrap();
                let mut res = Vec::with_capacity(s1.len());
                if c1 == 13 {
                    s1.str().unwrap().iter().for_each(|s| {
                        res.push(d0.get(s.unwrap_or("")).unwrap_or(&SpicyObj::Null).clone())
                    })
                } else {
                    let c1 = s1.cat32().unwrap();
                    c1.iter_str().for_each(|s| {
                        res.push(d0.get(s.unwrap_or("")).unwrap_or(&SpicyObj::Null).clone())
                    })
                };
                let res = SpicyObj::MixedList(res);
                match res.unify_series() {
                    Ok(obj) => Ok(obj),
                    Err(_) => Ok(res),
                }
            }
            _ => Err(err()),
        },
        SpicyObj::DataFrame(df0) => {
            let df_len = df0.height() as i64;
            match c1 {
                -5..=-1 => {
                    let i = arg1.to_i64().unwrap();
                    let i = if i < 0 { i + df_len } else { i };
                    let i = if i < 0 || i > df_len {
                        None
                    } else {
                        Some(i as u32)
                    };
                    let indices = IdxCa::new("idx".into(), &[i]);
                    Ok(SpicyObj::DataFrame(df0.take(&indices).unwrap()))
                }
                1..=5 => {
                    let indices = arg1.series().unwrap().cast(&DataType::Int64).unwrap();
                    let indices =
                        if indices.lt(0).unwrap().any() || indices.gt_eq(df_len).unwrap().any() {
                            indices
                                .i64()
                                .unwrap()
                                .iter()
                                .map(|i| {
                                    if let Some(i) = i {
                                        let i = if i < 0 { i + df_len } else { i };
                                        if i < 0 || i > df_len { None } else { Some(i) }
                                    } else {
                                        None
                                    }
                                })
                                .collect()
                        } else {
                            indices
                        };
                    let indices = indices.cast(&DataType::UInt32).unwrap();
                    Ok(SpicyObj::DataFrame(
                        df0.take(indices.u32().unwrap()).unwrap(),
                    ))
                }
                -13 | -14 => {
                    let column = arg1.str().unwrap();
                    df0.select([column])
                        .map_err(|e| SpicyError::Err(e.to_string()))
                        .map(|vec| SpicyObj::Series(vec[0].clone().take_materialized_series()))
                }
                13 | 14 => {
                    let s1 = arg1.series().unwrap();
                    let columns: Vec<String> = if c1 == 13 {
                        s1.str()
                            .unwrap()
                            .iter()
                            .map(|s| s.unwrap_or("").to_owned())
                            .collect()
                    } else {
                        s1.cat32()
                            .unwrap()
                            .iter_str()
                            .map(|s| s.unwrap_or("").to_owned())
                            .collect()
                    };
                    df0.select(columns)
                        .map_err(|e| SpicyError::Err(e.to_string()))
                        .map(SpicyObj::DataFrame)
                }
                _ => Err(err()),
            }
        }
        // not allow fn to use @, just use []
        _ => Err(err()),
    }
}
