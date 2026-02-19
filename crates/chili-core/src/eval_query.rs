use polars::lazy::dsl::col;
use polars::lazy::frame::IntoLazy;
use polars::prelude::Expr;
use polars::prelude::LazyFrame;
use polars::prelude::Selector;
use polars::prelude::lit;
use polars::prelude::when;

use crate::ast_node::AstNode;
use crate::ast_node::QueryOp;
use crate::errors::{SpicyError, SpicyResult};
use crate::eval::eval_by_node;
use crate::obj::SpicyObj;
use crate::par_df::DFType;
use crate::{Stack, engine_state::EngineState};

#[allow(clippy::too_many_arguments)]
pub fn eval_query(
    state: &EngineState,
    stack: &mut Stack,
    op: &QueryOp,
    op_exp: &[AstNode],
    by_exp: &[AstNode],
    args: SpicyObj,
    where_exp: &[AstNode],
    limited_exp: &Option<Box<AstNode>>,
    src: &str,
) -> SpicyResult<SpicyObj> {
    let mut skip_part_clause = false;
    let mut partitions = vec![];
    if let SpicyObj::ParDataFrame(par_df) = &args
        && !where_exp.is_empty()
        && let AstNode::BinaryExp { f2, lhs, rhs } = where_exp[0].clone()
    {
        skip_part_clause = true;
        if let AstNode::Id { name: par_unit, .. } = *lhs
            && ((par_df.df_type == DFType::ByDate && par_unit == "date")
                || (par_df.df_type == DFType::ByYear && par_unit == "year"))
        {
            let par = eval_by_node(state, stack, &rhs, src, None)?;
            if let AstNode::Id { name: op, .. } = *f2 {
                match op.as_str() {
                    "=" => partitions = vec![par.to_par_num()?],
                    ">=" | "<=" | ">" | "<" | "within" => {
                        let mut start = 0;
                        let mut end = -1;
                        if op == ">=" {
                            start = par.to_i64()? as i32;
                            end = i32::MAX;
                        } else if op == "<=" {
                            start = 0;
                            end = par.to_i64()? as i32;
                        } else if op == ">" {
                            start = 1 + par.to_i64()? as i32;
                            end = i32::MAX;
                        } else if op == "<" {
                            start = 0;
                            end = (par.to_i64()? as i32) - 1;
                        } else {
                            let pars = par.as_vec()?;
                            if pars.len() == 2 {
                                start = pars[0].to_i64()? as i32;
                                end = pars[1].to_i64()? as i32;
                            }
                        }
                        partitions = vec![start, end]
                    }
                    "in" => {
                        let pars = par
                            .as_vec()?
                            .iter()
                            .map(|p| p.to_i64())
                            .collect::<SpicyResult<Vec<_>>>()?;
                        partitions = pars.into_iter().map(|p| p as i32).collect::<Vec<i32>>();
                        if partitions.len() == 2 {
                            partitions.push(partitions[1]);
                        }
                    }
                    _ => {
                        return Err(SpicyError::Err(format!(
                            "Not support binary op '{}' on partition column",
                            op
                        )));
                    }
                }
            } else {
                return Err(SpicyError::Err(format!(
                    "mismatch partition unit '{}' with partition type '{:?}'",
                    par_unit, par_df.df_type
                )));
            }
        }
    }

    let limited = match limited_exp {
        Some(limited_exp) => eval_by_node(state, stack, limited_exp, src, None)?,
        None => SpicyObj::I64(0),
    };

    let (lf, columns) = make_it_lazy(&args, &partitions)?;

    let where_expr: Vec<SpicyObj> = where_exp
        .iter()
        .skip(if skip_part_clause { 1 } else { 0 })
        .map(|n| eval_by_node(state, stack, n, src, Some(&columns)))
        .map(|args| args.map_err(|e| SpicyError::EvalErr(e.to_string())))
        .collect::<Result<Vec<SpicyObj>, SpicyError>>()?;

    let op_expr: Vec<SpicyObj> = op_exp
        .iter()
        .map(|n| eval_by_node(state, stack, n, src, Some(&columns)))
        .collect::<Result<Vec<SpicyObj>, SpicyError>>()?;

    let by_expr: Vec<SpicyObj> = by_exp
        .iter()
        .map(|n| eval_by_node(state, stack, n, src, Some(&columns)))
        .collect::<Result<Vec<SpicyObj>, SpicyError>>()?;

    eval_fn_query(
        lf,
        op,
        &SpicyObj::MixedList(where_expr),
        &SpicyObj::MixedList(by_expr),
        &SpicyObj::MixedList(op_expr),
        &limited,
        &columns,
        state.is_lazy_mode(),
    )
}

pub fn eval_fn_query(
    mut lf: LazyFrame,
    query_op: &QueryOp,
    filter: &SpicyObj,
    group_by: &SpicyObj,
    op: &SpicyObj,
    limited: &SpicyObj,
    columns: &[String],
    is_lazy_mode: bool,
) -> SpicyResult<SpicyObj> {
    let where_exprs = filter
        .as_exprs()
        .map_err(|e| SpicyError::EvalErr(format!("where clause are not expressions, {}", e)))?;

    let where_exprs_len = where_exprs.len();

    if *query_op == QueryOp::Delete {
        if op.size() > 0 && where_exprs_len > 0 {
            return Err(SpicyError::EvalErr(
                "not support delete with both where clause and columns".to_owned(),
            ));
        }
        for exp in where_exprs.iter() {
            lf = lf.filter(exp.clone().not())
        }
    } else if *query_op == QueryOp::Select {
        for exp in where_exprs.iter() {
            lf = lf.filter(exp.clone())
        }
    }

    let op_exprs = op
        .as_exprs()
        .map_err(|e| SpicyError::EvalErr(format!("op expression are not expressions, {}", e)))?;

    let group_by_exprs = group_by.as_exprs().map_err(|e| {
        SpicyError::EvalErr(format!("group by expression are not expressions, {}", e))
    })?;

    if *query_op == QueryOp::Delete {
        let columns: Vec<&str> = op
            .to_str_vec()
            .map_err(|e| SpicyError::EvalErr(format!("requires columns(str) for delete, {}", e)))?;
        if columns.is_empty() && where_exprs_len == 0 {
            return lf
                .filter(lit(false))
                .collect()
                .map(SpicyObj::DataFrame)
                .map_err(|e| SpicyError::EvalErr(e.to_string()));
        } else if !columns.is_empty() {
            let selector = Selector::ByName {
                names: columns.into_iter().map(|c| c.into()).collect(),
                strict: true,
            };
            lf = lf.drop(selector);
            return lf
                .collect()
                .map(SpicyObj::DataFrame)
                .map_err(|e| SpicyError::EvalErr(e.to_string()));
        }
    } else {
        // update by => with_columns col("abc").over(partition_by);
        if *query_op == QueryOp::Select {
            if group_by.size() > 0 {
                if op.size() == 0 {
                    lf = lf.group_by_stable(group_by_exprs).agg(&[col("*").last()]);
                } else {
                    lf = lf.group_by_stable(group_by_exprs).agg(op_exprs);
                }
            } else if op_exprs.is_empty() {
                lf = lf.select(&[col("*")]);
            } else {
                lf = lf.select(op_exprs);
            }

            let limited_num = limited.to_i64().map_err(|e| {
                SpicyError::EvalErr(format!("limited expression must be a number, {}", e))
            })?;

            if limited_num > 0 {
                lf = lf.limit(limited_num as u32);
            } else if limited_num < 0 {
                lf = lf.tail(limited_num.unsigned_abs() as u32);
            }
        } else if *query_op == QueryOp::Update {
            let op_expr = if !group_by_exprs.is_empty() {
                op_exprs
                    .into_iter()
                    .map(|op| {
                        if let Expr::Alias(_, name) = &op {
                            // add another alias so that the update can handle "by columns"
                            op.clone().over(group_by_exprs.clone()).alias(name.clone())
                        } else {
                            op.over(group_by_exprs.clone())
                        }
                    })
                    .collect()
            } else {
                op_exprs
            };
            if where_exprs_len > 0 {
                let where_exp = where_exprs.into_iter().reduce(|a, b| a.and(b)).unwrap();
                lf = lf.with_columns(
                    op_expr
                        .into_iter()
                        .map(|op| {
                            let otherwise = if let Expr::Alias(_, name) = &op {
                                if columns.contains(&name.to_string()) {
                                    col(name.clone())
                                } else {
                                    SpicyObj::Null.as_expr().unwrap()
                                }
                            } else {
                                SpicyObj::Null.as_expr().unwrap()
                            };

                            when(where_exp.clone()).then(op).otherwise(otherwise)
                        })
                        .collect::<Vec<Expr>>(),
                )
            } else {
                lf = lf.with_columns(op_expr)
            }
        } else {
            return Err(SpicyError::NotYetImplemented(
                "eval query 'exec'".to_owned(),
            ));
        };
    }

    if is_lazy_mode {
        return Ok(SpicyObj::LazyFrame(lf));
    } else {
        lf.collect()
            .map(SpicyObj::DataFrame)
            .map_err(|e| SpicyError::EvalErr(e.to_string()))
    }
}

fn get_dataframe(state: &EngineState, args: &SpicyObj) -> SpicyResult<SpicyObj> {
    match args {
        SpicyObj::DataFrame(_) => Ok(args.clone()),
        SpicyObj::ParDataFrame(_) => Ok(args.clone()),
        SpicyObj::String(s) | SpicyObj::Symbol(s) => {
            if let Ok(obj) = state.get_var(s) {
                Ok(obj)
            } else {
                Ok(SpicyObj::ParDataFrame(state.get_par_df(s)?))
            }
        }
        _ => Err(SpicyError::EvalErr(format!(
            "not support query on type '{}'",
            args.get_type_name()
        ))),
    }
}

fn make_it_lazy(args: &SpicyObj, partitions: &[i32]) -> SpicyResult<(LazyFrame, Vec<String>)> {
    match args {
        SpicyObj::DataFrame(df) => {
            let columns = df
                .get_column_names()
                .iter()
                .map(|c| c.to_string())
                .collect();
            Ok((df.clone().lazy(), columns))
        }
        SpicyObj::ParDataFrame(par_df) => {
            let lf = if par_df.df_type == DFType::Single {
                par_df.scan_partition(0)?
            } else if partitions.is_empty() {
                return Err(SpicyError::EvalErr(format!(
                    "requires '{:?}' condition for this partitioned dataframe",
                    par_df.df_type
                )));
            } else if partitions.len() == 1 {
                let par_num = partitions[0];
                par_df.scan_partition(par_num)?
            } else if partitions.len() == 2 {
                let min_par = partitions[0];
                let max_par = partitions[1];
                par_df.scan_partition_by_range(min_par, max_par)?
            } else {
                par_df.scan_partitions(partitions)?
            };
            let columns = lf
                .clone()
                .filter(lit(false))
                .collect()
                .map_err(|e| {
                    SpicyError::Err(format!("failed to get column name for dataframe: {}", e))
                })?
                .get_column_names()
                .iter()
                .map(|c| c.to_string())
                .collect();
            Ok((lf, columns))
        }
        _ => Err(SpicyError::EvalErr(format!(
            "not support query on type '{}'",
            args.get_type_name()
        ))),
    }
}

// table, partitions, where, group, operations, limit
pub fn functional_select(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let obj = get_dataframe(state, arg0)?;
    let partitions = args[1].to_par_nums().unwrap_or_default();
    let (lf, columns) = make_it_lazy(&obj, &partitions)?;
    eval_fn_query(
        lf,
        &QueryOp::Select,
        args[2],
        args[3],
        args[4],
        args[5],
        &columns,
        state.is_lazy_mode(),
    )
}

// table, where, group, operations
pub fn functional_update(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let obj = get_dataframe(state, arg0)?;
    let (lf, columns) = make_it_lazy(&obj, &[])?;
    eval_fn_query(
        lf,
        &QueryOp::Update,
        args[1],
        args[2],
        args[3],
        &SpicyObj::I64(0),
        &columns,
        state.is_lazy_mode(),
    )
}

// table, where, columns
pub fn functional_delete(
    state: &EngineState,
    _stack: &mut Stack,
    args: &[&SpicyObj],
) -> SpicyResult<SpicyObj> {
    let arg0 = args[0];
    let obj = get_dataframe(state, arg0)?;
    let (lf, columns) = make_it_lazy(&obj, &[])?;
    eval_fn_query(
        lf,
        &QueryOp::Delete,
        args[1],
        &SpicyObj::MixedList(vec![]),
        args[2],
        &SpicyObj::I64(0),
        &columns,
        state.is_lazy_mode(),
    )
}
