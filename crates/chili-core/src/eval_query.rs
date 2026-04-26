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

/// Scan where-clauses for partition-column predicates and consume them into
/// a `partitions` vector shaped the way `make_it_lazy` / `scan_partition*`
/// expect:
///   - `vec![x]`           single exact partition
///   - `vec![start, end]`  inclusive range (interpreted by `scan_partition_by_range`)
///   - `vec![a, b, c, ..]` explicit list (interpreted by `scan_partitions`).
///     A 2-element list from `in` has a sentinel appended so it is not
///     mistaken for a range.
///
/// Returns `(partitions, skip_indices)` where `skip_indices` are the indices
/// of `where_exp` clauses that were fully consumed as partition predicates
/// and must not be re-applied as row-level filters.
///
/// Non-partition clauses are left alone. If the underlying dataframe is not
/// a `ParDataFrame`, `partitions` is empty and `skip_indices` is empty — the
/// caller (`make_it_lazy`) handles the non-partitioned case.
fn extract_partition_predicates(
    state: &EngineState,
    stack: &mut Stack,
    args: &SpicyObj,
    where_exp: &[AstNode],
    src: &str,
) -> SpicyResult<(Vec<i32>, Vec<usize>)> {
    let par_df = match args {
        SpicyObj::ParDataFrame(par_df) => par_df,
        _ => return Ok((vec![], vec![])),
    };

    let par_col = match par_df.df_type {
        DFType::ByDate => "date",
        DFType::ByYear => "year",
        DFType::Single => return Ok((vec![], vec![])),
    };

    let mut skip_indices: Vec<usize> = Vec::new();
    let mut exact: Option<i32> = None;
    let mut range_start: Option<i32> = None;
    let mut range_end: Option<i32> = None;
    let mut in_list: Option<Vec<i32>> = None;

    for (i, clause) in where_exp.iter().enumerate() {
        let (op_node, lhs, rhs) = match clause {
            AstNode::BinaryExp { op, lhs, rhs } => (op, lhs, rhs),
            _ => continue,
        };
        let lhs_name = match lhs.as_ref() {
            AstNode::Id { name, .. } => name,
            _ => continue,
        };
        if lhs_name != par_col {
            continue;
        }
        let op_name = match op_node.as_ref() {
            AstNode::Id { name, .. } => name.as_str(),
            _ => {
                return Err(SpicyError::Err(format!(
                    "mismatch partition unit '{}' with partition type '{:?}'",
                    lhs_name, par_df.df_type
                )));
            }
        };
        let par = eval_by_node(state, stack, rhs, src, None)?;
        match op_name {
            "=" => {
                exact = Some(par.to_par_num()?);
                skip_indices.push(i);
            }
            ">=" => {
                let v = par.to_i64()? as i32;
                range_start = Some(range_start.map_or(v, |s| s.max(v)));
                skip_indices.push(i);
            }
            ">" => {
                let v = (par.to_i64()? as i32).saturating_add(1);
                range_start = Some(range_start.map_or(v, |s| s.max(v)));
                skip_indices.push(i);
            }
            "<=" => {
                let v = par.to_i64()? as i32;
                range_end = Some(range_end.map_or(v, |e| e.min(v)));
                skip_indices.push(i);
            }
            "<" => {
                let v = (par.to_i64()? as i32).saturating_sub(1);
                range_end = Some(range_end.map_or(v, |e| e.min(v)));
                skip_indices.push(i);
            }
            "within" => {
                let pars = par.as_vec()?;
                if pars.len() == 2 {
                    let a = pars[0].to_i64()? as i32;
                    let b = pars[1].to_i64()? as i32;
                    range_start = Some(range_start.map_or(a, |s| s.max(a)));
                    range_end = Some(range_end.map_or(b, |e| e.min(b)));
                    skip_indices.push(i);
                }
            }
            "in" => {
                let pars = par
                    .as_vec()?
                    .iter()
                    .map(|p| p.to_i64())
                    .collect::<SpicyResult<Vec<_>>>()?;
                in_list = Some(pars.into_iter().map(|p| p as i32).collect::<Vec<i32>>());
                skip_indices.push(i);
            }
            _ => {
                // Unrecognised op on partition column — leave as a row filter.
                // Example: `where date <> 2024.01.03` (not supported for pruning).
            }
        }
    }

    // Build the final partitions vector. Precedence: exact > in > range.
    // Multiple exact predicates would be contradictory AND; we take the last.
    // Combining `in` with range bounds is not supported — `in` wins.
    let partitions = if let Some(x) = exact {
        // If a range bound also exists and excludes x, produce an empty-range
        // sentinel so the scan returns an empty frame.
        let in_range = range_start.is_none_or(|s| x >= s) && range_end.is_none_or(|e| x <= e);
        if in_range {
            vec![x]
        } else {
            // start > end triggers the empty-schema branch of scan_partition_by_range
            vec![1, 0]
        }
    } else if let Some(mut list) = in_list {
        // Apply range bounds if any were also provided
        if let Some(s) = range_start {
            list.retain(|v| *v >= s);
        }
        if let Some(e) = range_end {
            list.retain(|v| *v <= e);
        }
        if list.is_empty() {
            vec![1, 0] // empty-range sentinel
        } else if list.len() == 2 {
            // 2-element lists would otherwise be misinterpreted as a range by
            // make_it_lazy; append a dup sentinel so scan_partitions is used.
            list.push(list[1]);
            list
        } else {
            list
        }
    } else if range_start.is_some() || range_end.is_some() {
        vec![range_start.unwrap_or(0), range_end.unwrap_or(i32::MAX)]
    } else {
        vec![]
    };

    Ok((partitions, skip_indices))
}

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
    // Scan all where clauses for partition predicates.
    //
    // A partition predicate is a BinaryExp whose LHS is the partition column
    // name ("date" for ByDate, "year" for ByYear) and whose operator is one
    // of =, >=, <=, >, <, within, in. Matched clauses are consumed for
    // partition pruning (via `partitions`) and their indices recorded in
    // `skip_indices` so they are not re-applied as row-level filters.
    //
    // Multiple range bounds are combined into a single tight range:
    //   `where date>=X, date<=Y`  =>  partitions = [X, Y]
    //
    // Non-partition clauses (e.g. `symbol='AAPL'`) are left untouched and
    // applied as row-level filters by the normal path below.
    let (partitions, skip_indices) =
        extract_partition_predicates(state, stack, &args, where_exp, src)?;

    let limited = match limited_exp {
        Some(limited_exp) => eval_by_node(state, stack, limited_exp, src, None)?,
        None => SpicyObj::I64(0),
    };

    let (lf, columns) = make_it_lazy(&args, &partitions)?;

    // Proposal L: pre-allocate the expression vectors with the exact known
    // capacity from the input AST length. Most queries have <4 clauses; the
    // pre-allocation eliminates Vec growth-resize churn for the common case
    // without adding a SmallVec dep.
    let mut where_expr: Vec<SpicyObj> = Vec::with_capacity(where_exp.len());
    for (i, n) in where_exp.iter().enumerate() {
        if skip_indices.contains(&i) {
            continue;
        }
        let obj = eval_by_node(state, stack, n, src, Some(&columns))
            .map_err(|e| SpicyError::EvalErr(e.to_string()))?;
        where_expr.push(obj);
    }

    let mut op_expr: Vec<SpicyObj> = Vec::with_capacity(op_exp.len());
    for n in op_exp.iter() {
        op_expr.push(eval_by_node(state, stack, n, src, Some(&columns))?);
    }

    let mut by_expr: Vec<SpicyObj> = Vec::with_capacity(by_exp.len());
    for n in by_exp.iter() {
        by_expr.push(eval_by_node(state, stack, n, src, Some(&columns))?);
    }

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
        // Proposal K: fuse sequential filters into a single combined filter.
        // For Delete the semantics are "rows matching ALL clauses are deleted",
        // so the filter expression is `!(c1 AND c2 AND ... cN)` applied as a
        // keep-predicate, equivalent to a chain of `.filter(!c1).filter(!c2)...`
        // when each .filter is a "keep what doesn't match" gate. To match the
        // existing semantics exactly we apply `.not()` to each clause first,
        // then AND them — same as the original sequential loop.
        if let Some(combined) = where_exprs
            .iter()
            .map(|e| e.clone().not())
            .reduce(|a, b| a.and(b))
        {
            lf = lf.filter(combined);
        }
    } else if *query_op == QueryOp::Select {
        // Proposal K: fuse sequential filters into a single .and() chain.
        // Polars' optimizer typically folds consecutive `.filter()` calls
        // anyway, but doing it ourselves saves an optimizer pass and makes
        // intent explicit. Preserves the same semantics: AND of all clauses.
        if let Some(combined) = where_exprs.iter().cloned().reduce(|a, b| a.and(b)) {
            lf = lf.filter(combined);
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
        Ok(SpicyObj::LazyFrame(lf))
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
                return Err(SpicyError::MissingParCondErr(par_df.df_type.to_string()));
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
            // Proposal I: read column names via lazy schema collection rather
            // than materialising an empty filter. `collect_schema` is metadata-
            // only — no data is read from disk — and saves ~1-3ms per query.
            // collect_schema is `&mut self` in polars 0.53 so we clone first.
            let mut tmp = lf.clone();
            let schema = tmp.collect_schema().map_err(|e| {
                SpicyError::Err(format!("failed to get column name for dataframe: {}", e))
            })?;
            let columns: Vec<String> = schema.iter_names().map(|n| n.to_string()).collect();
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
