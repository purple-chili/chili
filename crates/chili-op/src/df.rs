use chili_core::{ArgType, SpicyError, SpicyObj, SpicyResult, validate_args};
use num::range;
use polars_lazy::frame::pivot::PivotExpr;
use std::sync::Arc;

use polars::{
    datatypes::DataType,
    frame::DataFrame,
    lazy::dsl::{col, lit, when},
    prelude::{
        AsofStrategy, Categories, DataTypeExpr, IntoColumn, IntoLazy, NamedFrom, QuantileMethod,
        Selector, SortMultipleOptions, SortOptions, UnpivotDF, int_ranges,
    },
    series::Series,
};
use polars_ops::{
    frame::{AsOfOptions, JoinArgs, JoinCoalesce, JoinType, JoinValidation},
    pivot::{PivotAgg, pivot_stable},
};

use crate::util::get_data_type_name;

// df, idColumns, valueColumns
pub fn unpivot(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[ArgType::DataFrame, ArgType::SymOrSyms, ArgType::SymOrSyms],
    )?;

    let df = args[0].df().unwrap();
    let indices = args[1].to_str_vec().unwrap();
    let on_cols = args[2].to_str_vec().unwrap();

    UnpivotDF::unpivot(df, on_cols, indices)
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(SpicyObj::DataFrame)
}
// df, idColumns, headerColumns, valueColumns, aggFn
pub fn pivot(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::DataFrame,
            ArgType::SymOrSyms,
            ArgType::SymOrSyms,
            ArgType::SymOrSyms,
            ArgType::Sym,
        ],
    )?;

    let df = args[0].df().unwrap();
    let indices = args[1].to_str_vec().unwrap();
    let on_cols = args[2].to_str_vec().unwrap();
    let values = args[3].to_str_vec().unwrap();
    let agg_fn_name = args[4].str().unwrap();

    let pivot_agg = match agg_fn_name {
        "" => None,
        "first" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").first())))),
        "last" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").last())))),
        "max" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").max())))),
        "mean" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").mean())))),
        "median" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").median())))),
        "min" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").min())))),
        "count" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").count())))),
        "sum" => Some(PivotAgg(Arc::new(PivotExpr::from_expr(col("").sum())))),
        _ => {
            return Err(SpicyError::Err(format!(
                "Expect 'min', 'max', 'first', 'last', 'sum', 'mean', 'median', 'len', got '{}'",
                agg_fn_name
            )));
        }
    };

    pivot_stable(
        df,
        on_cols,
        Some(indices),
        Some(values),
        true,
        pivot_agg,
        Some("_"),
    )
    .map_err(|e| SpicyError::Err(e.to_string()))
    .map(SpicyObj::DataFrame)
}

fn sort(args: &[&SpicyObj], descending: bool) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::SymOrSyms, ArgType::DataFrame])?;
    let arg0 = args[0];
    let df = args[1].df().unwrap();
    let columns: Vec<&str> = match arg0 {
        SpicyObj::Series(series) => series
            .cat32()
            .map_err(|_| SpicyError::EvalErr("requires symbols type".to_owned()))?
            .iter_str()
            .map(|s| s.unwrap_or(""))
            .collect(),
        SpicyObj::Symbol(s) => vec![s],
        _ => return Err(SpicyError::EvalErr("requires symbols type".to_owned())),
    };
    let mut options = SortMultipleOptions::default();
    options = options.with_maintain_order(true);
    if descending {
        options = SortMultipleOptions::with_order_descending(options, descending)
    }
    match df.sort(columns, options) {
        Ok(df) => Ok(SpicyObj::DataFrame(df)),
        Err(e) => Err(SpicyError::EvalErr(e.to_string())),
    }
}

// columns, df
pub fn x_asc(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    sort(args, false)
}

// columns, df
pub fn x_desc(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    sort(args, true)
}

pub fn join_op(args: &[&SpicyObj], join_args: JoinArgs) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::SymOrSyms,
            ArgType::DataFrameOrSeries,
            ArgType::DataFrameOrSeries,
        ],
    )?;

    let join_type = join_args.how.clone();
    let columns = args[0].to_str_vec().unwrap();
    let col_exprs = columns
        .iter()
        .map(|column| col(column.to_string()))
        .collect::<Vec<_>>();
    let df0 = args[1]
        .df()
        .cloned()
        .unwrap_or_else(|_| args[1].series().unwrap().clone().into_frame());
    let df1 = args[2]
        .df()
        .cloned()
        .unwrap_or_else(|_| args[2].series().unwrap().clone().into_frame());

    let mut same_columns = Vec::new();
    match join_type {
        JoinType::Inner | JoinType::Full | JoinType::Left => {
            for col in df0.get_column_names() {
                if columns.contains(&col.as_str()) {
                    continue;
                }
                if df1.get_column_names().contains(&col) {
                    same_columns.push(col)
                }
            }
        }
        JoinType::AsOf(option) => {
            let by = option.left_by.clone().unwrap_or_default();
            for col in df0.get_column_names() {
                if columns.contains(&col.as_str()) || by.contains(col) {
                    continue;
                }
                if df1.get_column_names().contains(&col) {
                    same_columns.push(col)
                }
            }
        }
        _ => (),
    }

    let df1 = if !same_columns.is_empty() {
        df1.clone().lazy().with_column(lit(true).alias("#"))
    } else {
        df1.clone().lazy()
    };

    let mut lazy = df0
        .clone()
        .lazy()
        .join(df1, col_exprs.clone(), col_exprs, join_args);

    if !same_columns.is_empty() {
        lazy = lazy
            .with_columns(
                same_columns
                    .iter()
                    .map(|c| {
                        when(col("#"))
                            .then(col(format!("{}_right", c)))
                            .otherwise(col(c.to_string()))
                            .alias(c.to_string())
                    })
                    .collect::<Vec<_>>(),
            )
            .drop(Selector::ByName {
                names: same_columns
                    .into_iter()
                    .map(|c| format!("{}_right", c).into())
                    .collect(),
                strict: true,
            })
            .drop(Selector::Matches("^#$".into()))
    }

    lazy.collect()
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(SpicyObj::DataFrame)
}

// df, idColumns, valueColumns
pub fn aj(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let mut asof_options = AsOfOptions::default();
    let arg0 = args[0];
    let mut syms = arg0.to_str_vec()?;
    if syms.len() > 1 {
        let mut args = args.to_vec();
        let as_of_col = SpicyObj::Symbol(syms.pop().unwrap().to_string());
        args[0] = &as_of_col;
        let by: Vec<polars::prelude::PlSmallStr> =
            syms.iter().map(|s| s.to_owned().into()).collect();
        asof_options.left_by = Some(by.clone());
        asof_options.right_by = Some(by);
        asof_options.allow_eq = true;
        let join_args = JoinArgs::new(JoinType::AsOf(asof_options.into()));
        join_op(&args, join_args)
    } else {
        let join_args = JoinArgs::new(JoinType::AsOf(asof_options.into()));
        join_op(args, join_args)
    }
}

pub fn cj(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let join_args = JoinArgs::new(JoinType::Cross);
    join_op(args, join_args)
}

pub fn ij(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let mut join_args = JoinArgs::new(JoinType::Inner).with_coalesce(JoinCoalesce::CoalesceColumns);
    join_args.validation = JoinValidation::ManyToOne;
    join_op(args, join_args)
}

pub fn lj(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let mut join_args = JoinArgs::new(JoinType::Left).with_coalesce(JoinCoalesce::CoalesceColumns);
    join_args.validation = JoinValidation::ManyToOne;
    join_op(args, join_args)
}

pub fn fj(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let mut join_args = JoinArgs::new(JoinType::Full).with_coalesce(JoinCoalesce::CoalesceColumns);
    join_args.validation = JoinValidation::ManyToOne;
    join_op(args, join_args)
}

// byColumns, time, start, end, aggregations, df0, df1,
pub fn wj(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(
        args,
        &[
            ArgType::StrLike,
            ArgType::StrOrSym,
            ArgType::StrOrSym,
            ArgType::StrOrSym,
            ArgType::Any,
            ArgType::DataFrame,
            ArgType::DataFrame,
        ],
    )?;

    let on_columns = args[0]
        .to_str_vec()
        .unwrap()
        .into_iter()
        .map(|s| s.to_string().into())
        .collect::<Vec<_>>();
    let time = args[1].str().unwrap();
    let start = args[2].str().unwrap();
    let end = args[3].str().unwrap();

    let aggregations = match args[4] {
        SpicyObj::Expr(expr) => vec![expr.clone()],
        SpicyObj::MixedList(exprs) => exprs
            .iter()
            .map(|e| e.as_expr())
            .collect::<SpicyResult<Vec<_>>>()?,
        _ => {
            return Err(SpicyError::EvalErr(
                "expected expression, got {}".to_owned(),
            ));
        }
    };
    let df0 = args[5].df().unwrap();
    let df1 = args[6].df().unwrap();

    let mut lf0 = df0.clone().lazy();

    let asof_options = AsOfOptions {
        left_by: Some(on_columns.clone()),
        right_by: Some(on_columns.clone()),
        ..Default::default()
    };

    let mut key_columns = on_columns.clone();
    key_columns.extend([time.into()]);

    let lf1 = df1
        .clone()
        .lazy()
        .sort(key_columns.clone(), SortMultipleOptions::default());

    let idx1 = lf1.clone().select(
        key_columns
            .iter()
            .map(|s| col(s.as_str()))
            .collect::<Vec<_>>(),
    );

    let mut asof_forward = asof_options.clone();
    asof_forward.strategy = AsofStrategy::Forward;

    asof_forward.allow_eq = true;

    lf0 = lf0.with_row_index("idx0", None);
    let mut lf = lf0
        .clone()
        .join(
            idx1.clone().with_row_index("min_idx", None),
            [start.into()],
            [time.into()],
            JoinArgs::new(JoinType::AsOf(Box::new(asof_forward))),
        )
        .drop(Selector::Matches("_right$".into()));

    let mut asof_backward = asof_options.clone();
    asof_backward.strategy = AsofStrategy::Backward;
    asof_backward.allow_eq = false;
    lf = lf
        .join(
            idx1.with_row_index("max_idx", None),
            [end.into()],
            [time.into()],
            JoinArgs::new(JoinType::AsOf(Box::new(asof_backward))),
        )
        .select([col("idx0"), col("min_idx"), col("max_idx")])
        .filter(col("min_idx").lt_eq(col("max_idx")));

    // Potentially loop by_columns groups to reduce memory usage
    let index_df = lf.collect().map_err(|e| SpicyError::Err(e.to_string()))?;
    let mut lf = index_df.clone().lazy();
    let index_length = index_df.height();
    let avg_window_size = lf
        .clone()
        .select([(col("max_idx") - col("min_idx")).alias("window_size")])
        .collect()
        .unwrap()
        .column("window_size")
        .unwrap()
        .as_materialized_series()
        .mean()
        .unwrap() as usize;

    // around 7GB => 200M rows
    let threshold = 200_000_000;
    if index_length * avg_window_size < threshold {
        lf = lf
            .with_column(
                int_ranges(
                    col("min_idx"),
                    col("max_idx") + lit(1),
                    lit(1),
                    DataType::UInt64,
                )
                .alias("idx1"),
            )
            .explode(Selector::ByName {
                names: Arc::new(["idx1".into()]),
                strict: true,
            })
            .select([col("idx0"), col("idx1")]);

        lf = lf
            .join(
                lf1.with_row_index("idx1", None),
                [col("idx1")],
                [col("idx1")],
                JoinArgs::new(JoinType::Inner),
            )
            .drop(Selector::ByName {
                names: key_columns.into(),
                strict: true,
            });

        if !aggregations.is_empty() {
            lf = lf.group_by([col("idx0")]).agg(aggregations);
        }

        lf = lf0.join(
            lf,
            [col("idx0")],
            [col("idx0")],
            JoinArgs::new(JoinType::Full),
        );

        lf = lf.drop(Selector::Matches("_right$".into()));

        Ok(SpicyObj::DataFrame(
            lf.collect().map_err(|e| SpicyError::Err(e.to_string()))?,
        ))
    } else {
        if aggregations.is_empty() && index_length * avg_window_size > threshold {
            return Err(SpicyError::EvalErr(format!(
                "wj requires aggregations when length '{}' * avg_window_size '{}' > '{}'",
                index_length, avg_window_size, threshold,
            )));
        }
        let patch_size = threshold / avg_window_size;
        let patches = index_length / patch_size;
        let mut aggs = range(0, patches + 1)
            .map(|i| {
                let length = (index_length - patch_size * i).min(patch_size);
                let mut patch_lf = index_df.slice((i * patch_size) as i64, length).lazy();
                patch_lf = patch_lf
                    .with_column(
                        int_ranges(
                            col("min_idx"),
                            col("max_idx") + lit(1),
                            lit(1),
                            DataType::UInt64,
                        )
                        .alias("idx1"),
                    )
                    .explode(Selector::ByName {
                        names: Arc::new(["idx1".into()]),
                        strict: true,
                    })
                    .select([col("idx0"), col("idx1")]);
                patch_lf
                    .join(
                        lf1.clone().with_row_index("idx1", None),
                        [col("idx1")],
                        [col("idx1")],
                        JoinArgs::new(JoinType::Left),
                    )
                    .group_by([col("idx0")])
                    .agg(aggregations.clone())
                    .collect()
                    .map_err(|e| SpicyError::Err(e.to_string()))
            })
            .collect::<SpicyResult<Vec<DataFrame>>>()?;
        let mut agg_df = aggs.remove(0);
        for agg in aggs.iter() {
            agg_df
                .extend(agg)
                .map_err(|e| SpicyError::Err(e.to_string()))?;
        }
        // Could use take instead of join
        let lf = lf0
            .join(
                agg_df.lazy(),
                [col("idx0")],
                [col("idx0")],
                JoinArgs::new(JoinType::Left),
            )
            .drop(Selector::Matches("^idx0$|^idx1$".into()));
        Ok(SpicyObj::DataFrame(
            lf.collect().map_err(|e| SpicyError::Err(e.to_string()))?,
        ))
    }
}

pub fn anti(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let join_args = JoinArgs::new(JoinType::Anti);
    join_op(args, join_args)
}

pub fn semi(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    let join_args = JoinArgs::new(JoinType::Semi);
    join_op(args, join_args)
}

pub fn x_reorder(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::SymOrSyms, ArgType::DataFrame])?;
    let mut columns = args[0].to_str_vec().unwrap();
    let df = args[1].df().unwrap();
    let original_columns = df.get_column_names();
    for col in original_columns {
        if columns.contains(&col.as_str()) {
            continue;
        }
        columns.push(col)
    }
    df.select(columns)
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(SpicyObj::DataFrame)
}

pub fn x_rename(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::SymOrSyms, ArgType::DataFrameOrSeries])?;

    let arg0 = args[0];
    let arg1 = args[1];

    match args[1] {
        SpicyObj::DataFrame(df) => {
            let columns = args[0].to_str_vec().unwrap();
            return df
                .clone()
                .lazy()
                .rename(&df.get_column_names()[..columns.len()], columns, false)
                .collect()
                .map_err(|e| SpicyError::Err(e.to_string()))
                .map(SpicyObj::DataFrame);
        }
        SpicyObj::Series(s) => {
            if arg0.is_sym() {
                return Ok(SpicyObj::Series(
                    s.clone().rename(arg0.sym().unwrap().into()).clone(),
                ));
            }
        }
        _ => unreachable!(),
    }

    Err(SpicyError::UnsupportedBinaryOpErr(
        "xrename".to_owned(),
        arg0.get_type_name(),
        arg1.get_type_name(),
    ))
}

pub fn cols(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame])?;
    let df = args[0].df().unwrap();
    Ok(SpicyObj::Series(
        Series::new(
            "".into(),
            df.get_column_names()
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
        )
        .cast(&DataType::Categorical(
            Categories::global(),
            Categories::global().mapping(),
        ))
        .unwrap(),
    ))
}

pub fn describe(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrameOrSeries])?;
    let arg0 = args[0];
    let df = match arg0 {
        // refer to https://github.com/pola-rs/polars/blob/1c6b7b70e935fe70384fc0d1ca8d07763011d8b8/py-polars/polars/lazyframe/frame.py#L680
        SpicyObj::DataFrame(df) => df
            .iter()
            .filter(|s| s.dtype().is_primitive_numeric() || s.dtype().is_temporal())
            .cloned()
            .collect(),
        SpicyObj::Series(s) => {
            if s.dtype().is_primitive_numeric() || s.dtype().is_temporal() {
                s.clone().into_frame()
            } else {
                return Ok(SpicyObj::Null);
            }
        }
        _ => unreachable!(),
    };
    let mut res = Series::new(
        "statistic".into(),
        [
            "count",
            "null_count",
            "mean",
            "std",
            "min",
            "10%",
            "25%",
            "50%",
            "75%",
            "90%",
            "max",
        ],
    )
    .into_frame();
    for s in df.iter() {
        let column = s.name().as_str();
        let exprs = &[
            col(column).count().alias("count"),
            col(column).null_count().alias("null_count"),
            col(column).mean().alias("mean"),
            col(column).to_physical().std(1).alias("std"),
            col(column).min().alias("min"),
            col(column)
                .to_physical()
                .quantile(lit(0.1), QuantileMethod::Midpoint)
                .alias("10%")
                .cast(s.dtype().clone()),
            col(column)
                .to_physical()
                .quantile(lit(0.25), QuantileMethod::Midpoint)
                .alias("25%")
                .cast(s.dtype().clone()),
            col(column)
                .to_physical()
                .quantile(lit(0.5), QuantileMethod::Midpoint)
                .alias("50%")
                .cast(s.dtype().clone()),
            col(column)
                .to_physical()
                .quantile(lit(0.75), QuantileMethod::Midpoint)
                .alias("75%")
                .cast(s.dtype().clone()),
            col(column)
                .to_physical()
                .quantile(lit(0.9), QuantileMethod::Midpoint)
                .alias("90%")
                .cast(s.dtype().clone()),
            col(column).max().alias("max"),
        ];
        let mut stat = df
            .clone()
            .lazy()
            .select([col(column).sort(SortOptions::default())]);
        if s.dtype().is_temporal() {
            stat = stat.select(
                exprs
                    .iter()
                    .map(|e| e.clone().cast(DataTypeExpr::Literal(DataType::String)))
                    .collect::<Vec<_>>(),
            );
        } else {
            stat = stat.select(exprs);
        }
        let mut stat = stat.collect().map_err(|e| SpicyError::Err(e.to_string()))?;
        let mut stat = stat
            .transpose(Some(column), None)
            .unwrap()
            .get_columns()
            .last()
            .unwrap()
            .clone();
        stat.rename(column.into());
        res = res
            .hstack(&[stat])
            .map_err(|e| SpicyError::Err(e.to_string()))?;
    }
    Ok(SpicyObj::DataFrame(res))
}

pub fn schema(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame])?;
    let df = args[0].df().unwrap();
    let columns = df.get_column_names();
    let columns = Series::new(
        "column".into(),
        columns.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    );
    let types = df
        .iter()
        .map(|s| get_data_type_name(s.dtype()))
        .collect::<Vec<_>>();
    let types = Series::new("datatype".into(), types);
    let flags = df
        .iter()
        .map(|s| match s.get_flags().is_sorted() {
            polars::series::IsSorted::Ascending => "asc",
            polars::series::IsSorted::Descending => "desc",
            polars::series::IsSorted::Not => "",
        })
        .collect::<Vec<_>>();
    let flags = Series::new("flag".into(), flags);
    Ok(SpicyObj::DataFrame(
        DataFrame::new(vec![columns.into(), types.into(), flags.into()]).unwrap(),
    ))
}

// may cause a reallocation
pub fn extend(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame, ArgType::DataFrame])?;
    let mut df0 = args[0].df().unwrap().clone();
    let df1 = args[1].df().unwrap();
    df0.extend(df1)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::DataFrame(df0))
}

pub fn hstack(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame, ArgType::DataFrameOrSeries])?;
    let df0 = args[0].df().unwrap();
    let arg1 = args[1];
    match arg1 {
        SpicyObj::Series(s1) => {
            let res = df0
                .hstack(&[s1.clone().into()])
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::DataFrame(res))
        }
        SpicyObj::DataFrame(df1) => {
            let res = df0
                .hstack(df1.get_columns())
                .map_err(|e| SpicyError::Err(e.to_string()))?;
            Ok(SpicyObj::DataFrame(res))
        }
        _ => unreachable!(),
    }
}

pub fn vstack(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame, ArgType::DataFrame])?;
    let df0 = args[0].df().unwrap();
    let df1 = args[1].df().unwrap();
    let res = df0
        .vstack(df1)
        .map_err(|e| SpicyError::Err(e.to_string()))?;
    Ok(SpicyObj::DataFrame(res))
}

pub fn flip(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::Dict])?;
    let arg0 = args[0].dict().unwrap();

    let df: DataFrame = arg0
        .iter()
        .map(|(k, v)| {
            if let SpicyObj::Series(s) = v {
                Ok(s.clone().rename(k.into()).clone().into_column())
            } else {
                Err(SpicyError::EvalErr(format!(
                    "Expected series, got {}",
                    v.get_type_name()
                )))
            }
        })
        .collect::<SpicyResult<DataFrame>>()?;
    Ok(SpicyObj::DataFrame(df))
}

pub fn explode(args: &[&SpicyObj]) -> SpicyResult<SpicyObj> {
    validate_args(args, &[ArgType::DataFrame, ArgType::SymOrSyms])?;
    let df = args[0].df().unwrap();
    let columns = args[1].to_str_vec().unwrap();
    df.explode(columns)
        .map_err(|e| SpicyError::Err(e.to_string()))
        .map(SpicyObj::DataFrame)
}
