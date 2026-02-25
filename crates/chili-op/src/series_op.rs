use polars::{
    error::{PolarsResult, polars_bail},
    prelude::{Column, DataType, IntoColumn, TimeUnit},
};
use polars_ops::series::RoundSeries;

pub(crate) fn xbar_expr(columns: &mut [Column]) -> PolarsResult<Column> {
    let s = columns[0].clone();
    let bar = columns[1].clone();
    xbar(s, bar)
}

pub(crate) fn xbar(s: Column, bar: Column) -> PolarsResult<Column> {
    let err = || polars_bail!(InvalidOperation: format!("'xbar' requires numeric/temporal bar size and series, got '{}' and '{}'", bar.dtype(), s.dtype()));

    if (bar.dtype().is_primitive_numeric() || bar.dtype().is_temporal())
        && (s.dtype().is_primitive_numeric() || s.dtype().is_temporal())
    {
        let s1 = if bar.dtype().is_float() && !s.dtype().is_float() {
            s.cast(bar.dtype()).unwrap()
        } else {
            s.clone()
        };

        let out = match s1.dtype() {
            DataType::Float32 | DataType::Float64 => {
                let bar_size = bar.cast(s1.dtype())?;
                ((s1 / bar_size.clone())?
                    .take_materialized_series()
                    .floor()?
                    .into_column()
                    * bar_size)?
            }
            DataType::Date => {
                let bar_size = bar.cast(&DataType::Int32)?;
                let s1 = s1.cast(&DataType::Int32).unwrap();
                ((s1 / bar_size.clone())? * bar_size)?.cast(&DataType::Date)?
            }
            DataType::Datetime(TimeUnit::Milliseconds, None) => {
                let bar_size = if bar.dtype().eq(&DataType::Duration(TimeUnit::Nanoseconds))
                    || bar.dtype().eq(&DataType::Time)
                {
                    bar.cast(&DataType::Int64).unwrap() / 1000000
                } else {
                    bar.cast(&DataType::Int64).unwrap()
                };
                ((s1.cast(&DataType::Int64).unwrap() / bar_size.clone())? * bar_size)?
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
            }
            DataType::Time
            | DataType::Datetime(TimeUnit::Nanoseconds, None)
            | DataType::Duration(TimeUnit::Nanoseconds) => {
                let bar_size = bar.cast(&DataType::Int64)?;
                ((s1.cast(&DataType::Int64).unwrap() / bar_size.clone())? * bar_size)?
                    .cast(s1.dtype())?
            }
            _ => {
                let bar_size = bar.cast(s1.dtype())?;
                ((s1 / bar_size.clone())? * bar_size)?
            }
        };
        Ok(out)
    } else {
        err()
    }
}
