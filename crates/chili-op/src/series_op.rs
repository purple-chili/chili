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
            // Floor on physical i64; restore unit + tz (tz-aware used to fall through
            // and fail casting Duration → Datetime(..., UTC)).
            DataType::Datetime(TimeUnit::Milliseconds, tz) => {
                let bar_size = if bar.dtype().eq(&DataType::Duration(TimeUnit::Nanoseconds))
                    || bar.dtype().eq(&DataType::Time)
                {
                    bar.cast(&DataType::Int64).unwrap() / 1000000
                } else {
                    bar.cast(&DataType::Int64).unwrap()
                };
                ((s1.cast(&DataType::Int64).unwrap() / bar_size.clone())? * bar_size)?
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, tz.clone()))?
            }
            DataType::Datetime(TimeUnit::Microseconds, tz) => {
                let bar_size = if bar.dtype().eq(&DataType::Duration(TimeUnit::Nanoseconds))
                    || bar.dtype().eq(&DataType::Time)
                {
                    bar.cast(&DataType::Int64).unwrap() / 1000
                } else {
                    bar.cast(&DataType::Int64).unwrap()
                };
                ((s1.cast(&DataType::Int64).unwrap() / bar_size.clone())? * bar_size)?
                    .cast(&DataType::Datetime(TimeUnit::Microseconds, tz.clone()))?
            }
            DataType::Time
            | DataType::Datetime(TimeUnit::Nanoseconds, _)
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

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{NamedFrom, TimeZone};
    use polars::series::Series;

    fn ns_utc(vals: Vec<i64>) -> Column {
        Series::new("t".into(), vals)
            .cast(&DataType::Datetime(
                TimeUnit::Nanoseconds,
                Some(TimeZone::UTC),
            ))
            .unwrap()
            .into_column()
    }

    fn duration_ns(ns: i64) -> Column {
        Series::new("b".into(), vec![ns])
            .cast(&DataType::Duration(TimeUnit::Nanoseconds))
            .unwrap()
            .into_column()
    }

    #[test]
    fn xbar_datetime_ns_utc_with_duration() {
        // 1s, 1.5s, 2.5s from epoch → 1s bars → 1s, 1s, 2s
        let s = ns_utc(vec![1_000_000_000, 1_500_000_000, 2_500_000_000]);
        let bar = duration_ns(1_000_000_000);
        let out = xbar(s, bar).expect("xbar tz datetime");
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Nanoseconds, Some(TimeZone::UTC))
        );
        let got: Vec<i64> = out
            .take_materialized_series()
            .cast(&DataType::Int64)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(got, vec![1_000_000_000, 1_000_000_000, 2_000_000_000]);
    }

    #[test]
    fn xbar_datetime_ms_utc_with_duration_ns() {
        let s = Series::new("t".into(), vec![1_000i64, 1_500, 2_500])
            .cast(&DataType::Datetime(
                TimeUnit::Milliseconds,
                Some(TimeZone::UTC),
            ))
            .unwrap()
            .into_column();
        // 1 second as ns duration → converted to 1000 ms bars
        let bar = duration_ns(1_000_000_000);
        let out = xbar(s, bar).expect("xbar ms utc");
        assert_eq!(
            out.dtype(),
            &DataType::Datetime(TimeUnit::Milliseconds, Some(TimeZone::UTC))
        );
        let got: Vec<i64> = out
            .take_materialized_series()
            .cast(&DataType::Int64)
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(got, vec![1_000, 1_000, 2_000]);
    }
}
