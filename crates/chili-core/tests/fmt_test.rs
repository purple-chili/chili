use polars::{
    prelude::{DataType, NamedFrom, TimeUnit},
    series::Series,
};

#[test]
fn fmt_duration_series() {
    let series = Series::new("duration".into(), vec![1000])
        .cast(&DataType::Duration(TimeUnit::Nanoseconds))
        .unwrap();
    let expected = "shape: (1,)\nSeries: 'duration' [duration[ns]]\n[\n\t0D00:00:00.000001000\n]";
    let actual = format!("{}", series);
    assert_eq!(expected, actual);
}

/*
// polars-core/src/fmt.rs
#[cfg(feature = "dtype-duration")]
pub fn fmt_duration_string<W: Write>(f: &mut W, v: i64, unit: TimeUnit) -> fmt::Result {
    let unit_in_s = match unit {
        TimeUnit::Nanoseconds => 1000_000_000,
        TimeUnit::Microseconds => 1000_000,
        TimeUnit::Milliseconds => 1000,
    };

    let sign = if v < 0 { "-" } else { "" };
    let v = v.abs();
    let n = v % unit_in_s;
    let secs = v / unit_in_s;
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;
    let secs = secs % 60;
    let mins = mins % 60;
    let hours = hours % 24;

    match unit {
        TimeUnit::Nanoseconds => write!(
            f,
            "{}{}D{:02}:{:02}:{:02}.{:09}",
            sign, days, hours, mins, secs, n,
        )?,
        TimeUnit::Microseconds => write!(
            f,
            "{}{}D{:02}:{:02}:{:02}.{:06}",
            sign, days, hours, mins, secs, n,
        )?,
        TimeUnit::Milliseconds => write!(
            f,
            "{}{}D{:02}:{:02}:{:02}.{:03}",
            sign, days, hours, mins, secs, n,
        )?,
    };
    Ok(())
}
*/
