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
