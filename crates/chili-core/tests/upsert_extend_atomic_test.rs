//! Failed mid-column `DataFrame::extend` must not tear the target;
//! `drain` must refuse to clear a non-rectangular frame.

use chili_core::{
    EngineState, SpicyObj, ensure_df_rectangular, extend_df_atomic,
};
use polars::prelude::*;

fn ts_fn_frame(ts: &[i64], fns: &[&str], categorical_fn: bool) -> DataFrame {
    let ts_col = Column::new("ts".into(), ts);
    let fn_s = Series::new("fn".into(), fns.to_vec());
    let fn_col = if categorical_fn {
        fn_s.cast(&DataType::Categorical(
            Categories::global(),
            Categories::global().mapping(),
        ))
        .unwrap()
        .into_column()
    } else {
        fn_s.into_column()
    };
    DataFrame::new(ts.len(), vec![ts_col, fn_col]).unwrap()
}

#[test]
fn extend_atomic_leaves_target_intact_on_dtype_mismatch() {
    let mut target = ts_fn_frame(&[1, 2], &["a", "b"], true);
    let incoming = ts_fn_frame(&[3], &["c"], false); // String fn vs Categorical
    let before_ts = target.column("ts").unwrap().len();
    let before_fn = target.column("fn").unwrap().len();

    let err = extend_df_atomic(&mut target, &incoming).expect_err("dtype mismatch");
    assert!(
        err.to_string().contains("extend failed"),
        "unexpected err: {err}"
    );
    assert_eq!(target.column("ts").unwrap().len(), before_ts);
    assert_eq!(target.column("fn").unwrap().len(), before_fn);
    assert_eq!(target.height(), before_ts);
    ensure_df_rectangular(&target).unwrap();
}

#[test]
fn raw_extend_can_tear_but_upsert_coerces_and_stays_rectangular() {
    // Demonstrate Polars extend tear without coerce.
    let mut raw = ts_fn_frame(&[1, 2], &["a", "b"], true);
    let incoming = ts_fn_frame(&[3], &["c"], false);
    let _ = raw.extend(&incoming); // Err — and may leave columns uneven
    assert!(
        ensure_df_rectangular(&raw).is_err(),
        "raw Polars extend must be able to tear (documents mid-column failure)"
    );

    // Chili DF upsert coerces dtypes then extends in place.
    let engine = EngineState::initialize();
    engine
        .upsert_var(
            ".usage",
            &SpicyObj::DataFrame(ts_fn_frame(&[1, 2], &["a", "b"], true)),
        )
        .unwrap();
    engine
        .upsert_var(
            ".usage",
            &SpicyObj::DataFrame(ts_fn_frame(&[3], &["c"], false)),
        )
        .expect("String fn should coerce to Categorical");
    let df = engine.get_var(".usage").unwrap().df().unwrap().clone();
    ensure_df_rectangular(&df).unwrap();
    assert_eq!(df.height(), 3);
}

#[test]
fn upsert_mixed_list_casts_string_fn_to_categorical() {
    let engine = EngineState::initialize();
    engine
        .upsert_var(
            ".usage",
            &SpicyObj::DataFrame(ts_fn_frame(&[1], &["seed"], true)),
        )
        .unwrap();

    let row = SpicyObj::MixedList(vec![
        SpicyObj::I64(2),
        SpicyObj::String("hook".into()),
    ]);
    engine.upsert_var(".usage", &row).expect("cast String→Categorical");
    let df = engine.get_var(".usage").unwrap().df().unwrap().clone();
    ensure_df_rectangular(&df).unwrap();
    assert_eq!(df.height(), 2);
}

#[test]
fn drain_refuses_torn_frame_without_clearing() {
    let engine = EngineState::initialize();
    let mut torn = ts_fn_frame(&[1, 2], &["a", "b"], true);
    let incoming = ts_fn_frame(&[3], &["c"], false);
    let _ = torn.extend(&incoming);
    assert!(ensure_df_rectangular(&torn).is_err());

    engine
        .set_var(".usage", SpicyObj::DataFrame(torn))
        .unwrap();

    let err = engine.drain(".usage").expect_err("torn drain");
    assert!(
        err.to_string().contains("not rectangular"),
        "unexpected: {err}"
    );

    let still = engine.get_var(".usage").unwrap().df().unwrap().clone();
    assert!(
        ensure_df_rectangular(&still).is_err(),
        "drain must not clear a torn buffer"
    );
}
