use chili_core::{EngineState, SpicyObj, Stack, compact_dataframe};
use chili_op::BUILT_IN_FN;
use polars::prelude::*;
use std::collections::BTreeSet;

fn engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.set_auto_compaction(0, 65_536).unwrap();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
}

fn eval(state: &EngineState, code: &str) -> SpicyObj {
    state
        .eval(
            &mut Stack::default(),
            &SpicyObj::String(code.into()),
            "compact.pep",
        )
        .unwrap()
}

#[test]
fn compact_merges_small_tail_and_preserves_large_prefix_and_snapshots() {
    let state = engine();
    state
        .set_var(
            "buf",
            SpicyObj::DataFrame(df!("seq" => (0..10_000i64).collect::<Vec<_>>()).unwrap()),
        )
        .unwrap();
    for seq in 10_000..11_000i64 {
        state
            .vstack_var("buf", &df!("seq" => [seq]).unwrap())
            .unwrap();
    }
    let snapshot = state.get_var("buf").unwrap();
    let pointer = snapshot.df().unwrap()["seq"]
        .i64()
        .unwrap()
        .downcast_iter()
        .next()
        .unwrap()
        .values()
        .as_ptr();
    assert_eq!(eval(&state, "compact[`buf; 256]"), SpicyObj::I64(11_000));
    let live = state.get_var("buf").unwrap();
    assert_eq!(live, snapshot);
    let seq = live.df().unwrap()["seq"].i64().unwrap();
    assert_eq!(seq.chunks().len(), 5);
    assert_eq!(
        seq.downcast_iter().next().unwrap().values().as_ptr(),
        pointer
    );
    assert_eq!(snapshot.df().unwrap()["seq"].n_chunks(), 1001);
    assert_eq!(
        eval(&state, "count select from buf where seq >= 10900"),
        SpicyObj::I64(100)
    );
}

#[test]
fn compact_preserves_nullable_and_categorical_values_with_unaligned_chunks() {
    let mut numbers = Series::new("seq".into(), [Some(1i64), None]);
    numbers
        .append(&Series::new("seq".into(), [Some(3i64), Some(4), None]))
        .unwrap();
    let mut strings = Series::new("sym".into(), [Some("a")]);
    strings
        .append(&Series::new(
            "sym".into(),
            [None, Some("b"), Some("c"), None],
        ))
        .unwrap();
    let categories = strings
        .cast(&DataType::Categorical(
            Categories::global(),
            Categories::global().mapping(),
        ))
        .unwrap()
        .with_name("category".into());
    let df = DataFrame::new(
        5,
        vec![
            numbers.into_column(),
            strings.into_column(),
            categories.into_column(),
        ],
    )
    .unwrap();
    let out = compact_dataframe(&df, 8).unwrap();
    assert_eq!(out, df);
    assert!(out.columns().iter().all(|c| c.n_chunks() == 1));
    assert_eq!(df["seq"].n_chunks(), 2);
}

#[test]
fn value_form_empty_frames_and_invalid_arguments() {
    let state = engine();
    eval(&state, "buf: ([]seq: 1 2)");
    eval(&state, "vstack[`buf; ([]seq: 3 4)]");
    let out = eval(&state, "compact[buf; 64]");
    assert_eq!(out.df().unwrap()["seq"].n_chunks(), 1);
    assert_eq!(
        state.get_var("buf").unwrap().df().unwrap()["seq"].n_chunks(),
        2
    );
    let empty = df!("seq" => Vec::<i64>::new()).unwrap();
    assert_eq!(compact_dataframe(&empty, 64).unwrap(), empty);
    let no_columns = DataFrame::new(4, vec![]).unwrap();
    assert_eq!(compact_dataframe(&no_columns, 64).unwrap(), no_columns);
    for target in [0, -1] {
        assert!(
            state
                .fn_call("compact", &[&out, &SpicyObj::I64(target)])
                .is_err()
        );
    }
    assert!(state.compact_var("missing", 64).is_err());
    state.set_var("scalar", SpicyObj::I64(1)).unwrap();
    assert!(state.compact_var("scalar", 64).is_err());
}

#[test]
fn concurrent_compaction_and_multiple_appenders_preserve_every_row() {
    let state = engine();
    state
        .vstack_var("buf", &df!("seq" => [-1i64]).unwrap())
        .unwrap();
    std::thread::scope(|scope| {
        for producer in 0..4i64 {
            let state = &state;
            scope.spawn(move || {
                for i in 0..1000i64 {
                    state
                        .vstack_var("buf", &df!("seq" => [producer * 1000 + i]).unwrap())
                        .unwrap();
                    if i % 16 == 0 {
                        std::thread::yield_now();
                    }
                }
            });
        }
        for _ in 0..30 {
            state.compact_var("buf", 128).unwrap();
        }
    });
    state.compact_var("buf", 128).unwrap();
    let live = state.get_var("buf").unwrap();
    let seq = live.df().unwrap()["seq"].i64().unwrap();
    assert_eq!(seq.len(), 4001);
    assert_eq!(
        seq.into_no_null_iter().collect::<BTreeSet<_>>(),
        (-1..4000).collect()
    );
    assert!(seq.chunks().len() <= 33);
}
