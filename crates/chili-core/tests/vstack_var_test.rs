use std::collections::BTreeSet;

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use polars::prelude::*;

fn engine() -> EngineState {
    let mut state = EngineState::initialize();
    // Exercise the append primitive independently of background maintenance.
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
            "vstack.pep",
        )
        .unwrap()
}

fn values(obj: &SpicyObj) -> Vec<i64> {
    obj.df().unwrap()["seq"]
        .i64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

#[test]
fn symbol_vstack_preserves_snapshot_buffers_across_repeated_appends() {
    let state = engine();
    let input = df!("seq" => (0..10_000i64).collect::<Vec<_>>()).unwrap();
    state.set_var("buf", SpicyObj::DataFrame(input)).unwrap();
    let snapshot = state.get_var("buf").unwrap();
    let original_ptr = snapshot.df().unwrap()["seq"]
        .i64()
        .unwrap()
        .downcast_iter()
        .next()
        .unwrap()
        .values()
        .as_ptr();
    for i in 10_000..10_032i64 {
        state
            .set_var("row", SpicyObj::DataFrame(df!("seq" => [i]).unwrap()))
            .unwrap();
        assert_eq!(eval(&state, "vstack[`buf; row]"), SpicyObj::I64(1));
        let live = state.get_var("buf").unwrap();
        let live_ptr = live.df().unwrap()["seq"]
            .i64()
            .unwrap()
            .downcast_iter()
            .next()
            .unwrap()
            .values()
            .as_ptr();
        assert_eq!(
            live_ptr, original_ptr,
            "historical payload must not be recopied"
        );
    }
    assert_eq!(values(&snapshot), (0..10_000).collect::<Vec<_>>());
    assert_eq!(
        values(&state.drain("buf").unwrap()),
        (0..10_032).collect::<Vec<_>>()
    );
    assert_eq!(state.get_var("buf").unwrap().df().unwrap().height(), 0);
    assert_eq!(eval(&state, "vstack[`buf; row]"), SpicyObj::I64(1));
    assert_eq!(values(&state.get_var("buf").unwrap()), vec![10_031]);
}

#[test]
fn symbol_vstack_creates_tables_and_coerces_incoming_schema_like_upsert() {
    let state = engine();
    assert_eq!(
        eval(&state, "vstack[`buf; ([]seq: 1 2; sym: `a`b)]"),
        SpicyObj::I64(2)
    );
    let incoming = df!("seq" => [3i32], "sym" => ["c"]).unwrap();
    state.vstack_var("buf", &incoming).unwrap();
    let out = state.get_var("buf").unwrap();
    assert_eq!(values(&out), vec![1, 2, 3]);
    assert!(matches!(
        out.df().unwrap()["sym"].dtype(),
        DataType::Categorical(..)
    ));
    assert_eq!(incoming["seq"].dtype(), &DataType::Int32);
    assert_eq!(incoming["sym"].dtype(), &DataType::String);
}

#[test]
fn failed_late_column_append_preserves_all_original_rows() {
    let state = engine();
    let original = df!("seq" => [1i64, 2], "value" => [10i64, 20]).unwrap();
    state
        .set_var("buf", SpicyObj::DataFrame(original.clone()))
        .unwrap();
    // Column one can append successfully before the second name fails.
    let wrong_name = df!("seq" => [3i64], "wrong" => [30i64]).unwrap();
    assert!(state.vstack_var("buf", &wrong_name).is_err());
    assert_eq!(
        state.get_var("buf").unwrap(),
        SpicyObj::DataFrame(original.clone())
    );
    let wrong_width = df!("seq" => [3i64]).unwrap();
    assert!(state.vstack_var("buf", &wrong_width).is_err());
    assert_eq!(state.get_var("buf").unwrap(), SpicyObj::DataFrame(original));
    state.set_var("scalar", SpicyObj::I64(42)).unwrap();
    assert!(state.vstack_var("scalar", &wrong_width).is_err());
    assert_eq!(state.get_var("scalar").unwrap(), SpicyObj::I64(42));
}

#[test]
fn dataframe_value_vstack_keeps_existing_nonmutating_behavior() {
    let state = engine();
    eval(&state, "buf: ([]seq: 1 2)");
    assert_eq!(
        values(&eval(&state, "vstack[buf; ([]seq: 3 4)]")),
        vec![1, 2, 3, 4]
    );
    assert_eq!(values(&state.get_var("buf").unwrap()), vec![1, 2]);
    let df = SpicyObj::DataFrame(df!("seq" => [3i32]).unwrap());
    assert!(
        state
            .fn_call("vstack", &[&state.get_var("buf").unwrap(), &df])
            .is_err()
    );
}

#[test]
fn concurrent_symbol_appends_and_drains_do_not_lose_rows() {
    let state = engine();
    state
        .vstack_var("buf", &df!("seq" => [-1i64]).unwrap())
        .unwrap();
    let mut seen = vec![];
    std::thread::scope(|scope| {
        for producer in 0..4i64 {
            let state = &state;
            scope.spawn(move || {
                for i in 0..100i64 {
                    let row = df!("seq" => [producer * 100 + i]).unwrap();
                    state.vstack_var("buf", &row).unwrap();
                }
            });
        }
        for _ in 0..100 {
            seen.extend(values(&state.drain("buf").unwrap()));
            std::thread::yield_now();
        }
    });
    seen.extend(values(&state.drain("buf").unwrap()));
    assert_eq!(seen.len(), 401);
    assert_eq!(
        seen.into_iter().collect::<BTreeSet<_>>(),
        (-1..400).collect()
    );
}

/// Run explicitly with POLARS_MAX_THREADS=1 and --ignored --nocapture.
/// This isolates snapshot-induced append work; it is not the TCP flush gate.
#[test]
#[ignore = "manual shared-snapshot append benchmark"]
fn profile_shared_snapshot_append() {
    use std::time::Instant;
    let rows: i64 = std::env::var("CHILI_SNAPSHOT_ROWS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(2_000_000);
    assert!(rows > 0);
    let row = SpicyObj::DataFrame(
        df!(
            "seq" => [rows], "price" => [1i64], "qty" => [1i64], "time" => [rows]
        )
        .unwrap(),
    );
    let mut extend_ms = vec![];
    let mut chunked_ms = vec![];
    for trial in 0..5 {
        // Alternate the order to reduce systematic warmup/allocator bias.
        for chunked in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let state = engine();
            let input = df!(
                "seq" => (0..rows).collect::<Vec<_>>(),
                "price" => (0..rows).collect::<Vec<_>>(),
                "qty" => (0..rows).collect::<Vec<_>>(),
                "time" => (0..rows).collect::<Vec<_>>()
            )
            .unwrap();
            state.set_var("buf", SpicyObj::DataFrame(input)).unwrap();
            for _ in 0..3 {
                state.upsert_var("buf", &row).unwrap();
            }
            let snapshot = state.get_var("buf").unwrap();
            let start = Instant::now();
            if chunked {
                state.vstack_var("buf", row.df().unwrap()).unwrap();
            } else {
                state.upsert_var("buf", &row).unwrap();
            }
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            if chunked {
                chunked_ms.push(ms);
            } else {
                extend_ms.push(ms);
            }
            assert_eq!(snapshot.df().unwrap().height(), rows as usize + 3);
            assert_eq!(
                state.get_var("buf").unwrap().df().unwrap().height(),
                rows as usize + 4
            );
        }
    }
    let median = |values: &[f64]| {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    println!(
        "{}",
        serde_json::json!({
            "rows": rows,
            "initial_bytes": rows * 32,
            "upsert_median_ms": median(&extend_ms),
            "vstack_median_ms": median(&chunked_ms),
            "upsert_ms": extend_ms,
            "vstack_ms": chunked_ms,
            "scope": "one append with a retained snapshot; no TCP or concurrent writer"
        })
    );
}
