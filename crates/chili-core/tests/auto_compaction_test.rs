use std::{
    collections::BTreeSet,
    thread,
    time::{Duration, Instant},
};

use chili_core::{EngineState, SpicyObj};
use chili_op::BUILT_IN_FN;
use polars::prelude::*;

fn chunks(state: &EngineState, id: &str) -> usize {
    state.get_var(id).unwrap().df().unwrap()["seq"].n_chunks()
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "background compaction did not finish"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn enabled_by_default_only_for_symbol_vstack_and_preserves_snapshots() {
    let state = EngineState::initialize();
    let mut original = df!("seq" => [1i64, 2]).unwrap();
    original.vstack_mut(&df!("seq" => [3i64]).unwrap()).unwrap();
    state
        .set_var("untouched", SpicyObj::DataFrame(original.clone()))
        .unwrap();
    state.vstack_var("buf", &original).unwrap();
    let snapshot = state.get_var("buf").unwrap();
    wait_until(|| chunks(&state, "buf") == 1);
    assert_eq!(state.get_var("buf").unwrap(), snapshot);
    assert_eq!(snapshot.df().unwrap()["seq"].n_chunks(), 2);
    assert_eq!(chunks(&state, "untouched"), 2);
    assert_eq!(state.get_interval(), 0, "no job scheduler is required");
    state.shutdown();
}

#[test]
fn disable_reenable_and_shutdown_are_respected() {
    let state = EngineState::initialize();
    state.register_fn(&BUILT_IN_FN);
    state
        .fn_call(
            ".compact.configure",
            &[&SpicyObj::I64(0), &SpicyObj::I64(64)],
        )
        .unwrap();
    for i in 0..100i64 {
        state
            .vstack_var("buf", &df!("seq" => [i]).unwrap())
            .unwrap();
    }
    thread::sleep(Duration::from_millis(300));
    assert_eq!(chunks(&state, "buf"), 100);
    for args in [(-1, 64), (10, 0), (10, -1)] {
        assert!(
            state
                .fn_call(
                    ".compact.configure",
                    &[&SpicyObj::I64(args.0), &SpicyObj::I64(args.1)]
                )
                .is_err()
        );
    }
    state.set_auto_compaction(10, 64).unwrap();
    wait_until(|| chunks(&state, "buf") == 2);
    state.shutdown();
    state.shutdown();
    assert!(state.set_auto_compaction(10, 64).is_err());
    for i in 100..120i64 {
        state
            .vstack_var("buf", &df!("seq" => [i]).unwrap())
            .unwrap();
    }
    thread::sleep(Duration::from_millis(50));
    assert_eq!(chunks(&state, "buf"), 22, "shutdown must stop maintenance");
}

#[test]
fn periodic_passes_preserve_all_concurrent_appends_on_multiple_tables() {
    let state = EngineState::initialize();
    state.set_auto_compaction(5, 64).unwrap();
    thread::scope(|scope| {
        for producer in 0..4i64 {
            let state = &state;
            scope.spawn(move || {
                for i in 0..500i64 {
                    let id = if producer % 2 == 0 { "left" } else { "right" };
                    state
                        .vstack_var(id, &df!("seq" => [producer * 500 + i]).unwrap())
                        .unwrap();
                    if i % 10 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            });
        }
    });
    for (id, producers) in [("left", [0, 2]), ("right", [1, 3])] {
        wait_until(|| chunks(&state, id) <= 16);
        let live = state.get_var(id).unwrap();
        let seq = live.df().unwrap()["seq"].i64().unwrap();
        assert_eq!(seq.len(), 1000);
        assert_eq!(
            seq.into_no_null_iter().collect::<BTreeSet<_>>(),
            producers
                .into_iter()
                .flat_map(|p| p * 500..(p + 1) * 500)
                .collect()
        );
    }
    state.shutdown();
}
