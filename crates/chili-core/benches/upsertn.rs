//! Compare equivalent rolling-table updates through the Pepper evaluator.
//! Run: cargo bench -p chili-core --bench upsertn
//! Fixture creation and correctness checks are outside the timed loop.

use std::{hint::black_box, time::Duration};

use chili_core::{EngineState, SpicyObj, Stack};
use chili_op::BUILT_IN_FN;
use criterion::{BenchmarkId, Criterion, criterion_group};
use polars::prelude::*;

#[path = "support/upsertn_resources.rs"]
mod resources;

const LIMIT: usize = 1000;
const METHODS: [(&str, &str); 3] = [
    ("upsertn", "upsertn[`t; data; 1000]"),
    ("value_upsert_take", "t: -1000#upsert[t; data]"),
    ("symbol_upsert_take", "upsert[`t; data]; t: -1000#t"),
];

fn frame(start: usize, rows: usize) -> DataFrame {
    DataFrame::new(
        rows,
        vec![
            Column::new(
                "seq".into(),
                (start as i64..(start + rows) as i64).collect::<Vec<_>>(),
            ),
            Column::new(
                "symbol".into(),
                (0..rows)
                    .map(|i| ["AAPL", "MSFT", "GOOG", "AMZN"][i % 4])
                    .collect::<Vec<_>>(),
            )
            .cast(&DataType::Categorical(
                Categories::global(),
                Categories::global().mapping(),
            ))
            .unwrap(),
            Column::new(
                "price".into(),
                (0..rows)
                    .map(|i| 100.0 + i as f64 * 0.01)
                    .collect::<Vec<_>>(),
            ),
            Column::new(
                "qty".into(),
                (0..rows).map(|i| 1 + i as i64 % 100).collect::<Vec<_>>(),
            ),
        ],
    )
    .unwrap()
}

fn engine(batch: usize) -> EngineState {
    engine_with_limit(batch, LIMIT)
}

fn engine_with_limit(batch: usize, limit: usize) -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&BUILT_IN_FN);
    state
        .set_var("t", SpicyObj::DataFrame(frame(0, limit)))
        .unwrap();
    state
        .set_var("data", SpicyObj::DataFrame(frame(limit, batch)))
        .unwrap();
    state
}

fn run(state: &EngineState, stack: &mut Stack, query: &SpicyObj) -> SpicyObj {
    state.eval(stack, query, "upsertn-bench.pep").unwrap()
}

fn bench_upsertn(c: &mut Criterion) {
    // The literal symbol expression takes from a row count, not a dataframe.
    let state = engine(10);
    let mut stack = Stack::new(None, 0, 0, "");
    let literal = SpicyObj::String("result: -1000#upsert[`t; data]".into());
    let out = run(&state, &mut stack, &literal);
    assert!(!out.is_df());
    assert_eq!(state.get_var("t").unwrap().df().unwrap().height(), 1010);
    eprintln!(
        "Literal symbol expression returns {}; table remains at 1010 rows.",
        out.get_type_name()
    );

    let mut group = c.benchmark_group("rolling_1000_four_columns");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for batch in [1, 10, 100, 1000, 10000] {
        for (name, source) in METHODS {
            let state = engine(batch);
            let query = SpicyObj::String(source.into());
            let mut stack = Stack::new(None, 0, 0, "");

            // Check all columns against append-then-tail, including subsequent
            // updates after the retained frame has become a slice.
            let mut expected = frame(0, LIMIT);
            let data = state.get_var("data").unwrap();
            for _ in 0..3 {
                expected.extend(data.df().unwrap()).unwrap();
                expected = expected.tail(Some(LIMIT));
                run(&state, &mut stack, &query);
                let actual = state.get_var("t").unwrap();
                assert!(
                    actual.df().unwrap().equals_missing(&expected),
                    "{name}, batch {batch}"
                );
            }
            drop(data);
            drop(expected);

            // Steady-state updates with a warm parse cache and no fixture
            // allocation or variable readback inside the timed loop.
            group.bench_function(BenchmarkId::new(name, batch), |b| {
                b.iter(|| black_box(run(&state, &mut stack, black_box(&query))));
            });
            assert_eq!(state.get_var("t").unwrap().df().unwrap().height(), LIMIT);
        }
    }
    group.finish();
}

criterion_group!(benches, bench_upsertn);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "--resources") {
        resources::main(&args[2..]);
    } else {
        benches();
        Criterion::default().configure_from_args().final_summary();
    }
}
