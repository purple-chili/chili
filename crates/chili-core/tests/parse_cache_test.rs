//! Parse cache regression tests.
//!
//! Verifies the LRU parse cache invariants:
//!   1. Cache hit returns the same parse result as a cold parse.
//!   2. Cache key is `(path, source)` — different paths or sources produce
//!      distinct entries, even if everything else matches.
//!   3. Errored parses never pollute the cache.
//!   4. Concurrent parses don't deadlock or corrupt the cache.

use std::sync::Arc;

use chili_core::EngineState;

const TEST_QUERY: &str = "select from t where date=2024.01.03";

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state
}

#[test]
fn parse_cache_hit_returns_equivalent_ast() {
    let engine = make_engine();

    let nodes_first = engine.parse("test.pep", TEST_QUERY).unwrap();
    let nodes_second = engine.parse("test.pep", TEST_QUERY).unwrap();

    assert_eq!(
        format!("{:?}", nodes_first),
        format!("{:?}", nodes_second),
        "cache hit should return identical AST"
    );
    assert_eq!(nodes_first.len(), nodes_second.len());
}

#[test]
fn parse_cache_records_one_entry_per_unique_query() {
    let engine = make_engine();
    assert_eq!(engine.parse_cache_len(), 0);

    engine.parse("test.pep", TEST_QUERY).unwrap();
    assert_eq!(engine.parse_cache_len(), 1);

    // Same query, same path → cache hit, no new entry
    engine.parse("test.pep", TEST_QUERY).unwrap();
    assert_eq!(engine.parse_cache_len(), 1);

    // Different query → new entry
    engine
        .parse("test.pep", "select from t where date=2024.01.04")
        .unwrap();
    assert_eq!(engine.parse_cache_len(), 2);
}

#[test]
fn parse_cache_distinguishes_by_path() {
    let engine = make_engine();

    engine.parse("ipc1.pep", TEST_QUERY).unwrap();
    engine.parse("ipc2.pep", TEST_QUERY).unwrap();

    // Same source text, different path → 2 distinct cache entries
    assert_eq!(engine.parse_cache_len(), 2);
}

#[test]
fn parse_cache_handles_invalid_queries_without_caching() {
    let engine = make_engine();

    let result = engine.parse("test.pep", "select from where 1");
    assert!(result.is_err(), "invalid query should fail");

    assert_eq!(
        engine.parse_cache_len(),
        0,
        "errored parses must not pollute the cache"
    );
}

#[test]
fn parse_cache_concurrent_access_is_safe() {
    use std::thread;

    let engine = Arc::new(make_engine());
    let mut handles = vec![];

    for _ in 0..8 {
        let engine = Arc::clone(&engine);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let nodes = engine.parse("test.pep", TEST_QUERY).unwrap();
                assert!(!nodes.is_empty());
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(engine.parse_cache_len(), 1);
}

#[test]
fn parse_cache_preserves_eval_correctness() {
    let engine = make_engine();

    let cold = engine.parse("test.pep", TEST_QUERY).unwrap();
    let hot = engine.parse("test.pep", TEST_QUERY).unwrap();

    let cold_dbg = format!("{:#?}", cold);
    let hot_dbg = format!("{:#?}", hot);

    assert_eq!(
        cold_dbg, hot_dbg,
        "cached AST must structurally match a fresh parse"
    );
}
