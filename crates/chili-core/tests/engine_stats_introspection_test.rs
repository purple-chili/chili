use chili_core::{EngineState, SpicyObj};
use indexmap::IndexMap;

fn make_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state
}

fn stats_dict(engine: &EngineState) -> IndexMap<String, SpicyObj> {
    match engine.stats().expect("stats() failed") {
        SpicyObj::Dict(d) => d,
        other => panic!("expected Dict, got {}", other.get_type_name()),
    }
}

#[test]
fn stats_exposes_new_introspection_fields() {
    let engine = make_engine();
    let s = stats_dict(&engine);

    for key in [
        "process_memory_rss_bytes",
        "queue_depth_total",
        "queue_depth_by_handle",
        "handle_count",
        "handle_nums",
        "vars_len",
        "topic_count",
    ] {
        assert!(s.contains_key(key), "stats() missing field '{}'", key);
    }

    for key in ["lazy_mode", "repl_lang", "partitioned_df_count", "parse_cache_len"] {
        assert!(s.contains_key(key), "stats() dropped legacy field '{}'", key);
    }
}

#[test]
fn stats_rss_is_positive_in_process() {
    let engine = make_engine();
    let s = stats_dict(&engine);
    match s.get("process_memory_rss_bytes") {
        Some(SpicyObj::I64(bytes)) => {
            assert!(*bytes > 0, "RSS should be > 0 for the running process, got {}", bytes);
        }
        other => panic!("process_memory_rss_bytes not an I64: {:?}", other.map(|o| o.get_type_name())),
    }
}

#[test]
fn stats_queue_depth_zero_with_no_subscribers() {
    let engine = make_engine();
    let s = stats_dict(&engine);
    match s.get("queue_depth_total") {
        Some(SpicyObj::I64(total)) => {
            assert_eq!(*total, 0, "no subscribers → total queue depth must be 0");
        }
        other => panic!("queue_depth_total not an I64: {:?}", other.map(|o| o.get_type_name())),
    }
    match s.get("handle_count") {
        Some(SpicyObj::I64(n)) => assert_eq!(*n, 0),
        other => panic!("handle_count not an I64: {:?}", other.map(|o| o.get_type_name())),
    }
}

#[test]
fn stats_vars_len_reflects_bound_variables() {
    let engine = make_engine();
    let s0 = stats_dict(&engine);
    let v0 = match s0.get("vars_len") {
        Some(SpicyObj::I64(n)) => *n,
        other => panic!("vars_len not an I64: {:?}", other.map(|o| o.get_type_name())),
    };

    let mut stack = chili_core::Stack::new(None, 0, 0, "");
    engine
        .eval(&mut stack, &SpicyObj::String("x:42".to_owned()), "test.chi")
        .expect("eval x:42 failed");

    let s1 = stats_dict(&engine);
    let v1 = match s1.get("vars_len") {
        Some(SpicyObj::I64(n)) => *n,
        other => panic!("vars_len not an I64: {:?}", other.map(|o| o.get_type_name())),
    };
    assert!(v1 > v0, "binding a var should grow vars_len ({} -> {})", v0, v1);
}
