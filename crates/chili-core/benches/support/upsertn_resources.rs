//! Linux resource measurements; one method/case per fresh process.
//! Driven by scripts/bench_upsertn_resources.py. No fixture allocation or
//! dataframe readback occurs in the measured update loop.

use std::{
    fs,
    hint::black_box,
    time::{Duration, Instant},
};

use chili_core::{SpicyObj, Stack};
use serde_json::json;

fn memory_kib(field: &str) -> u64 {
    fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix(field)?;
            value.split_whitespace().next()?.parse().ok()
        })
        .unwrap()
}

// /proc/self/stat reports user/system CPU for all process threads. The caller
// supplies SC_CLK_TCK; a >=1s sample keeps tick quantization small.
fn cpu_ticks() -> (u64, u64) {
    let stat = fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<_> = stat
        .rsplit_once(')')
        .unwrap()
        .1
        .split_whitespace()
        .collect();
    (fields[11].parse().unwrap(), fields[12].parse().unwrap())
}

pub fn main(args: &[String]) {
    assert_eq!(
        args.len(),
        6,
        "batch keep method seconds clk_tck verify|measure"
    );
    let batch: usize = args[0].parse().unwrap();
    let keep: usize = args[1].parse().unwrap();
    let method = args[2].as_str();
    let seconds: f64 = args[3].parse().unwrap();
    let hz: f64 = args[4].parse().unwrap();
    assert!(batch > 0 && keep > 0 && seconds >= 1.0 && hz > 0.0);
    let source = match method {
        "upsertn" => format!("upsertn[`t; data; {keep}]"),
        "value_upsert_take" => format!("t: -{keep}#upsert[t; data]"),
        "symbol_upsert_take" => format!("upsert[`t; data]; t: -{keep}#t"),
        _ => panic!("unknown method: {method}"),
    };
    let state = super::engine_with_limit(batch, keep);
    let query = SpicyObj::String(source.into());
    let mut stack = Stack::new(None, 0, 0, "");

    if args[5] == "verify" {
        let mut expected = super::frame(0, keep);
        let data = state.get_var("data").unwrap();
        for _ in 0..3 {
            expected.extend(data.df().unwrap()).unwrap();
            expected = expected.tail(Some(keep));
            super::run(&state, &mut stack, &query);
            assert!(
                state
                    .get_var("t")
                    .unwrap()
                    .df()
                    .unwrap()
                    .equals_missing(&expected)
            );
        }
        assert!(
            data.df()
                .unwrap()
                .equals_missing(&super::frame(keep, batch))
        );
        println!(
            "{}",
            json!({"verified": true, "batch": batch, "keep": keep, "method": method})
        );
        return;
    }
    assert_eq!(args[5], "measure");
    let fixture_rss_kib = memory_kib("VmRSS:");
    let warmup = Instant::now();
    let mut warmup_updates = 0_u64;
    while warmup.elapsed() < Duration::from_millis(250) || warmup_updates < 3 {
        black_box(super::run(&state, &mut stack, black_box(&query)));
        warmup_updates += 1;
    }
    let start_rss_kib = memory_kib("VmRSS:");
    // Linux resets the RSS high-water mark to current RSS, excluding fixture
    // creation and warmup peaks. Fail loudly if the kernel disallows this.
    fs::write("/proc/self/clear_refs", "5").expect("reset VmHWM");
    let cpu_before = cpu_ticks();
    let start = Instant::now();
    let duration = Duration::from_secs_f64(seconds);
    let mut updates = 0_u64;
    while start.elapsed() < duration {
        black_box(super::run(&state, &mut stack, black_box(&query)));
        updates += 1;
    }
    let wall_seconds = start.elapsed().as_secs_f64();
    let cpu_after = cpu_ticks();
    let peak_rss_kib = memory_kib("VmHWM:");
    let end_rss_kib = memory_kib("VmRSS:");
    let actual = state.get_var("t").unwrap();
    let df = actual.df().unwrap();
    assert_eq!(df.height(), keep);
    // Check the rolling sequence after the complete run without allocating
    // another dataframe. Full-column equivalence is checked by verify mode.
    let appended = (warmup_updates + updates) as usize * batch;
    let expected_seq = |i: usize| {
        let offset = appended + i;
        if offset < keep {
            offset
        } else {
            keep + (offset - keep) % batch
        }
    };
    let seq = df.column("seq").unwrap().i64().unwrap();
    assert_eq!(seq.get(0), Some(expected_seq(0) as i64));
    assert_eq!(seq.get(keep - 1), Some(expected_seq(keep - 1) as i64));
    println!(
        "{}",
        json!({
            "batch": batch, "keep": keep, "method": method, "updates": updates,
            "warmup_updates": warmup_updates, "wall_seconds": wall_seconds,
            "user_cpu_seconds": (cpu_after.0 - cpu_before.0) as f64 / hz,
            "system_cpu_seconds": (cpu_after.1 - cpu_before.1) as f64 / hz,
            "fixture_rss_kib": fixture_rss_kib, "start_rss_kib": start_rss_kib,
            "peak_rss_kib": peak_rss_kib, "end_rss_kib": end_rss_kib,
            "retained_estimated_bytes": df.estimated_size(),
        })
    );
}
