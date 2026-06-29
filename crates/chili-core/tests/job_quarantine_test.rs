//! Scheduled job quarantine-on-error tests for `execute_jobs`.

use chili_core::{EngineState, Job, SpicyObj, Stack, get_local_now_ns};
use chili_op::{BUILT_IN_FN, LOG_FN};

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    state
}

/// A repeating job, due now, whose end_time is far enough out that it would
/// normally reschedule (not naturally deactivate).
fn due_repeating_job(fn_name: &str) -> Job {
    let now = get_local_now_ns();
    let sec = 1_000_000_000i64;
    Job {
        fn_name: fn_name.to_owned(),
        start_time: now - sec,
        end_time: now + 3600 * sec,
        interval: sec,
        last_run_time: None,
        next_run_time: now - sec, // due
        is_active: true,
        description: "test".to_owned(),
    }
}

fn define_failing_fn(state: &EngineState) {
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(
            &mut s,
            &SpicyObj::String(".bad: {[] raise \"boom\"};".to_owned()),
            "t.pep",
        )
        .unwrap();
}

fn is_active(state: &EngineState, id: i64) -> bool {
    let df = state.list_job().unwrap();
    let ids = df.column("id").unwrap().i64().unwrap();
    let active = df.column("is_active").unwrap().bool().unwrap();
    for i in 0..df.height() {
        if ids.get(i) == Some(id) {
            return active.get(i).unwrap_or(false);
        }
    }
    panic!("job id {id} not found");
}

#[test]
fn failing_job_is_quarantined_when_enabled() {
    let state = new_engine();
    define_failing_fn(&state);
    state.set_jobs_deactivate_on_error(true);
    let id = state.add_job(due_repeating_job(".bad"));
    state.execute_jobs();
    assert!(
        !is_active(&state, id),
        "a failing job must be deactivated when quarantine is enabled"
    );
}

#[test]
fn failing_job_keeps_firing_by_default() {
    let state = new_engine();
    define_failing_fn(&state);
    assert!(!state.jobs_deactivate_on_error());
    let id = state.add_job(due_repeating_job(".bad"));
    state.execute_jobs();
    assert!(
        is_active(&state, id),
        "without quarantine a failing job stays active (rescheduled)"
    );
}

#[test]
fn healthy_job_stays_active_under_quarantine() {
    let state = new_engine();
    let mut s = Stack::new(None, 0, 0, "");
    state
        .eval(
            &mut s,
            &SpicyObj::String(".ok: {[] 1};".to_owned()),
            "t.pep",
        )
        .unwrap();
    state.set_jobs_deactivate_on_error(true);
    let id = state.add_job(due_repeating_job(".ok"));
    state.execute_jobs();
    assert!(
        is_active(&state, id),
        "a healthy job must NOT be quarantined"
    );
}

#[test]
fn toggle_round_trips() {
    let state = new_engine();
    assert!(!state.jobs_deactivate_on_error());
    state.set_jobs_deactivate_on_error(true);
    assert!(state.jobs_deactivate_on_error());
    state.set_jobs_deactivate_on_error(false);
    assert!(!state.jobs_deactivate_on_error());
}
