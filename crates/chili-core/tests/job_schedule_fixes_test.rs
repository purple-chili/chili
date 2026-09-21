//! Job scheduler: `.job.addAfter` delay, zero interval, write-back, panics.

use chili_core::{EngineState, Job, SpicyObj, Stack, get_local_now_ns};
use chili_op::{BUILT_IN_FN, LOG_FN};

const SEC: i64 = 1_000_000_000;

fn new_engine() -> EngineState {
    let mut state = EngineState::initialize();
    state.enable_pepper();
    state.register_fn(&LOG_FN);
    state.register_fn(&BUILT_IN_FN);
    state
}

fn eval(state: &EngineState, src: &str) -> SpicyObj {
    state
        .eval(&mut Stack::new(None, 0, 0, ""), &SpicyObj::String(src.to_owned()), "t.pep")
        .unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn job_field_bool(state: &EngineState, id: i64, col: &str) -> bool {
    let df = state.list_job().unwrap();
    let ids = df.column("id").unwrap().i64().unwrap();
    let vals = df.column(col).unwrap().bool().unwrap();
    (0..df.height())
        .find(|&i| ids.get(i) == Some(id))
        .map(|i| vals.get(i).unwrap())
        .unwrap_or_else(|| panic!("job {id} not found"))
}

#[test]
fn add_after_waits_for_the_delay() {
    let state = new_engine();
    eval(&state, "ran: 0; f: {[] ran:: ran + 1};");
    eval(&state, ".job.addAfter[`f; 0D01:00:00; \"later\"]");
    state.execute_jobs();
    assert_eq!(eval(&state, "ran"), SpicyObj::I64(0), "must not fire at the next poll");
    assert!(job_field_bool(&state, 1, "is_active"));
}

#[test]
fn add_rejects_a_zero_interval() {
    let state = new_engine();
    eval(&state, "f: {[] 1};");
    let res = state.eval(
        &mut Stack::new(None, 0, 0, ""),
        &SpicyObj::String(
            ".job.add[`f; 2030.01.01D00:00:00; 2030.01.02D00:00:00; 0D00:00:00; \"x\"]".into(),
        ),
        "t.pep",
    );
    assert!(res.is_err(), "interval 0 never advances and would fire on every poll");
}

#[test]
fn job_deactivated_while_running_stays_deactivated() {
    let state = new_engine();
    // The job switches itself off while it runs.
    eval(&state, "f: {[] .job.deactivate[1]};");
    let now = get_local_now_ns();
    let id = state.add_job(Job {
        fn_name: "f".into(),
        start_time: now - SEC,
        end_time: now + 3600 * SEC,
        interval: SEC,
        last_run_time: None,
        next_run_time: now - SEC,
        is_active: true,
        description: "self-off".into(),
    });
    assert_eq!(id, 1);
    state.execute_jobs();
    assert!(
        !job_field_bool(&state, 1, "is_active"),
        "the pre-run snapshot must not reactivate the job"
    );
}
