use std::sync::Mutex;

use rand::prelude::*;
use std::sync::LazyLock;

static GLOBAL_RNG_STATE: LazyLock<Mutex<SmallRng>> =
    LazyLock::new(|| Mutex::new(SmallRng::seed_from_u64(0)));

pub(crate) fn get_global_random_u64() -> u64 {
    GLOBAL_RNG_STATE.lock().unwrap().next_u64()
}

pub fn set_global_random_seed(seed: u64) {
    *GLOBAL_RNG_STATE.lock().unwrap() = SmallRng::seed_from_u64(seed);
}
