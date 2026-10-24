use std::{
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use parking_lot::{Condvar, Mutex, RwLock};

use crate::{
    SpicyError, SpicyResult,
    variables::{Variables, compact_and_publish},
};

struct Settings {
    interval_ms: u64,
    target_rows: usize,
    stopped: bool,
}

struct Control {
    settings: Mutex<Settings>,
    changed: Condvar,
}

/// One lazy worker per engine. It holds only a weak reference to variables and
/// never executes user code or runs compaction on an appending thread.
pub(crate) struct AutoCompaction {
    control: Arc<Control>,
    started: AtomicBool,
    enabled: AtomicBool,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Default for AutoCompaction {
    fn default() -> Self {
        Self {
            control: Arc::new(Control {
                settings: Mutex::new(Settings {
                    interval_ms: 250,
                    target_rows: 65_536,
                    stopped: false,
                }),
                changed: Condvar::new(),
            }),
            started: AtomicBool::new(false),
            enabled: AtomicBool::new(true),
            worker: Mutex::new(None),
        }
    }
}

impl AutoCompaction {
    pub(crate) fn configure(
        &self,
        vars: &Arc<RwLock<Variables>>,
        interval_ms: u64,
        target_rows: usize,
    ) -> SpicyResult<()> {
        if target_rows == 0 {
            return Err(SpicyError::Err(
                "compact target_rows must be positive".into(),
            ));
        }
        {
            let mut settings = self.control.settings.lock();
            if settings.stopped {
                return Err(SpicyError::Err(
                    "compaction worker has been shut down".into(),
                ));
            }
            settings.interval_ms = interval_ms;
            settings.target_rows = target_rows;
            self.enabled.store(interval_ms > 0, Ordering::Release);
            self.control.changed.notify_all();
        }
        // Existing dirty tables must also be picked up when re-enabled.
        if !vars.read().pending_compactions().is_empty() {
            self.ensure_started(vars);
        }
        Ok(())
    }

    pub(crate) fn ensure_started(&self, vars: &Arc<RwLock<Variables>>) {
        if self.started.load(Ordering::Acquire) || !self.enabled.load(Ordering::Acquire) {
            return;
        }
        let mut worker = self.worker.lock();
        if self.started.load(Ordering::Acquire) || self.control.settings.lock().stopped {
            return;
        }
        let vars = Arc::downgrade(vars);
        let control = Arc::clone(&self.control);
        match thread::Builder::new()
            .name("chili-compaction".into())
            .spawn(move || run(vars, control))
        {
            Ok(handle) => {
                *worker = Some(handle);
                self.started.store(true, Ordering::Release);
            }
            Err(error) => log::error!("could not start compaction worker: {error}"),
        }
    }

    pub(crate) fn shutdown(&self) {
        {
            let mut settings = self.control.settings.lock();
            settings.stopped = true;
            self.enabled.store(false, Ordering::Release);
            self.control.changed.notify_all();
        }
        // Serialize shutdown callers through the join. The worker itself never
        // takes this mutex, and the settings lock has already been released.
        let mut handle = self.worker.lock();
        if let Some(worker) = handle.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for AutoCompaction {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn run(vars: Weak<RwLock<Variables>>, control: Arc<Control>) {
    loop {
        let target_rows = {
            let mut settings = control.settings.lock();
            loop {
                if settings.stopped {
                    return;
                }
                if settings.interval_ms == 0 {
                    control.changed.wait(&mut settings);
                    continue;
                }
                let interval = Duration::from_millis(settings.interval_ms);
                if control
                    .changed
                    .wait_for(&mut settings, interval)
                    .timed_out()
                {
                    break;
                }
            }
            if settings.stopped {
                return;
            }
            settings.target_rows
        };
        let Some(vars) = vars.upgrade() else { return };
        let pending = vars.read().pending_compactions();
        for id in pending {
            {
                let settings = control.settings.lock();
                if settings.stopped {
                    return;
                }
                if settings.interval_ms == 0 {
                    break;
                }
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let snapshot = vars.write().begin_pending_compaction(&id)?;
                if let Some((snapshot, token)) = snapshot {
                    compact_and_publish(&vars, &id, snapshot, token, target_rows)?;
                }
                Ok::<_, SpicyError>(())
            }));
            match result {
                // A concurrent replacement/drain can legitimately cancel a
                // pass. It must not kill maintenance for other tables.
                Ok(Err(error)) => log::debug!("automatic compaction of '{id}': {error}"),
                Err(_) => log::error!("automatic compaction of '{id}' panicked"),
                Ok(Ok(())) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn dropping_worker_wakes_and_joins_it_without_retaining_control_or_variables() {
        let vars = Arc::new(RwLock::new(Variables::from(HashMap::new())));
        let control;
        {
            let worker = AutoCompaction::default();
            control = Arc::downgrade(&worker.control);
            worker.ensure_started(&vars);
            assert!(worker.started.load(Ordering::Acquire));
        }
        assert!(control.upgrade().is_none());
        assert_eq!(Arc::strong_count(&vars), 1);
    }
}
