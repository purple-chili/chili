"""`engine.add_at_time()` tests.

Thin PyO3 binding around chili's `.job.addAtTime` registered builtin.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest
from chili import ChiliEngine


class TestAddAtTime:
    """`engine.add_at_time()` schedules a pepper fn at a target wall time."""

    def test_returns_positive_job_id(self):
        """add_at_time returns a positive integer job id."""
        e = ChiliEngine(pepper=True)
        e.eval("eod_handler: {[d] eod_log: d}")
        when = datetime.now(timezone.utc) + timedelta(seconds=60)
        job_id = e.add_at_time("eod_handler", when, "add_at_time test")
        assert isinstance(job_id, int)
        assert job_id > 0

    def test_fires_near_target_time(self):
        """Schedule a fn 200ms in the future; verify it runs within 2s.

        Requires the chili-side scheduler to be active — constructed with
        ``job_interval > 0``.

        Pepper scheduled functions must be **nullary** — chili's
        ``execute_jobs`` invokes them as ``fn_name[]`` (no args). To
        access the firing time inside the handler, use ``today[]`` /
        ``now[]`` or query an engine variable.
        """
        e = ChiliEngine(pepper=True, job_interval=50)
        # Pepper fn flips a marker variable when invoked. Nullary form.
        e.eval(".test.fired: 0")
        e.eval(".test.marker: {[] .test.fired: 1}")
        when = datetime.now(timezone.utc) + timedelta(milliseconds=200)
        e.add_at_time(".test.marker", when, "fire-test")

        # Poll for the marker — scheduler polls at the engine's `interval`
        # (default; usually 100ms). Allow generous slack.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if e.get_var(".test.fired") == 1:
                break
            time.sleep(0.05)
        assert e.get_var(".test.fired") == 1, "scheduled handler did not fire within 2s"

    def test_default_description_is_empty(self):
        """description is optional; default is empty string."""
        e = ChiliEngine(pepper=True)
        e.eval("noop: {[] 0}")
        when = datetime.now(timezone.utc) + timedelta(seconds=60)
        # No description arg — should succeed and return a valid job id.
        job_id = e.add_at_time("noop", when)
        assert isinstance(job_id, int)
        assert job_id > 0

    def test_naive_datetime_rejected(self):
        """Naive datetime (no tz) raises TypeError — caller must pass tz-aware.

        Documents the hard requirement coming from pyo3-chrono's
        ``DateTime<Utc>::extract``. If callers want a UTC interpretation
        they should attach ``timezone.utc`` explicitly.
        """
        e = ChiliEngine(pepper=True)
        e.eval("noop: {[] 0}")
        when = datetime.now() + timedelta(seconds=60)  # tz-naive
        with pytest.raises(TypeError, match="tzinfo"):
            e.add_at_time("noop", when)
