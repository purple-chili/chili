"""Subscribe handshake vs concurrent publish: nothing falls between replay and live.

A tickerplant publishes 1-row frames with a monotonically increasing ``n``
while fresh subscribers churn against it. After each handshake settles the
subscriber's ``trade`` must hold a contiguous, duplicate-free run of ``n``:
the rows replayed up to the bound handed back by ``.tick.subscribe`` followed
by every live frame published after it.

Before the pending-frame buffer, an ``lpt`` that ran between the bound read
and activation was in neither half (roughly a third of handshakes at this
publish rate), so a handful of iterations is enough to catch a regression.
"""

import socket
import tempfile
import threading
import time
from datetime import date

import polars as pl
from chili import ChiliEngine


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_no_gap_between_replay_and_live_under_publish_churn():
    port = _free_port()
    tp = ChiliEngine(pepper=True)
    stop = threading.Event()

    with tempfile.TemporaryDirectory() as log_dir:
        schema = pl.DataFrame({"n": pl.Series([], dtype=pl.Int64)})
        tp.init_tick(
            schema={"trade": schema}, log_dir=log_dir + "/", filename=date.today()
        )
        tp.start_tcp_listener(port)
        time.sleep(0.1)

        def publisher():
            i = 0
            while not stop.is_set():
                tp.publish("trade", pl.DataFrame({"n": [i]}))
                i += 1

        th = threading.Thread(target=publisher, daemon=True)
        th.start()

        gaps: list[tuple[int, int]] = []
        dups = 0
        try:
            for i in range(40):
                # Roll the log so each fresh subscriber replays only the current
                # segment; the publisher never stalls now, so the log grows fast.
                # The roll happens under full publish load: rotation and the
                # tick[0] reset are one lpt_lock section.
                tp.roll_tick_log(log_dir + "/", f"seg{i}")
                s = ChiliEngine(pepper=True)
                s.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
                time.sleep(0.02)
                ns = s.get_var("trade")["n"].to_list()
                s.shutdown()
                if not ns:
                    continue
                if len(set(ns)) != len(ns):
                    dups += 1
                missing = sorted(set(range(min(ns), max(ns) + 1)) - set(ns))
                if missing:
                    gaps.append((missing[0], missing[-1]))
        finally:
            stop.set()
            th.join()
            tp.shutdown()

    assert not gaps, f"frames missing between replay and live: {gaps}"
    assert dups == 0, "frames delivered twice"
