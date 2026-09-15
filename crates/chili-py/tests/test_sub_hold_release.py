"""Subscriber-side hold buffer: `.handle.holding` parks live frames in the
subscriber until `.handle.release` applies them in order.

This is what lets `.sub.init` start reading before `replay`, so the backlog
of a slow replay sits in the subscriber rather than in the tickerplant's
outbound queue.
"""

import socket
import tempfile
import time
from datetime import date

import polars as pl
from chili import ChiliEngine


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rows(engine: ChiliEngine) -> list[int]:
    return engine.get_var("trade")["n"].to_list()


def test_holding_parks_frames_until_release_then_goes_live():
    port = _free_port()
    tp = ChiliEngine(pepper=True)
    with tempfile.TemporaryDirectory() as log_dir:
        schema = pl.DataFrame({"n": pl.Series([], dtype=pl.Int64)})
        tp.init_tick(
            schema={"trade": schema}, log_dir=log_dir + "/", filename=date.today()
        )
        tp.start_tcp_listener(port)
        time.sleep(0.1)

        s = ChiliEngine(pepper=True)
        s.load_sub()  # defines `upd`
        s.eval(
            f"""
            h: .handle.open "chili://127.0.0.1:{port}";
            info: h (`.tick.subscribe; enlist `trade);
            (set) each info[2];
            .handle.holding[h];
            """
        )

        for i in range(5):
            tp.publish("trade", pl.DataFrame({"n": [i]}))
        time.sleep(0.2)

        # Frames were read off the socket but not applied.
        assert _rows(s) == [], "held frames must not be applied before release"
        assert tp.stats()["queue_depth_total"] == 0, "backlog must not sit in the tickerplant"

        applied = s.eval(".handle.release[h]")
        assert applied == 5
        assert _rows(s) == [0, 1, 2, 3, 4]

        # Live now: later frames are applied by the reader thread, in order.
        for i in range(5, 8):
            tp.publish("trade", pl.DataFrame({"n": [i]}))
        time.sleep(0.2)
        assert _rows(s) == list(range(8))

        # Releasing twice is harmless.
        assert s.eval(".handle.release[h]") == 0

        s.shutdown()
        tp.shutdown()


def test_release_on_non_holding_handle_errors():
    s = ChiliEngine(pepper=True)
    try:
        s.eval(".handle.release[9999]")
    except Exception as e:  # noqa: BLE001
        assert "holding" in str(e)
    else:
        raise AssertionError("release on an unknown handle must error")
    s.shutdown()
