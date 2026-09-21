"""`.sub.recover` after the tickerplant goes away and comes back.

Every row must end up in the subscriber exactly once: nothing replayed on top
of rows it already holds, nothing published during the outage lost, and each
handle re-subscribing with its own topics and filter.
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


TRADE = pl.DataFrame(
    {"sym": pl.Series([], dtype=pl.Categorical), "n": pl.Series([], dtype=pl.Int64)}
)
QUOTE = pl.DataFrame({"n": pl.Series([], dtype=pl.Int64)})


def _tp(port: int, log_dir: str) -> ChiliEngine:
    tp = ChiliEngine(pepper=True)
    tp.init_tick(
        schema={"trade": TRADE, "quote": QUOTE},
        log_dir=log_dir + "/",
        filename=date.today(),
    )
    tp.start_tcp_listener(port)
    time.sleep(0.1)
    return tp


def _trade(tp: ChiliEngine, n: int, sym: str = "a") -> None:
    tp.publish(
        "trade",
        pl.DataFrame({"sym": pl.Series([sym], dtype=pl.Categorical), "n": [n]}),
    )


def _quote(tp: ChiliEngine, n: int) -> None:
    tp.publish("quote", pl.DataFrame({"n": [n]}))


def _await(fn, want, timeout=20.0):
    deadline = time.time() + timeout
    got = fn()
    while got != want and time.time() < deadline:
        time.sleep(0.1)
        got = fn()
    return got


def _ns(engine: ChiliEngine, table: str) -> list[int]:
    return sorted(engine.get_var(table)["n"].to_list())


def _restart(tp: ChiliEngine, port: int, log_dir: str) -> ChiliEngine:
    tp.shutdown()
    time.sleep(0.3)
    return _tp(port, log_dir)


def test_full_subscription_recovers_without_duplicates_or_loss():
    port = _free_port()
    with tempfile.TemporaryDirectory() as log_dir:
        tp = _tp(port, log_dir)
        for i in range(5):
            _trade(tp, i)
        s = ChiliEngine(pepper=True)
        s.subscribe(f"chili://127.0.0.1:{port}")  # all topics
        for i in range(5, 8):
            _trade(tp, i)
        _quote(tp, 100)
        assert _await(lambda: _ns(s, "trade"), list(range(8))) == list(range(8))

        tp = _restart(tp, port, log_dir)
        _trade(tp, 8)  # published while the subscriber is still away
        _quote(tp, 101)
        want = list(range(9))
        assert _await(lambda: _ns(s, "trade"), want) == want
        assert _ns(s, "quote") == [100, 101]

        _trade(tp, 9)  # live again after recover
        assert _await(lambda: _ns(s, "trade"), list(range(10))) == list(range(10))
        s.shutdown()
        tp.shutdown()


def test_partial_subscription_recovers_without_duplicates():
    port = _free_port()
    with tempfile.TemporaryDirectory() as log_dir:
        tp = _tp(port, log_dir)
        for i in range(4):
            _trade(tp, i)
            _quote(tp, 100 + i)
        s = ChiliEngine(pepper=True)
        s.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
        _trade(tp, 4)
        _quote(tp, 104)
        assert _await(lambda: _ns(s, "trade"), list(range(5))) == list(range(5))

        tp = _restart(tp, port, log_dir)
        _trade(tp, 5)
        want = list(range(6))
        assert _await(lambda: _ns(s, "trade"), want) == want
        s.shutdown()
        tp.shutdown()


def test_each_handle_recovers_its_own_topics_and_filter():
    port = _free_port()
    with tempfile.TemporaryDirectory() as log_dir:
        tp = _tp(port, log_dir)
        s = ChiliEngine(pepper=True)
        s.subscribe(
            f"chili://127.0.0.1:{port}",
            ["quote"],
            filters={"trade": ("sym", ["a"])},
        )
        _quote(tp, 100)
        _trade(tp, 0, "a")
        _trade(tp, 1, "b")  # filtered out live
        assert _await(lambda: _ns(s, "quote"), [100]) == [100]
        assert _await(lambda: _ns(s, "trade"), [0]) == [0]

        tp = _restart(tp, port, log_dir)
        # Wait for both handles to be back before publishing live rows.
        assert _await(lambda: tp.fn_call(".broker.list", []).height, 2) == 2
        _quote(tp, 101)
        _trade(tp, 2, "a")
        _trade(tp, 3, "b")
        # quote keeps flowing on its own handle, exactly once.
        assert _await(lambda: _ns(s, "quote"), [100, 101]) == [100, 101]
        # trade: replay is unfiltered (0, 1), live stays filtered (2 only), and
        # the row arrives once, not once per handle.
        assert _await(lambda: _ns(s, "trade"), [0, 1, 2]) == [0, 1, 2]
        s.shutdown()
        tp.shutdown()
