"""Acceptance test for subscriber-side ``eod`` dispatch.

Verifies that a subscriber receives and dispatches the ``eod`` message
when the publisher fires ``.tick.eod[date]``.

API notes:
- ``subscribe(uri, [topics])`` — single socket URI + topics list.
- ``has_var(id)`` returns ``bool``; ``get_var`` raises ``NameError``
  when the var is unset.
- ``::`` (pepper null literal) round-trips to Python ``None``.
- ``pub_port`` must be captured AFTER ``start_tcp_listener`` by reading
  the OS-assigned port back from the engine.
"""

import socket
import tempfile
import time
from datetime import date

import polars as pl
from chili import ChiliEngine


def _free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_subscriber_eod_shim_triggered_by_publisher_eod():
    """End-to-end: publisher fires .tick.eod[date]; subscriber's
    pepper-level ``eod`` handler should write ``.sub.eod.fired`` so
    Python can observe the message arrived."""
    port = _free_port()

    # -- publisher: tp engine with TCP listener --
    pub = ChiliEngine(pepper=True)
    with tempfile.TemporaryDirectory() as log_dir:
        trade_schema = pl.DataFrame(
            {
                "sym": pl.Series([], dtype=pl.Categorical),
                "price": pl.Series([], dtype=pl.Float64),
                "size": pl.Series([], dtype=pl.Int64),
            }
        )
        pub.init_tick(
            schema={"trade": trade_schema},
            log_dir=log_dir + "/",
            date=date.today(),
        )
        pub.start_tcp_listener(port)
        time.sleep(0.1)

        # -- subscriber: define eod shim BEFORE subscribing --
        sub = ChiliEngine(pepper=True)
        # `.sub.eod.fired: 0n` initializes the sentinel to pepper null
        # (which round-trips to Python None). `eod: {[msg] .sub.eod.fired: msg}`
        # defines the shim that captures the message into the global var.
        sub.eval(".sub.eod.fired: 0n")
        sub.eval("eod: {[msg] .sub.eod.fired: msg}")
        sub.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
        time.sleep(0.1)

        # -- publisher fires .tick.eod[date] --
        # This routes pub → .broker.eod → signal_eod → sync(h, (`eod; date))
        # to each Publishing handle. Subscriber's handle_chili_conn receives
        # the message and calls state.eval → eval_op → should dispatch
        # `eod[date]` invoking the shim.
        pub.eod(date.today())

        # -- wait for the eod shim to fire --
        eod_msg = None
        for _ in range(100):  # 100 * 50ms = 5s ceiling
            if sub.has_var(".sub.eod.fired"):
                got = sub.get_var(".sub.eod.fired")
                # Initial value is pepper null = Python None.
                # The shim writes a non-None value when it fires.
                if got is not None:
                    eod_msg = got
                    break
            time.sleep(0.05)

        # The bug: eod_msg stays None forever; .sub.eod.fired never gets
        # written by the eod shim. Either eod was never invoked (H1, H4)
        # or there's a timing race we should reconsider (H5).
        assert eod_msg is not None, (
            "subscriber's eod shim never fired — .sub.eod.fired is still "
            "None / unset after publisher's .tick.eod[date] broadcast. "
            "Expected the shim to have written the (`eod; <date>) message."
        )

        sub.shutdown()
    pub.shutdown()


def test_multi_message_subscriber_observes_upd_then_eod():
    """Verify a subscriber thread can process a multi-message sequence
    (an upd followed by an eod) and both side-effects land.
    """
    port = _free_port()

    pub = ChiliEngine(pepper=True)
    with tempfile.TemporaryDirectory() as log_dir:
        trade_schema = pl.DataFrame(
            {
                "sym": pl.Series([], dtype=pl.Categorical),
                "price": pl.Series([], dtype=pl.Float64),
                "size": pl.Series([], dtype=pl.Int64),
            }
        )
        pub.init_tick(
            schema={"trade": trade_schema},
            log_dir=log_dir + "/",
            date=date.today(),
        )
        pub.start_tcp_listener(port)
        time.sleep(0.1)

        sub = ChiliEngine(pepper=True)
        sub.eval(".sub.eod.fired: 0n")
        sub.eval("eod: {[msg] .sub.eod.fired: msg}")
        sub.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
        time.sleep(0.1)

        # Step 1 — publisher fires an upd; subscriber's `upd` (from
        # sub.pep) should upsert into `trade`.
        df = pl.DataFrame(
            {
                "sym": pl.Series(["AAPL"], dtype=pl.Categorical),
                "price": [150.0],
                "size": [100],
            }
        )
        pub.publish("trade", df)
        time.sleep(0.1)

        # Step 2 — publisher fires EOD; subscriber's `eod` shim should
        # write `.sub.eod.fired`.
        pub.eod(date.today())

        eod_msg = None
        for _ in range(100):
            if sub.has_var(".sub.eod.fired"):
                got = sub.get_var(".sub.eod.fired")
                if got is not None:
                    eod_msg = got
                    break
            time.sleep(0.05)

        # Both side-effects must have landed.
        sub_trade = sub.get_var("trade")
        assert sub_trade.shape[0] == 1, (
            f"subscriber's upd shim should have one row after the pub upd; "
            f"got shape {sub_trade.shape}"
        )
        assert eod_msg is not None, (
            "subscriber's eod shim should have fired after the multi-message "
            "sequence; `.sub.eod.fired` is still unset / None"
        )

        sub.shutdown()
    pub.shutdown()
