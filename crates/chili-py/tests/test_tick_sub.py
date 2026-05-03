"""Tests for tick/sub publish-subscribe flow between two engine instances."""

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


class TestTickSub:
    """Spin up a tick engine and a subscriber engine, publish data, and
    verify the subscriber receives it."""

    def test_publish_and_subscribe(self):
        port = _free_port()

        # -- tick engine (publisher) --
        t = ChiliEngine(pepper=True, debug=True)
        with tempfile.TemporaryDirectory() as log_dir:
            trade_schema = pl.DataFrame(
                {
                    "sym": pl.Series([], dtype=pl.Categorical),
                    "price": pl.Series([], dtype=pl.Float64),
                    "size": pl.Series([], dtype=pl.Int64),
                }
            )
            t.init_tick(
                schema={"trade": trade_schema},
                log_dir=log_dir + "/",
                date=date.today(),
            )

            # Start TCP listener so the subscriber can connect
            t.start_tcp_listener(port)
            time.sleep(0.1)

            # Publish first batch
            data1 = pl.DataFrame(
                {
                    "sym": pl.Series(["AAPL", "GOOG"], dtype=pl.Categorical),
                    "price": [150.0, 2800.0],
                    "size": [100, 200],
                }
            )
            t.publish("trade", data1)
            assert t.get_tick_count(0) >= 1

            # -- subscriber engine --
            s = ChiliEngine(pepper=True, debug=True)
            s.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
            assert s.get_var(".sub.topics") == ["trade"]
            time.sleep(0.1)

            # Diagnostics
            t_handles = t.list_handle()
            print(f"\n[DIAG] tick handles:\n{t_handles}")
            assert t_handles.shape[0] >= 2, f"tick handles:\n{t_handles}"

            # Check broker subscriber list on tick
            broker_list = t.fn_call(".broker.list", [])
            print(f"\n[DIAG] broker subscribers:\n{broker_list}")

            s_handles = s.list_handle()
            print(f"\n[DIAG] sub handles:\n{s_handles}")

            # Subscriber should have the replayed data
            sub_trade = s.get_var("trade")
            print(f"\n[DIAG] sub trade after replay:\n{sub_trade}")
            assert sub_trade.shape[0] == 2, f"expected 2 rows from replay:\n{sub_trade}"

            # Publish second batch directly (same thread)
            data2 = pl.DataFrame(
                {
                    "sym": pl.Series(["MSFT"], dtype=pl.Categorical),
                    "price": [300.0],
                    "size": [50],
                }
            )
            t.publish("trade", data2)
            tick_count = t.get_tick_count(0)
            print(f"\n[DIAG] tick count after 2nd publish: {tick_count}")
            time.sleep(0.1)

            sub_trade = s.get_var("trade")
            print(f"\n[DIAG] sub trade after 2nd publish:\n{sub_trade}")
            assert sub_trade.shape[0] == 3, f"expected 3 rows, got:\n{sub_trade}"

            s.shutdown()
        t.shutdown()
