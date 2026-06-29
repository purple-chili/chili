"""Filtered subscribe delivers only matching rows on live broadcasts."""

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


class TestTickSubFiltered:
    def test_filtered_vs_unfiltered_subscribe(self):
        port = _free_port()

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
                filename=date.today(),
            )
            t.start_tcp_listener(port)
            time.sleep(0.1)

            s_all = ChiliEngine(pepper=True, debug=True)
            s_all.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
            time.sleep(0.1)

            s_flt = ChiliEngine(pepper=True, debug=True)
            s_flt.subscribe(
                f"chili://127.0.0.1:{port}",
                filters={"trade": ("sym", ["AAPL", "GOOG"])},
            )
            time.sleep(0.1)

            assert s_flt.get_var(".sub.filterColumn") == "sym"
            assert sorted(s_flt.get_var(".sub.filterValues")) == ["AAPL", "GOOG"]

            batch = pl.DataFrame(
                {
                    "sym": pl.Series(
                        ["AAPL", "MSFT", "GOOG", "TSLA"], dtype=pl.Categorical
                    ),
                    "price": [150.0, 300.0, 2800.0, 700.0],
                    "size": [100, 50, 200, 75],
                }
            )
            t.publish("trade", batch)
            time.sleep(0.2)

            all_trade = s_all.get_var("trade")
            flt_trade = s_flt.get_var("trade")

            assert all_trade.shape[0] == 4, f"unfiltered expected 4:\n{all_trade}"
            assert flt_trade.shape[0] == 2, f"filtered expected 2:\n{flt_trade}"
            got = sorted(str(x) for x in flt_trade["sym"].to_list())
            assert got == ["AAPL", "GOOG"], f"filtered syms wrong: {got}"

            assert flt_trade.columns == ["sym", "price", "size"]

            batch2 = pl.DataFrame(
                {
                    "sym": pl.Series(["GOOG", "NVDA"], dtype=pl.Categorical),
                    "price": [2810.0, 900.0],
                    "size": [10, 20],
                }
            )
            t.publish("trade", batch2)
            time.sleep(0.2)

            all_trade = s_all.get_var("trade")
            flt_trade = s_flt.get_var("trade")
            assert all_trade.shape[0] == 6, f"unfiltered expected 6:\n{all_trade}"
            assert flt_trade.shape[0] == 3, f"filtered expected 3:\n{flt_trade}"

            s_flt.shutdown()
            s_all.shutdown()
        t.shutdown()
