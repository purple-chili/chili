"""Log-aware replies preserve Python str paths alongside message bounds."""

import socket

import polars as pl
import pytest

from chili import ChiliEngine


@pytest.mark.parametrize("filtered", [False, True])
def test_subscription_path_is_str_and_replay_end_counts_messages(tmp_path, filtered):
    tp = ChiliEngine(pepper=True)
    client = ChiliEngine(pepper=True)
    path = tmp_path / "log with spaces λ"
    try:
        tp.init_tick(
            {"trade": pl.DataFrame(schema={"sym": pl.String, "seq": pl.UInt64})},
            str(tmp_path) + "/", path.name,
        )
        tp.eval('.tick.upd: {[table; data] lpt[table; data; `seq; .tick.msgHandle]};')
        tp.publish("trade", pl.DataFrame({"sym": ["a", "b", "a"]}))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        tp.start_tcp_listener(port)
        client.set_var("h", client.open_handle(f"chili://127.0.0.1:{port}"))
        request = (
            'h (`.tick.subscribeFiltered; `trade; `sym; enlist `a)'
            if filtered else 'h (`.tick.subscribe; enlist `trade)'
        )
        info = client.eval(request)
        assert type(info[0]) is str
        assert info[0] == str(path)
        assert info[1] == 3  # stamped row count
        assert info[3] == 1  # exclusive message end
        # Pepper strings elsewhere must retain their existing Python bytes type.
        assert client.eval('"unchanged"') == b"unchanged"
    finally:
        client.shutdown()
        tp.shutdown()
