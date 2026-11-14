"""Regression: row-stamped subscriptions must replay message bounds."""
import collections
import socket
import tempfile
import time

import pytest
import polars as pl
from chili import ChiliEngine


def await_value(fn, want, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = fn()
        if got == want:
            return got
        time.sleep(0.01)
    raise AssertionError((got, want))


def run(stamped, inject_cold=False, inject_count=1):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as directory:
        tp = ChiliEngine(pepper=True)
        sub = ChiliEngine(pepper=True)
        try:
            schema = {"id": pl.Series([], dtype=pl.Int64)}
            if stamped:
                schema["seq"] = pl.Series([], dtype=pl.UInt64)
            tp.init_tick({"trade": pl.DataFrame(schema)}, directory + "/", "log")
            if stamped:
                tp.eval(".tick.upd: {[table; data] lpt[table; data; `seq; .tick.msgHandle]};")
            tp.fn_call("tock", [0, 1000])
            tp.set_var(".probe.frames", inject_count)
            tp.eval("""
                .probe.mode: 0;
                .probe.calls: 0;
                .probe.inject: {[ ]
                  .tick.upd[`trade; .probe.frame0];
                  if[.probe.frames = 3;
                    .tick.upd[`trade; .probe.frame1];
                    .tick.upd[`trade; .probe.frame2]]
                };
                .probe.original: .tick.subscribe;
                .tick.subscribe: {[topics]
                  if[.probe.mode = 2; .probe.inject[]];
                  info: .probe.original[topics];
                  if[.probe.mode = 1; .probe.inject[]];
                  .probe.serverHandle: this.h;
                  .probe.bound: info[3];
                  .probe.calls: .probe.calls + 1;
                  info
                };
            """)
            tp.start_tcp_listener(port)
            next_id = 0
            def publish_batch():
                nonlocal next_id
                tp.publish("trade", pl.DataFrame({"id": list(range(next_id, next_id + 50))}))
                next_id += 50
            for _ in range(10):
                publish_batch()
            if inject_cold:
                for i in range(inject_count):
                    tp.set_var(f".probe.frame{i}", pl.DataFrame({"id": [next_id + i]}))
                next_id += inject_count
                tp.set_var(".probe.mode", 1)
            sub.subscribe(f"chili://127.0.0.1:{port}", ["trade"])
            h = int(sub.list_handle().filter(pl.col("conn_type") == "Subscribing")["num"][0])
            def snapshot(label):
                ids = sub.get_var("trade")["id"].to_list()
                counts = collections.Counter(ids)
                return {"step": label, "published": next_id, "rows": len(ids),
                        "duplicates": [x for x, n in sorted(counts.items()) if n > 1],
                        "missing": sorted(set(range(next_id)) - counts.keys()),
                        "cursor": sub.fn_call("tick", [h, 0]),
                        "messages": tp.fn_call(".broker.validateSeqStrict", [directory + "/log", True]),
                        "bound": tp.get_var(".probe.bound")}
            results = [snapshot("cold")]
            if inject_cold:
                await_value(lambda: sub.get_var("trade").height, next_id)
                results = [snapshot("cold_injected")]
                assert results[0]["cursor"] == results[0]["messages"] == 10 + inject_count
                assert results[0]["rows"] == next_id
                assert not results[0]["duplicates"] and not results[0]["missing"]
                return results
            for _ in range(10):
                publish_batch()
            await_value(lambda: sub.get_var("trade").height, 1000)
            results.append(snapshot("live"))
            for mode, label in [(1, "recover_publish_after_registration"), (2, "recover_publish_before_registration")]:
                if mode == 2:
                    tp.set_var(".probe.mode", 0)
                    for _ in range(10):
                        publish_batch()
                    await_value(lambda: sub.get_var("trade").height, 1500 + inject_count)
                for i in range(inject_count):
                    tp.set_var(f".probe.frame{i}", pl.DataFrame({"id": [next_id + i]}))
                next_id += inject_count
                tp.set_var(".probe.mode", mode)
                prior = tp.get_var(".probe.calls")
                tp.fn_call(".handle.close", [tp.get_var(".probe.serverHandle")])
                await_value(lambda: tp.get_var(".probe.calls"), prior + 1)
                await_value(lambda: sub.fn_call("tick", [h, 0]), 20 + inject_count if mode == 1 else 30 + 2 * inject_count)
                results.append(snapshot(label))
            for result in results:
                assert result["rows"] == result["published"], result
                assert result["cursor"] == result["messages"], result
                assert not result["duplicates"] and not result["missing"], result
            return results
        finally:
            sub.shutdown()
            tp.shutdown()


@pytest.mark.parametrize("inject_count", [1, 3])
@pytest.mark.parametrize("stamped", [False, True])
@pytest.mark.parametrize("inject_cold", [False, True])
def test_message_bound_prevents_duplicate_then_skip(stamped, inject_cold, inject_count):
    run(stamped, inject_cold, inject_count)


def test_legacy_handshake_rejected_before_replay():
    engine = ChiliEngine(pepper=True)
    try:
        engine.load_sub()
        with pytest.raises(Exception, match="message replay bound"):
            engine.eval('.sub.replayEnd[("log"; 500; {})]')
    finally:
        engine.shutdown()


def test_filtered_stamped_handshake_uses_message_end():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory() as directory:
        tp, sub = ChiliEngine(pepper=True), ChiliEngine(pepper=True)
        try:
            tp.init_tick({"trade": pl.DataFrame(schema={"id": pl.Int64, "sym": pl.String, "seq": pl.UInt64})}, directory + "/", "log")
            tp.eval("""
                .tick.upd: {[table; data] lpt[table; data; `seq; .tick.msgHandle]};
                .probe.original: .tick.subscribeFiltered;
                .tick.subscribeFiltered: {[topic; column; values]
                  info: .probe.original[topic; column; values];
                  .tick.upd[`trade; .probe.frame];
                  info
                };
            """)
            tp.publish("trade", pl.DataFrame({"id": [0, 1], "sym": ["x", "y"]}))
            tp.set_var(".probe.frame", pl.DataFrame({"id": [2], "sym": ["x"]}))
            tp.start_tcp_listener(port)
            sub.subscribe(f"chili://127.0.0.1:{port}", [], filters={"trade": ("sym", ["x"])})
            await_value(lambda: sub.get_var("trade").height, 3)
            assert sub.get_var("trade")["id"].to_list() == [0, 1, 2]
        finally:
            sub.shutdown()
            tp.shutdown()
