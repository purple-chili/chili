"""Python FFI, ownership and concurrency coverage for compaction and writing."""

import gc
import threading
import time

import polars as pl
import pytest
from chili import ChiliEngine, ChiliError


def test_compact_snapshot_survives_engine_and_parquet_roundtrip(tmp_path):
    engine = ChiliEngine(pepper=True, compaction_interval=0)
    original = pl.DataFrame({"seq": range(5000), "value": ["large prefix"] * 5000})
    engine.set_var("buf", original)
    for i in range(5000, 5500):
        engine.fn_call(
            "vstack",
            [
                "buf",
                pl.DataFrame(
                    {"seq": [i], "value": [None if i % 7 == 0 else "tail"]},
                    schema={"seq": pl.Int64, "value": pl.String},
                ),
            ],
        )
    snapshot = engine.get_var("buf")
    assert engine.fn_call("compact", ["buf", 128]) == 5500
    compacted = engine.get_var("buf")
    assert compacted.equals(snapshot)
    path = tmp_path / "compacted.parquet"
    size = engine.fn_call("wparquet", [str(path), compacted, 3])
    assert size == path.stat().st_size
    assert pl.read_parquet(path).equals(snapshot)
    engine.shutdown()
    del engine, original
    gc.collect()
    # Exported Arrow buffers must retain the right owner/deallocator, including
    # when the Python extension and Python Polars use different allocators.
    assert compacted.equals(snapshot)
    assert snapshot["seq"].to_list() == list(range(5500))


def test_python_compaction_preserves_concurrent_appends():
    engine = ChiliEngine(pepper=True, compaction_interval=0)
    engine.set_var("buf", pl.DataFrame(schema={"seq": pl.Int64}))
    errors = []
    ready = threading.Barrier(3)

    def append(offset):
        try:
            ready.wait()
            for i in range(500):
                engine.fn_call("vstack", ["buf", pl.DataFrame({"seq": [offset + i]})])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(offset,)) for offset in [0, 500]]
    try:
        for thread in threads:
            thread.start()
        ready.wait()
        for _ in range(20):
            engine.fn_call("compact", ["buf", 128])
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert not errors
        values = engine.get_var("buf")["seq"].to_list()
        assert len(values) == 1000
        assert sorted(values) == list(range(1000))
    finally:
        for thread in threads:
            thread.join(timeout=10)
        engine.shutdown()


def test_default_background_compaction_keeps_large_prefix_and_snapshot():
    engine = ChiliEngine(pepper=True)
    try:
        original = pl.DataFrame({"seq": range(100_000)})
        engine.set_var("buf", original)
        for i in range(100_000, 100_500):
            engine.fn_call("vstack", ["buf", pl.DataFrame({"seq": [i]})])
        snapshot = engine.get_var("buf")
        deadline = time.monotonic() + 3
        while engine.get_var("buf").n_chunks() > 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        live = engine.get_var("buf")
        assert live.equals(snapshot)
        assert live["seq"].get_chunks()[0].len() == 100_000
        assert live["seq"].to_list() == list(range(100_500))
    finally:
        engine.shutdown()


def test_background_configuration_and_shutdown():
    engine = ChiliEngine(pepper=True, compaction_interval=0)
    try:
        for i in range(500):
            engine.fn_call("vstack", ["buf", pl.DataFrame({"seq": [i]})])
        time.sleep(0.3)
        assert engine.get_var("buf").n_chunks() == 500
        with pytest.raises(ChiliError, match="positive"):
            engine.set_auto_compaction(target_rows=0)
        with pytest.raises(OverflowError):
            engine.set_auto_compaction(interval_ms=-1)
        engine.set_auto_compaction(interval_ms=10, target_rows=128)
        deadline = time.monotonic() + 3
        while engine.get_var("buf").n_chunks() > 4:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert engine.get_var("buf")["seq"].to_list() == list(range(500))
    finally:
        engine.shutdown()
    with pytest.raises(ChiliError, match="shut down"):
        engine.set_auto_compaction()
