"""Keep process RSS in .os.mem when refreshing only the needed metrics."""

import pytest

from chili import ChiliEngine


def test_os_mem_reports_process_rss_and_system_ram(monkeypatch):
    monkeypatch.setenv("CHILI_MEMORY_LIMIT", "2048.5")
    engine = ChiliEngine(pepper=True)
    try:
        memory = engine.fn_call(".os.mem", [])
        rss_mib = engine.stats()["process_memory_rss_bytes"] / (1024 * 1024)

        assert set(memory) == {"used", "limit", "total", "avail"}
        assert all(isinstance(value, float) for value in memory.values())
        assert memory["used"] > 0
        assert memory["used"] == pytest.approx(rss_mib, abs=8)
        assert memory["limit"] == 2048.5
        assert 0 <= memory["avail"] <= memory["total"]
        assert memory["total"] > 0
    finally:
        engine.shutdown()
