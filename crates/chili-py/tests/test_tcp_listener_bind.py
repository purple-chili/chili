"""Tests for synchronous ``start_tcp_listener`` bind behavior."""

import socket

import pytest
from chili import ChiliEngine, ChiliError


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_bind_taken_port_raises_synchronously():
    port = _free_port()
    hog = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    hog.bind(("127.0.0.1", port))
    hog.listen(1)
    try:
        eng = ChiliEngine()
        with pytest.raises(ChiliError) as ei:
            eng.start_tcp_listener(port)
        assert "in use" in str(ei.value).lower() or "bind failed" in str(ei.value).lower()
    finally:
        hog.close()


def test_clean_bind_on_free_port_returns():
    port = _free_port()
    eng = ChiliEngine()
    eng.start_tcp_listener(port)


def test_second_live_bind_same_port_raises():
    port = _free_port()
    eng1 = ChiliEngine()
    eng1.start_tcp_listener(port)
    eng2 = ChiliEngine()
    with pytest.raises(ChiliError):
        eng2.start_tcp_listener(port)


def test_so_reuseaddr_rebind_after_close():
    port = _free_port()
    eng1 = ChiliEngine()
    eng1.start_tcp_listener(port)
    c = socket.create_connection(("127.0.0.1", port), timeout=2.0)
    c.close()
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    p2 = _free_port()
    probe.bind(("127.0.0.1", p2))
    probe.close()
