"""Tests for :class:`chili.engine.ChiliEngine`."""

import pytest
import polars as pl

from chili import ChiliEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Create a fresh engine for each test and shut it down afterwards."""
    e = ChiliEngine()
    yield e
    e.shutdown()


@pytest.fixture()
def lazy_engine():
    """Engine with lazy evaluation enabled."""
    e = ChiliEngine(lazy=True)
    yield e
    e.shutdown()


@pytest.fixture()
def pepper_engine():
    """Engine using Pepper syntax."""
    e = ChiliEngine(pepper=True)
    yield e
    e.shutdown()


# ---------------------------------------------------------------------------
# Construction & mode flags
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_modes(self, engine: ChiliEngine):
        assert engine.is_lazy_mode() is False
        assert engine.is_repl_use_chili_syntax() is True

    def test_lazy_mode(self, lazy_engine: ChiliEngine):
        assert lazy_engine.is_lazy_mode() is True

    def test_pepper_mode(self, pepper_engine: ChiliEngine):
        assert pepper_engine.is_repl_use_chili_syntax() is False


# ---------------------------------------------------------------------------
# eval – basic type round-trips
# ---------------------------------------------------------------------------


class TestEval:
    def test_eval_int(self, engine: ChiliEngine):
        assert engine.eval("1 + 2") == 3

    def test_eval_float(self, engine: ChiliEngine):
        assert engine.eval("3.14") == pytest.approx(3.14)

    def test_eval_bool_true(self, engine: ChiliEngine):
        assert engine.eval("true") is True

    def test_eval_bool_false(self, engine: ChiliEngine):
        assert engine.eval("false") is False

    def test_eval_string(self, engine: ChiliEngine):
        """Chili strings map to Python bytes."""
        result = engine.eval('"hello"')
        assert result == b"hello"

    def test_eval_arithmetic(self, engine: ChiliEngine):
        assert engine.eval("10 * 5 - 8") == 42

    def test_eval_error_raises(self, engine: ChiliEngine):
        with pytest.raises(Exception):
            engine.eval("undefined_var_xyz")


# ---------------------------------------------------------------------------
# Variable management
# ---------------------------------------------------------------------------


class TestVariables:
    def test_set_get_int(self, engine: ChiliEngine):
        engine.set_var("x", 42)
        assert engine.get_var("x") == 42

    def test_set_get_float(self, engine: ChiliEngine):
        engine.set_var("pi", 3.14)
        assert engine.get_var("pi") == pytest.approx(3.14)

    def test_set_get_str(self, engine: ChiliEngine):
        engine.set_var("name", "chili")
        assert engine.get_var("name") == "chili"

    def test_set_get_bool(self, engine: ChiliEngine):
        engine.set_var("flag", True)
        assert engine.get_var("flag") is True

    def test_set_get_none(self, engine: ChiliEngine):
        engine.set_var("empty", None)
        assert engine.get_var("empty") is None

    def test_set_get_list(self, engine: ChiliEngine):
        engine.set_var("items", [1, 2, 3])
        result = engine.get_var("items")
        assert result == [1, 2, 3]

    def test_set_get_dict(self, engine: ChiliEngine):
        engine.set_var("d", {"a": 1, "b": 2})
        result = engine.get_var("d")
        assert result["a"] == 1
        assert result["b"] == 2

    def test_set_get_dataframe(self, engine: ChiliEngine):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        engine.set_var("df", df)
        result = engine.get_var("df")
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_has_var_true(self, engine: ChiliEngine):
        engine.set_var("exists", 1)
        assert engine.has_var("exists") is True

    def test_has_var_false(self, engine: ChiliEngine):
        assert engine.has_var("nope_not_here") is False

    def test_del_var(self, engine: ChiliEngine):
        engine.set_var("to_delete", 99)
        assert engine.has_var("to_delete") is True
        result = engine.del_var("to_delete")
        assert result == 99
        assert engine.has_var("to_delete") is False

    def test_get_var_missing_raises(self, engine: ChiliEngine):
        with pytest.raises(Exception):
            engine.get_var("no_such_var")

    def test_overwrite_var(self, engine: ChiliEngine):
        engine.set_var("v", 1)
        assert engine.get_var("v") == 1
        engine.set_var("v", 2)
        assert engine.get_var("v") == 2


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------


class TestSource:
    def test_set_and_get_source(self, engine: ChiliEngine):
        idx = engine.set_source("test.chi", "1 + 1")
        path, src = engine.get_source(idx)
        assert path == "test.chi"
        assert src == "1 + 1"

    def test_multiple_sources(self, engine: ChiliEngine):
        idx1 = engine.set_source("a.chi", "10")
        idx2 = engine.set_source("b.chi", "20")
        assert idx1 != idx2
        assert engine.get_source(idx1) == ("a.chi", "10")
        assert engine.get_source(idx2) == ("b.chi", "20")


# ---------------------------------------------------------------------------
# Import source path
# ---------------------------------------------------------------------------


class TestImportSource:
    def test_import_missing_file_raises(self, engine: ChiliEngine):
        with pytest.raises(Exception):
            engine.import_source_path("", "/tmp/nonexistent_chili_file.chi")


# ---------------------------------------------------------------------------
# Tick counter
# ---------------------------------------------------------------------------


class TestTick:
    def test_initial_tick(self, engine: ChiliEngine):
        assert engine.get_tick_count(0) == 0

    def test_tick_increment(self, engine: ChiliEngine):
        engine.tick(0, 5)
        assert engine.get_tick_count(0) == 5

    def test_tick_multiple(self, engine: ChiliEngine):
        engine.tick(0, 3)
        engine.tick(0, 7)
        assert engine.get_tick_count(0) == 10


# ---------------------------------------------------------------------------
# Parse cache
# ---------------------------------------------------------------------------


class TestParseCache:
    def test_initial_cache_empty(self, engine: ChiliEngine):
        assert engine.parse_cache_len() == 0

    def test_cache_grows_after_eval(self, engine: ChiliEngine):
        engine.eval("1 + 2")
        assert engine.parse_cache_len() >= 1


# ---------------------------------------------------------------------------
# Displayed variables & list_vars
# ---------------------------------------------------------------------------


class TestVarListing:
    def test_get_displayed_vars_type(self, engine: ChiliEngine):
        result = engine.get_displayed_vars()
        assert isinstance(result, dict)

    def test_displayed_vars_contains_user_var(self, engine: ChiliEngine):
        engine.set_var("mytest", 123)
        dv = engine.get_displayed_vars()
        assert "mytest" in dv

    def test_list_vars_returns_dataframe(self, engine: ChiliEngine):
        result = engine.list_vars("")
        assert isinstance(result, pl.DataFrame)

    def test_list_vars_has_expected_columns(self, engine: ChiliEngine):
        result = engine.list_vars("")
        expected_cols = {"name", "display", "type", "columns", "is_built_in"}
        assert expected_cols.issubset(set(result.columns))

    def test_list_vars_pattern_filter(self, engine: ChiliEngine):
        engine.set_var("abc_test", 1)
        engine.set_var("xyz_test", 2)
        result = engine.list_vars("abc")
        names = result["name"].to_list()
        assert "abc_test" in names
        assert "xyz_test" not in names


# ---------------------------------------------------------------------------
# fn_call
# ---------------------------------------------------------------------------


class TestFnCall:
    def test_fn_call_type(self, engine: ChiliEngine):
        """Call the built-in 'type' function to check a value's type name."""
        result = engine.fn_call("type", [42])
        assert isinstance(result, str)

    def test_fn_call_count(self, engine: ChiliEngine):
        """count([1,2,3]) should return 3."""
        result = engine.fn_call("count", [[1, 2, 3]])
        assert result == 3


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_is_idempotent(self, engine: ChiliEngine):
        """Calling shutdown multiple times should not raise."""
        engine.shutdown()
        engine.shutdown()


# ---------------------------------------------------------------------------
# stats & start_tcp_listener
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_returns_dict(self, engine: ChiliEngine):
        s = engine.stats()
        assert isinstance(s, dict)

    def test_stats_contains_expected_keys(self, engine: ChiliEngine):
        s = engine.stats()
        assert "lazy_mode" in s
        assert "repl_lang" in s
        assert "parse_cache_len" in s


class TestTcpListener:
    def test_start_tcp_listener_binds_port(self):
        import socket
        import time

        e = ChiliEngine()
        # Find an available port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        e.start_tcp_listener(port)
        # Give the background thread a moment to bind
        time.sleep(0.2)
        # Verify something is listening on that port
        with socket.socket() as s:
            result = s.connect_ex(("127.0.0.1", port))
            assert result == 0, f"Nothing listening on port {port}"
        e.shutdown()
