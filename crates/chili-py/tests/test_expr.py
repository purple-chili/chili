"""Polars expressions passed between Python and Chili."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from chili import ChiliEngine
from chili.engine_state import ChiliEvalError, TypeMismatchError


@pytest.fixture()
def engine():
    engine = ChiliEngine(pepper=True)
    yield engine
    engine.shutdown()


@pytest.fixture()
def trades():
    return pl.DataFrame(
        {
            "sym": ["a", "b", "a", "c"],
            "qty": [1, 2, 3, 4],
            "price": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_functional_select_groups_by_polars_column(engine, trades):
    engine.set_var("trades", trades)
    result = engine.fn_call(".fn.select", ["trades", [], [], [pl.col("sym")], [], 0])
    expected = trades.group_by("sym", maintain_order=True).agg(pl.all().last())
    assert_frame_equal(result, expected)


def test_functional_select_filters_and_aggregates(engine, trades):
    engine.set_var("trades", trades)
    where = [pl.col("qty") > 0, pl.col("price") < 35.0]
    group = [pl.col("sym")]
    operations = [(pl.col("qty") * pl.col("price")).sum().alias("value")]
    result = engine.fn_call(".fn.select", ["trades", [], where, group, operations, 0])
    expected = trades.filter(*where).group_by(group, maintain_order=True).agg(operations)
    assert_frame_equal(result, expected)


def test_expr_variable_round_trip(engine, trades):
    expr = (pl.col("qty") + 1).alias("next_qty")
    engine.set_var("expr", expr)
    restored = engine.get_var("expr")
    assert isinstance(restored, pl.Expr)
    assert_frame_equal(trades.select(restored), trades.select(expr))


def test_chili_column_expression_returns_polars_expr(engine, trades):
    expr = engine.fn_call("col", ["qty"])
    assert isinstance(expr, pl.Expr)
    assert_frame_equal(trades.select(expr), trades.select("qty"))
    assert isinstance(engine.eval("col `qty"), pl.Expr)


def test_chili_aggregation_returns_correct_polars_expr(engine, trades):
    expr = engine.eval("sum[col `qty]")
    assert isinstance(expr, pl.Expr)
    assert_frame_equal(trades.select(expr), trades.select(pl.col("qty").sum()))


def test_expr_in_nested_containers(engine, trades):
    expr = pl.col("qty").sum()
    engine.set_var("expressions", {"ops": [expr]})
    restored = engine.get_var("expressions")["ops"][0]
    assert isinstance(restored, pl.Expr)
    assert_frame_equal(trades.select(restored), trades.select(expr))


def test_plain_strings_remain_literals(engine, trades):
    engine.set_var("trades", trades)
    result = engine.fn_call(".fn.select", ["trades", [], [], ["sym"], [], 0])
    assert result.height == 1
    assert result["literal"].to_list() == ["sym"]


def test_missing_column_propagates_query_error(engine, trades):
    engine.set_var("trades", trades)
    with pytest.raises(ChiliEvalError, match="missing_column"):
        engine.fn_call(
            ".fn.select", ["trades", [], [], [pl.col("missing_column")], [], 0]
        )


def test_incompatible_expression_format_raises(engine):
    class IncompatibleExpr:
        _pyexpr = object()

        @property
        def meta(self):
            return self

        def serialize(self, *, format):
            assert format == "json"
            return '{"UnknownExprVariant":"qty"}'

    with pytest.raises(TypeMismatchError, match="incompatible expression format"):
        engine.set_var("expr", IncompatibleExpr())
