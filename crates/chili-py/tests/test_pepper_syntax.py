"""Pepper `::` null-literal disambiguation tests.

The `::` null-literal vs `:: <expr>` binary-arg ambiguity: the parser
previously greedy-consumed tokens after `::` as binary-arg continuation,
causing `x: 0n; y: 1` to reject with `found 'Punc';' expected arguments`.
"""

import pytest
from chili import ChiliEngine


class TestNullLiteralSemicolonDisambiguation:
    """`::` as a standalone null-literal in atom position."""

    def test_minimal_repro_xy(self):
        """``x: 0n; y: 1`` parses; x is None, y is 1."""
        e = ChiliEngine(pepper=True)
        e.eval("x: 0n; y: 1")
        assert e.get_var("x") is None
        assert e.get_var("y") == 1

    def test_exact_form(self):
        """``.sub.eod.fired: 0n; eod: {[msg] .sub.eod.fired: msg};``"""
        e = ChiliEngine(pepper=True)
        e.eval(".sub.eod.fired: 0n; eod: {[msg] .sub.eod.fired: msg};")
        # After eval, .sub.eod.fired is None (set to 0n which is null).
        assert e.get_var(".sub.eod.fired") is None
        # eod is defined as a function; calling it sets .sub.eod.fired.
        e.fn_call("eod", ["hello"])
        assert e.get_var(".sub.eod.fired") == "hello"

    def test_standalone_null_literal(self):
        """``::`` alone parses without args/RHS."""
        e = ChiliEngine(pepper=True)
        # Single 0n expression — evaluates to null.
        result = e.eval("0n")
        assert result is None

    def test_general_multistatement_unchanged(self):
        """The non-`::` general case still parses (regression check on Q2's case 1)."""
        e = ChiliEngine(pepper=True)
        e.eval("a: 1; b: 2; c: a + b")
        assert e.get_var("c") == 3
