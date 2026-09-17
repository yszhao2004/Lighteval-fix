"""Output comparison: the same normalisation LightEval does, plus stated slack.

No LightEval import needed -- this half is self-contained.
"""

from __future__ import annotations

import pytest

from lighteval_fix.lcb_stdin import compare_output

EQUAL = [
    ("3\n", "3", "print() always emits the newline the gold lacks"),
    ("3\n", "3\n", "identical"),
    ("1 2 3\n", "1 2 3", "spacing preserved"),
    ("1  2\n", "1 2", "interior spacing between numeric tokens is not meaningful"),
    ("1\n2\n", "1\n2", "multiple lines"),
    ("  4  \n", "4", "surrounding whitespace"),
    ("0.5000001\n", "0.5", "relative float tolerance"),
    ("1000000.1\n", "1000000.0", "tolerance is relative, so it scales"),
]

DIFFERENT = [
    ("0.6\n", "0.5", "genuinely different value"),
    ("1\n2\n", "2\n1", "order matters"),
    ("1\n", "1\n2", "line count matters"),
    ("YES\n", "yes", "case matters for non-numeric output"),
    ("1 2\n", "1 2 3", "token count matters"),
]


@pytest.mark.parametrize("got,want,why", EQUAL)
def test_equal(got, want, why):
    assert compare_output(got, want), why


@pytest.mark.parametrize("got,want,why", DIFFERENT)
def test_different(got, want, why):
    assert not compare_output(got, want), why


def test_zero_tolerance_reproduces_lighteval_strictness():
    """tolerance=0 is LightEval's exact Decimal comparison."""
    assert compare_output("0.5000001\n", "0.5")
    assert not compare_output("0.5000001\n", "0.5", tolerance=0)
    assert compare_output("0.5\n", "0.5", tolerance=0)
