"""Running a LiveCodeBench stdin candidate the way it was written.

LightEval's stdin path does not execute the candidate program as written. It
rewrites it first (``tasks/tasks/lcb/codegen_metrics.py:265-268``):

* ``clean_if_name`` (line 86) strips a trailing ``if __name__ == '__main__':``
  block and splices its body back at module level via ``ast.unparse``.
* ``make_function`` (line 102) collects every statement that is *not* an
  import and re-emits them as the body of ``def wrapped_function():``.

The rewritten source is then exec'd in-process with ``sys.stdout`` and
``sys.stdin`` monkeypatched (``Capturing``, line 72; ``call_method``, line 132).

That transformation is not semantics-preserving for the shape competitive
Python is usually written in. Module-level names become locals of
``wrapped_function``, so anything that reaches them through the module
namespace rather than a closure -- a ``global`` declaration, a class body, an
``exec``/``eval`` against ``globals()`` -- resolves differently or not at all.

**What is measured and what is not.** Running the same candidate programs as
ordinary subprocesses, with LightEval's own 6 s per-test timeout and a
comparison that matches LightEval's (outer strip, per-line strip, then numeric
token comparison), recovers a large fraction of the stdin problems LightEval
scores as failures. The timeout is ruled out: repeating the re-run at 6 s
rather than a more generous 10 s changes the outcome for not one problem. The
whitespace story is ruled out too -- LightEval already strips, at
``codegen_metrics.py:192-196``, so the often-repeated "it compares bytes and a
trailing newline fails it" is simply not what the code does. That leaves the
rewrite as the remaining difference. It has not been reduced to a minimal
failing program here, so it is the strong suspect, not a proven cause.

The one real difference in comparison semantics is deliberate and is called
out rather than hidden: LightEval compares numeric lines with exact
``Decimal`` equality (``codegen_metrics.py:342``), which rejects
``0.5000001`` against ``0.5``. ``compare_output`` below applies a relative
tolerance, because a float-valued problem otherwise fails on the last digit.
Pass ``tolerance=0`` for exactly LightEval's behaviour.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

__all__ = ["compare_output", "run_stdin_case"]


def _lines(value: str) -> list[str]:
    """Outer strip, then per-line strip -- the same shape LightEval uses."""
    return [line.strip() for line in value.strip().splitlines()]


def compare_output(got: str, want: str, tolerance: float = 1e-6) -> bool:
    """Is this program's stdout the expected output?

    Three attempts, cheapest first:

    1. whole-output equality after stripping;
    2. line-by-line equality after stripping each line;
    3. numeric comparison per line, token by token, with a *relative*
       tolerance -- absolute tolerance is wrong across magnitudes, and exact
       equality is wrong for anything a float formatter touched.

    ``tolerance=0`` disables step 3's slack and reproduces LightEval's exact
    numeric comparison.
    """
    if got.strip() == want.strip():
        return True
    got_lines, want_lines = _lines(got), _lines(want)
    if got_lines == want_lines:
        return True
    if len(got_lines) != len(want_lines):
        return False
    for got_line, want_line in zip(got_lines, want_lines):
        if got_line == want_line:
            continue
        got_tokens, want_tokens = got_line.split(), want_line.split()
        if len(got_tokens) != len(want_tokens):
            return False
        for got_token, want_token in zip(got_tokens, want_tokens):
            if got_token == want_token:
                continue
            try:
                a, b = float(got_token), float(want_token)
            except ValueError:
                return False
            if abs(a - b) > tolerance * max(1.0, abs(b)):
                return False
    return True


def run_stdin_case(code: str, stdin: str, expected: str, timeout: float = 6.0,
                   tolerance: float = 1e-6) -> bool:
    """One (program, stdin, expected) trial, as a subprocess.

    A subprocess rather than an in-process exec for three reasons: the program
    runs with the module semantics it was written against, a crash or a
    ``sys.exit`` cannot take the grader with it, and a runaway loop is bounded
    by the kernel rather than by a signal handler that a busy C extension can
    ignore.

    ``timeout`` defaults to 6 s, which is LightEval's own value, so this is not
    quietly more generous than the harness it is compared against.
    """
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    try:
        handle.write(code)
        handle.close()
        try:
            done = subprocess.run([sys.executable, handle.name], input=stdin,
                                  capture_output=True, text=True,
                                  timeout=timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return False
        return compare_output(done.stdout, expected, tolerance)
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
