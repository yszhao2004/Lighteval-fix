"""Is LightEval's stdin rewrite semantics-preserving? For one shape, no.

This test lifts `clean_if_name` and `make_function` out of LightEval and runs
a correct program both ways. It is the minimal reproduction behind §3 of the
README: a program that reads input at module level and reaches it through a
`global` declaration is turned into a NameError.

Skipped, not failed, when LightEval is not importable -- unlike the answer
tests, this one is about someone else's code, so its absence is not a
correctness signal about ours.
"""

from __future__ import annotations

import ast
import io
import sys
import textwrap

import pytest

pytest.importorskip("lighteval")


def _rewriters():
    """The two functions LightEval applies to a stdin candidate."""
    import inspect

    from lighteval.tasks.tasks.lcb import codegen_metrics

    source = inspect.getsource(codegen_metrics)
    start = source.index("def clean_if_name")
    end = source.index("def call_method")
    namespace = {"ast": ast, "import_string": ""}
    exec(compile(source[start:end], "lcb_rewriters", "exec"), namespace)
    return namespace["clean_if_name"], namespace["make_function"]


def _run(code: str, stdin: str, call_wrapped: bool):
    out = io.StringIO()
    saved_out, saved_in = sys.stdout, sys.stdin
    sys.stdout, sys.stdin = out, io.StringIO(stdin)
    try:
        scope = {"__name__": "__main__"}
        exec(compile(code, "candidate", "exec"), scope)
        if call_wrapped:
            scope["wrapped_function"]()
    except Exception as error:
        return out.getvalue().strip(), f"{type(error).__name__}: {error}"
    finally:
        sys.stdout, sys.stdin = saved_out, saved_in
    return out.getvalue().strip(), None


GLOBAL_DECL = textwrap.dedent("""
    import sys
    N = int(sys.stdin.readline())
    def solve():
        global N
        print(N * 2)
    solve()
""").strip()

PLAIN_TOP_LEVEL = textwrap.dedent("""
    import sys
    n = int(sys.stdin.readline())
    print(n * 2)
""").strip()

CLASS_READS_MODULE_NAME = textwrap.dedent("""
    import sys
    K = int(sys.stdin.readline())
    class C:
        limit = K
    print(C.limit)
""").strip()


def test_global_declaration_is_broken_by_the_rewrite():
    """The reproduction: correct as written, NameError after rewriting."""
    clean_if_name, make_function = _rewriters()
    written, written_error = _run(GLOBAL_DECL, "21\n", call_wrapped=False)
    assert (written, written_error) == ("42", None)

    rewritten = make_function(clean_if_name(GLOBAL_DECL))
    after, after_error = _run(rewritten, "21\n", call_wrapped=True)
    assert after_error is not None and "NameError" in after_error
    assert after != "42"


@pytest.mark.parametrize("program,expected",
                         [(PLAIN_TOP_LEVEL, "42"),
                          (CLASS_READS_MODULE_NAME, "21")])
def test_rewrite_is_harmless_for_these(program, expected):
    """The rewrite is not uniformly destructive, which is why it went unnoticed."""
    clean_if_name, make_function = _rewriters()
    assert _run(program, "21\n", call_wrapped=False) == (expected, None)
    rewritten = make_function(clean_if_name(program))
    assert _run(rewritten, "21\n", call_wrapped=True) == (expected, None)


def test_subprocess_runner_agrees_with_the_program_as_written():
    from lighteval_fix.lcb_stdin import run_stdin_case

    assert run_stdin_case(GLOBAL_DECL, "21\n", "42", timeout=10)
    assert run_stdin_case(PLAIN_TOP_LEVEL, "21\n", "42", timeout=10)
    assert not run_stdin_case(PLAIN_TOP_LEVEL, "21\n", "43", timeout=10)
