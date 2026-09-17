"""Eval-side fixes for LightEval: find the answer, then compare it properly.

Two independent halves.

**Grading** (no LightEval source change needed, works on finished runs):

    from lighteval_fix.extraction import extract_final_answer, answers_equivalent
    from lighteval_fix.lcb_stdin import run_stdin_case, compare_output

**Running** (install before importing LightEval's model modules, so a fresh
environment does not lose a run to a logging crash or silently change its
sampling):

    import lighteval_fix.harness as harness
    harness.apply_all()
"""

from lighteval_fix.extraction import (  # noqa: F401
    answers_equivalent,
    extract_all_candidates,
    extract_final_answer,
)

__all__ = [
    "extract_final_answer",
    "extract_all_candidates",
    "answers_equivalent",
]
