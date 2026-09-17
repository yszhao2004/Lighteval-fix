"""Eval-side fixes for LightEval: find the answer, then compare it properly.

Nothing here changes generation. Every fix is downstream of the model output,
which is the point: a run that already cost hours should not be re-run because
the grader misread what it produced.

    from lighteval_fix.extraction import extract_final_answer, answers_equivalent
    from lighteval_fix.lcb_stdin import run_stdin_case
"""

from lighteval_fix.extraction import (  # noqa: F401
    answers_equivalent,
    extract_all_candidates,
    extract_final_answer,
)

__all__ = ["extract_final_answer", "extract_all_candidates", "answers_equivalent"]
