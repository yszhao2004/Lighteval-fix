"""Answer extraction and equivalence, for grading that is automatic and right.

The problem this solves is narrow and it is not about mathematics. A reasoning
model produces a long generation; somewhere in it is the answer; the grader has
to find that span and decide whether it means the same thing as the gold. Both
halves go wrong in ways that look like the model being wrong:

* **Extraction picks the wrong span.** LightEval's extractor ranks candidate
  patterns by priority, and ``\\boxed{...}`` ranks at 55 while the ``ANSWER:``
  final line ranks at 100 (``metrics/utils/extractive_match_utils.py``,
  ``lazy_latex_regex`` / ``lazy_expr_regex``). Lower wins. So for a task whose
  prompt asks for ``ANSWER: $ANSWER`` -- ``tasks/tasks/math_500.py:34-39``
  does exactly that -- a ``\\boxed{}`` that appears anywhere earlier, including
  inside a discarded line of reasoning, outranks the line the prompt asked for.

* **Equivalence is decided by string equality for anything non-numeric.**
  There is no string extraction target: ``ExtractionTarget`` is
  ``LatexExtractionConfig | ExprExtractionConfig | IndicesExtractionConfig``
  (``extractive_match_utils.py:96``), and LightEval's own comment says so --
  "There is currently no StringExtractionConfig, so if the gold is
  \\boxed{\\text{Friday}} and model outputs Friday it will not match, because
  nothing will be extracted" (``metrics/dynamic_metrics.py:174``). MATH-500
  does contain such answers.

The fix is deliberately *not* a normaliser. It is easy to write a regex pass
that makes ``288\\pi`` and ``288π`` compare equal, and every such pass also
makes things equal that are not: strip brackets to reconcile ``\\left(3\\right)``
with ``(3)`` and you have also equated the ordered pair ``(1,2)`` with the set
``{1,2}``. So:

    extraction decides *which string*; sympy decides *whether it means the same*.

Equivalence routes through LightEval's own ``compare_gold_target``, which is a
vendored copy of math-verify's logic (``metrics/utils/math_comparison.py``) and
already handles fractions against decimals, unsimplified radicals, intervals,
sets, relations and percentages. Nothing new is added to the dependency set --
``latex2sympy2_extended`` is already a base dependency of LightEval
(``pyproject.toml``), and ``math_verify`` itself is *not* used by LightEval at
all, so importing it would add a second, divergent comparator.

What this module adds on top is the one branch LightEval is missing: a text
answer compared as text, after removing only the wrappers that carry no value
(``\\text{}``, ``\\mathrm{}``, surrounding ``$``, case, whitespace). Brackets,
commas and operators are never touched.
"""

from __future__ import annotations

import re

__all__ = ["extract_final_answer", "answers_equivalent", "extract_all_candidates"]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
# The instruction families in use. Both spellings appear across tasks:
# math_500 asks for "ANSWER: $ANSWER", gpqa for "Answer: $LETTER".
# The bold markers matter: models emit "**Answer:** C" and "**Answer**: C" as
# often as the plain form, and a pattern that does not allow them either misses
# the line or captures the asterisks as part of the answer.
_ANSWER_LINE = re.compile(
    r"(?im)^[^\S\n]*\*{0,2}answer\*{0,2}[^\S\n]*:[^\S\n]*\*{0,2}[^\S\n]*"
    r"(?P<a>.+?)[^\S\n]*\*{0,2}[^\S\n]*$")
_THINK_CLOSE = "</think>"
_LATEX_ENV = re.compile(
    r"\$\$(?P<d>.+?)\$\$|\\\[(?P<b>.+?)\\\]|\$(?P<s>.+?)\$|\\\((?P<p>.+?)\\\)",
    re.S)


def _last_boxed(text: str) -> str | None:
    """The last ``\\boxed{...}``, brace-matched.

    A regex cannot do this: MATH answers nest braces (``\\boxed{\\frac{a}{b}}``)
    and a greedy or lazy pattern gets either too much or too little.
    """
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    while j < len(text) and depth:
        depth += (text[j] == "{") - (text[j] == "}")
        j += 1
    return text[i + len("\\boxed{"):j - 1] if depth == 0 else None


def extract_all_candidates(text: str) -> list[str]:
    """Every answer-shaped span, best first, for diagnostics."""
    out: list[str] = []
    after = text.split(_THINK_CLOSE)[-1] if _THINK_CLOSE in text else text
    for scope in (after, text):
        matches = list(_ANSWER_LINE.finditer(scope))
        if matches:
            out.append(matches[-1].group("a").strip())
        boxed = _last_boxed(scope)
        if boxed is not None:
            out.append(boxed.strip())
        envs = list(_LATEX_ENV.finditer(scope))
        if envs:
            groups = [g for g in envs[-1].groupdict().values() if g]
            if groups:
                out.append(groups[0].strip())
        if out:
            break
    seen, unique = set(), []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def extract_final_answer(text: str) -> str | None:
    """The span a grader should compare, or None if the answer is not stated.

    Priority, and the reasons for this order rather than LightEval's:

    1. **The requested format wins.** If the prompt asked for an ``ANSWER:``
       line and the model produced one, that is the model's answer -- not a
       ``\\boxed{}`` it wrote earlier while thinking. LightEval ranks boxed
       above it, which inverts the model's own intent.
    2. **Only after the thinking block.** A reasoning trace contains discarded
       candidates; anything before ``</think>`` is working, not an answer. The
       search runs on the post-``</think>`` text first and falls back to the
       whole string for models that emit no such marker.
    3. Then ``\\boxed{}``, then the last inline latex environment, which covers
       tasks whose prompt does mandate boxing (the AIME family does).

    Trailing punctuation is stripped because "ANSWER: 42." is the same answer
    as "ANSWER: 42"; nothing else about the span is altered.
    """
    candidates = extract_all_candidates(text)
    if not candidates:
        return None
    answer = candidates[0]
    answer = answer.strip().rstrip(".").strip()
    # A model that writes "ANSWER: \boxed{42}" gets unwrapped to the value.
    inner = _last_boxed(answer)
    if inner is not None:
        answer = inner.strip()
    return answer or None


# --------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------
# Wrappers that carry no value. Deliberately short: every entry here is a
# presentation command, never an operator, a bracket or a separator.
_TEXT_WRAPPER = re.compile(r"\\(?:text|mathrm|mathbf|mbox|textbf|operatorname)\s*\{([^{}]*)\}")
_TRIM = str.maketrans({"$": "", "\u00a0": " "})


def _as_text(value: str) -> str:
    """The string form of a word answer, for the branch sympy cannot take."""
    out = value.strip().translate(_TRIM)
    for _ in range(3):  # \text{\textbf{x}} does occur
        out = _TEXT_WRAPPER.sub(r"\1", out)
    out = out.replace("\\left", "").replace("\\right", "")
    out = re.sub(r"\s+", " ", out).strip()
    # "(C)" and "C" are the same choice; this is a parenthesis around a single
    # token only, never around an expression, so it cannot equate a pair with
    # a scalar.
    single = re.fullmatch(r"\(\s*([A-Za-z])\s*\)", out)
    if single:
        out = single.group(1)
    return out.casefold()


def _looks_textual(value: str) -> bool:
    """True for answers that are words, not mathematics.

    Guarded tightly: a single run of letters, optionally hyphenated or spaced,
    with no digits, operators, brackets or commas. "even", "ellipse",
    "Evelyn", "C" qualify; "x+1", "(1,2)", "2pi" do not.
    """
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z\s\-']*", _as_text(value)))


# A bracketed, comma-separated list: an ordered pair, an interval, or a set,
# depending entirely on which bracket it wears.
_BRACKETED = re.compile(r"^\s*\\?([\(\[\{])(?P<body>.*[,;].*?)\\?([\)\]\}])\s*$", re.S)
_BRACKET_KIND = {"(": "open", "[": "closed", "{": "set"}


def _bracket_kinds_conflict(gold: str, pred: str) -> bool:
    """True when two bracketed lists wear different brackets.

    This guard exists because LightEval's own comparator is more permissive
    here than it should be: ``sympy_compare_sets``
    (``metrics/utils/math_comparison.py``) coerces between intervals and
    tuples, so it reports ``(1,2)`` equal to ``{1,2}`` -- an ordered pair, or
    an open interval, equal to a two-element set. Under every reading those
    are different answers, and a grader that equates them will score a wrong
    answer correct.

    The check is syntactic and deliberately narrow: it fires only when *both*
    sides are a bracketed list containing a separator, and their outer
    brackets are of different kinds. ``\\left(1,2\\right)`` against ``(1,2)``
    is unaffected, because the wrappers are stripped before comparison and
    both are then "open". Nothing else about either string is inspected.
    """
    g_match = _BRACKETED.match(gold.replace("\\left", "").replace("\\right", ""))
    p_match = _BRACKETED.match(pred.replace("\\left", "").replace("\\right", ""))
    if not g_match or not p_match:
        return False
    return _BRACKET_KIND[g_match.group(1)] != _BRACKET_KIND[p_match.group(1)]


_TARGETS_CACHE: list | None = None


def _extraction_targets() -> list:
    """LightEval's ``target_res``: compiled regexes paired with their config.

    ``extract_target_from_pred`` does **not** take the config dataclasses --
    it takes ``list[tuple[list[tuple[Pattern, int]], ExtractionTarget]]``, the
    already-compiled patterns with their priorities. Passing configs makes the
    third positional argument land in ``fallback_mode`` and fails with a
    TypeError, which is how this was first got wrong.

    The regexes are built from ``lazy_expr_regex`` / ``lazy_latex_regex``
    rather than through ``get_extraction_regexes``, because that helper needs
    a ``Doc`` purely to size the multiple-choice letter list -- and no indices
    target is used here.

    ``boxed_match_priority=0`` promotes ``\\boxed{}`` to the top for *this*
    call only: the caller has already chosen the span, and it is handed to the
    parser wrapped in a box, so that wrapper should win over anything the
    value itself happens to look like. This does not change how any task is
    graded; it is local to comparing two answers that are already isolated.
    """
    global _TARGETS_CACHE
    if _TARGETS_CACHE is None:
        from lighteval.metrics.utils.extractive_match_utils import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            lazy_expr_regex,
            lazy_latex_regex,
        )
        from lighteval.utils.language import Language

        expr = ExprExtractionConfig()
        latex = LatexExtractionConfig(boxed_match_priority=0)
        _TARGETS_CACHE = [
            (lazy_expr_regex(expr, Language.ENGLISH), expr),
            (lazy_latex_regex(latex, Language.ENGLISH), latex),
        ]
    return _TARGETS_CACHE


def answers_equivalent(gold: str, pred: str, precision: int = 6,
                       timeout_seconds: int = 5) -> bool:
    """Do these two answer strings mean the same thing?

    Numbers, expressions, sets, intervals and relations go to LightEval's own
    ``compare_gold_target``; word answers take the text branch that LightEval
    has no target for. Returns False rather than raising when neither branch
    can decide, so a grader built on this never scores an item it did not
    understand as correct.
    """
    if gold is None or pred is None:
        return False
    g, p = str(gold).strip(), str(pred).strip()
    if not g or not p:
        return False

    # Text answers first: sympy would parse "even" as a symbol and "C" as a
    # symbol too, and then compare them structurally -- which happens to work
    # for single words but silently misbehaves for multi-word answers.
    if _looks_textual(g) and _looks_textual(p):
        return _as_text(g) == _as_text(p)
    if _looks_textual(g) != _looks_textual(p):
        return False

    if _bracket_kinds_conflict(g, p):
        return False

    from lighteval.metrics.utils.extractive_match_utils import extract_target_from_pred
    from lighteval.metrics.utils.math_comparison import compare_gold_target

    targets = _extraction_targets()

    def parsed(value: str):
        # The extractor expects a generation, not a bare answer, so give it
        # one in the shape its highest-priority pattern recognises. Wrapping
        # in \boxed first is what makes a bare "1/2" parse at all: the
        # unanchored patterns sit at priority 300 and are easy to miss.
        #
        # fallback_mode is "no_fallback" here, unlike the harness's
        # "first_match". A fallback appends the raw string when parsing fails,
        # and a raw string only ever compares byte-equal (math_comparison.py's
        # str/str branch) -- which is exactly the string comparison this
        # module exists to avoid. Better to report "could not parse" and let
        # the caller's own byte comparison decide.
        for shape in (f"\\boxed{{{value}}}", f"${value}$", value):
            found = extract_target_from_pred(
                shape, targets, "no_fallback", "any_match", timeout_seconds)
            if found:
                return found
        return []

    g_parsed, p_parsed = parsed(g), parsed(p)
    if g_parsed and p_parsed:
        return bool(compare_gold_target(g_parsed, p_parsed, precision,
                                        timeout_seconds=timeout_seconds))
    # Nothing parsed on one side: fall back to an exact comparison of the
    # trimmed strings. This is the same fallback LightEval uses, and it is
    # conservative -- it can only ever agree with a byte-identical answer.
    return g == p
