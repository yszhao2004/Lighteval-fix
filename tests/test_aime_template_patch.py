"""Does LightEval grade the AIME prompt's own template as the answer? Yes.

This is the minimal reproduction behind §5 of the README. The AIME prompt asks
for a last line of the form "Therefore, the final answer is:
$\\boxed{ANSWER}$. I hope it is correct". A reasoning model that restates that
line while it thinks produces a span matching the extractor's priority-0
pattern, and the unpatched extractor takes it: "ANSWER" parses as the product
A*E*N*R*S*W, and a correct answer written afterwards is never tried.

`patches/0002` is applied to a copy of the installed extractor, so this also
fails loudly -- rather than testing something else -- when the installed
LightEval is not the commit the patch was written against.

Skipped, not failed, when LightEval is not importable, like the LCB
reproduction: it is about someone else's code.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess

import pytest

pytest.importorskip("lighteval")

PATCH = (pathlib.Path(__file__).resolve().parents[1] / "patches"
         / "0002-extraction-skip-restated-answer-placeholder.patch")
MODULE = "src/lighteval/metrics/utils/extractive_match_utils.py"

# The model restates the template while reasoning, then answers after
# </think> in a form that ranks below the template's: the box is not wrapped
# in $...$, so only the lower-priority patterns match it. LightEval's
# reasoning-tag removal needs both <think> and </think> in the generation;
# with <think> supplied by the chat template it removes nothing, which is why
# the extractor below sees the whole text.
RESTATED = (
    "The format asks: Therefore, the final answer is: $\\boxed{ANSWER}$. "
    "I hope it is correct.\nWorking... so the result is 321.\n</think>\n"
    "Therefore, the final answer is \\(\\boxed{321}\\).")
TEMPLATE_ONLY = "The last line should read: the final answer is: $\\boxed{ANSWER}$. I hope"
CLEAN = "Therefore, the final answer is: $\\boxed{42}$. I hope it is correct"


@pytest.fixture(scope="module")
def extractors(tmp_path_factory):
    """(upstream, patched) copies of LightEval's extractive_match_utils."""
    from lighteval.metrics.utils import extractive_match_utils as upstream

    root = tmp_path_factory.mktemp("lighteval")
    target = root / MODULE
    target.parent.mkdir(parents=True)
    shutil.copy(upstream.__file__, target)
    applied = subprocess.run(["git", "apply", str(PATCH)], cwd=root,
                             capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    spec = importlib.util.spec_from_file_location("extractive_match_utils_0002", target)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    return upstream, patched


def _math_prediction(module, text):
    """What `pass_at_k_math` extracts from a generation."""
    from lighteval.utils.language import Language

    targets = [module.ExprExtractionConfig(), module.LatexExtractionConfig()]
    regexes = module.get_extraction_regexes(None, targets, Language.ENGLISH)
    return module.extract_target_from_pred(text, regexes, "first_match", "any_match", 5)


def _is_correct(module, gold, text):
    from lighteval.metrics.utils.math_comparison import compare_gold_target

    return compare_gold_target(_math_prediction(module, gold),
                               _math_prediction(module, text), 6, timeout_seconds=5)


def test_unpatched_extractor_takes_the_restated_template(extractors):
    """The reproduction: the template wins, the correct answer scores zero."""
    upstream, _ = extractors
    prediction = _math_prediction(upstream, RESTATED)
    assert any("ANSWER" in str(item) for item in prediction)
    assert not _is_correct(upstream, "321", RESTATED)


def test_patched_extractor_takes_the_answer(extractors):
    _, patched = extractors
    prediction = _math_prediction(patched, RESTATED)
    assert not any("ANSWER" in str(item) for item in prediction)
    assert _is_correct(patched, "321", RESTATED)


def test_template_alone_extracts_no_placeholder(extractors):
    _, patched = extractors
    assert not any("ANSWER" in str(item) for item in _math_prediction(patched, TEMPLATE_ONLY))


def test_patch_changes_nothing_without_a_restated_template(extractors):
    upstream, patched = extractors
    assert [str(x) for x in _math_prediction(upstream, CLEAN)] == \
           [str(x) for x in _math_prediction(patched, CLEAN)]
    assert _is_correct(patched, "42", CLEAN)


@pytest.mark.parametrize("text,letter", [("Answer: B", "B"), ("Answer: A", "A"),
                                         ("reasoning...\nAnswer: $LETTER\n...\nAnswer: C", "C")])
def test_letter_answers_are_still_read(extractors, text, letter):
    """GPQA's template word is skipped; a real letter, including "A", is not."""
    from lighteval.tasks.requests import Doc
    from lighteval.utils.language import Language

    _, patched = extractors
    doc = Doc(query="q", choices=["w", "x", "y", "z"], gold_index=0)
    targets = [patched.IndicesExtractionConfig(prefix_for_extraction="NativeLetters",
                                               try_extract_without_anchor=True)]
    regexes = patched.get_extraction_regexes(doc, targets, Language.ENGLISH)
    prediction = patched.extract_target_from_pred(text, regexes, "first_match", "any_match", 5)
    assert [str(x) for x in prediction][:1] == [letter]
