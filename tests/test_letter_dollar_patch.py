"""Does LightEval read "Answer: $C", the GPQA template copied literally? No.

This is the minimal reproduction behind §4 of the README. GPQA's prompt asks
for "Answer: $LETTER", and some models copy the dollar sign. The letter
patterns want a space or the start of the text right before the letter, so
the requested line is not read at priority 100 and a weaker pattern elsewhere
in the text decides the grade.

`patches/0004` is applied to a copy of the installed extractor, so this also
fails loudly when the installed LightEval is not the commit the patch was
written against. Skipped, not failed, when LightEval is not importable.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess

import pytest

pytest.importorskip("lighteval")

PATCH = (pathlib.Path(__file__).resolve().parents[1] / "patches"
         / "0004-indices-accept-dollar-wrapped-letter.patch")
MODULE = "src/lighteval/metrics/utils/extractive_match_utils.py"


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
    spec = importlib.util.spec_from_file_location("extractive_match_utils_0004", target)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    return upstream, patched


def _letter(module, text):
    """What `gpqa_instruct_pass_at_k` extracts from a generation."""
    from lighteval.tasks.requests import Doc
    from lighteval.utils.language import Language

    doc = Doc(query="q", choices=["w", "x", "y", "z"], gold_index=0)
    targets = [module.IndicesExtractionConfig(prefix_for_extraction="NativeLetters",
                                              try_extract_without_anchor=True)]
    regexes = module.get_extraction_regexes(doc, targets, Language.ENGLISH)
    found = module.extract_target_from_pred(text, regexes, "first_match", "any_match", 5)
    return str(found[0]) if found else None


# An earlier, weaker mention of another letter, then the requested line with
# the template's dollar sign kept.
COPIED = ["At first the answer is D, but that ignores the ring strain.\nAnswer: $C",
          "At first the answer is D, but that ignores the ring strain.\nAnswer: $C$"]


@pytest.mark.parametrize("text", COPIED)
def test_unpatched_extractor_misses_the_copied_template(extractors, text):
    """The reproduction: the weaker mention decides the grade."""
    upstream, _ = extractors
    assert _letter(upstream, text) == "D"


@pytest.mark.parametrize("text", COPIED)
def test_patched_extractor_reads_the_answer_line(extractors, text):
    _, patched = extractors
    assert _letter(patched, text) == "C"


@pytest.mark.parametrize("text,letter", [
    ("At first the answer is D.\nAnswer: C", "C"),
    ("Answer: **B**", "B"),
    ("Let $A$ be the set of products.\nAnswer: C", "C"),
])
def test_patch_changes_nothing_on_ordinary_lines(extractors, text, letter):
    upstream, patched = extractors
    assert _letter(upstream, text) == _letter(patched, text) == letter
