"""Does LightEval remove reasoning when the prompt opened the block? No.

This is the minimal reproduction behind §5 of the README. `remove_reasoning_tags`
loops `while start_tag in result and end_tag in result`, so it needs both tags
in the generation. Reasoning models' chat templates put `<think>` at the end
of the prompt, the generation carries only `</think>`, and nothing is removed.

`patches/0003` is applied to a copy of the installed function, so this also
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
         / "0003-reasoning-tags-closing-tag-only.patch")
MODULE = "src/lighteval/utils/utils.py"
TAGS = [("<think>", "</think>")]


@pytest.fixture(scope="module")
def removers(tmp_path_factory):
    """(upstream, patched) remove_reasoning_tags."""
    from lighteval.utils import utils as upstream

    root = tmp_path_factory.mktemp("lighteval")
    target = root / MODULE
    target.parent.mkdir(parents=True)
    shutil.copy(upstream.__file__, target)
    applied = subprocess.run(["git", "apply", str(PATCH)], cwd=root,
                             capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    spec = importlib.util.spec_from_file_location("lighteval_utils_0003", target)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    return upstream.remove_reasoning_tags, patched.remove_reasoning_tags


# What the model generates when the prompt already ended with "<think>".
PROMPT_OPENED = "the final answer is A, or maybe not\n</think>\nAnswer: D"


def test_unpatched_removes_nothing_when_the_prompt_opened_the_block(removers):
    """The reproduction: the reasoning survives into what is graded."""
    upstream, _ = removers
    assert upstream(PROMPT_OPENED, TAGS) == PROMPT_OPENED


def test_patched_removes_the_reasoning(removers):
    _, patched = removers
    assert patched(PROMPT_OPENED, TAGS) == "\nAnswer: D"


@pytest.mark.parametrize("text", [
    "<think>reasoning</think> Answer: D",          # both tags: upstream already works
    "no reasoning tags at all. Answer: D",          # neither tag
    "reasoning cut off by the token limit, final",  # never closed: nothing is removed
    "a <think>r1</think> b <think>r2</think> c",    # several complete blocks
])
def test_patched_matches_upstream_where_upstream_works(removers, text):
    upstream, patched = removers
    assert patched(text, TAGS) == upstream(text, TAGS)


def test_prompt_opened_block_followed_by_a_complete_one(removers):
    _, patched = removers
    assert patched("r0</think>a <think>r1</think>b", TAGS) == "a b"
