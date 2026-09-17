# Lighteval-fix

Eval-side fixes for [LightEval](https://github.com/huggingface/lighteval): the
grading that happens **after** the model output exists.

Nothing here changes generation, sampling, or the model. Every item is a case
where a correct answer is scored zero because of how it was written, or because
of how the candidate program was executed. That matters more than it sounds:
these failures do not look like bugs. The run completes, the report is
well-formed, and the number is simply lower than the model deserves — which is
indistinguishable, from the outside, from the model being worse.

Three areas, in descending order of how well the cause is pinned down:

| Area | Cause | Confidence |
|---|---|---|
| MATH-500 answer extraction | requested answer format outranked by `\boxed`; no string-answer target | **confirmed from source** |
| LiveCodeBench stdin execution | candidate program is rewritten before execution | **minimal reproduction included** |
| GPQA letter extraction | unknown | **not diagnosed — verifier only** |

Everything is verified by a test corpus that runs offline, with no model and no
GPU. This repository contains no accuracy figures for any model: it ships the
tools and the reasoning, and you produce numbers on your own runs.

---

## 1. Environment

LightEval is pinned by commit here, not by release. The line numbers quoted
throughout refer to **`6ba40c4`** on `main`. Releases lag: `0.13.0` also
carries the issues described below, but at different offsets.

```bash
python -m venv .venv            # 3.10 or newer; latex2sympy2_extended needs it
source .venv/bin/activate
pip install -U pip

# LightEval with the math extra, at the pinned commit.
pip install "lighteval[math,extended_tasks] @ \
  git+https://github.com/huggingface/lighteval.git@6ba40c4"

# This repository, in editable mode.
pip install -e .
pip install pytest                 # to run the corpus
```

`latex2sympy2_extended==1.0.6` arrives as a base dependency of LightEval, and
the `[math]` extra pins the same version, so the symbolic comparison is
available without anything further.

**`math-verify` is deliberately *not* a dependency.** It is easy to assume
LightEval uses it — the AIME task even mentions it in a comment
(`src/lighteval/tasks/tasks/aime.py:43`) — but there is no `import math_verify`
anywhere in the tree. LightEval carries a vendored copy of that logic in
`src/lighteval/metrics/utils/{extractive_match_utils,math_comparison}.py`.
Adding the real package would create a second comparator that disagrees with
the harness in ways nobody tracks. This repository calls LightEval's own
`compare_gold_target` instead.

### Checking the environment

```bash
pytest tests/ -q                   # the corpus; needs lighteval importable
python -m tools.regrade --help
```

If `tests/test_answer_equivalence.py` cannot import LightEval it will fail
rather than skip. That is intentional: a silently skipped correctness test is
how a broken grader ships.

---

## 2. MATH-500: the answer format the prompt asks for is outranked

### What the task asks for

`src/lighteval/tasks/tasks/math_500.py:34-39`

```python
MATH_QUERY_TEMPLATE = """
Solve the following problem. The final line of your response MUST be of the following format:
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the final answer. Think step by step before answering.

{prompt}
""".strip()
```

So the model is instructed to end with `ANSWER: <value>`. No `\boxed{}` is
requested. (The AIME tasks *do* mandate boxing — `tasks/tasks/aime.py:43` —
which is why this only bites MATH-500 and tasks like it.)

### What the metric accepts

`src/lighteval/metrics/metrics.py:469-480` attaches
`MultilingualExtractiveMatchMetric` with `ExprExtractionConfig()` and
`LatexExtractionConfig()`, both at their defaults. Extraction then ranks
candidate patterns by priority and **the lowest number wins**
(`metrics/utils/extractive_match_utils.py:591-632`):

| pattern | priority |
|---|---|
| `final answer is … I hope` | 0 |
| `final answer … is …` | 50 |
| **`\boxed{…}`** | **55** (`LatexExtractionConfig.boxed_match_priority`, line 67) |
| **`answer:` + value** | **100** |
| `answer` + value | 200 |
| bare expression / latex env | 300 |

Two consequences:

1. **A `\boxed{}` anywhere outranks the requested final line.** A reasoning
   model that boxes a candidate mid-derivation and then writes the requested
   `ANSWER:` line is graded on the box it abandoned. Within one priority tier
   the rightmost match wins (line 607-608), but a tier-55 box beats a tier-100
   answer line no matter where either sits.
2. **A word answer extracts nothing at all.** The extraction targets are
   `LatexExtractionConfig | ExprExtractionConfig | IndicesExtractionConfig`
   (line 96) — there is no string target. LightEval says so itself, at
   `metrics/dynamic_metrics.py:174`:

   > There is currently no StringExtractionConfig, so if the gold is
   > `\boxed{\text{Friday}}` and model outputs `Friday` it will not match,
   > because nothing will be extracted.

   MATH-500 contains such answers (`even`, `ellipse`, a person's name, a
   multiple-choice letter). They score zero whatever the model writes.

A third, quieter one: the gold for this task is the **entire worked solution**
prefixed with `ANSWER: ` (`math_500.py:44-49`,
`choices=[f"ANSWER: {line['solution']}"]`), so gold extraction is itself
searching prose for the answer, and falls back to the raw solution string when
it finds nothing (`dynamic_metrics.py:241-243`).

### The fix, and why it is not a normaliser

The temptation is to normalise: strip `\left`/`\right`, map `\pi` to `π`,
turn `\frac{1}{2}` into `0.5`, and compare strings. Every such pass is also
wrong, because the transformations that reconcile notation also erase meaning.
Delete brackets so that `\left(3\right)` matches `(3)`, and you have equated
the ordered pair `(1,2)` with the set `{1,2}`. A normaliser permissive enough
to be useful is permissive enough to be unsound, and there is no way to tell
from the score which it was being.

So the split is:

> **Extraction decides *which string*. Symbolic comparison decides *whether it
> means the same*.** Neither step guesses.

`lighteval_fix/extraction.py` implements both halves:

* `extract_final_answer(text)` — searches after `</think>` first (a reasoning
  trace's discarded candidates are working, not answers), prefers the
  requested `ANSWER:` line over `\boxed{}`, then falls back to `\boxed{}` and
  to the last inline latex environment. Handles `**Answer:** C` and
  `**Answer**: C`, nested braces inside `\boxed{}`, `ANSWER: \boxed{42}`, and
  multiple boxes (last wins).
* `answers_equivalent(gold, pred)` — routes numbers, expressions, sets,
  intervals and relations to LightEval's own `compare_gold_target`
  (`metrics/utils/math_comparison.py:578`), which already handles fractions
  against decimals, unsimplified radicals, percentages, endpoint-wise interval
  equality and set/tuple coercion. Word answers take a text branch that
  removes **only** presentation wrappers — `\text{}`, `\mathrm{}`, `\mathbf{}`,
  `$`, case, whitespace, and a parenthesis around a single letter so that
  `(C)` matches `C`. Brackets, commas and operators are never touched, and the
  text branch is gated by `_looks_textual`, which admits only a run of letters
  with no digits, operators, brackets or commas.

The negative half of `tests/test_answer_equivalence.py` is the part worth
reading. It asserts that `(1,2) ≠ {1,2}`, `(1,2) ≠ [1,2]`,
`[-2,7] ≠ (-2,7)`, `(15,-29) ≠ (-29,15)`, `288π ≠ 289π` and `0.5 ≠ 5` — the
cases a permissive normaliser gets wrong. Nothing in the loop asks a human to
look at a diff.

### Two things found by writing those tests

**LightEval's comparator equates an ordered pair with a set.** Handed `(1,2)`
and `{1,2}`, `compare_gold_target` returns true: `sympy_compare_sets`
(`metrics/utils/math_comparison.py`) coerces between intervals and tuples.
Under every reading of that notation — ordered pair, open interval, or
two-element set — those are different answers, so a grader that accepts it
will score a wrong answer correct. `answers_equivalent` therefore refuses the
case before calling the comparator, with a syntactic guard
(`_bracket_kinds_conflict`) that fires only when both sides are bracketed
lists whose outer brackets differ in kind. `\left(1,2\right)` against
`(1,2)` is unaffected. This is the one place this repository overrides
upstream rather than deferring to it, and it is narrow on purpose.

**Percent is a value, not a decoration.** `50\%` parses as `50/100`
(`is_atomic_or_pct_atomic`), so it does not equal `50`. That is correct
symbolically, and it is left alone — even though a question that already asks
"what percent" makes a bare `50` a plausible answer. Deciding otherwise would
require knowing the question, which a comparator does not. It is recorded as
`KNOWN_TENSION` in the corpus rather than silently resolved either way.

---

## 3. LiveCodeBench: the stdin path does not run the program you were given

For stdin-style problems (`fn_name` absent —
`tasks/tasks/lcb/codegen_metrics.py:369-376`) the candidate is rewritten before
execution (lines 265-268):

* `clean_if_name` (line 86) strips a trailing `if __name__ == '__main__':` and
  splices its body to module level through `ast.unparse`.
* `make_function` (line 102) takes every statement that is not an import and
  re-emits it as the body of `def wrapped_function():`.

The result is exec'd in-process with `sys.stdout`/`sys.stdin` monkeypatched
(`Capturing`, line 72; `call_method`, line 132).

That rewrite is not semantics-preserving. `tests/test_lcb_rewrite.py` carries
the minimal case:

```python
import sys
N = int(sys.stdin.readline())
def solve():
    global N
    print(N * 2)
solve()
```

As written it prints `42` for input `21`. After the rewrite, `N` is a local of
`wrapped_function`, while `global N` in `solve` still names the module
namespace — where nothing was ever assigned — so it raises
`NameError: name 'N' is not defined`. A correct program, scored zero.

A class body that reads a module-level name survives (closures cover it), and
plain top-level code survives. So the rewrite is not uniformly destructive; it
breaks the subset that reaches the module namespace explicitly.

### What has been ruled out

Two explanations circulate for LCB stdin failures and neither holds:

* **"It compares bytes, so a trailing newline fails it."** It does not.
  `get_stripped_lines` (line 192) strips the whole output and then each line,
  and a non-matching line falls through to a `Decimal` token comparison
  (line 332-343).
* **"The 6-second timeout is too short."** Re-running the same candidates at
  6 s and at 10 s changed the verdict for not one problem in the runs this was
  developed against.

`lighteval_fix/lcb_stdin.py` therefore runs the program **as written**, as a
subprocess, at LightEval's own 6 s default so the comparison is not quietly
more generous. `compare_output` matches LightEval's normalisation (outer strip,
per-line strip, whitespace-tokenised numeric comparison) with one stated
difference: a **relative** float tolerance, because LightEval's exact `Decimal`
equality (line 342) rejects `0.5000001` against `0.5`. Pass `tolerance=0` for
LightEval's exact behaviour.

**Scope of the claim.** The rewrite is demonstrated to break one concrete class
of correct program. It is not established that every stdin failure has that
cause; `extract_code` taking the text between the *last two* fences
(line 659-665) and the exact numeric comparison are also candidates.

---

## 4. GPQA: a verifier, because the cause is not known

GPQA answers are single letters, extracted with
`IndicesExtractionConfig(prefix_for_extraction="NativeLetters",
try_extract_without_anchor=True)` (`metrics/metrics.py:607-619`).

Long generations that end with a clean `Answer: X` are sometimes scored zero.
The mechanism is **not** the one usually blamed. The `timeout_seconds=5`
default (`metrics/dynamic_metrics.py:165`) is never consumed on this path:
`extract_indices` (`extractive_match_utils.py:531-537`) is a regex group
lookup with no sympy parsing and does not take a timeout. The alarms that do
exist are a 2 s one around writing to `doc.specific` (`dynamic_metrics.py:208`,
logged as a warning, does not zero a score) and a 5 s one inside
`compare_gold_target`, whose letter-versus-letter path is a plain string
compare (`math_comparison.py:595-601`).

Remaining candidates, none confirmed:

* `pattern.finditer(pred)` at `extractive_match_utils.py:604` is not
  timeout-guarded and is run once per pattern over the whole generation;
* `try_extract_without_anchor=True` admits a bare capital letter at priority
  250/300, which a long answer can supply by accident — though a clean
  `Answer: X` at priority 100 should outrank it;
* the `signal.alarm` timeouts are main-thread only (`utils/timeout.py:40-51`),
  so they behave differently under a threaded runner.

Until one of these is demonstrated, this repository ships **no GPQA patch**.
What it ships is `tools/regrade.py`, which re-extracts and re-compares every
item and reports the disagreements. A disagreement is evidence; a patch built
on a guess is not.

---

## 5. Re-grading a finished run

Scoring can be redone from the details files, offline, without the model:

```bash
python -m tools.regrade --details path/to/run/details --task math500
python -m tools.regrade --details 'path/to/**/*.parquet' --task gpqa --show 20
```

It prints how many items the harness scored correct, how many the fixed
extraction recovers, how many are genuinely different, and both percentages.
The recovered items are listed so a reader can audit them, but the verdict is
computed, not eyeballed — the point is that the correction is a script anyone
can rerun on the same artefacts rather than a judgement somebody made while
reading diffs.

---

## 6. Layout

```
lighteval_fix/
  extraction.py      answer extraction + equivalence (MATH-500 and friends)
  lcb_stdin.py       run a stdin candidate as written, compare its output
patches/             the same changes as diffs against lighteval 6ba40c4
tools/regrade.py     re-score a finished run from its details files
tests/               the corpus: extraction, equivalence, and the LCB rewrite
```

## 7. Limitations

* The line numbers are `6ba40c4`. Upstream moves; `patches/` will need
  refreshing, and the modules under `lighteval_fix/` are written to fail
  loudly rather than silently when the shape they expect has changed.
* The text branch of `answers_equivalent` is a string comparison, not a
  semantic one. `"an ellipse"` and `"ellipse"` are different strings and will
  be reported as different answers.
* GPQA is undiagnosed. See §4.
* The LCB rewrite is shown to break one class of program, not all of them.
  See §3.
