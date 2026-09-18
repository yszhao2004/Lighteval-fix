# Lighteval-fix

Eval-side fixes for [LightEval](https://github.com/huggingface/lighteval): the
grading that happens **after** the model output exists.

Nothing here changes generation, sampling, or the model. Every item is a case
where a correct answer is scored zero because of how it was written, or because
of how the candidate program was executed. That matters more than it sounds:
these failures do not look like bugs. The run completes, the report is
well-formed, and the number is simply lower than the model deserves — which is
indistinguishable, from the outside, from the model being worse.

| Area | Cause | Confidence | What ships |
|---|---|---|---|
| MATH-500 answer extraction | the requested `ANSWER:` line is read only inside math delimiters and is outranked by `\boxed`; no string-answer target | **confirmed from source** | re-grader + extraction module |
| LiveCodeBench stdin execution | candidate program is rewritten before execution | **minimal reproduction** | re-grader + upstream patch |
| GPQA letter extraction | a `final answer … is X` phrase in the reasoning outranks the answer line | **confirmed on real runs** | upstream patch 0003 + re-grader |
| AIME answer extraction | the prompt's own format template, restated while reasoning, outranks the answer | **minimal reproduction** | upstream patch 0002 + re-grader rule |
| Reasoning removal | does nothing when the chat template opens `<think>` in the prompt, so all of the above search the reasoning | **minimal reproduction** | upstream patch 0003 |
| Run-killing harness faults | three separate ones, §8 | **confirmed from source** | import-time patches |

Two independent halves, and which you need depends on what you are doing:

* **Re-grading finished runs** (`tools/`) — needs only the details files. This
  is the complete path: it does not require LightEval to be patched, and every
  correction is a script anyone can rerun on the same artefacts.
* **Running new evaluations** (`lighteval_fix.harness`) — install before
  LightEval's model modules import, so a run is not lost to a logging crash or
  silently sampled differently than configured. See §8.

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
python -m tools.regrade_lcb --help
```

If `tests/test_answer_equivalence.py` cannot import LightEval it will fail
rather than skip. That is intentional: a silently skipped correctness test is
how a broken grader ships.

### Re-grading runs made elsewhere

The details files are all that is needed — no model, no GPU, no server:

```bash
# answers: MATH-500, GPQA, AIME, anything answer-shaped
python -m tools.regrade     --details path/to/<task>/details --task math500

# LiveCodeBench stdin problems: this one executes candidate programs
python -m tools.regrade_lcb --details path/to/lcb/details
```

`tools/regrade_lcb.py` runs model-generated code as a subprocess with nothing
but a timeout around it. Run it in a container or on a throwaway machine, never
on a host holding anything you care about. LightEval's in-process path calls
`reliability_guard` to disable `os.system` and friends; a subprocess cannot be
guarded that way, and running the program as written is the whole point.

Both tools print the harness's own number beside the re-graded one, so the
correction is always visible as a delta rather than replacing the original
silently.

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
requested. (The AIME tasks *do* mandate boxing — `tasks/tasks/aime.py:45` —
which is why this inversion only bites MATH-500 and tasks like it. AIME fails
differently; see §5.)

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

Three consequences:

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
3. **The requested line is read only inside math delimiters.** The `answer:`
   pattern (lines 256-259) wraps `latex_envs_re`, which is `$…$`, `$$…$$`,
   `\(…\)`, `\[…\]` or ` [ … ] ` (lines 210-218); the only bare form it
   accepts is a numeric `\frac{a}{b}` (lines 219-221). In the prompt,
   `$ANSWER` is a placeholder, and models write the value bare. The
   expression target (`lazy_expr_regex`, lines 101-123) then reads a leading
   number, or nothing. LightEval's own extractor, on single lines:

   | generation | extracted |
   |---|---|
   | `ANSWER: -\sqrt{3}` | nothing |
   | `ANSWER: \pi` | nothing |
   | `ANSWER: 3R^2` | `3` |
   | `ANSWER: 288\pi` | `288` |
   | `ANSWER: $3R^2$` | `3R^2` |
   | `then 7 apples.` / `ANSWER: \sqrt{66}` | `7` |

   When the line yields nothing, another number or box anywhere in the text
   is taken instead, including one inside the reasoning, which is not
   removed (§5). This is how a grade can compare the gold against a span the
   model never offered as its answer.

A last, quieter one: the gold for this task is the **entire worked solution**
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

## 4. GPQA: a phrase in the reasoning outranks the answer line

GPQA answers are single letters, extracted with
`IndicesExtractionConfig(prefix_for_extraction="NativeLetters",
try_extract_without_anchor=True)` (`metrics/metrics.py:607-619`). The letter
patterns are ranked the same way as the maths ones
(`metrics/utils/extractive_match_utils.py:311-341`), lowest number first:

| pattern | priority |
|---|---|
| `final answer is X. I hope` | 0 |
| `final answer`, up to 100 characters, then `is X` | **50** |
| `answer:` + X, which is what the prompt asks for | **100** |
| `answer` + X, a letter at the start of the text or of a line, a bare letter | 150–300 |

The priority-50 pattern (line 314) is loose: "final answer", then anything
up to 100 characters, then "is" and a letter. A long reasoning trace
produces such a phrase without meaning it as an answer. These are the spans
LightEval graded in three real generations, each of which ended with a clean
`Answer:` line giving the correct letter and scored zero:

> Need be careful: if final answer is A
>
> The final answer A. If the expected answer is C
>
> in final answer could mention "One product is B

All three sit inside the reasoning, which LightEval was meant to remove and
did not (§5). Priority 50 beats 100 wherever the two sit, so the phrase is
graded. The longer the trace, the more chances there are for such a phrase,
which is why these misreads sit in the tail of the length distribution.

The explanation usually offered, a timeout, is ruled out. The
`timeout_seconds=5` default (`metrics/dynamic_metrics.py:165`) is never used
on this path: `extract_indices` (`extractive_match_utils.py:531-537`) is a
regex group lookup with no sympy parsing and takes no timeout. The alarms
that do exist are a 2 s one around writing to `doc.specific`
(`dynamic_metrics.py:208`, logged as a warning, does not zero a score) and a
5 s one inside `compare_gold_target`, whose letter-versus-letter path is a
plain string compare (`math_comparison.py:595-601`). Extraction on the
generations above takes milliseconds.

What ships: patch 0003 (§5) makes the reasoning removal work, which removes
these phrases before extraction, and `tools/regrade.py` re-extracts every item
from its final answer and reports the disagreements.

---

## 5. AIME: the prompt's own template is graded as the answer

### What the task asks for

`src/lighteval/tasks/tasks/aime.py:45`:

> The last line of your response should be of the following format:
> 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct'
> (without quotes) where ANSWER is just the final number or expression that
> solves the problem.

### What the extractor does with it

The highest-priority pattern, priority 0, is that same sentence wrapped around
a LaTeX environment (`metrics/utils/extractive_match_utils.py:250-252`):

```python
final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{latex_re}\.?\s?I hope"
```

Within a priority tier, matches are tried rightmost first and the first one
that parses ends the search (lines 607-611).

Some reasoning models restate the instruction while they think: "the last
line should be: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it
is correct'". The restatement carries the `$…$` delimiters, so it matches the
priority-0 pattern. The model's own last line often does not: `\boxed{321}`
or `\(\boxed{321}\)` without the dollar signs falls to a lower tier. So the
restatement wins, and it parses, because sympy reads `ANSWER` as the product
`A·E·N·R·S·W`. Nothing else is tried, and the item scores zero with the
correct answer written as the last line.

### Why the reasoning is searched at all

LightEval does remove reasoning before grading. `remove_reasoning_tags` is on
by default with the pair `('<think>', '</think>')` (`pipeline.py:93-94`), it
is applied in `_post_process_outputs` (`pipeline.py:351-358`), and the metric
reads `final_text`, which returns the post-processed text
(`models/model_output.py:142-144`). But the removal loop is
`while start_tag in result and end_tag in result` (`utils/utils.py:308-309`),
so it needs both tags in the generation. Current reasoning models' chat
templates put `<think>` at the end of the prompt, so the generation holds only
`</think>`. Nothing is removed, nothing is logged, and extraction runs over
the whole trace.

### Patch 0002: skip the template

`patches/0002-extraction-skip-restated-answer-placeholder.patch` makes
`extract_target_from_pred` skip a match when every span it captured is a
template word: `ANSWER` (AIME and the `ANSWER: $ANSWER` family) or `LETTER`
(GPQA's `Answer: $LETTER`). The check ignores LaTeX commands, `$`, braces and
whitespace. Anything else in the span is kept, whether a digit, a real letter
answer such as `A`, or `\text{Answer: 5}`. Priorities, ordering and parsing
are untouched, so a generation that restates no template is graded exactly as
upstream grades it.

One consequence is stated rather than hidden. If a generation is cut off
before `</think>`, the patched extractor falls through to whatever LightEval
would have taken without the restatement, which can be a value inside the
reasoning. That is LightEval's existing rule for truncated generations. The
patch does not add it.

`tests/test_aime_template_patch.py` applies the patch to a copy of the
installed extractor. It checks the minimal reproduction under both versions,
a text holding only the template, a text with no template (unchanged), and
GPQA letters.

**Validated against real runs.** The patch was run over every AIME and
MATH-500 details file available, 85 runs and 11,010 items across nine models.
There, the unpatched extractor, called the way `pass_at_k_math` calls it,
reproduces every recorded per-item score. With the patch, the only changes
are from 0 to 1, all on items whose extracted span had been the placeholder,
and no item goes from 1 to 0. In the changed items the model had written the
correct final answer after `</think>`, except for a few generations cut off
at the token limit, which were graded under the truncation rule above. The
restatement came from Qwen3.8-Flash-Next, in unquantized and quantized runs
alike, and from both GLM models. It is a model habit, not a property of a
checkpoint.

### Patch 0003: make the removal work

`patches/0003-reasoning-tags-closing-tag-only.patch` makes
`remove_reasoning_tags` treat everything before a closing tag that has no
opening tag ahead of it as reasoning, and then runs the original loop. A
generation with both tags, or with neither, is handled exactly as before,
and a generation cut off before `</think>` still has nothing removed.

It was validated on the same runs plus every GPQA run: 118 runs and 17,846
items, on which the unpatched implementation reproduces every recorded
score.

* **GPQA and AIME.** Grades now follow the stated final answer. The misreads
  of §4 and the template misreads above are corrected. Some items also move
  from 1 to 0, and those are corrections too. A few had been credited
  through a phrase in the reasoning while the stated answer was a different,
  wrong letter. A few more were cut off after `</think>`, before their final
  answer, and had been credited with a value from the reasoning.
* **MATH-500.** Changes go both ways. Most are gains. Most of the losses are
  correct final answers written in a form §2 shows LightEval cannot read
  (`ANSWER: 3R^2`), which had been credited only because the same value also
  appeared, delimited, in the reasoning; the rest had a wrong or cut-off
  final answer. On MATH-500 the patch therefore
  swaps accidental credit for the §2 failures. Use it there together with the
  re-grader, not alone.

`tests/test_reasoning_tags_patch.py` applies it to a copy of the installed
function.

0002 is kept alongside because 0003 does nothing for a generation cut off
before `</think>`. The whole trace is still searched there, and a restated
template anywhere in it still wins at priority 0.

### The re-grader

`lighteval_fix.extraction` applies the same rule in all three places it looks
for an answer: `\boxed{}`, the `ANSWER:` line, and the LaTeX-environment
fallback. A generation cut off before `</think>` with a restated template
after its last real answer is therefore still read at the answer.

---

## 6. What validating against real runs changed

The 66-case corpus was green before any of this repository's code had been
pointed at an actual evaluation. Running `tools/regrade.py` against the details
files of finished runs found two defects the synthetic cases could not:

**The gold needed the opposite extraction rule from the prediction.** MATH-500's
gold is literally `"ANSWER: " + solution` (`tasks/tasks/math_500.py:47`), so
preferring the requested `ANSWER:` line — right for a generation — captured the
*first line of the derivation* instead of the answer boxed at its end. One gold
came out as a 500-character paragraph. Extraction is now side-aware:
`extract_final_answer(text, prefer="boxed")` for a reference solution,
`prefer="requested"` for a generation, plus a length guard so a paragraph is
never accepted as an answer. Before the fix the re-grader recovered *fewer*
items than a crude hand-written normaliser; after it, more.

**Unicode maths was silently unparseable.** `288\pi` against `288π` happens to
parse, and the corpus contained exactly that case — so it hid the fact that
`2\sqrt{113}` against `2√113`, `16\sqrt{3}` against `16√3`, and
`(-\infty,2)\cup(3,\infty)` against `(-∞,2)∪(3,∞)` all failed. `_transliterate`
now rewrites unicode maths into its LaTeX spelling before parsing. This is
transliteration, not normalisation: every character mapped has one unambiguous
LaTeX equivalent, and no bracket, separator or grouping is touched. The five
real cases are now in the corpus.

**Agreement as evidence.** On GPQA the re-grader's verdicts match an
independently written checker on every cell tested — same recovered counts,
same residual. The two implementations share no code and disagree in method
(one defers to LightEval's symbolic comparator, the other used a hand-written
normaliser), so agreement is not an artefact of either.

### What the residual looks like, and why it is left alone

After both fixes a small number of items are still scored wrong, and they fall
into classes that a comparator cannot settle by itself:

| class | example (gold vs prediction) |
|---|---|
| genuinely wrong | `2\sqrt{113}` vs `2\sqrt{106}` |
| units appended | `12\pi` vs `12π inches per second` |
| mixed number | `137 \frac{1}{2}` vs `137 1/2` (which parses as 137 × ½) |
| `±` expanded to a list | `1 \pm \sqrt{19}` vs `1 - sqrt(19), 1 + sqrt(19)` |
| more roots than the gold lists | `3 \pm 2\sqrt{2}` vs four explicit values |
| column vector vs tuple | `\begin{pmatrix}-18\\-49\\96\end{pmatrix}` vs `(-18, -49, 96)` |
| interval vs inequality with a named variable | `(3,4]` vs `3 < λ ≤ 4` |

Each of these needs either the question text (are units expected? was the
answer meant to list every root?) or a judgement about notation
(is `137 1/2` a mixed number or a product?). Accepting them would mean
guessing, and a grader that guesses in the model's favour is worse than one
that is merely strict. They are recorded here instead.

---

## 7. Re-grading a finished run

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

## 8. Faults that end a run before grading

Not extraction, but the first things to hit in a new environment. `harness.py`
installs all three; each raises if the upstream shape it depends on changed,
because a silently inert patch is worse than none.

```python
import lighteval_fix.harness as harness
harness.apply_all()                    # before lighteval's model modules load
```

**`xxhash` is handed `str` where it accepts only bytes.** The details logger
hashes three `str` values, so the run raises
`TypeError: Strings must be encoded before hashing` **after every sample has
been generated and scored**, and writes no result file. The whole generation
budget is lost to a logging detail.

**Sampling parameters are silently dropped.** `to_litellm_dict` forwards only
seven fields; `top_k`, `min_p` and `presence_penalty` are accepted by the
config model and then not sent. A protocol that fixes any of them reports
numbers produced under different sampling, with no warning. A separate case:
`top_k=-1` means "disabled" to an OpenAI-compatible SGLang server, but the
config model constrains the field to >= 0 and rejects the config before a
request is sent — pass such values through `forced=`. And
`chat_template_kwargs` gates reasoning in some templates, so omitting it can
evaluate a reasoning model with reasoning off, which looks like a weaker model
rather than like a mistake.

One trap worth stating plainly: those parameters ride in `extra_body`, which
**litellm** unpacks. That is litellm's behaviour, not the server's. Code that
POSTs raw JSON to `/v1/chat/completions` must put the fields at the **top
level** — no server reads a nested `extra_body`, and the request is accepted
while the parameters are ignored.

**An unguarded SGLang import.** LightEval's SGLang backend does
`from sglang.srt.hf_transformers_utils import get_tokenizer` at import time,
and importing any LightEval model module pulls it in — including runs that use
the endpoint backend and never touch SGLang. Current SGLang moved that module
under `srt/utils/`, so the import fails before the first request. The patch
leaves a forwarding shim rather than editing a package we do not own.

---

## 9. What an upstream MATH-500 fix would have to change

The correction here is delivered by re-grading, not as a patch to
`patches/`, and that is a deliberate choice rather than an omission. Fixing it
upstream means three changes, and only the first is small:

* **The priority inversion** is small: `math_500` shares `Metrics.pass_at_k_math`
  (`metrics/metrics.py:469-480`) with the AIME tasks, whose prompts *do*
  mandate `\boxed{}`. Raising `boxed_match_priority` above the `ANSWER:` tier
  would fix MATH-500 and break AIME, so it needs a second metric entry used
  only by tasks whose prompt asks for a final line.
* **The answer line** is read only inside math delimiters (§2, item 3). It
  needs a reading of the bare text after `ANSWER:` up to the end of the line,
  parsed as LaTeX.
* **The missing string target** is not small. `ExtractionTarget` is a union of
  three configs (`extractive_match_utils.py:96`) with no string member, which
  is why a word answer extracts nothing at all; LightEval's own comment asks
  for a `StringExtractionConfig` (`dynamic_metrics.py:174`). Adding one touches
  extraction, comparison and every task that uses them.

None of them can be validated here the way the LCB patch was — that one has a
minimal reproduction that runs in milliseconds, while these need a full
evaluation against a model to show they changed the right scores. So they are
specified rather than guessed at, and the re-grade path is what this repository
stands behind.

---

## 10. Limitations

* The line numbers are `6ba40c4`. Upstream moves; `patches/` will need
  refreshing, and the modules under `lighteval_fix/` are written to fail
  loudly rather than silently when the shape they expect has changed.
* The text branch of `answers_equivalent` is a string comparison, not a
  semantic one. `"an ellipse"` and `"ellipse"` are different strings and will
  be reported as different answers.
* Patch 0003 alone trades one set of MATH-500 errors for another (§5). Use
  it there with the re-grader.
* The template words are exactly two, `ANSWER` and `LETTER`. A task whose
  template names its slot differently needs the word added to both the patch
  and `extraction.py`.
* The LCB rewrite is shown to break one class of program, not all of them.
  See §3.

---

## 11. Layout

```
lighteval_fix/
  extraction.py      answer extraction + equivalence (MATH-500 and friends)
  lcb_stdin.py       run a stdin candidate as written, compare its output
  harness.py         the three run-killing faults of §8
patches/             upstream diffs against lighteval 6ba40c4
tools/
  regrade.py         re-score answers from a finished run's details
  regrade_lcb.py     re-run LiveCodeBench stdin candidates and re-score
tests/               extraction, equivalence, the LCB rewrite, patches 0002
                     and 0003, the harness
```
