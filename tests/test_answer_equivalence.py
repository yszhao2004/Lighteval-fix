"""The corpus that decides whether extraction and equivalence are automatic.

Every case here is a real shape seen in reasoning-model output, written as
(gold, prediction, should_be_equal). The point of the file is the negative
cases: it is easy to write a normaliser that accepts every notation variant,
and such a normaliser also accepts genuinely different answers. A hand-rolled
regex pass that strips braces and parentheses, for instance, happily equates
an ordered pair with a set.

So the rule this repo follows is: **do not normalise, compare symbolically.**
Extraction decides *what string* to compare; LightEval's own symbolic
comparator decides whether two strings mean the same number, expression,
interval or word. Nothing in the loop needs a human to look at a diff.

One exception is carried deliberately: the comparator equates an ordered pair
with a set, so `answers_equivalent` refuses that case itself before handing
the strings over. See `_bracket_kinds_conflict`.

Run with:  pytest tests/ -q
"""

from __future__ import annotations

import pytest

from lighteval_fix.extraction import answers_equivalent, extract_final_answer


# --------------------------------------------------------------------------
# Extraction: which span of the generation is the answer
# --------------------------------------------------------------------------
EXTRACTION = [
    # \boxed is the usual case and must keep working, including nesting.
    (r"...therefore \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\boxed{\frac{a}{b+\frac{c}{d}}}", r"\frac{a}{b+\frac{c}{d}}"),
    # The last one wins: a model that boxes an intermediate result and then
    # boxes its conclusion must be read at the conclusion.
    (r"first \boxed{7}, after correcting: \boxed{42}", "42"),
    # The shape a prompt asking for "ANSWER: $ANSWER" actually produces. An
    # extractor that requires \boxed scores these zero however right they are.
    ("Working... so the value is 288pi.\n\nANSWER: 288\\pi", r"288\pi"),
    ("ANSWER: -13x+3", "-13x+3"),
    ("answer: [-2, 7]", "[-2, 7]"),
    # Reasoning traces close a thinking block first; the answer is after it.
    ("<think>maybe 5, no, 6</think>\n\nANSWER: 6", "6"),
    # A boxed answer inside the thinking block must not beat the final line.
    ("<think>guess \\boxed{5}</think>\n\nANSWER: 6", "6"),
    # Nothing answer-shaped at all.
    ("I am not sure how to proceed.", None),
    # A restated format template is not an answer. A generation cut off
    # before </think> is searched whole, so an echo written after the real
    # answer must not win -- in a box, an answer line, or a latex environment.
    (r"so the result is \boxed{321}. The format wants: Therefore, the final "
     r"answer is: $\boxed{ANSWER}$. I hope it is correct", "321"),
    ("value is 7.\nANSWER: 7\nThe last line must be:\nANSWER: $ANSWER", "7"),
    ("Answer: B\nThe required format is\nAnswer: $LETTER", "B"),
    # A template and nothing else is no answer at all.
    (r"The last line should read $\boxed{ANSWER}$.", None),
    # Only the two template words are skipped: a single letter is a real
    # answer, and a box carrying a value next to the word is read as it is.
    (r"\boxed{A}", "A"),
    (r"\boxed{\text{Answer: 5}}", r"\text{Answer: 5}"),
]


@pytest.mark.parametrize("text,expected", EXTRACTION)
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected


# --------------------------------------------------------------------------
# Equivalence: same value, different notation -> equal
# --------------------------------------------------------------------------
EQUIVALENT = [
    # Fractions against each other and against decimals.
    (r"\frac{1}{2}", r"\dfrac{1}{2}"),
    (r"\frac{1}{2}", "0.5"),
    (r"\tfrac{3}{4}", "0.75"),
    # Unicode where the gold writes LaTeX. This is the single most common
    # cosmetic difference and the reason a string comparison is unusable.
    (r"288\pi", "288π"),
    (r"12\pi", "12π"),
    (r"\frac{\pi}{6}", "π/6"),
    (r"-\frac{\pi}{6}", "-π/6"),
    # Unicode maths the parser cannot read, against its LaTeX spelling. Every
    # one of these was a false rejection on a real run before transliteration
    # was added; the synthetic corpus had only "288\\pi" vs "288π", which
    # happens to parse and so hid the gap.
    (r"2\sqrt{113}", "2√113"),
    (r"16 \sqrt{3}", "16√3"),
    (r"11 \sqrt{5} + 11", "11(√5+1)"),
    (r"(-\infty, 2) \cup (3, \infty)", "(-∞,2)∪(3,∞)"),
    (r"1 \pm \sqrt{19}", "1 ± √19"),
    # Radicals, simplified or not.
    (r"2\sqrt{3}", r"\sqrt{12}"),
    (r"\sqrt{66}", "sqrt(66)"),
    # Wrappers that carry no value.
    (r"\left(3\right)", "(3)"),
    (r"45^\circ", "45"),
    (r"\$12", "12"),
    (r"1{,}024", "1024"),
    # Word answers, which MATH500 does contain.
    (r"\text{even}", "even"),
    (r"\text{ellipse}", "ellipse"),
    (r"\text{(C)}", "C"),
    # Polynomials that differ only by term order or spacing.
    ("x^3+3x-6", "x^3 + 3x - 6"),
    ("-13x+3", "3-13x"),
    # Equations of a plane, scaled consistently.
    ("5x - 7y + 11z + 4 = 0", "5x-7y+11z+4=0"),
    # Trailing punctuation and stray whitespace.
    ("42", " 42. "),
]


@pytest.mark.parametrize("gold,pred", EQUIVALENT)
def test_equivalent(gold, pred):
    assert answers_equivalent(gold, pred), f"{gold!r} should equal {pred!r}"


# --------------------------------------------------------------------------
# Non-equivalence: the cases a permissive normaliser gets wrong
# --------------------------------------------------------------------------
# Percent is a value, not a decoration: LightEval parses "50\\%" as 50/100
# (`is_atomic_or_pct_atomic`, math_comparison.py), so it does not equal "50".
# That is the right call symbolically and it is kept, even though a MATH-500
# question that already says "what percent" makes the bare "50" a plausible
# answer. Scoring it correct would require knowing the question, which a
# comparator does not.
KNOWN_TENSION = [
    (r"50\%", "50", False),
]


@pytest.mark.parametrize("gold,pred,equal", KNOWN_TENSION)
def test_known_tension(gold, pred, equal):
    assert answers_equivalent(gold, pred) is equal


DIFFERENT = [
    # An ordered pair is not a set. A normaliser that deletes brackets to make
    # notation differences vanish also makes this difference vanish.
    ("(1,2)", "{1,2}"),
    # Nor is it an interval.
    ("(1,2)", "[1,2]"),
    ("[-2, 7]", "(-2, 7)"),
    # Plain wrong values, including ones that are close.
    ("1/3", "1/4"),
    ("288\\pi", "289\\pi"),
    ("2\\sqrt{3}", "3\\sqrt{2}"),
    ("42", "-42"),
    # Sign and order inside a pair.
    ("(15,-29)", "(-15,29)"),
    ("(15,-29)", "(-29,15)"),
    # Different words.
    (r"\text{even}", "odd"),
    (r"\text{(C)}", "D"),
    # Same digits, different magnitude.
    ("0.5", "5"),
    ("1024", "1,024,000"),
]


@pytest.mark.parametrize("gold,pred", DIFFERENT)
def test_different(gold, pred):
    assert not answers_equivalent(gold, pred), f"{gold!r} must not equal {pred!r}"


# --------------------------------------------------------------------------
# The two must compose: extract, then compare
# --------------------------------------------------------------------------
END_TO_END = [
    (r"\boxed{288\pi}", "ANSWER: 288π", True),
    (r"\boxed{\text{even}}", "ANSWER: even", True),
    (r"\boxed{(15,-29)}", "ANSWER: (15,-29)", True),
    (r"\boxed{(15,-29)}", "ANSWER: (-29,15)", False),
    (r"\boxed{\frac{1}{2}}", "<think>...</think>\n\nANSWER: 0.5", True),
    # The AIME shape: the template is restated inside the thinking block, the
    # answer follows it in the requested form.
    (r"\boxed{321}", "The last line should be: 'Therefore, the final answer is: "
                      "$\\boxed{ANSWER}$. I hope it is correct'. ... so it is 321.\n"
                      "</think>\nTherefore, the final answer is: \\boxed{321}. "
                      "I hope it is correct", True),
]


@pytest.mark.parametrize("gold_text,pred_text,equal", END_TO_END)
def test_end_to_end(gold_text, pred_text, equal):
    gold = extract_final_answer(gold_text)
    pred = extract_final_answer(pred_text)
    assert gold is not None and pred is not None
    assert answers_equivalent(gold, pred) is equal
