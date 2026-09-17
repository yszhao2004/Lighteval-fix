"""Re-grade a finished LightEval run from its details files. No model needed.

Why this exists as a separate tool, next to the patches: a run that has
already cost hours of generation should not have to be repeated because the
grader misread its output. The details parquet holds the full generation and
the gold for every sample, so the scoring can be redone offline in seconds,
and the two numbers can be reported side by side -- what the harness scored,
and what the same answers score once extraction is fixed.

It is also the honest way to publish a corrected score: the correction is a
script anyone can rerun on the same artefacts, not a judgement someone made
while reading diffs.

    python -m tools.regrade --details path/to/details/**/*.parquet --task math500
    python -m tools.regrade --details path/to/run --task gpqa --show 20

Prints a breakdown and exits non-zero if nothing could be read, so it is safe
to put in CI.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from lighteval_fix.extraction import answers_equivalent, extract_final_answer

# LightEval stores the gold either directly or as choices[gold_index]; the
# latter can be a whole worked solution whose own \boxed holds the answer.
_GOLD_KEYS = ("gold", "answer", "solution")


def _load(details: str):
    paths = sorted(glob.glob(details)) if any(c in details for c in "*?[") else None
    if paths is None:
        paths = ([details] if details.endswith(".parquet")
                 else sorted(glob.glob(os.path.join(details, "**", "*.parquet"),
                                       recursive=True)))
    if not paths:
        raise SystemExit(f"no parquet found under {details!r}")
    import pyarrow.parquet as pq

    return pq.read_table(paths[-1]).to_pydict(), paths[-1]


def _as_dict(value):
    return json.loads(value) if isinstance(value, str) else value


def _prediction_text(row) -> str:
    items = row if isinstance(row, list) else [row]
    for item in items:
        text = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, list):
            text = text[0] if text else ""
        if isinstance(text, str):
            return text
    return ""


def _gold_text(doc) -> str:
    for key in _GOLD_KEYS:
        value = doc.get(key)
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, str) and value.strip():
            return value
    choices, index = doc.get("choices") or [], doc.get("gold_index")
    if isinstance(index, list):
        index = index[0] if index else None
    if choices and index is not None and int(index) < len(choices):
        return choices[int(index)]
    return ""


def _harness_score(metric) -> float | None:
    metric = _as_dict(metric) or {}
    for key, value in metric.items():
        if any(k in key for k in ("pass", "acc", "match", "score")):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True,
                    help="a details .parquet, a directory holding one, or a glob")
    ap.add_argument("--task", default="",
                    help="label for the report only; extraction is task-agnostic")
    ap.add_argument("--show", type=int, default=10,
                    help="how many recovered items to print for audit")
    args = ap.parse_args(argv)

    data, path = _load(args.details)
    print(f"[regrade] {path}")
    total = len(data["model_response"])

    passed = recovered = wrong = unreadable = 0
    examples = []
    for i in range(total):
        score = _harness_score(data["metric"][i])
        if score is not None and score >= 1.0:
            passed += 1
            continue
        doc = _as_dict(data["doc"][i]) or {}
        gold = extract_final_answer(_gold_text(doc)) or _gold_text(doc).strip()
        pred = extract_final_answer(_prediction_text(data["model_response"][i]))
        if not gold or pred is None:
            unreadable += 1
            continue
        if answers_equivalent(gold, pred):
            recovered += 1
            if len(examples) < args.show:
                examples.append((i, gold, pred))
        else:
            wrong += 1

    graded = passed / total if total else 0.0
    fixed = (passed + recovered) / total if total else 0.0
    label = args.task or "task"
    print(f"[regrade] {label}: {total} items")
    print(f"  harness scored correct        {passed}")
    print(f"  recovered by extraction       {recovered}")
    print(f"  genuinely different           {wrong}")
    print(f"  no answer found on either side{unreadable:>4}")
    print(f"  --")
    print(f"  as graded   {graded:.2%}")
    print(f"  re-graded   {fixed:.2%}")
    if recovered:
        print(f"  recovered examples (verdict is automatic; shown to audit):")
        for i, gold, pred in examples:
            print(f"    #{i}: gold={gold!r} predicted={pred!r}")
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
