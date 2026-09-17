"""Re-grade the LiveCodeBench stdin problems of a finished run.

Separate from ``tools/regrade.py`` because the two do different work.
``regrade.py`` compares answer strings and needs nothing but the details
file. This one **executes candidate programs**, so it needs a machine it is
allowed to run untrusted code on, several cores, and a few minutes.

Only stdin problems are re-run. Call-based problems (those whose spec carries
a ``fn_name``) take a different path in LightEval that invokes the function
directly and compares return values (``tasks/tasks/lcb/codegen_metrics.py:199``);
that path does not rewrite the candidate and there is no evidence it is wrong,
so its verdicts are taken from the harness unchanged.

    python -m tools.regrade_lcb --details path/to/run/details
    python -m tools.regrade_lcb --details path/to/run/details --tolerance 0

``--tolerance 0`` reproduces LightEval's exact ``Decimal`` comparison, so the
contribution of the float slack can be measured rather than assumed.

**Safety.** This runs model-generated code as a subprocess with no sandbox
beyond a timeout. Run it where that is acceptable -- a container or a throwaway
machine -- and never on a host holding anything you care about. LightEval's own
in-process path calls ``reliability_guard`` to disable ``os.system`` and
friends; a subprocess cannot be guarded that way, which is the price of running
the program as written.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import sys

from lighteval_fix.lcb_stdin import run_stdin_case


def _as_dict(value):
    return json.loads(value) if isinstance(value, str) else value


def _text(row) -> str:
    items = row if isinstance(row, list) else [row]
    for item in items:
        value = item.get("text") if isinstance(item, dict) else item
        if isinstance(value, list):
            value = value[0] if value else ""
        if isinstance(value, str):
            return value
    return ""


def _extract_code(text: str) -> str:
    """LightEval's own fenced-block extraction, if importable; else a fallback.

    Using the harness's extractor keeps this comparison about *execution*
    rather than about who finds the code block. Its rule -- the text between
    the last two fences (``codegen_metrics.py:659-665``) -- is itself a
    candidate cause of failures, so changing it here would confound the two.
    """
    try:
        from lighteval.tasks.tasks.lcb.codegen_metrics import extract_code

        return extract_code(text) or ""
    except Exception:
        parts = text.split("```")
        if len(parts) < 3:
            return ""
        block = parts[-2]
        return block.split("\n", 1)[1] if "\n" in block else ""


def _one(payload):
    code, inputs, outputs, timeout, tolerance = payload
    if not code.strip():
        return False
    for stdin, expected in zip(inputs, outputs):
        if not run_stdin_case(code, stdin, expected, timeout=timeout,
                              tolerance=tolerance):
            return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--timeout", type=float, default=6.0,
                    help="per test case; 6 s is LightEval's own default")
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="relative float slack; 0 reproduces LightEval exactly")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args(argv)

    paths = ([args.details] if args.details.endswith(".parquet")
             else sorted(glob.glob(os.path.join(args.details, "**", "*.parquet"),
                                   recursive=True)))
    if not paths:
        raise SystemExit(f"no parquet found under {args.details!r}")
    import pyarrow.parquet as pq

    data = pq.read_table(paths[-1]).to_pydict()
    print(f"[regrade-lcb] {paths[-1]}")

    jobs, harness_stdin, call_based, call_pass, no_code = [], 0, 0, 0, 0
    for i in range(len(data["model_response"])):
        doc = _as_dict(data["doc"][i]) or {}
        spec = _as_dict(doc.get("specific")) or {}
        metric = _as_dict(data["metric"][i]) or {}
        score = next((v for k, v in metric.items()
                      if "pass" in k or "codegen" in k), 0)
        scored = float(score or 0) >= 1.0
        if spec.get("fn_name"):
            call_based += 1
            call_pass += scored
            continue
        if not spec.get("inputs"):
            continue
        harness_stdin += scored
        code = _extract_code(_text(data["model_response"][i]))
        if not code.strip():
            no_code += 1
        jobs.append((code, spec["inputs"], spec["outputs"],
                     args.timeout, args.tolerance))

    if not jobs:
        print("[regrade-lcb] no stdin problems in this run")
        return 1
    with multiprocessing.Pool(args.workers) as pool:
        results = pool.map(_one, jobs)
    fixed = sum(results)
    total = call_based + len(jobs)

    print(f"[regrade-lcb] {total} problems: {call_based} call-based, "
          f"{len(jobs)} stdin ({no_code} with no extractable code)")
    print(f"  call-based, from the harness   {call_pass}/{call_based}")
    print(f"  stdin, as the harness scored   {harness_stdin}/{len(jobs)}")
    print(f"  stdin, run as written          {fixed}/{len(jobs)}")
    print("  --")
    if total:
        print(f"  as graded   {(call_pass + harness_stdin) / total:.2%}")
        print(f"  re-graded   {(call_pass + fixed) / total:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
