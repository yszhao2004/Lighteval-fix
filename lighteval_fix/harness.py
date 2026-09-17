"""Fixes for things that break a run before grading ever happens.

These are not about answer extraction, and they are in this repository for one
reason: if you are applying these fixes in a fresh environment, these are the
faults you will hit first. One of them destroys a completed run. Two of them
let a run finish while measuring something other than what you configured,
which is worse, because the report looks fine.

Installed by monkeypatching at import time, so no LightEval source is edited:

    import lighteval_fix.harness as harness
    harness.apply_all()                      # before importing lighteval models

Each patch self-checks and raises if the shape it depends on has changed
upstream. A silently inert patch is worse than no patch, because the run still
produces numbers.
"""

from __future__ import annotations

import json
import os

__all__ = ["apply_all", "patch_xxhash_accepts_str",
           "patch_generation_parameters", "patch_sglang_transformers_import"]


def patch_xxhash_accepts_str() -> None:
    """Let the details logger hash the strings it already passes it.

    ``logging/info_loggers.py`` hashes three values that are all ``str``
    (``xxhash.xxh64(doc.query)`` and ``xxhash.xxh64(str(...))`` twice more),
    but xxhash accepts only bytes, so the run raises

        TypeError: Strings must be encoded before hashing

    **after every sample has been generated and scored**, and writes no result
    file. The entire generation budget is lost to a logging detail. Encoding on
    the way in fixes all three call sites and is a no-op for callers that
    already pass bytes.
    """
    import xxhash

    original = xxhash.xxh64

    def xxh64(data=b"", *args, **kwargs):
        if isinstance(data, str):
            data = data.encode("utf-8")
        return original(data, *args, **kwargs)

    xxhash.xxh64 = xxh64
    if xxhash.xxh64("selftest").hexdigest() != original(b"selftest").hexdigest():
        raise RuntimeError("xxhash str-encoding patch did not take effect")


# Accepted by LightEval's config model and then not forwarded.
_DROPPED = ("top_k", "min_p", "presence_penalty")


def patch_generation_parameters(extra_keys=_DROPPED, forced=None,
                                chat_template_kwargs=None) -> None:
    """Forward the sampling parameters the endpoint backend leaves behind.

    ``GenerationParameters.to_litellm_dict`` forwards only
    ``max_completion_tokens``, ``stop``, ``temperature``, ``top_p``, ``seed``,
    ``repetition_penalty`` and ``frequency_penalty``. A protocol that also
    fixes ``top_k``, ``min_p`` or ``presence_penalty`` loses them **in
    silence**: the run completes and reports numbers produced under different
    sampling than the one configured. Nothing warns.

    Three distinct cases needing different handling:

    * **Accepted, then dropped** -- ``top_k``, ``min_p``,
      ``presence_penalty``. Re-added here from the config object.
    * **Refused by the config model** -- ``top_k=-1``, which an
      OpenAI-compatible SGLang server reads as "disabled" while LightEval
      constrains the field to >= 0 and rejects the config before a request is
      sent. Such values cannot go in the model yaml at all; pass them in
      ``forced``.
    * **Not sampling** -- ``chat_template_kwargs``, which gates reasoning in
      some templates (``enable_thinking``, or ``thinking`` for others).
      Omitting it can evaluate a reasoning model with reasoning off, which
      looks like a much weaker model rather than like a mistake.

    All three ride in ``extra_body``, which **litellm** unpacks into the
    request body. That is a property of litellm, not of the server: code that
    POSTs raw JSON to ``/v1/chat/completions`` must put these fields at the
    **top level**, because no server reads a nested ``extra_body``. Getting
    that wrong is silent in the same way -- the request is accepted and the
    parameters ignored.
    """
    from lighteval.models.model_input import GenerationParameters

    forced = dict(forced or {})
    chat_template_kwargs = dict(chat_template_kwargs or {})
    original = GenerationParameters.to_litellm_dict

    def to_litellm_dict(self) -> dict:
        args = original(self)

        def body():
            return dict(args.get("extra_body") or {})

        dropped = {k: getattr(self, k, None) for k in extra_keys}
        dropped = {k: v for k, v in dropped.items() if v is not None}
        if dropped:
            args["extra_body"] = {**body(), **dropped}
            print(f"[lighteval-fix] restored dropped sampling params: {dropped}",
                  flush=True)
        if forced:
            args["extra_body"] = {**body(), **forced}
            print(f"[lighteval-fix] forced params the config model refuses: "
                  f"{forced}", flush=True)
        if chat_template_kwargs:
            args["extra_body"] = {**body(),
                                  "chat_template_kwargs": dict(chat_template_kwargs)}
        return args

    GenerationParameters.to_litellm_dict = to_litellm_dict

    # If upstream starts forwarding these itself the patch would send them
    # twice, and that should surface here rather than in whatever the server
    # does with a duplicated field.
    probe = GenerationParameters(temperature=1.0, top_p=0.95, top_k=20,
                                 min_p=0.0, presence_penalty=1.5,
                                 repetition_penalty=1.0, max_new_tokens=8,
                                 seed=0)
    still = [k for k in extra_keys if k in original(probe)]
    if still:
        raise RuntimeError(
            f"LightEval now forwards {still} itself; this patch would "
            "duplicate them and must be updated")


def patch_sglang_transformers_import() -> None:
    """Restore ``sglang.srt.hf_transformers_utils`` for LightEval's importer.

    LightEval's SGLang backend does an unguarded

        from sglang.srt.hf_transformers_utils import get_tokenizer

    at import time, and importing *any* LightEval model module pulls it in --
    including runs that use the endpoint backend and never touch SGLang in
    process. SGLang moved that module to ``srt/utils/hf_transformers_utils``,
    so a current SGLang turns any LightEval run into an ImportError before the
    first request.

    Writes a forwarding shim at the old path rather than editing a package we
    do not own. ``sglang.srt`` is a namespace package, so its ``__file__`` is
    None and the directory comes from ``__path__``. No-op when SGLang is
    absent, or when the old path already imports.
    """
    import pathlib
    import sys

    try:
        import sglang.srt
    except Exception:
        return
    try:
        import sglang.srt.hf_transformers_utils  # noqa: F401
        return
    except Exception:
        pass

    target = pathlib.Path(list(sglang.srt.__path__)[0]) / "hf_transformers_utils.py"
    target.write_text(
        '"""Compatibility shim: this module now lives in sglang.srt.utils."""\n'
        "from sglang.srt.utils.hf_transformers_utils import *  # noqa: F401,F403\n"
    )
    sys.modules.pop("sglang.srt.hf_transformers_utils", None)
    from sglang.srt.hf_transformers_utils import get_tokenizer  # noqa: F401


def apply_all(forced=None, chat_template_kwargs=None) -> None:
    """Install every harness fix.

    ``forced`` and ``chat_template_kwargs`` default to the
    ``LIGHTEVAL_FIX_FORCED_SAMPLING`` and
    ``LIGHTEVAL_FIX_CHAT_TEMPLATE_KWARGS`` environment variables (JSON), so a
    runner can set them without importing anything.
    """
    if forced is None:
        forced = json.loads(os.environ.get("LIGHTEVAL_FIX_FORCED_SAMPLING") or "{}")
    if chat_template_kwargs is None:
        chat_template_kwargs = json.loads(
            os.environ.get("LIGHTEVAL_FIX_CHAT_TEMPLATE_KWARGS") or "{}")
    patch_sglang_transformers_import()
    patch_xxhash_accepts_str()
    patch_generation_parameters(forced=forced,
                                chat_template_kwargs=chat_template_kwargs)
