"""The harness patches, exercised without a model or a server.

Each one is meant to raise rather than sit inert if upstream's shape changed,
so the tests check both that it takes effect and that its self-check is real.
"""

from __future__ import annotations

import pytest


def test_xxhash_accepts_str_and_matches_bytes():
    xxhash = pytest.importorskip("xxhash")
    from lighteval_fix.harness import patch_xxhash_accepts_str

    with pytest.raises(TypeError):
        xxhash.xxh64("before the patch")

    patch_xxhash_accepts_str()
    assert (xxhash.xxh64("selftest").hexdigest()
            == xxhash.xxh64(b"selftest").hexdigest())


def test_generation_parameters_forwards_the_dropped_ones():
    pytest.importorskip("lighteval")
    from lighteval.models.model_input import GenerationParameters

    from lighteval_fix.harness import patch_generation_parameters

    params = GenerationParameters(temperature=1.0, top_p=0.95, top_k=20,
                                  min_p=0.0, presence_penalty=1.5,
                                  repetition_penalty=1.0, max_new_tokens=8,
                                  seed=0)
    before = GenerationParameters.to_litellm_dict(params)
    # The premise: upstream drops these three. If this assertion fails,
    # upstream has started forwarding them and the patch must be retired.
    assert not any(k in before for k in ("top_k", "min_p", "presence_penalty"))
    assert not (before.get("extra_body") or {}).get("top_k")

    original = GenerationParameters.to_litellm_dict
    try:
        patch_generation_parameters(forced={"top_k": -1},
                                    chat_template_kwargs={"thinking": True})
        after = GenerationParameters.to_litellm_dict(params)
        body = after.get("extra_body") or {}
        assert body.get("min_p") == 0.0
        assert body.get("presence_penalty") == 1.5
        # forced wins over the config value, which is the point of forcing it
        assert body.get("top_k") == -1
        assert body.get("chat_template_kwargs") == {"thinking": True}
        # and the standard fields are untouched
        assert after.get("temperature") == 1.0
    finally:
        GenerationParameters.to_litellm_dict = original


def test_sglang_shim_is_a_noop_without_sglang():
    """Absent SGLang it must do nothing, not raise: the endpoint backend
    does not need SGLang at all."""
    from lighteval_fix.harness import patch_sglang_transformers_import

    patch_sglang_transformers_import()
