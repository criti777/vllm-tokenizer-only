from pathlib import Path

import pytest

from vllm_text_oracle import ProfileResolutionError, TextOracle
from vllm_text_oracle.renderers import RendererUnavailableError


pytestmark = pytest.mark.model("glm-5.2")

ASSETS_ROOT = Path("model_assets")


def test_from_model_returns_profile_aware_result() -> None:
    oracle = TextOracle.from_model("glm-5.2", assets_root=ASSETS_ROOT)

    result = oracle.process(
        {
            "model": "zai-org/GLM-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        },
        case_id="profile.001",
        include_token_ids=True,
    )

    assert result.status == "ok"
    assert result.model_profile == "glm-5.2"
    assert result.renderer == "hf"
    assert result.rendered_text == (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>你好<|assistant|><think>"
    )


def test_request_model_must_resolve_to_oracle_profile() -> None:
    oracle = TextOracle.from_model("glm-5.2", assets_root=ASSETS_ROOT)

    result = oracle.process(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "hello"}],
        },
        case_id="profile.mismatch",
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.stage == "profile_resolution"
    assert result.error.type == "model_profile_mismatch"


def test_unknown_model_is_rejected_before_assets_are_loaded() -> None:
    with pytest.raises(ProfileResolutionError, match="unknown model alias"):
        TextOracle.from_model("unknown/model", assets_root=ASSETS_ROOT)


def test_specialized_renderer_never_falls_back_to_hf() -> None:
    with pytest.raises(RendererUnavailableError, match="deepseek_v4"):
        TextOracle.from_model("deepseek-v4", assets_root=ASSETS_ROOT)
