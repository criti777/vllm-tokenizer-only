from pathlib import Path

import pytest

from vllm_text_oracle import TextOracle


ASSETS_ROOT = Path("model_assets")


@pytest.mark.model("kimi-k2.6")
def test_kimi_renders_image_part_to_official_text_placeholder() -> None:
    oracle = TextOracle.from_model("kimi-k2.6", assets_root=ASSETS_ROOT)
    result = oracle.process(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://invalid.example/image.png"},
                        },
                        {"type": "text", "text": "after"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://invalid.example/second.png"},
                        },
                    ],
                }
            ]
        },
        case_id="kimi.image-placeholder",
    )

    assert result.status == "ok", result.error
    assert result.rendered_text is not None
    assert "before<|media_begin|>image<|media_content|>" in result.rendered_text
    assert "<|media_pad|><|media_end|>\nafter" in result.rendered_text
    assert result.rendered_text.count("<|media_begin|>image") == 2
    assert "invalid.example" not in result.rendered_text


@pytest.mark.parametrize(
    "profile_id",
    [
        pytest.param("glm-5.2", marks=pytest.mark.model("glm-5.2")),
        pytest.param("deepseek-v3.2", marks=pytest.mark.model("deepseek-v3.2")),
        pytest.param("deepseek-v4", marks=pytest.mark.model("deepseek-v4")),
    ],
)
def test_processor_dependent_media_has_stable_error_stage(profile_id: str) -> None:
    oracle = TextOracle.from_model(profile_id, assets_root=ASSETS_ROOT)
    result = oracle.process(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,not-decoded"},
                        },
                    ],
                }
            ]
        },
        case_id=f"{profile_id}.processor-required",
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.stage == "processor_required"
    assert result.error.type == "multimodal_processor_required"
