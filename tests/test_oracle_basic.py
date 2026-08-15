from pathlib import Path

import pytest

from vllm_text_oracle import TextOracle as PublicTextOracle
from vllm_text_oracle.oracle import TextOracle


MODEL_PATH = Path(
    "model_assets/zai-org--GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9"
)


def test_text_oracle_is_exported_from_public_package() -> None:
    assert PublicTextOracle is TextOracle


@pytest.fixture(scope="module")
def oracle() -> TextOracle:
    return TextOracle.from_local_assets(MODEL_PATH)


def test_glm52_default_render_and_ids(oracle: TextOracle) -> None:
    result = oracle.process(
        {
            "model": "zai-org/GLM-5.2",
            "messages": [{"role": "user", "content": "你好"}],
        },
        case_id="basic.001",
        include_token_ids=True,
    )

    assert result.status == "ok"
    assert result.rendered_text == (
        "[gMASK]<sop><|system|>Reasoning Effort: Max"
        "<|user|>你好<|assistant|><think>"
    )
    assert result.token_ids == (
        154822,
        154824,
        154826,
        25062,
        287,
        29905,
        371,
        25,
        7487,
        154827,
        109377,
        154828,
        154841,
    )


def test_request_thinking_kwarg_changes_generation_prompt(oracle: TextOracle) -> None:
    result = oracle.process(
        {
            "messages": [{"role": "user", "content": "你好"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        case_id="thinking.off.001",
        include_token_ids=True,
    )

    assert result.rendered_text == (
        "[gMASK]<sop><|user|>你好<|assistant|><think></think>"
    )
    assert result.token_ids_length == 7


def test_invalid_message_returns_stable_validation_error(oracle: TextOracle) -> None:
    result = oracle.process(
        {"messages": [{"content": "missing role"}]},
        case_id="invalid.001",
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.stage == "request_validation"
    assert result.error.type == "validation_error"
