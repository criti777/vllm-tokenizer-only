from pathlib import Path
import json

import pytest

from vllm_text_oracle import TextOracle


ASSETS_ROOT = Path("model_assets")


@pytest.mark.parametrize(
    "profile_id",
    [
        pytest.param("deepseek-v3", marks=pytest.mark.model("deepseek-v3")),
        pytest.param("kimi-k2.6", marks=pytest.mark.model("kimi-k2.6")),
        pytest.param("glm-5.1", marks=pytest.mark.model("glm-5.1")),
        pytest.param("glm-5.2", marks=pytest.mark.model("glm-5.2")),
        pytest.param("minimax-m2.7", marks=pytest.mark.model("minimax-m2.7")),
    ],
)
def test_basic_request_renders_and_encodes_for_hf_profile(profile_id: str) -> None:
    oracle = TextOracle.from_model(profile_id, assets_root=ASSETS_ROOT)

    result = oracle.process(
        {"messages": [{"role": "user", "content": "你好, world 🌍"}]},
        case_id=f"{profile_id}.basic",
        include_token_ids=True,
    )

    assert result.status == "ok", result.error
    assert result.model_profile == profile_id
    assert result.renderer == "hf"
    expected = json.loads(
        (Path("tests/model_cases") / profile_id / "basic.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.rendered_text == expected["rendered_text"]
    assert result.token_ids == tuple(expected["token_ids"])
    assert result.token_ids_length == len(result.token_ids)


@pytest.mark.model("glm-5.1")
def test_glm51_does_not_reuse_glm52_template_bytes() -> None:
    glm51 = TextOracle.from_model("glm-5.1", assets_root=ASSETS_ROOT)
    glm52 = TextOracle.from_model("glm-5.2", assets_root=ASSETS_ROOT)
    request = {"messages": [{"role": "user", "content": "compare templates"}]}

    result51 = glm51.process(request, case_id="glm51.template", include_token_ids=True)
    result52 = glm52.process(request, case_id="glm52.template", include_token_ids=True)

    assert result51.rendered_text != result52.rendered_text
    assert result51.token_ids != result52.token_ids
