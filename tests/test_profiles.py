import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vllm_text_oracle.errors import OracleStage, ProfileResolutionError
from vllm_text_oracle.profiles import ModelProfile, ModelRegistry


PROFILE_FILE = Path(__file__).parents[1] / "models" / "profiles.json"


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry.from_file(PROFILE_FILE)


def test_registry_contains_exactly_the_confirmed_official_profiles(
    registry: ModelRegistry,
) -> None:
    assert registry.profile_ids == (
        "deepseek-v3",
        "deepseek-v3.2",
        "deepseek-v4",
        "kimi-k2.6",
        "glm-5.1",
        "glm-5.2",
        "minimax-m2.7",
    )


@pytest.mark.parametrize(
    ("profile_id", "repository", "renderer"),
    [
        ("deepseek-v3", "deepseek-ai/DeepSeek-V3", "hf"),
        ("deepseek-v3.2", "deepseek-ai/DeepSeek-V3.2", "deepseek_v32"),
        ("deepseek-v4", "deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4"),
        ("kimi-k2.6", "moonshotai/Kimi-K2.6", "hf"),
        ("glm-5.1", "zai-org/GLM-5.1", "hf"),
        ("glm-5.2", "zai-org/GLM-5.2", "hf"),
        ("minimax-m2.7", "MiniMaxAI/MiniMax-M2.7", "hf"),
    ],
)
def test_resolves_canonical_profiles(
    registry: ModelRegistry,
    profile_id: str,
    repository: str,
    renderer: str,
) -> None:
    profile = registry.resolve(profile_id)

    assert profile.repository == repository
    assert profile.renderer == renderer
    assert len(profile.revision) == 40


def test_resolves_declared_request_alias(registry: ModelRegistry) -> None:
    assert registry.resolve("zai-org/GLM-5.2").profile_id == "glm-5.2"


def test_remote_tokenizer_code_is_enabled_only_for_kimi(registry: ModelRegistry) -> None:
    enabled = tuple(
        profile_id
        for profile_id in registry.profile_ids
        if registry.resolve(profile_id).trust_remote_code
    )

    assert enabled == ("kimi-k2.6",)


def test_unknown_model_never_falls_back(registry: ModelRegistry) -> None:
    with pytest.raises(ProfileResolutionError, match="unknown model alias"):
        registry.resolve("deepseek-v99")


def test_duplicate_aliases_are_rejected() -> None:
    common = {
        "repository": "owner/model",
        "revision": "a" * 40,
        "renderer": "hf",
        "asset_manifest": "models/manifests/model.json",
        "capabilities": {},
    }
    first = ModelProfile(profile_id="one", aliases=("shared",), **common)
    second = ModelProfile(profile_id="two", aliases=("shared",), **common)

    with pytest.raises(ValueError, match="duplicate model alias"):
        ModelRegistry((first, second))


def test_profile_rejects_floating_revision() -> None:
    with pytest.raises(ValidationError, match="revision"):
        ModelProfile(
            profile_id="bad",
            aliases=(),
            repository="owner/model",
            revision="main",
            renderer="hf",
            asset_manifest="models/manifests/bad.json",
            capabilities={},
        )


def test_oracle_stage_values_are_stable() -> None:
    assert [stage.value for stage in OracleStage] == [
        "profile_resolution",
        "request_validation",
        "message_normalization",
        "template_render",
        "encode",
        "processor_required",
        "asset_integrity",
    ]


def test_profile_file_is_valid_json() -> None:
    payload = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))

    assert payload[0]["profile_id"] == "deepseek-v3"
