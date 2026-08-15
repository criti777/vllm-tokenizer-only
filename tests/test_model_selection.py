from pathlib import Path

import pytest

from tools.generate_results import build_parser as build_generate_parser
from tools.verify_results import build_parser as build_verify_parser
from vllm_text_oracle.errors import ProfileResolutionError
from vllm_text_oracle.model_selection import parse_model_selection
from vllm_text_oracle.profiles import ModelRegistry


REGISTRY = ModelRegistry.from_file(
    Path(__file__).parents[1] / "models" / "profiles.json"
)


def ids(values: list[str] | None) -> tuple[str, ...]:
    return tuple(
        profile.profile_id for profile in parse_model_selection(values, REGISTRY)
    )


def test_no_selection_means_core_only() -> None:
    assert ids(None) == ()
    assert ids([]) == ()


def test_repeated_selection_preserves_order_and_removes_duplicates() -> None:
    assert ids(["glm-5.2", "deepseek-v4", "glm-5.2"]) == (
        "glm-5.2",
        "deepseek-v4",
    )


def test_all_expands_in_registry_order() -> None:
    assert ids(["all"]) == REGISTRY.profile_ids


def test_all_cannot_be_combined_with_profiles() -> None:
    with pytest.raises(ValueError, match="cannot combine 'all'"):
        ids(["all", "glm-5.2"])


def test_unknown_selection_is_a_hard_failure() -> None:
    with pytest.raises(ProfileResolutionError, match="unknown model alias"):
        ids(["not-a-model"])


def test_generate_cli_accepts_repeatable_model_selection() -> None:
    args = build_generate_parser().parse_args(
        ["--model", "glm-5.2", "--model", "deepseek-v4"]
    )

    assert args.model == ["glm-5.2", "deepseek-v4"]


def test_verify_cli_accepts_repeatable_model_selection() -> None:
    args = build_verify_parser().parse_args(
        [
            "--results",
            "result-dir",
            "--model",
            "glm-5.2",
            "--model",
            "deepseek-v4",
        ]
    )

    assert args.model == ["glm-5.2", "deepseek-v4"]


@pytest.mark.model("glm-5.2")
def test_selected_model_marker_runs_when_requested() -> None:
    assert True
