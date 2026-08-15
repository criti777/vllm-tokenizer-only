from pathlib import Path

import pytest

from vllm_text_oracle.model_selection import parse_model_selection
from vllm_text_oracle.profiles import ModelRegistry


_REGISTRY_PATH = Path(__file__).parents[1] / "models" / "profiles.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="append",
        default=None,
        metavar="PROFILE",
        help="run tests for a model profile; repeat or pass 'all'",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model(profile): test requires the named model profile assets",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    registry = ModelRegistry.from_file(_REGISTRY_PATH)
    try:
        selected = parse_model_selection(config.getoption("model"), registry)
    except (ValueError, LookupError) as error:
        raise pytest.UsageError(str(error)) from error
    selected_ids = {profile.profile_id for profile in selected}

    for item in items:
        marker = item.get_closest_marker("model")
        if marker is None:
            continue
        required = set(marker.args)
        if not selected_ids:
            item.add_marker(pytest.mark.skip(reason="model tests require --model"))
        elif not required & selected_ids:
            item.add_marker(
                pytest.mark.skip(reason=f"requires model: {', '.join(sorted(required))}")
            )
