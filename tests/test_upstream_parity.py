from pathlib import Path

import pytest

from tools.verify_upstream_parity import verify_parity


pytestmark = pytest.mark.model("glm-5.2")


def test_structural_cases_match_independent_reference() -> None:
    assert verify_parity(Path("datasets/requests"), ultrachat_sample=10) == {
        "handwritten": 300,
        "generated": 2000,
        "ultrachat": 10,
    }
