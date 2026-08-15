from pathlib import Path

from tools.verify_upstream_parity import verify_parity


def test_structural_cases_match_independent_reference() -> None:
    assert verify_parity(Path("datasets/requests"), ultrachat_sample=10) == {
        "handwritten": 300,
        "generated": 2000,
        "ultrachat": 10,
    }
