import json
from pathlib import Path

import pytest

from tools.verify_reproducibility import compare_directories
from tools.verify_requests import verify_requests


def test_committed_requests_are_valid() -> None:
    summary = verify_requests(Path("datasets/requests"))
    assert summary == {"handwritten": 300, "generated": 2000, "ultrachat": 10000}


def test_request_verifier_rejects_duplicate_ids(tmp_path: Path) -> None:
    record = {"case_id": "same", "request": {"messages": []}}
    for relative in ("handwritten.jsonl", "generated/combinatorial.jsonl", "imported/ultrachat.jsonl"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate case_id"):
        verify_requests(tmp_path)


def test_reproducibility_comparison_detects_byte_difference(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    expected.mkdir()
    actual.mkdir()
    (expected / "x").write_bytes(b"a")
    (actual / "x").write_bytes(b"b")
    with pytest.raises(ValueError, match="byte mismatch"):
        compare_directories(expected, actual, ["x"])
