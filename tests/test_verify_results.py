import json
from pathlib import Path

import pytest

from tools.verify_results import verify_result_set


def test_verify_committed_result_set() -> None:
    summary = verify_result_set(
        Path("datasets/requests"),
        Path(
            "datasets/results/zai-org--GLM-5.2/"
            "b4734de4facf877f85769a911abafc5283eab3d9"
        ),
    )

    assert summary["total"] == 12_300
    assert summary["ok"] == 12_264
    assert summary["error"] == 36


def test_verify_rejects_tampered_rendered_text(tmp_path: Path) -> None:
    requests = tmp_path / "requests"
    results = tmp_path / "results"
    requests.mkdir()
    results.mkdir()
    request = {
        "case_id": "x",
        "tags": ["handwritten"],
        "request": {"messages": [{"role": "user", "content": "x"}]},
    }
    (requests / "handwritten.jsonl").write_text(json.dumps(request) + "\n")
    (requests / "generated").mkdir()
    (requests / "imported").mkdir()
    (requests / "generated/combinatorial.jsonl").write_text("")
    (requests / "imported/ultrachat.jsonl").write_text("")
    result = {
        "case_id": "x",
        "status": "ok",
        "request_sha256": "wrong",
        "rendered_text": "tampered",
        "rendered_text_sha256": "wrong",
        "token_ids_length": 1,
        "token_ids_sha256": "wrong",
        "token_ids": [1],
    }
    (results / "handwritten.results.jsonl").write_text(json.dumps(result) + "\n")
    (results / "generated.results.jsonl").write_text("")
    (results / "ultrachat.results.jsonl").write_text("")
    (results / "manifest.json").write_text(json.dumps({"counts": {}}))

    with pytest.raises(ValueError, match="request hash mismatch"):
        verify_result_set(requests, results)
