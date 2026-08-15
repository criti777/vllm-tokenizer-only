import json
from pathlib import Path

import pytest

from tools.generate_results import generate_profile_result_set
from tools.verify_results import verify_profile_result_set, verify_result_set
from vllm_text_oracle.jsonl import write_jsonl_atomic
from vllm_text_oracle.oracle import TextOracle
from vllm_text_oracle.profiles import ModelRegistry


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


@pytest.mark.model("glm-5.2")
def test_verify_layered_profile_result_set_and_file_hashes(tmp_path: Path) -> None:
    requests = tmp_path / "requests"
    (requests / "generated").mkdir(parents=True)
    (requests / "imported").mkdir()
    record = {
        "case_id": "portable.handwritten.001",
        "tags": ["handwritten"],
        "request": {
            "model": "corpus-placeholder",
            "messages": [{"role": "user", "content": "hello"}],
        },
    }
    write_jsonl_atomic([record], requests / "handwritten.jsonl")
    write_jsonl_atomic(
        [{**record, "case_id": "portable.combinatorial.001"}],
        requests / "generated/combinatorial.jsonl",
    )
    write_jsonl_atomic(
        [{**record, "case_id": "portable.ultrachat.001"}],
        requests / "imported/ultrachat.jsonl",
    )
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve("glm-5.2")
    result_dir = generate_profile_result_set(
        profile,
        TextOracle.from_model("glm-5.2", assets_root=Path("model_assets")),
        request_dir=requests,
        output_root=tmp_path / "results",
    )

    assert verify_profile_result_set(requests, result_dir, profile) == {
        "total": 3,
        "ok": 3,
        "error": 0,
    }

    with (result_dir / "combinatorial.jsonl").open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_profile_result_set(requests, result_dir, profile)
