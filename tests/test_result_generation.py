from pathlib import Path

import pytest

from tools.generate_results import generate_records, reuse_complete_shard
from vllm_text_oracle.jsonl import write_jsonl_atomic
from vllm_text_oracle.oracle import TextOracle


pytestmark = pytest.mark.model("glm-5.2")


MODEL_PATH = Path(
    "model_assets/zai-org--GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9"
)


@pytest.fixture(scope="module")
def oracle() -> TextOracle:
    return TextOracle.from_local_assets(MODEL_PATH)


def test_generate_records_keeps_one_result_per_request(oracle: TextOracle) -> None:
    requests = [
        {
            "case_id": "handwritten.basic.001",
            "tags": ["handwritten"],
            "request": {"messages": [{"role": "user", "content": "hello"}]},
        },
        {
            "case_id": "generated.001",
            "tags": ["generated"],
            "request": {"messages": [{"role": "user", "content": "world"}]},
        },
    ]

    results = generate_records(requests, oracle)

    assert [result["case_id"] for result in results] == [
        "handwritten.basic.001",
        "generated.001",
    ]
    assert "token_ids" in results[0]
    assert "token_ids" not in results[1]
    assert results[0]["token_ids_length"] == len(results[0]["token_ids"])


def test_generate_records_rejects_duplicate_case_ids(oracle: TextOracle) -> None:
    requests = [
        {
            "case_id": "duplicate",
            "tags": [],
            "request": {"messages": [{"role": "user", "content": "a"}]},
        },
        {
            "case_id": "duplicate",
            "tags": [],
            "request": {"messages": [{"role": "user", "content": "b"}]},
        },
    ]

    with pytest.raises(ValueError, match="duplicate case_id"):
        generate_records(requests, oracle)


def test_generate_records_preserves_stable_error_shape(oracle: TextOracle) -> None:
    results = generate_records(
        [{"case_id": "invalid", "tags": ["handwritten"], "request": {"messages": [{}]}}],
        oracle,
    )

    assert results[0]["status"] == "error"
    assert results[0]["error"]["stage"] == "request_validation"
    assert results[0]["error"]["type"] == "validation_error"


def test_reuse_complete_shard_requires_matching_ids_and_request_hashes(
    tmp_path: Path, oracle: TextOracle
) -> None:
    requests = [
        {
            "case_id": "case.001",
            "tags": [],
            "request": {"messages": [{"role": "user", "content": "hello"}]},
        }
    ]
    results = generate_records(requests, oracle)
    result_path = tmp_path / "results.jsonl"
    write_jsonl_atomic(results, result_path)

    reused = reuse_complete_shard(requests, result_path)

    assert reused == results

    changed = [
        {
            "case_id": "case.001",
            "tags": [],
            "request": {"messages": [{"role": "user", "content": "changed"}]},
        }
    ]
    with pytest.raises(ValueError, match="request hash mismatch"):
        reuse_complete_shard(changed, result_path)
