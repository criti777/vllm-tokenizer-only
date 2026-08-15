from pathlib import Path

import pytest

from tools.generate_results import (
    generate_profile_result_set,
    generate_records,
    reuse_complete_shard,
)
from vllm_text_oracle.hashing import canonical_json_sha256
from vllm_text_oracle.jsonl import read_jsonl, write_jsonl_atomic
from vllm_text_oracle.oracle import TextOracle
from vllm_text_oracle.profiles import ModelRegistry


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


def test_generate_records_can_override_model_and_token_retention(
    oracle: TextOracle,
) -> None:
    records = [
        {
            "case_id": "portable.001",
            "tags": ["handwritten"],
            "request": {
                "model": "source-corpus-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        }
    ]

    results = generate_records(
        records,
        oracle,
        include_token_ids=False,
        model_override="zai-org/GLM-5.2",
    )

    assert results[0]["status"] == "ok"
    assert "token_ids" not in results[0]
    assert results[0]["request_sha256"] != canonical_json_sha256(
        records[0]["request"]
    )


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


def test_generate_profile_result_set_uses_layered_model_layout(
    tmp_path: Path,
) -> None:
    request_dir = tmp_path / "requests"
    (request_dir / "generated").mkdir(parents=True)
    (request_dir / "imported").mkdir()
    records = [
        {
            "case_id": "one",
            "tags": ["handwritten"],
            "request": {"messages": [{"role": "user", "content": "hello"}]},
        }
    ]
    write_jsonl_atomic(records, request_dir / "handwritten.jsonl")
    write_jsonl_atomic(records, request_dir / "generated/combinatorial.jsonl")
    write_jsonl_atomic(records, request_dir / "imported/ultrachat.jsonl")
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    profile = registry.resolve("glm-5.2")
    profile_oracle = TextOracle.from_model("glm-5.2", assets_root=Path("model_assets"))

    result_dir = generate_profile_result_set(
        profile,
        profile_oracle,
        request_dir=request_dir,
        output_root=tmp_path / "results",
        ultrachat_limit=1,
    )

    assert result_dir == tmp_path / "results/glm-5.2"
    assert (result_dir / "handwritten.jsonl").is_file()
    assert (result_dir / "combinatorial.jsonl").is_file()
    assert (result_dir / "ultrachat.jsonl.gz").is_file()
    combinatorial = list(read_jsonl(result_dir / "combinatorial.jsonl"))[0]
    ultrachat = list(read_jsonl(result_dir / "ultrachat.jsonl.gz"))[0]
    assert ultrachat["model_profile"] == "glm-5.2"
    assert "token_ids" not in combinatorial
    assert "token_ids" not in ultrachat
    manifest = __import__("json").loads(
        (result_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["profile_id"] == "glm-5.2"
    assert manifest["request_model_override"] == "zai-org/GLM-5.2"
    assert manifest["selection"]["ultrachat_limit"] == 1
    assert manifest["counts"]["ultrachat"] == {"error": 0, "ok": 1, "total": 1}
    assert len(manifest["files"]["ultrachat.jsonl.gz"]["sha256"]) == 64
    assert len(manifest["files"]["ultrachat.jsonl.gz"]["content_sha256"]) == 64


def test_profile_result_set_is_byte_reproducible(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    (request_dir / "generated").mkdir(parents=True)
    (request_dir / "imported").mkdir()
    for index, path in enumerate(
        (
            request_dir / "handwritten.jsonl",
            request_dir / "generated/combinatorial.jsonl",
            request_dir / "imported/ultrachat.jsonl",
        )
    ):
        write_jsonl_atomic(
            [
                {
                    "case_id": f"repro.{index}",
                    "tags": [],
                    "request": {
                        "messages": [{"role": "user", "content": "same"}]
                    },
                }
            ],
            path,
        )
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve(
        "glm-5.2"
    )
    profile_oracle = TextOracle.from_model(
        "glm-5.2", assets_root=Path("model_assets")
    )

    first = generate_profile_result_set(
        profile,
        profile_oracle,
        request_dir=request_dir,
        output_root=tmp_path / "first",
    )
    second = generate_profile_result_set(
        profile,
        profile_oracle,
        request_dir=request_dir,
        output_root=tmp_path / "second",
    )

    for name in (
        "handwritten.jsonl",
        "combinatorial.jsonl",
        "ultrachat.jsonl.gz",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    before_resume = {
        name: (first / name).read_bytes()
        for name in (
            "handwritten.jsonl",
            "combinatorial.jsonl",
            "ultrachat.jsonl.gz",
        )
    }
    (first / "manifest.json").unlink()
    resumed = generate_profile_result_set(
        profile,
        profile_oracle,
        request_dir=request_dir,
        output_root=tmp_path / "first",
    )
    assert {
        name: (resumed / name).read_bytes() for name in before_resume
    } == before_resume


def test_profile_result_set_applies_ultrachat_limit(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    (request_dir / "generated").mkdir(parents=True)
    (request_dir / "imported").mkdir()
    write_jsonl_atomic([], request_dir / "handwritten.jsonl")
    write_jsonl_atomic([], request_dir / "generated/combinatorial.jsonl")
    write_jsonl_atomic(
        [
            {
                "case_id": f"ultrachat.{index}",
                "tags": [],
                "request": {
                    "messages": [{"role": "user", "content": str(index)}]
                },
            }
            for index in range(3)
        ],
        request_dir / "imported/ultrachat.jsonl",
    )
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve(
        "glm-5.2"
    )

    result_dir = generate_profile_result_set(
        profile,
        TextOracle.from_model("glm-5.2", assets_root=Path("model_assets")),
        request_dir=request_dir,
        output_root=tmp_path / "results",
        ultrachat_limit=2,
    )

    results = list(read_jsonl(result_dir / "ultrachat.jsonl.gz"))
    assert [result["case_id"] for result in results] == [
        "ultrachat.0",
        "ultrachat.1",
    ]


@pytest.mark.model("deepseek-v4")
def test_profile_result_set_supports_specialized_renderer(tmp_path: Path) -> None:
    request_dir = tmp_path / "requests"
    (request_dir / "generated").mkdir(parents=True)
    (request_dir / "imported").mkdir()
    records = [
        {
            "case_id": "deepseek-v4.basic",
            "tags": [],
            "request": {
                "messages": [{"role": "user", "content": "hello"}]
            },
        }
    ]
    for path in (
        request_dir / "handwritten.jsonl",
        request_dir / "generated/combinatorial.jsonl",
        request_dir / "imported/ultrachat.jsonl",
    ):
        write_jsonl_atomic(records, path)
    profile = ModelRegistry.from_file(Path("models/profiles.json")).resolve(
        "deepseek-v4"
    )

    result_dir = generate_profile_result_set(
        profile,
        TextOracle.from_model("deepseek-v4", assets_root=Path("model_assets")),
        request_dir=request_dir,
        output_root=tmp_path / "results",
    )

    result = list(read_jsonl(result_dir / "handwritten.jsonl"))[0]
    assert result["status"] == "ok"
    assert result["renderer"] == "deepseek_v4"
