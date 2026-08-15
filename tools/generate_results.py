"""Generate GLM-5.2 reference results from request JSONL files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vllm_text_oracle.jsonl import read_jsonl, write_jsonl_atomic
from vllm_text_oracle.oracle import TextOracle
from vllm_text_oracle.hashing import canonical_json_sha256


MODEL_ID = "zai-org/GLM-5.2"
MODEL_REVISION = "b4734de4facf877f85769a911abafc5283eab3d9"
VLLM_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"


def generate_records(
    records: Iterable[Mapping[str, Any]], oracle: TextOracle
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each request record requires a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        request = record.get("request")
        if not isinstance(request, Mapping):
            raise ValueError(f"{case_id}: request must be an object")
        tags = record.get("tags", [])
        include_token_ids = isinstance(tags, list) and "handwritten" in tags
        result = oracle.process(
            request,
            case_id=case_id,
            include_token_ids=include_token_ids,
        )
        results.append(result.to_dict())
    return results


def reuse_complete_shard(
    request_records: Iterable[Mapping[str, Any]], result_path: Path
) -> list[dict[str, Any]]:
    requests = list(request_records)
    results = list(read_jsonl(result_path))
    if len(requests) != len(results):
        raise ValueError(
            f"incomplete result shard {result_path}: "
            f"expected {len(requests)}, found {len(results)}"
        )
    for request_record, result in zip(requests, results):
        case_id = request_record.get("case_id")
        if result.get("case_id") != case_id:
            raise ValueError(
                f"case_id mismatch in {result_path}: expected {case_id}, "
                f"found {result.get('case_id')}"
            )
        request = request_record.get("request")
        if not isinstance(request, Mapping):
            raise ValueError(f"{case_id}: request must be an object")
        expected_hash = canonical_json_sha256(dict(request))
        if result.get("request_sha256") != expected_hash:
            raise ValueError(f"request hash mismatch for {case_id} in {result_path}")
    return results


def dependency_versions() -> dict[str, str]:
    names = [
        "huggingface-hub",
        "jinja2",
        "pydantic",
        "tokenizers",
        "transformers",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(f"model_assets/zai-org--GLM-5.2/{MODEL_REVISION}"),
    )
    parser.add_argument(
        "--requests", type=Path, default=Path("datasets/requests")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"datasets/results/zai-org--GLM-5.2/{MODEL_REVISION}"),
    )
    args = parser.parse_args()
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite completed baseline: {args.output}")

    inputs = {
        "handwritten": args.requests / "handwritten.jsonl",
        "generated": args.requests / "generated" / "combinatorial.jsonl",
        "ultrachat": args.requests / "imported" / "ultrachat.jsonl",
    }
    oracle = TextOracle.from_local_assets(args.assets)
    counts: dict[str, dict[str, int]] = {}
    for name, source in inputs.items():
        request_records = list(read_jsonl(source))
        result_path = args.output / f"{name}.results.jsonl"
        if result_path.exists():
            results = reuse_complete_shard(request_records, result_path)
        else:
            results = generate_records(request_records, oracle)
            write_jsonl_atomic(results, result_path)
        counts[name] = {
            "total": len(results),
            "ok": sum(result["status"] == "ok" for result in results),
            "error": sum(result["status"] == "error" for result in results),
        }

    asset_manifest = json.loads((args.assets / "manifest.json").read_text("utf-8"))
    manifest = {
        "schema_version": 1,
        "oracle_version": "0.1.0",
        "vllm": {"tag": "v0.26.0", "commit": VLLM_COMMIT},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "asset_hashes": asset_manifest["files"],
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "counts": counts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
