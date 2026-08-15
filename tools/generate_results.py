"""Generate GLM-5.2 reference results from request JSONL files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vllm_text_oracle.jsonl import (
    read_jsonl,
    write_jsonl_atomic,
    write_jsonl_gzip_atomic,
)
from vllm_text_oracle.oracle import TextOracle
from vllm_text_oracle.hashing import canonical_json_sha256
from vllm_text_oracle.model_selection import parse_model_selection
from vllm_text_oracle.profiles import ModelRegistry


VLLM_COMMIT = "568afb3a13806beb53bb2e6bd518269357b237c0"
PROFILE_FILE = Path("models/profiles.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gzip_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_records(
    records: Iterable[Mapping[str, Any]],
    oracle: TextOracle,
    *,
    include_token_ids: bool | None = None,
    model_override: str | None = None,
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
        source_request = record.get("request")
        if not isinstance(source_request, Mapping):
            raise ValueError(f"{case_id}: request must be an object")
        request = dict(source_request)
        if model_override is not None:
            request["model"] = model_override
        tags = record.get("tags", [])
        keep_ids = include_token_ids
        if keep_ids is None:
            keep_ids = isinstance(tags, list) and "handwritten" in tags
        result = oracle.process(
            request,
            case_id=case_id,
            include_token_ids=keep_ids,
        )
        results.append(result.to_dict())
    return results


def reuse_complete_shard(
    request_records: Iterable[Mapping[str, Any]],
    result_path: Path,
    *,
    model_override: str | None = None,
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
        source_request = request_record.get("request")
        if not isinstance(source_request, Mapping):
            raise ValueError(f"{case_id}: request must be an object")
        request = dict(source_request)
        if model_override is not None:
            request["model"] = model_override
        expected_hash = canonical_json_sha256(request)
        if result.get("request_sha256") != expected_hash:
            raise ValueError(f"request hash mismatch for {case_id} in {result_path}")
    return results


def dependency_versions() -> dict[str, str]:
    names = [
        "huggingface-hub",
        "jinja2",
        "pydantic",
        "tiktoken",
        "tokenizers",
        "transformers",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def generate_profile_result_set(
    profile,
    oracle: TextOracle,
    *,
    request_dir: Path,
    output_root: Path,
    ultrachat_limit: int | None = None,
) -> Path:
    if ultrachat_limit is not None and ultrachat_limit <= 0:
        raise ValueError("ultrachat_limit must be positive")
    output = output_root / profile.profile_id
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError(f"refusing to overwrite completed baseline: {output}")
    inputs = {
        "handwritten": request_dir / "handwritten.jsonl",
        "combinatorial": request_dir / "generated" / "combinatorial.jsonl",
        "ultrachat": request_dir / "imported" / "ultrachat.jsonl",
    }
    names = {
        "handwritten": "handwritten.jsonl",
        "combinatorial": "combinatorial.jsonl",
        "ultrachat": "ultrachat.jsonl.gz",
    }
    counts: dict[str, dict[str, int]] = {}
    files: dict[str, dict[str, int | str]] = {}
    for shard, source in inputs.items():
        request_records = list(read_jsonl(source))
        if shard == "ultrachat" and ultrachat_limit is not None:
            request_records = request_records[:ultrachat_limit]
        result_path = output / names[shard]
        if result_path.exists():
            results = reuse_complete_shard(
                request_records,
                result_path,
                model_override=profile.repository,
            )
        else:
            results = generate_records(
                request_records,
                oracle,
                include_token_ids=shard == "handwritten",
                model_override=profile.repository,
            )
            if result_path.suffix == ".gz":
                write_jsonl_gzip_atomic(results, result_path)
            else:
                write_jsonl_atomic(results, result_path)
        counts[shard] = {
            "total": len(results),
            "ok": sum(result["status"] == "ok" for result in results),
            "error": sum(result["status"] == "error" for result in results),
        }
        file_info: dict[str, int | str] = {
            "size": result_path.stat().st_size,
            "sha256": _sha256_file(result_path),
        }
        if result_path.suffix == ".gz":
            file_info["content_sha256"] = _gzip_content_sha256(result_path)
        files[result_path.name] = file_info

    manifest = {
        "schema_version": 2,
        "oracle_version": "0.1.0",
        "vllm": {"tag": "v0.26.0", "commit": VLLM_COMMIT},
        "profile_id": profile.profile_id,
        "request_model_override": profile.repository,
        "selection": {"ultrachat_limit": ultrachat_limit},
        "model": {"id": profile.repository, "revision": profile.revision},
        "renderer": profile.renderer,
        "dependencies": dependency_versions(),
        "counts": counts,
        "files": files,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", metavar="PROFILE")
    parser.add_argument("--assets-root", type=Path, default=Path("model_assets"))
    parser.add_argument(
        "--requests", type=Path, default=Path("datasets/requests")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/results/by-profile"),
    )
    parser.add_argument("--ultrachat-limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = ModelRegistry.from_file(PROFILE_FILE)
    selected = parse_model_selection(args.model or ["glm-5.2"], registry)
    for profile in selected:
        oracle = TextOracle.from_model(
            profile.profile_id,
            assets_root=args.assets_root,
        )
        output = generate_profile_result_set(
            profile,
            oracle,
            request_dir=args.requests,
            output_root=args.output_root,
            ultrachat_limit=args.ultrachat_limit,
        )
        print(f"generated {profile.profile_id}: {output}")


if __name__ == "__main__":
    main()
