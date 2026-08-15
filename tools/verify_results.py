"""Verify request/result correspondence and result integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vllm_text_oracle.hashing import (
    canonical_json_sha256,
    text_sha256,
    token_ids_sha256,
)
from vllm_text_oracle.jsonl import read_jsonl
from vllm_text_oracle.model_selection import parse_model_selection
from vllm_text_oracle.profiles import ModelRegistry


SHARDS = {
    "handwritten": "handwritten.jsonl",
    "generated": "generated/combinatorial.jsonl",
    "ultrachat": "imported/ultrachat.jsonl",
}
PROFILE_FILE = Path("models/profiles.json")
PROFILE_SHARDS = {
    "handwritten": ("handwritten.jsonl", "handwritten.jsonl"),
    "combinatorial": (
        "generated/combinatorial.jsonl",
        "combinatorial.jsonl",
    ),
    "ultrachat": ("imported/ultrachat.jsonl", "ultrachat.jsonl.gz"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_profile_result_set(
    request_dir: Path, result_dir: Path, profile: Any
) -> dict[str, int]:
    manifest = json.loads((result_dir / "manifest.json").read_text("utf-8"))
    if manifest.get("profile_id") != profile.profile_id:
        raise ValueError("manifest profile mismatch")
    if manifest.get("request_model_override") != profile.repository:
        raise ValueError("manifest model override mismatch")
    if manifest.get("model") != {
        "id": profile.repository,
        "revision": profile.revision,
    }:
        raise ValueError("manifest model identity mismatch")

    for filename, expected in manifest.get("files", {}).items():
        path = result_dir / filename
        if not path.is_file() or _sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"file hash mismatch: {filename}")

    total = ok = error = 0
    seen: set[str] = set()
    for shard, (request_name, result_name) in PROFILE_SHARDS.items():
        requests = list(read_jsonl(request_dir / request_name))
        results = list(read_jsonl(result_dir / result_name))
        if len(requests) != len(results):
            raise ValueError(f"{shard}: request/result count mismatch")
        shard_ok = shard_error = 0
        for request_record, result in zip(requests, results):
            case_id = request_record["case_id"]
            if case_id in seen:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            if result.get("case_id") != case_id:
                raise ValueError(f"case_id mismatch: {case_id}")
            effective_request = dict(request_record["request"])
            effective_request["model"] = profile.repository
            if result.get("request_sha256") != canonical_json_sha256(
                effective_request
            ):
                raise ValueError(f"request hash mismatch: {case_id}")
            if result.get("model_profile") != profile.profile_id:
                raise ValueError(f"result profile mismatch: {case_id}")
            if result.get("status") == "ok":
                _verify_success(case_id, result, shard == "handwritten")
                if shard != "handwritten" and "token_ids" in result:
                    raise ValueError(f"unexpected full token ids: {case_id}")
                shard_ok += 1
            elif result.get("status") == "error":
                _verify_error(case_id, result)
                shard_error += 1
            else:
                raise ValueError(f"invalid status: {case_id}")
        actual = {"total": len(results), "ok": shard_ok, "error": shard_error}
        if manifest.get("counts", {}).get(shard) != actual:
            raise ValueError(f"{shard}: manifest count mismatch")
        total += len(results)
        ok += shard_ok
        error += shard_error
    return {"total": total, "ok": ok, "error": error}


def verify_result_set(request_dir: Path, result_dir: Path) -> dict[str, int]:
    manifest = json.loads((result_dir / "manifest.json").read_text("utf-8"))
    total = ok = error = 0
    seen: set[str] = set()
    for shard, relative_request in SHARDS.items():
        requests = list(read_jsonl(request_dir / relative_request))
        results = list(read_jsonl(result_dir / f"{shard}.results.jsonl"))
        if len(requests) != len(results):
            raise ValueError(f"{shard}: request/result count mismatch")
        shard_ok = shard_error = 0
        for request_record, result in zip(requests, results):
            case_id = request_record["case_id"]
            if case_id in seen:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            if result.get("case_id") != case_id:
                raise ValueError(f"case_id mismatch: {case_id}")
            expected_request_hash = canonical_json_sha256(request_record["request"])
            if result.get("request_sha256") != expected_request_hash:
                raise ValueError(f"request hash mismatch: {case_id}")
            if result.get("status") == "ok":
                _verify_success(case_id, result, shard == "handwritten")
                shard_ok += 1
            elif result.get("status") == "error":
                _verify_error(case_id, result)
                shard_error += 1
            else:
                raise ValueError(f"invalid status: {case_id}")
        expected = manifest["counts"].get(shard)
        actual = {"total": len(results), "ok": shard_ok, "error": shard_error}
        if expected != actual:
            raise ValueError(f"{shard}: manifest count mismatch")
        total += len(results)
        ok += shard_ok
        error += shard_error
    return {"total": total, "ok": ok, "error": error}


def _verify_success(case_id: str, result: dict[str, Any], require_ids: bool) -> None:
    rendered = result.get("rendered_text")
    if not isinstance(rendered, str):
        raise ValueError(f"missing rendered text: {case_id}")
    if result.get("rendered_text_sha256") != text_sha256(rendered):
        raise ValueError(f"rendered text hash mismatch: {case_id}")
    count = result.get("token_ids_length")
    if not isinstance(count, int) or count < 0:
        raise ValueError(f"invalid token count: {case_id}")
    if not isinstance(result.get("token_ids_sha256"), str):
        raise ValueError(f"missing token hash: {case_id}")
    ids = result.get("token_ids")
    if require_ids and not isinstance(ids, list):
        raise ValueError(f"missing handwritten token ids: {case_id}")
    if ids is not None:
        if len(ids) != count:
            raise ValueError(f"token count mismatch: {case_id}")
        if result["token_ids_sha256"] != token_ids_sha256(ids):
            raise ValueError(f"token hash mismatch: {case_id}")


def _verify_error(case_id: str, result: dict[str, Any]) -> None:
    value = result.get("error")
    if not isinstance(value, dict) or not all(
        isinstance(value.get(key), str) and value[key]
        for key in ("stage", "type", "message")
    ):
        raise ValueError(f"invalid error record: {case_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", metavar="PROFILE")
    parser.add_argument("--requests", type=Path, default=Path("datasets/requests"))
    parser.add_argument(
        "--results", type=Path, default=Path("datasets/results/by-profile")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = ModelRegistry.from_file(PROFILE_FILE)
    selected = parse_model_selection(args.model or ["glm-5.2"], registry)
    summaries = {
        profile.profile_id: verify_profile_result_set(
            args.requests, args.results / profile.profile_id, profile
        )
        for profile in selected
    }
    print(json.dumps(summaries, sort_keys=True))


if __name__ == "__main__":
    main()
