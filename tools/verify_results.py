"""Verify request/result correspondence and result integrity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_text_oracle.hashing import (
    canonical_json_sha256,
    text_sha256,
    token_ids_sha256,
)
from vllm_text_oracle.jsonl import read_jsonl


SHARDS = {
    "handwritten": "handwritten.jsonl",
    "generated": "generated/combinatorial.jsonl",
    "ultrachat": "imported/ultrachat.jsonl",
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=Path("datasets/requests"))
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_result_set(args.requests, args.results), sort_keys=True))


if __name__ == "__main__":
    main()
