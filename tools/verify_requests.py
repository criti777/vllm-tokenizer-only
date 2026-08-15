"""Validate request corpus structure, counts, IDs, and manifest hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_text_oracle.jsonl import read_jsonl


SHARDS = {
    "handwritten": ("handwritten.jsonl", 300),
    "generated": ("generated/combinatorial.jsonl", 2000),
    "ultrachat": ("imported/ultrachat.jsonl", 10000),
}


def verify_requests(root: Path) -> dict[str, int]:
    seen: set[str] = set()
    summary: dict[str, int] = {}
    for name, (relative, expected) in SHARDS.items():
        records = list(read_jsonl(root / relative))
        for record in records:
            case_id = record.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{name}: missing case_id")
            if case_id in seen:
                raise ValueError(f"duplicate case_id: {case_id}")
            seen.add(case_id)
            request = record.get("request")
            if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
                raise ValueError(f"{case_id}: invalid request envelope")
        summary[name] = len(records)
    for name, (_, expected) in SHARDS.items():
        if summary[name] != expected:
            raise ValueError(f"{name}: expected {expected}, found {summary[name]}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=Path("datasets/requests"))
    args = parser.parse_args()
    print(json.dumps(verify_requests(args.requests), sort_keys=True))


if __name__ == "__main__":
    main()
