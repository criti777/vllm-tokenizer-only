"""Deterministically import OpenAI-style requests from an UltraChat shard."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

from vllm_text_oracle.hashing import canonical_json_sha256
from vllm_text_oracle.jsonl import write_jsonl_atomic


DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
MODEL = "zai-org/GLM-5.2"


def select_ultrachat_records(
    rows: Iterable[Mapping[str, Any]], *, limit: int, source_shard: str
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, int, list[dict[str, Any]]]] = []
    for source_index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        content_hash = canonical_json_sha256(messages)
        candidates.append((content_hash, source_index, messages))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < limit:
        raise ValueError(f"requested {limit} records, found {len(candidates)}")
    return [
        {
            "case_id": f"ultrachat.{content_hash}.{source_index}",
            "tags": ["imported", "ultrachat"],
            "source": {
                "kind": "imported",
                "dataset": DATASET,
                "revision": DATASET_REVISION,
                "split": "train_sft",
                "shard": source_shard,
                "source_index": source_index,
                "content_sha256": content_hash,
            },
            "request": {"model": MODEL, "messages": messages},
        }
        for content_hash, source_index, messages in candidates[:limit]
    ]


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    table = parquet.read_table(path, columns=["messages"])
    return table.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/requests/imported/ultrachat.jsonl"),
    )
    args = parser.parse_args()
    rows = read_parquet_rows(args.parquet)
    records = select_ultrachat_records(
        rows, limit=args.limit, source_shard=args.source_shard
    )
    write_jsonl_atomic(records, args.output)


if __name__ == "__main__":
    main()
