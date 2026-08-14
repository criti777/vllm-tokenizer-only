import json
from pathlib import Path

from tools.build_requests import build_combinatorial, build_handwritten
from tools.import_ultrachat import select_ultrachat_records
from vllm_text_oracle.jsonl import read_jsonl, write_jsonl_atomic


def test_handwritten_builder_has_exact_count_unique_ids_and_categories() -> None:
    records = build_handwritten()

    assert len(records) == 300
    assert len({record["case_id"] for record in records}) == 300
    tags = {tag for record in records for tag in record["tags"]}
    assert {
        "basic",
        "roles",
        "content-parts",
        "tools",
        "reasoning",
        "unicode-whitespace",
        "long-context",
        "invalid",
    } <= tags


def test_combinatorial_builder_is_deterministic_and_has_exact_count() -> None:
    first = build_combinatorial(seed=20260814)
    second = build_combinatorial(seed=20260814)

    assert len(first) == 2000
    assert first == second
    assert len({record["case_id"] for record in first}) == 2000


def test_jsonl_atomic_round_trip_preserves_text(tmp_path: Path) -> None:
    records = [
        {
            "case_id": "whitespace.001",
            "request": {"messages": [{"role": "user", "content": " x\r\n "}]},
        }
    ]
    destination = tmp_path / "records.jsonl"

    write_jsonl_atomic(records, destination)

    assert list(read_jsonl(destination)) == records
    raw = destination.read_text(encoding="utf-8")
    assert json.loads(raw)["request"]["messages"][0]["content"] == " x\r\n "


def test_ultrachat_selection_is_hash_sorted_and_preserves_messages() -> None:
    rows = [
        {"prompt": "p2", "messages": [{"role": "user", "content": " x\r\n "}]},
        {"prompt": "p1", "messages": [{"role": "user", "content": "你好"}]},
        {"prompt": "p3", "messages": [{"role": "user", "content": "third"}]},
    ]

    first = select_ultrachat_records(rows, limit=2, source_shard="test.parquet")
    second = select_ultrachat_records(rows, limit=2, source_shard="test.parquet")

    assert first == second
    assert len(first) == 2
    assert [record["source"]["content_sha256"] for record in first] == sorted(
        record["source"]["content_sha256"] for record in first
    )
    selected_contents = {
        record["request"]["messages"][0]["content"] for record in first
    }
    assert selected_contents <= {" x\r\n ", "你好", "third"}
