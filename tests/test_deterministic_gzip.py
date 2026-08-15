import gzip
from pathlib import Path

from vllm_text_oracle.jsonl import read_jsonl, write_jsonl_gzip_atomic


def test_gzip_jsonl_is_byte_deterministic(tmp_path: Path) -> None:
    records = [{"text": "你好", "id": 1}, {"id": 2, "text": "world"}]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "different-name.jsonl.gz"

    write_jsonl_gzip_atomic(records, first)
    write_jsonl_gzip_atomic(records, second)

    assert first.read_bytes() == second.read_bytes()
    assert list(read_jsonl(first)) == records


def test_gzip_header_has_zero_mtime_and_no_filename(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl.gz"

    write_jsonl_gzip_atomic([{"id": 1}], path)

    payload = path.read_bytes()
    assert int.from_bytes(payload[4:8], "little") == 0
    assert payload[3] & gzip.FNAME == 0
