"""Byte-for-byte comparison for independently generated baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RESULT_FILES = [
    "handwritten.results.jsonl",
    "generated.results.jsonl",
    "ultrachat.results.jsonl",
    "manifest.json",
]


def compare_directories(expected: Path, actual: Path, files: list[str] = RESULT_FILES) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in files:
        left = expected / name
        right = actual / name
        if not left.is_file() or not right.is_file():
            raise ValueError(f"missing reproducibility file: {name}")
        left_bytes = left.read_bytes()
        right_bytes = right.read_bytes()
        if left_bytes != right_bytes:
            raise ValueError(f"byte mismatch: {name}")
        hashes[name] = hashlib.sha256(left_bytes).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_directories(args.expected, args.actual), sort_keys=True))


if __name__ == "__main__":
    main()
