"""Cross-language deterministic hashing contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON deterministically without altering array order or text."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return _sha256(canonical_json_bytes(value))


def text_sha256(text: str) -> str:
    return _sha256(text.encode("utf-8"))


def token_ids_sha256(ids: Sequence[int]) -> str:
    return _sha256(",".join(str(token_id) for token_id in ids).encode("utf-8"))

