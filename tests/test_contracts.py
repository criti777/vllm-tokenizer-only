import hashlib
import json

from vllm_text_oracle.contracts import OracleError, OracleResult
from vllm_text_oracle.hashing import (
    canonical_json_sha256,
    text_sha256,
    token_ids_sha256,
)


def test_canonical_json_hash_ignores_object_key_order() -> None:
    left = {"messages": [{"content": "你好", "role": "user"}], "model": "m"}
    right = {"model": "m", "messages": [{"role": "user", "content": "你好"}]}

    assert canonical_json_sha256(left) == canonical_json_sha256(right)


def test_text_hash_uses_exact_utf8_bytes() -> None:
    text = "你好\r\n"

    assert text_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_token_hash_uses_comma_decimal_form() -> None:
    assert token_ids_sha256([154800, 42, 7]) == hashlib.sha256(
        b"154800,42,7"
    ).hexdigest()


def test_success_result_serializes_stably() -> None:
    result = OracleResult.success(
        case_id="basic.001",
        request_sha256="request-hash",
        rendered_text="hello",
        token_ids=[1, 2],
        include_token_ids=True,
        diagnostics={"chat_template_content_format": "string"},
    )

    payload = result.to_dict()

    assert payload["status"] == "ok"
    assert payload["token_ids_length"] == 2
    assert payload["token_ids"] == [1, 2]
    assert json.dumps(payload, sort_keys=True, ensure_ascii=False)


def test_error_result_has_stable_stage_and_type() -> None:
    result = OracleResult.failure(
        case_id="invalid.001",
        request_sha256="request-hash",
        error=OracleError(
            stage="request_validation",
            type="validation_error",
            message="role is required",
        ),
    )

    assert result.to_dict()["error"] == {
        "stage": "request_validation",
        "type": "validation_error",
        "message": "role is required",
    }
