"""Serializable oracle result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .hashing import text_sha256, token_ids_sha256


@dataclass(frozen=True)
class OracleError:
    stage: str
    type: str
    message: str


@dataclass(frozen=True)
class OracleResult:
    case_id: str
    status: Literal["ok", "error"]
    request_sha256: str
    rendered_text: str | None = None
    rendered_text_sha256: str | None = None
    token_ids_length: int | None = None
    token_ids_sha256: str | None = None
    token_ids: tuple[int, ...] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: OracleError | None = None

    @classmethod
    def success(
        cls,
        *,
        case_id: str,
        request_sha256: str,
        rendered_text: str,
        token_ids: list[int],
        include_token_ids: bool,
        diagnostics: dict[str, Any],
    ) -> "OracleResult":
        return cls(
            case_id=case_id,
            status="ok",
            request_sha256=request_sha256,
            rendered_text=rendered_text,
            rendered_text_sha256=text_sha256(rendered_text),
            token_ids_length=len(token_ids),
            token_ids_sha256=token_ids_sha256(token_ids),
            token_ids=tuple(token_ids) if include_token_ids else None,
            diagnostics=diagnostics,
        )

    @classmethod
    def failure(
        cls,
        *,
        case_id: str,
        request_sha256: str,
        error: OracleError,
    ) -> "OracleResult":
        return cls(
            case_id=case_id,
            status="error",
            request_sha256=request_sha256,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = (
            list(self.token_ids) if self.token_ids is not None else None
        )
        return {key: value for key, value in payload.items() if value is not None}
