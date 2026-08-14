"""Stable public API for the extracted vLLM text preprocessing path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from transformers import AutoTokenizer

from vendor.vllm.extracted.chat_utils import UnsupportedMultimodalError
from vendor.vllm.extracted.hf_renderer import render_and_encode
from vendor.vllm.extracted.protocol import ChatCompletionRequest

from .contracts import OracleError, OracleResult
from .hashing import canonical_json_sha256


class TextOracle:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_local_assets(cls, path: Path) -> "TextOracle":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(tokenizer)

    def process(
        self,
        request: Mapping[str, Any],
        *,
        case_id: str,
        include_token_ids: bool = False,
    ) -> OracleResult:
        request_dict = dict(request)
        request_hash = canonical_json_sha256(request_dict)
        try:
            parsed = ChatCompletionRequest.model_validate(request_dict)
        except ValidationError as error:
            return OracleResult.failure(
                case_id=case_id,
                request_sha256=request_hash,
                error=OracleError(
                    stage="request_validation",
                    type="validation_error",
                    message=str(error),
                ),
            )
        tools = parsed.tools
        try:
            _, rendered, token_ids, resolved_format = render_and_encode(
                tokenizer=self.tokenizer,
                messages=parsed.messages,
                tools=tools,
                chat_template=parsed.chat_template,
                content_format=parsed.chat_template_content_format,
                template_kwargs=parsed.template_kwargs(tools),
                add_special_tokens=parsed.add_special_tokens,
            )
        except UnsupportedMultimodalError as error:
            return OracleResult.failure(
                case_id=case_id,
                request_sha256=request_hash,
                error=OracleError(
                    stage="message_normalization",
                    type="unsupported_multimodal",
                    message=str(error),
                ),
            )
        except Exception as error:
            return OracleResult.failure(
                case_id=case_id,
                request_sha256=request_hash,
                error=OracleError(
                    stage="render_or_encode",
                    type=error.__class__.__name__,
                    message=str(error),
                ),
            )
        return OracleResult.success(
            case_id=case_id,
            request_sha256=request_hash,
            rendered_text=rendered,
            token_ids=token_ids,
            include_token_ids=include_token_ids,
            diagnostics={
                "chat_template_content_format": resolved_format,
                "add_generation_prompt": parsed.add_generation_prompt,
                "continue_final_message": parsed.continue_final_message,
                "add_special_tokens": parsed.add_special_tokens,
            },
        )
