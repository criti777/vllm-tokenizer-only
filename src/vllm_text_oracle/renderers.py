"""Renderer construction without model-weight loading or silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from transformers import AutoTokenizer

from vendor.vllm.extracted.hf_renderer import render_and_encode
from vendor.vllm.extracted.chat_utils import parse_chat_messages
from vendor.vllm.extracted.deepseek_v32_encoding import (
    encode_messages as encode_deepseek_v32_messages,
)
from vendor.vllm.extracted.deepseek_v4_encoding import (
    encode_messages as encode_deepseek_v4_messages,
)

from .profiles import ModelProfile


_REMOTE_CODE_LOCK = Lock()


class RendererUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    token_ids: list[int]
    diagnostics: dict[str, Any]


class HFRenderer:
    name = "hf"

    def __init__(
        self, tokenizer: Any, *, content_format_override: str | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.content_format_override = content_format_override

    @classmethod
    def from_local_assets(
        cls,
        path: Path,
        *,
        trust_remote_code: bool = False,
        content_format_override: str | None = None,
    ) -> "HFRenderer":
        if trust_remote_code:
            import transformers.dynamic_module_utils as dynamic_modules

            with _REMOTE_CODE_LOCK:
                original_cache = dynamic_modules.HF_MODULES_CACHE
                with TemporaryDirectory(prefix="vllm-oracle-hf-modules-") as cache:
                    dynamic_modules.HF_MODULES_CACHE = cache
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            path,
                            local_files_only=True,
                            trust_remote_code=True,
                        )
                    finally:
                        dynamic_modules.HF_MODULES_CACHE = original_cache
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
            )
        return cls(tokenizer, content_format_override=content_format_override)

    def render(self, parsed: Any) -> RenderedPrompt:
        _, rendered, token_ids, resolved_format = render_and_encode(
            tokenizer=self.tokenizer,
            messages=parsed.messages,
            tools=parsed.tools,
            chat_template=parsed.chat_template,
            content_format=(
                self.content_format_override or parsed.chat_template_content_format
            ),
            template_kwargs=parsed.template_kwargs(parsed.tools),
            add_special_tokens=parsed.add_special_tokens,
        )
        return RenderedPrompt(
            text=rendered,
            token_ids=token_ids,
            diagnostics={
                "chat_template_content_format": resolved_format,
                "add_generation_prompt": parsed.add_generation_prompt,
                "continue_final_message": parsed.continue_final_message,
                "add_special_tokens": parsed.add_special_tokens,
            },
        )


class DeepSeekV32Renderer:
    name = "deepseek_v32"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_local_assets(cls, path: Path) -> "DeepSeekV32Renderer":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(tokenizer)

    def render(self, parsed: Any) -> RenderedPrompt:
        conversation = parse_chat_messages(parsed.messages, "string")
        if parsed.tools:
            conversation.insert(0, {"role": "system", "content": "", "tools": parsed.tools})
        kwargs = parsed.template_kwargs(parsed.tools)
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        drop_thinking = bool(conversation and conversation[-1]["role"] == "user")
        text = encode_deepseek_v32_messages(
            conversation,
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=drop_thinking,
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return RenderedPrompt(
            text=text,
            token_ids=list(token_ids),
            diagnostics={
                "thinking_mode": "thinking" if thinking else "chat",
                "drop_thinking": drop_thinking,
            },
        )


class DeepSeekV4Renderer:
    name = "deepseek_v4"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_local_assets(cls, path: Path) -> "DeepSeekV4Renderer":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(tokenizer)

    def render(self, parsed: Any) -> RenderedPrompt:
        conversation = parse_chat_messages(parsed.messages, "string")
        if parsed.tools:
            conversation.insert(0, {"role": "system", "content": "", "tools": parsed.tools})
        kwargs = parsed.template_kwargs(parsed.tools)
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        effort = kwargs.get("reasoning_effort")
        if effort == "none":
            thinking = False
            effort = None
        elif isinstance(effort, str):
            effort = "max" if effort in {"max", "xhigh"} else "high"
        else:
            effort = None
        drop_thinking = bool(kwargs.get("drop_thinking", True))
        text = encode_deepseek_v4_messages(
            conversation,
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=drop_thinking,
            reasoning_effort=effort,
        )
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return RenderedPrompt(
            text=text,
            token_ids=list(token_ids),
            diagnostics={
                "thinking_mode": "thinking" if thinking else "chat",
                "drop_thinking": drop_thinking,
                "reasoning_effort": effort,
            },
        )


def build_renderer(
    profile: ModelProfile, asset_path: Path
) -> HFRenderer | DeepSeekV32Renderer | DeepSeekV4Renderer:
    if profile.renderer == "hf":
        return HFRenderer.from_local_assets(
            asset_path,
            trust_remote_code=profile.trust_remote_code,
            content_format_override=(
                "openai" if profile.capabilities.get("content_parts") else None
            ),
        )
    if profile.renderer == "deepseek_v32":
        return DeepSeekV32Renderer.from_local_assets(asset_path)
    if profile.renderer == "deepseek_v4":
        return DeepSeekV4Renderer.from_local_assets(asset_path)
    raise RendererUnavailableError(f"unknown renderer {profile.renderer}")
