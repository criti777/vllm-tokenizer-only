"""Renderer construction without model-weight loading or silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from vendor.vllm.extracted.hf_renderer import render_and_encode

from .profiles import ModelProfile


class RendererUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    token_ids: list[int]
    diagnostics: dict[str, Any]


class HFRenderer:
    name = "hf"

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_local_assets(cls, path: Path) -> "HFRenderer":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(tokenizer)

    def render(self, parsed: Any) -> RenderedPrompt:
        _, rendered, token_ids, resolved_format = render_and_encode(
            tokenizer=self.tokenizer,
            messages=parsed.messages,
            tools=parsed.tools,
            chat_template=parsed.chat_template,
            content_format=parsed.chat_template_content_format,
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


def build_renderer(profile: ModelProfile, asset_path: Path) -> HFRenderer:
    if profile.renderer == "hf":
        return HFRenderer.from_local_assets(asset_path)
    raise RendererUnavailableError(
        f"renderer {profile.renderer} is not installed for {profile.profile_id}"
    )
