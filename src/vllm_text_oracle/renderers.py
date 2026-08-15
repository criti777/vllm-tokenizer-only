"""Renderer construction without model-weight loading or silent fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from transformers import AutoTokenizer

from vendor.vllm.extracted.hf_renderer import render_and_encode

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

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_local_assets(
        cls, path: Path, *, trust_remote_code: bool = False
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
        return HFRenderer.from_local_assets(
            asset_path,
            trust_remote_code=profile.trust_remote_code,
        )
    raise RendererUnavailableError(
        f"renderer {profile.renderer} is not installed for {profile.profile_id}"
    )
