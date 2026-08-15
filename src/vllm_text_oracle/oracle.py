"""Stable public API for the extracted vLLM text preprocessing path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from transformers import AutoTokenizer

from vendor.vllm.extracted.chat_utils import UnsupportedMultimodalError
from vendor.vllm.extracted.protocol import ChatCompletionRequest

from .assets import verify_asset_directory
from .contracts import OracleError, OracleResult
from .errors import ProfileResolutionError
from .hashing import canonical_json_sha256
from .profiles import ModelProfile, ModelRegistry
from .renderers import HFRenderer, build_renderer


PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "models" / "profiles.json"


class TextOracle:
    def __init__(
        self,
        tokenizer: Any,
        *,
        profile: ModelProfile | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.profile = profile
        self.registry = registry
        self._renderer = HFRenderer(tokenizer)

    @classmethod
    def from_local_assets(cls, path: Path) -> "TextOracle":
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(tokenizer)

    @classmethod
    def from_model(
        cls,
        model_alias: str,
        *,
        assets_root: Path,
        profile_file: Path = DEFAULT_PROFILES,
    ) -> "TextOracle":
        registry = ModelRegistry.from_file(profile_file)
        profile = registry.resolve(model_alias)
        asset_path = (
            assets_root / profile.repository.replace("/", "--") / profile.revision
        )
        manifest_path = PROJECT_ROOT / profile.asset_manifest
        verify_asset_directory(manifest_path, asset_path)
        renderer = build_renderer(profile, asset_path)
        oracle = cls(renderer.tokenizer, profile=profile, registry=registry)
        oracle._renderer = renderer
        return oracle

    def process(
        self,
        request: Mapping[str, Any],
        *,
        case_id: str,
        include_token_ids: bool = False,
    ) -> OracleResult:
        request_dict = dict(request)
        request_hash = canonical_json_sha256(request_dict)
        profile_id = self.profile.profile_id if self.profile else None
        renderer_name = self.profile.renderer if self.profile else None
        request_model = request_dict.get("model")
        if self.profile is not None and isinstance(request_model, str):
            try:
                requested_profile = self.registry.resolve(request_model)  # type: ignore[union-attr]
            except ProfileResolutionError as error:
                return OracleResult.failure(
                    case_id=case_id,
                    request_sha256=request_hash,
                    model_profile=profile_id,
                    renderer=renderer_name,
                    error=OracleError(
                        stage="profile_resolution",
                        type="unknown_request_model",
                        message=str(error),
                    ),
                )
            if requested_profile.profile_id != self.profile.profile_id:
                return OracleResult.failure(
                    case_id=case_id,
                    request_sha256=request_hash,
                    model_profile=profile_id,
                    renderer=renderer_name,
                    error=OracleError(
                        stage="profile_resolution",
                        type="model_profile_mismatch",
                        message=(
                            f"request model {request_model} resolves to "
                            f"{requested_profile.profile_id}, not {profile_id}"
                        ),
                    ),
                )
        try:
            parsed = ChatCompletionRequest.model_validate(request_dict)
        except ValidationError as error:
            return OracleResult.failure(
                case_id=case_id,
                request_sha256=request_hash,
                model_profile=profile_id,
                renderer=renderer_name,
                error=OracleError(
                    stage="request_validation",
                    type="validation_error",
                    message=str(error),
                ),
            )
        tools = parsed.tools
        try:
            rendered = self._renderer.render(parsed)
        except UnsupportedMultimodalError as error:
            return OracleResult.failure(
                case_id=case_id,
                request_sha256=request_hash,
                model_profile=profile_id,
                renderer=renderer_name,
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
                model_profile=profile_id,
                renderer=renderer_name,
                error=OracleError(
                    stage="render_or_encode",
                    type=error.__class__.__name__,
                    message=str(error),
                ),
            )
        return OracleResult.success(
            case_id=case_id,
            request_sha256=request_hash,
            rendered_text=rendered.text,
            token_ids=rendered.token_ids,
            include_token_ids=include_token_ids,
            diagnostics=rendered.diagnostics,
            model_profile=profile_id,
            renderer=renderer_name,
        )
