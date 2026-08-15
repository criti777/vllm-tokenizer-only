"""Immutable model profiles and strict alias resolution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .errors import ProfileResolutionError


RendererName = Literal["hf", "deepseek_v32", "deepseek_v4"]


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    repository: str = Field(pattern=r"^[^/]+/[^/]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    renderer: RendererName
    trust_remote_code: bool = False
    asset_manifest: str = Field(min_length=1)
    capabilities: dict[str, bool]


class ModelRegistry:
    def __init__(self, profiles: Iterable[ModelProfile]) -> None:
        ordered = tuple(profiles)
        by_alias: dict[str, ModelProfile] = {}
        for profile in ordered:
            for alias in (profile.profile_id, profile.repository, *profile.aliases):
                if alias in by_alias:
                    raise ValueError(f"duplicate model alias: {alias}")
                by_alias[alias] = profile
        self._profiles = ordered
        self._by_alias = by_alias

    @classmethod
    def from_file(cls, path: Path) -> "ModelRegistry":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(ModelProfile.model_validate(item) for item in payload)

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.profile_id for profile in self._profiles)

    def resolve(self, alias: str) -> ModelProfile:
        try:
            return self._by_alias[alias]
        except KeyError as error:
            raise ProfileResolutionError(f"unknown model alias: {alias}") from error
