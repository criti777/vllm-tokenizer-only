"""Offline model text-asset integrity verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AssetIntegrityError(RuntimeError):
    pass


class AssetFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=1)


class AssetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    profile_id: str
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: dict[str, AssetFile]


@dataclass(frozen=True)
class VerifiedAssets:
    profile_id: str
    path: Path
    files: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset_directory(manifest_path: Path, asset_path: Path) -> VerifiedAssets:
    manifest = AssetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    tracked = set(manifest.files)
    for relative_name, expected in manifest.files.items():
        path = asset_path / relative_name
        if not path.is_file():
            raise AssetIntegrityError(f"missing asset: {relative_name}")
        actual_size = path.stat().st_size
        if actual_size != expected.size:
            raise AssetIntegrityError(
                f"size mismatch for {relative_name}: "
                f"expected {expected.size}, got {actual_size}"
            )
        actual_hash = _file_sha256(path)
        if actual_hash != expected.sha256:
            raise AssetIntegrityError(
                f"SHA-256 mismatch for {relative_name}: "
                f"expected {expected.sha256}, got {actual_hash}"
            )

    for path in asset_path.rglob("*.py"):
        relative_name = path.relative_to(asset_path).as_posix()
        if relative_name.startswith(".cache/"):
            continue
        if relative_name not in tracked:
            raise AssetIntegrityError(f"untracked Python asset: {relative_name}")

    return VerifiedAssets(
        profile_id=manifest.profile_id,
        path=asset_path,
        files=tuple(sorted(tracked)),
    )
