"""Fetch only pinned tokenizer/template/config assets for selected profiles."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from tools.fetch_common import fetch_checked
from vllm_text_oracle.assets import verify_asset_directory
from vllm_text_oracle.model_selection import parse_model_selection
from vllm_text_oracle.profiles import ModelProfile, ModelRegistry


PROJECT_ROOT = Path(__file__).parents[1]
PROFILE_FILE = PROJECT_ROOT / "models" / "profiles.json"
SOURCE_FILES = PROJECT_ROOT / "models" / "source_files.json"


def asset_url(profile: ModelProfile, relative_name: str) -> str:
    encoded_name = quote(relative_name, safe="/")
    return (
        f"https://huggingface.co/{profile.repository}/resolve/"
        f"{profile.revision}/{encoded_name}?download=true"
    )


def validate_asset_names(names: list[str]) -> None:
    for name in names:
        lowered = name.lower()
        if lowered.endswith((".safetensors", ".bin", ".pt", ".pth")):
            raise ValueError(f"weight file is forbidden: {name}")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe asset path: {name}")


def _write_json_atomic(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def fetch_profile(
    profile: ModelProfile,
    names: list[str],
    *,
    assets_root: Path,
    manifest_path: Path,
) -> Path:
    validate_asset_names(names)
    destination = (
        assets_root / profile.repository.replace("/", "--") / profile.revision
    )
    files: dict[str, dict[str, int | str]] = {}
    for name in names:
        target = destination / name
        digest = fetch_checked(asset_url(profile, name), target)
        files[name] = {"sha256": digest, "size": target.stat().st_size}
    _write_json_atomic(
        {
            "schema_version": 1,
            "profile_id": profile.profile_id,
            "repository": profile.repository,
            "revision": profile.revision,
            "files": files,
        },
        manifest_path,
    )
    verify_asset_directory(manifest_path, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="PROFILE")
    parser.add_argument(
        "--assets-root", type=Path, default=PROJECT_ROOT / "model_assets"
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    registry = ModelRegistry.from_file(PROFILE_FILE)
    profiles = parse_model_selection(args.model, registry)
    source_files = json.loads(SOURCE_FILES.read_text(encoding="utf-8"))
    for profile in profiles:
        manifest_path = PROJECT_ROOT / profile.asset_manifest
        asset_path = (
            args.assets_root
            / profile.repository.replace("/", "--")
            / profile.revision
        )
        if args.verify_only:
            verify_asset_directory(manifest_path, asset_path)
        else:
            fetch_profile(
                profile,
                source_files[profile.profile_id],
                assets_root=args.assets_root,
                manifest_path=manifest_path,
            )
        print(f"verified {profile.profile_id}: {asset_path}")


if __name__ == "__main__":
    main()
