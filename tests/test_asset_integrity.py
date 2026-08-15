import hashlib
import json
from pathlib import Path

import pytest

from tools.fetch_model_assets import asset_url, validate_asset_names
from vllm_text_oracle.assets import AssetIntegrityError, verify_asset_directory
from vllm_text_oracle.profiles import ModelProfile


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: Path, files: dict[str, bytes]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "fixture",
                "repository": "owner/model",
                "revision": "a" * 40,
                "files": {
                    name: {"sha256": sha256(payload), "size": len(payload)}
                    for name, payload in files.items()
                },
            }
        ),
        encoding="utf-8",
    )


def test_verifies_complete_offline_asset_directory(tmp_path: Path) -> None:
    files = {"tokenizer.json": b"tokenizer", "config.json": b"{}"}
    assets = tmp_path / "assets"
    assets.mkdir()
    for name, payload in files.items():
        (assets / name).write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, files)

    verified = verify_asset_directory(manifest, assets)

    assert verified.profile_id == "fixture"
    assert verified.path == assets
    assert verified.files == ("config.json", "tokenizer.json")


def test_missing_asset_is_a_hard_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, {"tokenizer.json": b"expected"})
    assets = tmp_path / "assets"
    assets.mkdir()

    with pytest.raises(AssetIntegrityError, match="missing asset"):
        verify_asset_directory(manifest, assets)


def test_hash_mismatch_is_a_hard_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, {"tokenizer.json": b"expected"})
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_bytes(b"tampered")

    with pytest.raises(AssetIntegrityError, match="SHA-256 mismatch"):
        verify_asset_directory(manifest, assets)


def test_untracked_python_cannot_be_executed_implicitly(tmp_path: Path) -> None:
    files = {"tokenizer.json": b"expected"}
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, files)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_bytes(files["tokenizer.json"])
    (assets / "tokenization_remote.py").write_text("raise SystemExit", encoding="utf-8")

    with pytest.raises(AssetIntegrityError, match="untracked Python"):
        verify_asset_directory(manifest, assets)


def test_size_mismatch_is_reported_before_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, {"tokenizer.json": b"expected"})
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_bytes(b"x")

    with pytest.raises(AssetIntegrityError, match="size mismatch"):
        verify_asset_directory(manifest, assets)


def test_asset_url_is_bound_to_immutable_profile_revision() -> None:
    profile = ModelProfile(
        profile_id="fixture",
        aliases=(),
        repository="owner/model",
        revision="a" * 40,
        renderer="hf",
        asset_manifest="models/manifests/fixture.json",
        capabilities={},
    )

    assert asset_url(profile, "tokenizer.json") == (
        "https://huggingface.co/owner/model/resolve/"
        f"{'a' * 40}/tokenizer.json?download=true"
    )


@pytest.mark.parametrize(
    "name",
    ["model.safetensors", "model-00001-of-00002.safetensors", "pytorch_model.bin"],
)
def test_weight_files_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="weight file"):
        validate_asset_names([name])
