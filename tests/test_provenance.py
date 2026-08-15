import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools.fetch_common import DownloadError, fetch_checked
from vllm_text_oracle.profiles import ModelRegistry


UPSTREAM_MANIFEST = Path("vendor/vllm/upstream-files.json")
UPSTREAM_ROOT = Path("vendor/vllm/upstream")
COVERAGE_MANIFEST = Path("vendor/vllm/coverage.json")
PROFILE_MANIFEST = Path("models/profiles.json")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        status, payload = {
            "/ok": (200, b"asset"),
            "/empty": (200, b""),
            "/missing": (404, b"missing"),
        }.get(self.path, (500, b"unexpected"))
        self.send_response(status)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def server_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def test_fetch_checked_writes_verified_nonempty_file(
    tmp_path: Path, server_url: str
) -> None:
    destination = tmp_path / "asset.bin"
    expected = hashlib.sha256(b"asset").hexdigest()

    actual = fetch_checked(f"{server_url}/ok", destination, expected)

    assert actual == expected
    assert destination.read_bytes() == b"asset"


@pytest.mark.parametrize("path", ["missing", "empty"])
def test_fetch_checked_rejects_bad_responses(
    tmp_path: Path, server_url: str, path: str
) -> None:
    with pytest.raises(DownloadError):
        fetch_checked(f"{server_url}/{path}", tmp_path / "asset.bin")


def test_fetch_checked_rejects_hash_mismatch(
    tmp_path: Path, server_url: str
) -> None:
    with pytest.raises(DownloadError, match="SHA-256"):
        fetch_checked(f"{server_url}/ok", tmp_path / "asset.bin", "0" * 64)

    assert not (tmp_path / "asset.bin").exists()


def test_specialized_deepseek_sources_are_pinned_and_byte_exact() -> None:
    manifest = json.loads(UPSTREAM_MANIFEST.read_text(encoding="utf-8"))
    entries = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    expected = {
        "vllm/tokenizers/deepseek_v32.py",
        "vllm/tokenizers/deepseek_v32_encoding.py",
        "vllm/renderers/deepseek_v32.py",
        "vllm/tokenizers/deepseek_v4.py",
        "vllm/tokenizers/deepseek_v4_encoding.py",
        "vllm/renderers/deepseek_v4.py",
    }

    assert expected <= entries.keys()
    for upstream_path in expected:
        local_path = UPSTREAM_ROOT / upstream_path.removeprefix("vllm/")
        assert local_path.is_file()
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == entries[
            upstream_path
        ]


def test_deepseek_encoding_extractions_are_unmodified() -> None:
    for name in ("deepseek_v32_encoding.py", "deepseek_v4_encoding.py"):
        upstream = UPSTREAM_ROOT / "tokenizers" / name
        extracted = Path("vendor/vllm/extracted") / name

        assert extracted.read_bytes() == upstream.read_bytes()


def test_every_profile_has_source_extraction_and_behavior_coverage() -> None:
    registry = ModelRegistry.from_file(PROFILE_MANIFEST)
    coverage = json.loads(COVERAGE_MANIFEST.read_text(encoding="utf-8"))

    assert coverage["vllm_commit"] == (
        "568afb3a13806beb53bb2e6bd518269357b237c0"
    )
    assert tuple(coverage["profiles"]) == registry.profile_ids
    for profile_id, entry in coverage["profiles"].items():
        profile = registry.resolve(profile_id)
        assert entry["renderer"] == profile.renderer
        assert entry["upstream"]
        assert entry["extracted"]
        assert entry["tests"]
        for relative_path in entry["upstream"]:
            assert (UPSTREAM_ROOT / relative_path).is_file(), relative_path
        for relative_path in entry["extracted"] + entry["tests"]:
            assert Path(relative_path).is_file(), relative_path


def test_specialized_profiles_cannot_claim_generic_hf_fallback() -> None:
    coverage = json.loads(COVERAGE_MANIFEST.read_text(encoding="utf-8"))

    for profile_id in ("deepseek-v3.2", "deepseek-v4"):
        entry = coverage["profiles"][profile_id]
        assert entry["renderer"].startswith("deepseek_")
        assert "vendor/vllm/extracted/hf_renderer.py" not in entry["extracted"]
