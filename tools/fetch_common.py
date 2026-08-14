"""Checked, atomic HTTP downloads."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


class DownloadError(RuntimeError):
    pass


def fetch_checked(
    url: str,
    destination: Path,
    expected_sha256: str | None = None,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        try:
            with urllib.request.urlopen(url) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise DownloadError(f"HTTP {status} for {url}")
                payload = response.read()
        except urllib.error.HTTPError as error:
            raise DownloadError(f"HTTP {error.code} for {url}") from error
        except urllib.error.URLError as error:
            raise DownloadError(f"request failed for {url}: {error.reason}") from error

        if not payload:
            raise DownloadError(f"empty response for {url}")

        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise DownloadError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )

        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
        return actual_sha256
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
