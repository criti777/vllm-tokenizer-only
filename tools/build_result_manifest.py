"""Build a deterministic aggregate manifest for per-profile baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from vllm_text_oracle.profiles import ModelRegistry


PROFILE_FILE = Path("models/profiles.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_aggregate_manifest(
    result_root: Path, registry: ModelRegistry
) -> dict[str, object]:
    profiles: dict[str, object] = {}
    grand_total = grand_ok = grand_error = 0
    limits: set[int | None] = set()
    for profile_id in registry.profile_ids:
        path = result_root / profile_id / "manifest.json"
        if not path.is_file():
            raise ValueError(f"missing profile manifest: {profile_id}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("profile_id") != profile_id:
            raise ValueError(f"profile manifest mismatch: {profile_id}")
        counts = manifest.get("counts", {})
        total = sum(value["total"] for value in counts.values())
        ok = sum(value["ok"] for value in counts.values())
        error = sum(value["error"] for value in counts.values())
        limit = manifest.get("selection", {}).get("ultrachat_limit")
        limits.add(limit)
        profiles[profile_id] = {
            "manifest": f"{profile_id}/manifest.json",
            "manifest_sha256": _sha256(path),
            "total": total,
            "ok": ok,
            "error": error,
        }
        grand_total += total
        grand_ok += ok
        grand_error += error
    if len(limits) != 1:
        raise ValueError("profile baselines use inconsistent UltraChat limits")
    return {
        "schema_version": 1,
        "profile_count": len(profiles),
        "selection": {"ultrachat_limit": next(iter(limits))},
        "counts": {
            "total": grand_total,
            "ok": grand_ok,
            "error": grand_error,
        },
        "profiles": profiles,
    }


def write_aggregate_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".manifest.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", type=Path, default=Path("datasets/results/by-profile")
    )
    args = parser.parse_args()
    payload = build_aggregate_manifest(
        args.results, ModelRegistry.from_file(PROFILE_FILE)
    )
    write_aggregate_manifest(args.results / "manifest.json", payload)
    print(json.dumps(payload["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
