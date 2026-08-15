import json
from pathlib import Path

import pytest

from tools.build_result_manifest import build_aggregate_manifest
from vllm_text_oracle.profiles import ModelRegistry


def test_aggregate_manifest_requires_every_profile_and_consistent_limits(
    tmp_path: Path,
) -> None:
    registry = ModelRegistry.from_file(Path("models/profiles.json"))
    for profile_id in registry.profile_ids:
        directory = tmp_path / profile_id
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "profile_id": profile_id,
                    "selection": {"ultrachat_limit": 10},
                    "counts": {
                        "handwritten": {"total": 1, "ok": 1, "error": 0},
                        "combinatorial": {"total": 2, "ok": 1, "error": 1},
                        "ultrachat": {"total": 10, "ok": 10, "error": 0},
                    },
                }
            ),
            encoding="utf-8",
        )

    aggregate = build_aggregate_manifest(tmp_path, registry)

    assert aggregate["profile_count"] == 7
    assert aggregate["selection"] == {"ultrachat_limit": 10}
    assert aggregate["counts"] == {"total": 91, "ok": 84, "error": 7}

    (tmp_path / registry.profile_ids[0] / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing profile manifest"):
        build_aggregate_manifest(tmp_path, registry)
