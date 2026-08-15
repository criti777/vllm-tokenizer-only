"""Shared model selection semantics for tests and data tools."""

from __future__ import annotations

from collections.abc import Sequence

from .profiles import ModelProfile, ModelRegistry


def parse_model_selection(
    values: Sequence[str] | None,
    registry: ModelRegistry,
) -> tuple[ModelProfile, ...]:
    if not values:
        return ()
    if "all" in values:
        if len(values) != 1:
            raise ValueError("cannot combine 'all' with individual model profiles")
        return tuple(registry.resolve(profile_id) for profile_id in registry.profile_ids)

    selected: list[ModelProfile] = []
    seen: set[str] = set()
    for value in values:
        profile = registry.resolve(value)
        if profile.profile_id not in seen:
            selected.append(profile)
            seen.add(profile.profile_id)
    return tuple(selected)
