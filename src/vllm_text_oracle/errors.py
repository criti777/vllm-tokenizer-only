"""Stable public error stages and typed profile failures."""

from enum import StrEnum


class OracleStage(StrEnum):
    PROFILE_RESOLUTION = "profile_resolution"
    REQUEST_VALIDATION = "request_validation"
    MESSAGE_NORMALIZATION = "message_normalization"
    TEMPLATE_RENDER = "template_render"
    ENCODE = "encode"
    PROCESSOR_REQUIRED = "processor_required"
    ASSET_INTEGRITY = "asset_integrity"


class ProfileResolutionError(ValueError):
    """Raised when a request model cannot be resolved without ambiguity."""
