"""CPU-only vLLM-compatible text input oracle."""

from .contracts import OracleError, OracleResult
from .errors import OracleStage, ProfileResolutionError
from .oracle import TextOracle
from .profiles import ModelProfile, ModelRegistry

__all__ = [
    "ModelProfile",
    "ModelRegistry",
    "OracleError",
    "OracleResult",
    "OracleStage",
    "ProfileResolutionError",
    "TextOracle",
]
