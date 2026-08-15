"""CPU-only vLLM-compatible text input oracle."""

from .contracts import OracleError, OracleResult
from .oracle import TextOracle

__all__ = ["OracleError", "OracleResult", "TextOracle"]
