"""Typed, safe failures for deterministic M3 preprocessing."""

from __future__ import annotations


class PreprocessingError(RuntimeError):
    """A failure whose public fields are safe to serialize in local evidence."""

    def __init__(self, *, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.safe_message = message


class PreprocessingInputError(PreprocessingError):
    """The verified M2 product is incompatible with the selected M3 recipe."""


class PreprocessingArtifactError(PreprocessingError):
    """An M3 artifact could not be safely published or reloaded."""
