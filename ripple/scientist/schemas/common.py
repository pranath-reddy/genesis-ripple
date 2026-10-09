"""Shared immutable contracts for the RIPPLe scientist control plane."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
MEDIA_TYPE_PATTERN = r"^[a-z0-9.+-]+/[a-z0-9.+-]+$"


class FrozenModel(BaseModel):
    """Strict base model used at scientific and execution boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json_sha256(value: BaseModel | dict[str, Any]) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_relative_artifact_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or ".." in candidate.parts
        or value != candidate.as_posix()
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise ValueError("artifact path must be normalized and run-relative")
    return value


class ArtifactRef(FrozenModel):
    """Content-addressed reference to one immutable run artifact."""

    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    role: str = Field(pattern=IDENTIFIER_PATTERN)
    relative_path: str = Field(min_length=1, max_length=1024)
    media_type: str = Field(pattern=MEDIA_TYPE_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=0, le=10 * 1024**4)
    producer: str = Field(pattern=IDENTIFIER_PATTERN)
    configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    input_artifact_ids: tuple[str, ...] = ()

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("input_artifact_ids")
    @classmethod
    def _unique_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("input artifact IDs must be unique")
        if any(re.fullmatch(IDENTIFIER_PATTERN, item) is None for item in value):
            raise ValueError("invalid input artifact ID")
        return value


class ComputeBudget(FrozenModel):
    """Hard upper bounds enforced by workflow code, not by the language model."""

    max_llm_requests: int = Field(default=8, ge=0, le=100)
    max_tool_calls: int = Field(default=32, ge=0, le=10_000)
    max_simulations: int = Field(default=10, ge=0, le=1_000_000)
    max_training_runs: int = Field(default=3, ge=0, le=1_000)
    max_gpu_seconds: int = Field(default=900, ge=0, le=31_536_000)
    max_storage_bytes: int = Field(default=2 * 1024**3, ge=0, le=100 * 1024**4)


class BudgetUsage(FrozenModel):
    llm_requests: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    simulations: int = Field(default=0, ge=0)
    training_runs: int = Field(default=0, ge=0)
    gpu_seconds: float = Field(default=0.0, ge=0)
    storage_bytes: int = Field(default=0, ge=0)

    def fits(self, budget: ComputeBudget) -> bool:
        return (
            self.llm_requests <= budget.max_llm_requests
            and self.tool_calls <= budget.max_tool_calls
            and self.simulations <= budget.max_simulations
            and self.training_runs <= budget.max_training_runs
            and self.gpu_seconds <= budget.max_gpu_seconds
            and self.storage_bytes <= budget.max_storage_bytes
        )


class ScientificGate(FrozenModel):
    """One explicit permission boundary with traceable reasons."""

    gate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["open", "closed", "not_applicable"]
    evidence_artifact_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _closed_gate_has_reason(self) -> "ScientificGate":
        if self.status == "closed" and not self.reasons:
            raise ValueError("a closed gate must record at least one reason")
        return self


class SourceIdentity(FrozenModel):
    name: str = Field(min_length=1, max_length=128)
    revision: str = Field(min_length=7, max_length=128)
    repository: str = Field(min_length=1, max_length=512)
    source_tree_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class SkyCoordinate(FrozenModel):
    ra_deg: float = Field(ge=0.0, lt=360.0)
    dec_deg: float = Field(ge=-90.0, le=90.0)
