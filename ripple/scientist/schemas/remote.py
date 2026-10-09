"""Non-secret contracts for an offline scientific worker."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import FrozenModel, SHA256_PATTERN


class RemoteWorkerSettings(FrozenModel):
    host: str = Field(pattern=r"^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$")
    python: str = Field(pattern=r"^/[A-Za-z0-9._/-]+$")
    remote_root: str = Field(pattern=r"^/[A-Za-z0-9._/-]+$")
    connect_timeout_seconds: int = Field(default=10, ge=1, le=60)

    @field_validator("python")
    @classmethod
    def _safe_python(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.as_posix() != value or ".." in path.parts or value == "/":
            raise ValueError("worker Python must be a normalized absolute path")
        return value

    @field_validator("remote_root")
    @classmethod
    def _safe_remote_root(cls, value: str) -> str:
        normalized = value.rstrip("/")
        path = PurePosixPath(normalized)
        broad_roots = {
            "/",
            "/home",
            "/opt",
            "/root",
            "/srv",
            "/tmp",
            "/usr",
            "/var",
        }
        if (
            path.as_posix() != normalized
            or ".." in path.parts
            or normalized in broad_roots
        ):
            raise ValueError("remote root must be a dedicated child directory")
        return normalized


class RemoteCommandResult(FrozenModel):
    operation: Literal["environment", "simulate", "build-dataset", "train", "evaluate"]
    exit_code: int
    stdout: str = Field(max_length=50_000)
    stderr: str = Field(max_length=50_000)
    stdout_sha256: str = Field(pattern=SHA256_PATTERN)
    elapsed_seconds: float = Field(ge=0.0)
    expected_source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    verified_source_manifest_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    source_verification_succeeded: bool
    succeeded: bool

    @model_validator(mode="after")
    def _bind_execution_to_source_verification(self) -> "RemoteCommandResult":
        if self.source_verification_succeeded:
            if (
                self.verified_source_manifest_sha256
                != self.expected_source_manifest_sha256
            ):
                raise ValueError("verified source hash must equal the expected hash")
        elif self.verified_source_manifest_sha256 is not None:
            raise ValueError(
                "failed source verification cannot publish a verified hash"
            )
        expected_success = self.exit_code == 0 and self.source_verification_succeeded
        if self.succeeded != expected_success:
            raise ValueError(
                "remote success must include successful source verification"
            )
        return self
