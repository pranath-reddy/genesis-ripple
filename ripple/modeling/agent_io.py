"""Bounded JSON persistence for model-onboarding requests and agent outcomes."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .agent_contracts import ModelOnboardingAgentOutcome
from .onboarding import ModelOnboardingRequest, SourceInventory


_ModelT = TypeVar("_ModelT", bound=BaseModel)
_MAX_REQUEST_BYTES = 1024 * 1024
_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_OUTCOME_BYTES = 32 * 1024 * 1024


class AgentArtifactIOError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def load_onboarding_request(path: Path) -> ModelOnboardingRequest:
    return _load_json_model(
        path,
        model_type=ModelOnboardingRequest,
        maximum_bytes=_MAX_REQUEST_BYTES,
        artifact_name="onboarding request",
    )


def load_source_inventory(path: Path) -> SourceInventory:
    return _load_json_model(
        path,
        model_type=SourceInventory,
        maximum_bytes=_MAX_INVENTORY_BYTES,
        artifact_name="source inventory",
    )


def load_agent_outcome(path: Path) -> ModelOnboardingAgentOutcome:
    return _load_json_model(
        path,
        model_type=ModelOnboardingAgentOutcome,
        maximum_bytes=_MAX_OUTCOME_BYTES,
        artifact_name="onboarding agent outcome",
    )


def write_onboarding_request(path: Path, request: ModelOnboardingRequest) -> None:
    _write_json_model(
        path,
        request,
        maximum_bytes=_MAX_REQUEST_BYTES,
        artifact_name="onboarding request",
    )


def write_source_inventory(path: Path, inventory: SourceInventory) -> None:
    _write_json_model(
        path,
        inventory,
        maximum_bytes=_MAX_INVENTORY_BYTES,
        artifact_name="source inventory",
    )


def write_agent_outcome(path: Path, outcome: ModelOnboardingAgentOutcome) -> None:
    _write_json_model(
        path,
        outcome,
        maximum_bytes=_MAX_OUTCOME_BYTES,
        artifact_name="onboarding agent outcome",
    )


def _load_json_model(
    path: Path,
    *,
    model_type: type[_ModelT],
    maximum_bytes: int,
    artifact_name: str,
) -> _ModelT:
    encoded = _read_bounded_regular_file(Path(path), maximum_bytes=maximum_bytes)
    try:
        return model_type.model_validate_json(encoded)
    except (ValueError, ValidationError) as exc:
        raise AgentArtifactIOError(
            code="invalid_agent_artifact",
            message=f"The {artifact_name} is invalid ({type(exc).__name__}).",
        ) from None


def _write_json_model(
    path: Path,
    model: BaseModel,
    *,
    maximum_bytes: int,
    artifact_name: str,
) -> None:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise AgentArtifactIOError(
            code="invalid_agent_artifact_filename",
            message=f"The {artifact_name} filename must end in .json.",
        )
    encoded = (
        model.model_dump_json(indent=2, exclude_none=False).encode("utf-8") + b"\n"
    )
    if len(encoded) > maximum_bytes:
        raise AgentArtifactIOError(
            code="agent_artifact_too_large",
            message=f"The {artifact_name} exceeds its fixed byte bound.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise AgentArtifactIOError(
            code="agent_artifact_exists",
            message=f"The {artifact_name} is immutable and cannot overwrite a path.",
        )

    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path, follow_symlinks=False)
        path.chmod(0o600)
        temporary.unlink()
        temporary = None
    except FileExistsError:
        raise AgentArtifactIOError(
            code="agent_artifact_exists",
            message=f"The {artifact_name} is immutable and cannot overwrite a path.",
        ) from None
    except AgentArtifactIOError:
        raise
    except OSError as exc:
        raise AgentArtifactIOError(
            code="agent_artifact_write_failed",
            message=f"The {artifact_name} could not be published ({type(exc).__name__}).",
        ) from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    reloaded = _load_json_model(
        path,
        model_type=type(model),
        maximum_bytes=maximum_bytes,
        artifact_name=artifact_name,
    )
    if reloaded != model:
        raise AgentArtifactIOError(
            code="agent_artifact_round_trip_mismatch",
            message=f"The {artifact_name} changed during its strict JSON round trip.",
        )


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        initial = os.lstat(path)
    except FileNotFoundError:
        raise AgentArtifactIOError(
            code="agent_artifact_not_found",
            message="The requested agent artifact does not exist.",
        ) from None
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise AgentArtifactIOError(
            code="invalid_agent_artifact_file",
            message="An agent artifact must be a regular non-symlink file.",
        )
    if initial.st_size <= 0 or initial.st_size > maximum_bytes:
        raise AgentArtifactIOError(
            code="agent_artifact_size_out_of_bounds",
            message="An agent artifact is outside its fixed byte bound.",
        )

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_state(initial, opened):
            raise AgentArtifactIOError(
                code="agent_artifact_changed",
                message="An agent artifact changed before it could be read.",
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise AgentArtifactIOError(
                    code="agent_artifact_size_out_of_bounds",
                    message="An agent artifact exceeded its byte bound while reading.",
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if not _same_file_state(opened, final) or byte_count != final.st_size:
            raise AgentArtifactIOError(
                code="agent_artifact_changed",
                message="An agent artifact changed while it was being read.",
            )
        return b"".join(chunks)
    except AgentArtifactIOError:
        raise
    except OSError as exc:
        raise AgentArtifactIOError(
            code="agent_artifact_read_failed",
            message=f"An agent artifact could not be read ({type(exc).__name__}).",
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
    )


__all__ = [
    "AgentArtifactIOError",
    "load_agent_outcome",
    "load_onboarding_request",
    "load_source_inventory",
    "write_agent_outcome",
    "write_onboarding_request",
    "write_source_inventory",
]
