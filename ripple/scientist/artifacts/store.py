"""Atomic, run-scoped artifact storage with digest verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..paths import (
    UnsafePathError,
    checked_absolute_path,
    checked_real_directory,
    reject_symlink_components,
)
from ..schemas.common import (
    ArtifactRef,
    IDENTIFIER_PATTERN,
    canonical_json_sha256,
    validate_relative_artifact_path,
)


class ArtifactStoreError(RuntimeError):
    pass


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    """A directory boundary that never writes outside a single run root."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        try:
            self.root = checked_real_directory(root, create=create)
        except UnsafePathError:
            raise ArtifactStoreError("artifact root must be a real directory")

    def resolve(self, relative_path: str) -> Path:
        normalized = validate_relative_artifact_path(relative_path)
        candidate = (self.root / normalized).absolute()
        if candidate != self.root and self.root not in candidate.parents:
            raise ArtifactStoreError("artifact path escaped the run root")
        try:
            reject_symlink_components(candidate)
        except UnsafePathError:
            raise ArtifactStoreError(
                "symlink artifact paths are not permitted"
            ) from None
        return candidate

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            reject_symlink_components(checked_absolute_path(path))
        except UnsafePathError:
            raise ArtifactStoreError("symlink artifact paths are not permitted")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path, follow_symlinks=False)
            path.chmod(0o444)
        except FileExistsError:
            raise ArtifactStoreError("immutable artifact already exists") from None
        except OSError as exc:
            raise ArtifactStoreError(
                f"immutable artifact publication failed ({type(exc).__name__})"
            ) from None
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def json_bytes(value: BaseModel | dict[str, Any]) -> bytes:
        """Return the exact bytes used for an immutable JSON artifact."""

        payload = (
            value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        )
        return (
            json.dumps(
                payload,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

    def write_json(self, relative_path: str, value: BaseModel | dict[str, Any]) -> Path:
        encoded = self.json_bytes(value)
        path = self.resolve(relative_path)
        if os.path.lexists(path):
            raise ArtifactStoreError(
                f"immutable artifact already exists: {relative_path}"
            )
        self._atomic_bytes(path, encoded)
        return path

    def write_bytes(self, relative_path: str, payload: bytes) -> Path:
        path = self.resolve(relative_path)
        if os.path.lexists(path):
            raise ArtifactStoreError(
                f"immutable artifact already exists: {relative_path}"
            )
        self._atomic_bytes(path, payload)
        return path

    def reference(
        self,
        *,
        artifact_id: str,
        role: str,
        relative_path: str,
        media_type: str,
        producer: str,
        configuration: BaseModel | dict[str, Any],
        input_artifact_ids: tuple[str, ...] = (),
    ) -> ArtifactRef:
        if re.fullmatch(IDENTIFIER_PATTERN, artifact_id) is None:
            raise ArtifactStoreError("invalid artifact ID")
        path = self.resolve(relative_path)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError("referenced artifact is missing or unsafe")
        return ArtifactRef(
            artifact_id=artifact_id,
            role=role,
            relative_path=relative_path,
            media_type=media_type,
            sha256=sha256_file(path),
            byte_count=path.stat().st_size,
            producer=producer,
            configuration_sha256=canonical_json_sha256(configuration),
            input_artifact_ids=input_artifact_ids,
        )

    def verify(self, reference: ArtifactRef) -> Path:
        path = self.resolve(reference.relative_path)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError(f"artifact is missing: {reference.artifact_id}")
        if path.stat().st_size != reference.byte_count:
            raise ArtifactStoreError(
                f"artifact byte count changed: {reference.artifact_id}"
            )
        if sha256_file(path) != reference.sha256:
            raise ArtifactStoreError(
                f"artifact digest changed: {reference.artifact_id}"
            )
        return path
