"""Bounded serialization helpers for versioned model manifests."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .contracts import ModelManifest, ModelManifestRef


_MAX_MANIFEST_BYTES = 2 * 1024 * 1024


class ModelManifestIOError(ValueError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def load_model_manifest(path: Path) -> ModelManifest:
    """Load one regular, bounded JSON manifest through the strict Pydantic schema."""

    path = Path(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise ModelManifestIOError(
            code="manifest_not_found",
            message="The requested model manifest does not exist.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ModelManifestIOError(
            code="invalid_manifest_file",
            message="A model manifest must be a regular non-symlink file.",
        )
    if details.st_size <= 0 or details.st_size > _MAX_MANIFEST_BYTES:
        raise ModelManifestIOError(
            code="manifest_size_out_of_bounds",
            message="The model manifest size is outside the allowed bound.",
        )
    try:
        payload = path.read_bytes()
        json.loads(payload)
        return ModelManifest.model_validate_json(payload)
    except (OSError, ValueError, ValidationError) as exc:
        raise ModelManifestIOError(
            code="invalid_model_manifest",
            message=f"The model manifest could not be validated ({type(exc).__name__}).",
        ) from None


def write_model_manifest_atomic(
    path: Path, manifest: ModelManifest
) -> ModelManifestRef:
    """Publish a manifest atomically and verify its exact typed round trip.

    This function serializes an already constructed contract.  It does not
    approve, qualify, or otherwise change the manifest's execution gates.
    """

    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ModelManifestIOError(
            code="invalid_manifest_filename",
            message="A model manifest filename must end in .json.",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ModelManifestIOError(
            code="manifest_already_exists",
            message="Model manifests are immutable and cannot overwrite an existing path.",
        )

    encoded = (
        manifest.model_dump_json(indent=2, exclude_none=False).encode("utf-8") + b"\n"
    )
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ModelManifestIOError(
            code="manifest_size_out_of_bounds",
            message="The serialized model manifest exceeds the allowed bound.",
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        os.link(temporary_path, path, follow_symlinks=False)
        path.chmod(0o600)
        temporary_path.unlink()
        temporary_path = None
    except FileExistsError:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ModelManifestIOError(
            code="manifest_already_exists",
            message="Model manifests are immutable and cannot overwrite an existing path.",
        ) from None
    except Exception as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ModelManifestIOError(
            code="manifest_write_failed",
            message=f"The model manifest could not be published ({type(exc).__name__}).",
        ) from None

    reloaded = load_model_manifest(path)
    if reloaded != manifest:
        raise ModelManifestIOError(
            code="manifest_round_trip_mismatch",
            message="The published model manifest did not round-trip exactly.",
        )
    return ModelManifestRef.from_manifest(reloaded)


def builtin_manifest_paths() -> tuple[Path, ...]:
    """Return the explicit built-in manifest allowlist in stable order."""

    directory = Path(__file__).with_name("manifests")
    return (directory / "mriganka-domain-adaptation.provisional.v1.json",)


def load_builtin_manifests() -> tuple[ModelManifest, ...]:
    return tuple(load_model_manifest(path) for path in builtin_manifest_paths())


__all__ = [
    "ModelManifestIOError",
    "builtin_manifest_paths",
    "load_builtin_manifests",
    "load_model_manifest",
    "write_model_manifest_atomic",
]
