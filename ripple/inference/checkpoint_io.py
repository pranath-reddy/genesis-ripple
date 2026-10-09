"""Digest-first, weights-only loading for the recovered ENN checkpoint pair."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .contracts import CheckpointSpec, FileIdentity
from .mriganka_enn import (
    MrigankaENNContractError,
    validate_classifier_state_dict,
    validate_encoder_state_dict,
)

_MAX_ENCODER_BYTES = 192 * 1024 * 1024
_MAX_CLASSIFIER_BYTES = 1024 * 1024


class CheckpointLoadError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class LoadedCheckpoint:
    identity: FileIdentity
    state_dict: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class LoadedCheckpointBundle:
    encoder: LoadedCheckpoint
    classifier: LoadedCheckpoint


def load_checkpoint_bundle(
    *,
    bundle_root: Path,
    encoder_spec: CheckpointSpec,
    classifier_spec: CheckpointSpec,
) -> LoadedCheckpointBundle:
    root = _checked_directory(bundle_root)
    encoder = _load_checkpoint(root / encoder_spec.filename, encoder_spec)
    classifier = _load_checkpoint(root / classifier_spec.filename, classifier_spec)
    return LoadedCheckpointBundle(encoder=encoder, classifier=classifier)


def _load_checkpoint(path: Path, spec: CheckpointSpec) -> LoadedCheckpoint:
    maximum = _MAX_ENCODER_BYTES if spec.role == "encoder" else _MAX_CLASSIFIER_BYTES
    candidate = _checked_regular_file(path)
    before = os.lstat(candidate)
    if before.st_size != spec.byte_count:
        raise CheckpointLoadError(
            code=f"{spec.role}_checkpoint_size_mismatch",
            message=f"The {spec.role} checkpoint byte count does not match its manifest.",
        )
    if before.st_size > maximum:
        raise CheckpointLoadError(
            code=f"{spec.role}_checkpoint_too_large",
            message=f"The {spec.role} checkpoint exceeds its fixed size limit.",
        )

    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise CheckpointLoadError(
                code=f"{spec.role}_checkpoint_changed_before_read",
                message=f"The {spec.role} checkpoint changed before it was opened.",
            )

        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            byte_count += len(chunk)
            if byte_count > maximum:
                raise CheckpointLoadError(
                    code=f"{spec.role}_checkpoint_size_changed",
                    message=f"The {spec.role} checkpoint grew beyond its size limit.",
                )
            digest.update(chunk)
        observed_sha256 = digest.hexdigest()
        if byte_count != spec.byte_count or observed_sha256 != spec.sha256:
            raise CheckpointLoadError(
                code=f"{spec.role}_checkpoint_digest_mismatch",
                message=f"The {spec.role} checkpoint does not match the pinned artifact.",
            )

        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            try:
                loaded: Any = torch.load(
                    handle,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as exc:  # noqa: BLE001 - sanitize loader failures
                raise CheckpointLoadError(
                    code=f"{spec.role}_checkpoint_load_failed",
                    message=(
                        f"The {spec.role} checkpoint could not be loaded with the "
                        f"weights-only loader ({type(exc).__name__})."
                    ),
                ) from None

        after = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(after):
            raise CheckpointLoadError(
                code=f"{spec.role}_checkpoint_changed_during_read",
                message=f"The {spec.role} checkpoint changed while it was loaded.",
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if not isinstance(loaded, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, torch.Tensor)
        for key, value in loaded.items()
    ):
        raise CheckpointLoadError(
            code=f"{spec.role}_checkpoint_not_state_dict",
            message=f"The {spec.role} checkpoint is not a tensor-only state dictionary.",
        )
    state_dict = dict(loaded)
    try:
        if spec.role == "encoder":
            validate_encoder_state_dict(state_dict)
        else:
            validate_classifier_state_dict(state_dict)
    except MrigankaENNContractError as exc:
        raise CheckpointLoadError(
            code=f"{spec.role}_checkpoint_structure_mismatch",
            message=f"The {spec.role} checkpoint does not match the recovered ENN graph: {exc}",
        ) from None

    return LoadedCheckpoint(
        identity=FileIdentity(
            filename=spec.filename,
            byte_count=byte_count,
            sha256=observed_sha256,
        ),
        state_dict=state_dict,
    )


def _checked_directory(path: Path) -> Path:
    candidate = _absolute_without_parent_traversal(path)
    _reject_symlink_components(candidate)
    try:
        details = os.lstat(candidate)
    except FileNotFoundError:
        raise CheckpointLoadError(
            code="checkpoint_root_not_found",
            message="The checkpoint bundle root is unavailable.",
        ) from None
    if not stat.S_ISDIR(details.st_mode):
        raise CheckpointLoadError(
            code="checkpoint_root_not_directory",
            message="The checkpoint bundle root is not a directory.",
        )
    return candidate


def _checked_regular_file(path: Path) -> Path:
    candidate = _absolute_without_parent_traversal(path)
    _reject_symlink_components(candidate)
    try:
        details = os.lstat(candidate)
    except FileNotFoundError:
        raise CheckpointLoadError(
            code="checkpoint_not_found",
            message="A required checkpoint file is unavailable.",
        ) from None
    if not stat.S_ISREG(details.st_mode):
        raise CheckpointLoadError(
            code="checkpoint_not_regular_file",
            message="A required checkpoint is not a regular file.",
        )
    return candidate


def _absolute_without_parent_traversal(path: Path) -> Path:
    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise CheckpointLoadError(
            code="checkpoint_parent_traversal_forbidden",
            message="Parent traversal is not permitted in checkpoint paths.",
        )
    return (candidate if candidate.is_absolute() else Path.cwd() / candidate).absolute()


def _reject_symlink_components(path: Path) -> None:
    for component in tuple(reversed(path.parents)) + (path,):
        if component == Path(component.anchor):
            continue
        try:
            details = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise CheckpointLoadError(
                code="checkpoint_symlink_forbidden",
                message="Symlinked checkpoint paths are not permitted.",
            )


def _file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
    )


__all__ = [
    "CheckpointLoadError",
    "LoadedCheckpoint",
    "LoadedCheckpointBundle",
    "load_checkpoint_bundle",
]
