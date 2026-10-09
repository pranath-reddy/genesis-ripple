"""Lexical path checks shared by the scientist control plane."""

from __future__ import annotations

import os
import stat
from pathlib import Path


class UnsafePathError(ValueError):
    """A caller-owned path traverses a symlink or has an unexpected type."""


def checked_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute path only after rejecting traversal and symlink components."""

    candidate = Path(path).expanduser()
    if ".." in candidate.parts:
        raise UnsafePathError("parent traversal is not permitted")
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    absolute = absolute.absolute()
    reject_symlink_components(absolute)
    return absolute


def checked_real_file(path: str | os.PathLike[str]) -> Path:
    """Require a regular file whose existing path components are not symlinks."""

    candidate = checked_absolute_path(path)
    try:
        details = os.lstat(candidate)
    except FileNotFoundError:
        raise UnsafePathError("required file is unavailable") from None
    if not stat.S_ISREG(details.st_mode):
        raise UnsafePathError("required path is not a regular file")
    reject_symlink_components(candidate)
    return candidate


def checked_real_directory(
    path: str | os.PathLike[str],
    *,
    create: bool = False,
    mode: int = 0o700,
) -> Path:
    """Require or create a directory without accepting symlinked components."""

    candidate = checked_absolute_path(path)
    if create:
        candidate.mkdir(parents=True, exist_ok=True, mode=mode)
    reject_symlink_components(candidate)
    try:
        details = os.lstat(candidate)
    except FileNotFoundError:
        raise UnsafePathError("required directory is unavailable") from None
    if not stat.S_ISDIR(details.st_mode):
        raise UnsafePathError("required path is not a directory")
    return candidate


def reject_symlink_components(path: str | os.PathLike[str]) -> None:
    """Reject every existing symlink from the filesystem root through ``path``."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise UnsafePathError("path must be absolute before component validation")
    chain = tuple(reversed(candidate.parents)) + (candidate,)
    for component in chain:
        if component == Path(component.anchor):
            continue
        try:
            details = os.lstat(component)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise UnsafePathError("symlinked path components are not permitted")


__all__ = [
    "UnsafePathError",
    "checked_absolute_path",
    "checked_real_directory",
    "checked_real_file",
    "reject_symlink_components",
]
