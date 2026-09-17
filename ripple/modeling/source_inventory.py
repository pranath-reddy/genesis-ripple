"""Deterministic, read-only inventory of explicitly selected local model sources.

The inventory boundary is intentionally narrow:

* callers provide an absolute repository root and explicit relative files or
  directories;
* paths must remain beneath that root and no symlink is ever followed;
* secret-shaped and hidden paths are rejected or excluded before opening;
* only a fixed executable/configuration/checkpoint-metadata suffix allowlist is
  considered, with Markdown and other narrative documentation excluded;
* files are bounded, opened read-only with no-follow semantics, and streamed
  only to compute their digest.  Source content is never returned or executed;
* this module performs no network access and imports no agent/LLM framework.

The resulting :class:`SourceInventory` is evidence for an onboarding proposal.
It does not authorize preprocessing, model execution, or scientific use.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import EvidenceSource
from .onboarding import ModelOnboardingRequest, SourceArtifact, SourceInventory


DEFAULT_SOURCE_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".pyi",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
)
EXPLICIT_CHECKPOINT_SUFFIXES: frozenset[str] = frozenset(
    {".ckpt", ".pt", ".pth", ".safetensors"}
)
EXPLICITLY_EXCLUDED_DOCUMENT_SUFFIXES: frozenset[str] = frozenset(
    {".md", ".markdown", ".mdown", ".mkd", ".rst"}
)

_SECRET_FILE_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "secrets.json",
        "token.json",
    }
)
_SECRET_SUFFIXES = frozenset({".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"})
_SECRET_COMPONENT_PATTERN = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|credential|credentials|private[_-]?key|"
    r"secret|secrets|token|tokens)(?:$|[._-])",
    flags=re.IGNORECASE,
)
_CHECKPOINT_METADATA_PATTERN = re.compile(
    r"(?:^|[._-])(?:checkpoint|model[_-]?state|weights)(?:$|[._-])",
    flags=re.IGNORECASE,
)
_MEDIA_TYPES = {
    ".py": "text/x-python",
    ".pyi": "text/x-python",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".toml": "application/toml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
}
_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
}


class SourceInventoryError(RuntimeError):
    """Base error for a rejected or failed local inventory."""


class InvalidInventoryPathError(SourceInventoryError):
    """A requested path was invalid, outside the root, or not a regular source."""


class SymlinkRejectedError(SourceInventoryError):
    """A symlink was encountered at the inventory boundary."""


class SecretPathRejectedError(SourceInventoryError):
    """A path looked capable of containing credentials or private key material."""


class UnsupportedSourceTypeError(SourceInventoryError):
    """An explicit file was not in the fixed source-format allowlist."""


class InventoryLimitError(SourceInventoryError):
    """The declared file, byte, or directory-depth budget was exceeded."""


class SourceChangedError(SourceInventoryError):
    """A file changed between validation and bounded hashing."""


class LocalInventoryPolicy(BaseModel):
    """Strict resource and suffix bounds for a local inventory operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    allowed_suffixes: tuple[str, ...] = DEFAULT_SOURCE_SUFFIXES
    max_files: int = Field(default=128, ge=1, le=4096)
    max_checkpoint_files: int = Field(default=32, ge=0, le=1024)
    max_scanned_entries: int = Field(default=4096, ge=1, le=100_000)
    max_file_bytes: int = Field(default=4 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_total_bytes: int = Field(default=32 * 1024 * 1024, ge=1, le=512 * 1024 * 1024)
    max_checkpoint_file_bytes: int = Field(
        default=20 * 1024 * 1024 * 1024,
        ge=1,
        le=100 * 1024 * 1024 * 1024,
    )
    max_checkpoint_total_bytes: int = Field(
        default=40 * 1024 * 1024 * 1024,
        ge=1,
        le=200 * 1024 * 1024 * 1024,
    )
    max_directory_depth: int = Field(default=12, ge=0, le=64)
    hash_chunk_bytes: int = Field(default=1024 * 1024, ge=4096, le=8 * 1024 * 1024)
    reject_hidden_paths: bool = True

    @field_validator("allowed_suffixes")
    @classmethod
    def _validate_suffixes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(suffix.lower() for suffix in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("allowed suffixes must be non-empty and unique")
        if any(
            not suffix.startswith(".")
            or "/" in suffix
            or "\\" in suffix
            or suffix != suffix.strip()
            for suffix in normalized
        ):
            raise ValueError("allowed suffixes must be normalized filename suffixes")
        if set(normalized) & EXPLICITLY_EXCLUDED_DOCUMENT_SUFFIXES:
            raise ValueError(
                "Markdown and narrative documentation are never inventory sources"
            )
        if not set(normalized) <= set(DEFAULT_SOURCE_SUFFIXES):
            raise ValueError(
                "allowed suffixes may only narrow the fixed safe allowlist"
            )
        return normalized

    @model_validator(mode="after")
    def _validate_byte_budget(self) -> "LocalInventoryPolicy":
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("per-file byte limit cannot exceed the total byte limit")
        if self.max_checkpoint_file_bytes > self.max_checkpoint_total_bytes:
            raise ValueError(
                "per-checkpoint byte limit cannot exceed the checkpoint total byte limit"
            )
        return self


@dataclass(frozen=True)
class _Candidate:
    path: Path
    relative_path: str
    stat_result: os.stat_result
    role: Literal["source", "checkpoint"] = "source"


def inventory_local_sources(
    request: ModelOnboardingRequest,
    *,
    repository_root: Path,
    relative_sources: tuple[str, ...],
    relative_checkpoints: tuple[str, ...] = (),
    repository_revision: str | None = None,
    created_at_utc: datetime | None = None,
    policy: LocalInventoryPolicy | None = None,
) -> SourceInventory:
    """Inventory explicitly selected local sources without executing their content.

    ``repository_root`` must be absolute. ``relative_sources`` may contain files
    or directories, but every item must be a normalized repository-relative
    path. Directories are traversed in lexical order without following symlinks.
    Unsupported files encountered inside a selected directory are counted as
    exclusions; an unsupported explicitly selected file is rejected.

    The request's locators are retained as researcher intent only. This function
    never resolves or downloads them; the local root and paths are independent,
    explicit trust-boundary inputs. Checkpoints are accepted only through the
    separate explicit ``relative_checkpoints`` argument; directory traversal
    never discovers or deserializes model weights.
    """

    if not isinstance(request, ModelOnboardingRequest):
        raise TypeError("request must be a validated ModelOnboardingRequest")
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    if not isinstance(relative_sources, tuple) or not relative_sources:
        raise InvalidInventoryPathError(
            "relative_sources must be a non-empty tuple of explicit local paths"
        )
    if len(relative_sources) != len(set(relative_sources)):
        raise InvalidInventoryPathError("relative source paths must be unique")
    if not isinstance(relative_checkpoints, tuple):
        raise InvalidInventoryPathError("relative_checkpoints must be a tuple")
    if len(relative_checkpoints) != len(set(relative_checkpoints)):
        raise InvalidInventoryPathError("relative checkpoint paths must be unique")

    effective_policy = policy or LocalInventoryPolicy()
    if len(relative_checkpoints) > effective_policy.max_checkpoint_files:
        raise InventoryLimitError("explicit checkpoint count exceeds its fixed limit")
    root = _validate_repository_root(repository_root)
    created = created_at_utc or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware")
    if created.utcoffset() != timezone.utc.utcoffset(created):
        raise ValueError("created_at_utc must use UTC")
    if repository_revision is not None:
        if (
            not repository_revision
            or repository_revision != repository_revision.strip()
            or len(repository_revision) > 256
            or any(ord(character) < 32 for character in repository_revision)
        ):
            raise ValueError("repository_revision must be trimmed safe text")

    candidates: dict[str, _Candidate] = {}
    excluded_count = 0
    for raw_relative in relative_sources:
        relative = _validate_relative_source(raw_relative)
        target = root.joinpath(*PurePosixPath(relative).parts)
        _assert_no_symlink_chain(root, target)
        try:
            target_stat = os.lstat(target)
        except FileNotFoundError as exc:
            raise InvalidInventoryPathError(
                f"requested source does not exist: {relative}"
            ) from exc
        if stat.S_ISLNK(target_stat.st_mode):
            raise SymlinkRejectedError(f"requested source is a symlink: {relative}")
        if _is_rejected_path(PurePosixPath(relative), effective_policy):
            raise SecretPathRejectedError(
                f"requested source is hidden or secret-shaped: {relative}"
            )

        if stat.S_ISREG(target_stat.st_mode):
            if not _is_allowed_source(target, effective_policy):
                raise UnsupportedSourceTypeError(
                    _unsupported_message(relative, target.suffix.lower())
                )
            candidates[relative] = _Candidate(target, relative, target_stat)
        elif stat.S_ISDIR(target_stat.st_mode):
            directory_candidates, directory_excluded = _walk_directory(
                root=root,
                directory=target,
                directory_relative=PurePosixPath(relative),
                policy=effective_policy,
            )
            for candidate in directory_candidates:
                candidates.setdefault(candidate.relative_path, candidate)
            excluded_count += directory_excluded
        else:
            raise InvalidInventoryPathError(
                f"requested source is not a regular file or directory: {relative}"
            )

    checkpoint_total = 0
    for raw_relative in relative_checkpoints:
        relative = _validate_relative_source(raw_relative)
        if relative in candidates:
            raise InvalidInventoryPathError(
                "a path cannot be both a source and an explicit checkpoint"
            )
        target = root.joinpath(*PurePosixPath(relative).parts)
        _assert_no_symlink_chain(root, target)
        try:
            target_stat = os.lstat(target)
        except FileNotFoundError as exc:
            raise InvalidInventoryPathError(
                f"requested checkpoint does not exist: {relative}"
            ) from exc
        if stat.S_ISLNK(target_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode):
            raise InvalidInventoryPathError(
                f"requested checkpoint is not a regular non-symlink file: {relative}"
            )
        if _is_rejected_path(PurePosixPath(relative), effective_policy):
            raise SecretPathRejectedError(
                f"requested checkpoint is hidden or secret-shaped: {relative}"
            )
        if target.suffix.lower() not in EXPLICIT_CHECKPOINT_SUFFIXES:
            raise UnsupportedSourceTypeError(
                f"explicit checkpoint has a non-allowlisted suffix: {relative}"
            )
        if (
            target_stat.st_size <= 0
            or target_stat.st_size > effective_policy.max_checkpoint_file_bytes
        ):
            raise InventoryLimitError(
                f"checkpoint exceeds its per-file byte limit: {relative}"
            )
        checkpoint_total += target_stat.st_size
        if checkpoint_total > effective_policy.max_checkpoint_total_bytes:
            raise InventoryLimitError(
                "explicit checkpoints exceed their total byte limit"
            )
        candidates[relative] = _Candidate(
            target,
            relative,
            target_stat,
            role="checkpoint",
        )

    ordered_candidates = tuple(candidates[key] for key in sorted(candidates))
    _enforce_candidate_bounds(ordered_candidates, effective_policy)
    if not ordered_candidates:
        raise InvalidInventoryPathError(
            "the explicit source selection contained no allowlisted source artifacts"
        )

    artifacts: list[SourceArtifact] = []
    evidence_sources: list[EvidenceSource] = []
    for candidate in ordered_candidates:
        digest = _hash_regular_file(candidate, effective_policy)
        artifact, evidence = _models_for_candidate(
            candidate,
            digest=digest,
            repository_revision=repository_revision,
            accessed_at_utc=created,
        )
        artifacts.append(artifact)
        evidence_sources.append(evidence)

    inventory_id = _inventory_id(
        request_id=request.request_id,
        repository_revision=repository_revision,
        artifacts=artifacts,
    )
    return SourceInventory(
        inventory_id=inventory_id,
        request_id=request.request_id,
        created_at_utc=created,
        repository_revision=repository_revision,
        artifacts=tuple(artifacts),
        evidence_sources=tuple(evidence_sources),
        excluded_path_count=excluded_count,
        inventory_complete_for_declared_scope=True,
    )


def _validate_repository_root(repository_root: Path) -> Path:
    if not repository_root.is_absolute():
        raise InvalidInventoryPathError(
            "repository_root must be an explicit absolute path"
        )
    root = Path(os.path.abspath(os.fspath(repository_root)))
    _assert_path_components_not_symlinks(root)
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError as exc:
        raise InvalidInventoryPathError("repository_root does not exist") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise InvalidInventoryPathError("repository_root must be a directory")
    return root


def _validate_relative_source(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidInventoryPathError(
            "relative source paths must be non-empty trimmed strings"
        )
    if "\\" in value or "://" in value or "\x00" in value:
        raise InvalidInventoryPathError(
            "relative source paths must be local POSIX paths"
        )
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or value != parsed.as_posix()
        or ".." in parsed.parts
        or parsed.as_posix() in {"", "."}
    ):
        raise InvalidInventoryPathError(
            "source paths must be normalized repository-relative paths"
        )
    return value


def _assert_path_components_not_symlinks(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise SymlinkRejectedError(f"symlink path component rejected: {current}")


def _assert_no_symlink_chain(root: Path, target: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise InvalidInventoryPathError(
            "requested source escaped the repository root"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise SymlinkRejectedError(
                f"symlink path component rejected: {current.relative_to(root).as_posix()}"
            )


def _is_rejected_path(relative: PurePosixPath, policy: LocalInventoryPolicy) -> bool:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    if policy.reject_hidden_paths and any(
        part.startswith(".") for part in lowered_parts
    ):
        return True
    for part in lowered_parts:
        if part in _SECRET_FILE_NAMES or _SECRET_COMPONENT_PATTERN.search(part):
            return True
    return relative.suffix.lower() in _SECRET_SUFFIXES


def _is_allowed_source(path: Path, policy: LocalInventoryPolicy) -> bool:
    suffix = path.suffix.lower()
    return (
        suffix in policy.allowed_suffixes
        and suffix not in EXPLICITLY_EXCLUDED_DOCUMENT_SUFFIXES
    )


def _unsupported_message(relative: str, suffix: str) -> str:
    if suffix in EXPLICITLY_EXCLUDED_DOCUMENT_SUFFIXES:
        return f"Markdown/narrative documentation is excluded from source inventory: {relative}"
    return f"explicit source has a non-allowlisted suffix {suffix!r}: {relative}"


def _walk_directory(
    *,
    root: Path,
    directory: Path,
    directory_relative: PurePosixPath,
    policy: LocalInventoryPolicy,
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    excluded_count = 0
    scanned_entries = 0

    def visit(current: Path, relative: PurePosixPath, depth: int) -> None:
        nonlocal excluded_count, scanned_entries
        if depth > policy.max_directory_depth:
            raise InventoryLimitError(
                f"directory depth exceeds {policy.max_directory_depth}: {relative.as_posix()}"
            )
        _assert_no_symlink_chain(root, current)
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as exc:
            raise InvalidInventoryPathError(
                f"could not read selected directory metadata: {relative.as_posix()}"
            ) from exc

        for entry in entries:
            scanned_entries += 1
            if scanned_entries > policy.max_scanned_entries:
                raise InventoryLimitError(
                    f"directory scan exceeds {policy.max_scanned_entries} entries"
                )
            child_relative = relative / entry.name
            if _is_rejected_path(child_relative, policy):
                excluded_count += 1
                continue
            try:
                if entry.is_symlink():
                    raise SymlinkRejectedError(
                        f"symlink encountered in selected directory: {child_relative.as_posix()}"
                    )
                entry_stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise SourceChangedError(
                    f"source disappeared during inventory: {child_relative.as_posix()}"
                ) from exc

            child = current / entry.name
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(child, child_relative, depth + 1)
            elif stat.S_ISREG(entry_stat.st_mode):
                if _is_allowed_source(child, policy):
                    if entry_stat.st_size == 0:
                        excluded_count += 1
                        continue
                    candidates.append(
                        _Candidate(child, child_relative.as_posix(), entry_stat)
                    )
                    if len(candidates) > policy.max_files:
                        raise InventoryLimitError(
                            f"source inventory exceeds {policy.max_files} files"
                        )
                else:
                    excluded_count += 1
            else:
                raise InvalidInventoryPathError(
                    "non-regular filesystem entry encountered in selected directory: "
                    f"{child_relative.as_posix()}"
                )

    visit(directory, directory_relative, 0)
    return candidates, excluded_count


def _enforce_candidate_bounds(
    candidates: tuple[_Candidate, ...], policy: LocalInventoryPolicy
) -> None:
    source_candidates = tuple(
        candidate for candidate in candidates if candidate.role == "source"
    )
    if len(source_candidates) > policy.max_files:
        raise InventoryLimitError(f"source inventory exceeds {policy.max_files} files")
    total = 0
    for candidate in source_candidates:
        size = candidate.stat_result.st_size
        if size <= 0 or size > policy.max_file_bytes:
            raise InventoryLimitError(
                f"source exceeds per-file byte limit: {candidate.relative_path}"
            )
        total += size
        if total > policy.max_total_bytes:
            raise InventoryLimitError(
                f"source inventory exceeds {policy.max_total_bytes} total bytes"
            )


def _hash_regular_file(candidate: _Candidate, policy: LocalInventoryPolicy) -> str:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate.path, flags)
    except OSError as exc:
        raise InvalidInventoryPathError(
            f"could not open source safely for hashing: {candidate.relative_path}"
        ) from exc

    digest = hashlib.sha256()
    bytes_read = 0
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise InvalidInventoryPathError(
                f"opened source is not a regular file: {candidate.relative_path}"
            )
        if not _same_file_state(candidate.stat_result, opened_stat):
            raise SourceChangedError(
                f"source changed before hashing: {candidate.relative_path}"
            )
        while True:
            chunk = os.read(descriptor, policy.hash_chunk_bytes)
            if not chunk:
                break
            bytes_read += len(chunk)
            byte_limit = (
                policy.max_checkpoint_file_bytes
                if candidate.role == "checkpoint"
                else policy.max_file_bytes
            )
            if bytes_read > byte_limit:
                raise InventoryLimitError(
                    f"source exceeded its byte limit while hashing: {candidate.relative_path}"
                )
            digest.update(chunk)
        final_stat = os.fstat(descriptor)
        if (
            not _same_file_state(opened_stat, final_stat)
            or bytes_read != final_stat.st_size
        ):
            raise SourceChangedError(
                f"source changed during hashing: {candidate.relative_path}"
            )
    finally:
        os.close(descriptor)
    try:
        path_stat = os.lstat(candidate.path)
    except FileNotFoundError as exc:
        raise SourceChangedError(
            f"source path disappeared after hashing: {candidate.relative_path}"
        ) from exc
    if not _same_file_state(candidate.stat_result, path_stat):
        raise SourceChangedError(
            f"source path changed during hashing: {candidate.relative_path}"
        )
    return digest.hexdigest()


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


def _models_for_candidate(
    candidate: _Candidate,
    *,
    digest: str,
    repository_revision: str | None,
    accessed_at_utc: datetime,
) -> tuple[SourceArtifact, EvidenceSource]:
    suffix = candidate.path.suffix.lower()
    identity_material = f"{candidate.relative_path}\0{digest}".encode("utf-8")
    identity = hashlib.sha256(identity_material).hexdigest()[:24]
    artifact_id = f"artifact-{identity}"
    source_id = f"source-{identity}"

    if candidate.role == "checkpoint":
        kind = "checkpoint"
        inspection_mode = "hash_only"
        source_kind = "checkpoint_metadata"
        authority = "context_only"
        media_type = "application/octet-stream"
        language = None
    elif suffix in {".py", ".pyi"}:
        kind = "python_source"
        inspection_mode = "hash_only"
        source_kind = "executable_source"
        authority = "context_only"
        media_type = _MEDIA_TYPES[suffix]
        language = _LANGUAGES[suffix]
    elif _CHECKPOINT_METADATA_PATTERN.search(candidate.path.stem):
        kind = "checkpoint_metadata"
        inspection_mode = "hash_only"
        source_kind = "checkpoint_metadata"
        authority = "context_only"
        media_type = _MEDIA_TYPES[suffix]
        language = _LANGUAGES[suffix]
    else:
        kind = "training_configuration"
        inspection_mode = "hash_only"
        source_kind = "training_configuration"
        authority = "context_only"
        media_type = _MEDIA_TYPES[suffix]
        language = _LANGUAGES[suffix]

    artifact = SourceArtifact(
        artifact_id=artifact_id,
        evidence_source_id=source_id,
        kind=kind,
        repository_relative_path=candidate.relative_path,
        media_type=media_type,
        byte_count=candidate.stat_result.st_size,
        sha256=digest,
        inspection_mode=inspection_mode,
        language=language,
    )
    evidence = EvidenceSource(
        source_id=source_id,
        kind=source_kind,
        authority=authority,
        title=candidate.relative_path,
        locator=candidate.relative_path,
        revision=repository_revision,
        sha256=digest,
        accessed_at_utc=accessed_at_utc,
    )
    return artifact, evidence


def _inventory_id(
    *,
    request_id: str,
    repository_revision: str | None,
    artifacts: Iterable[SourceArtifact],
) -> str:
    digest = hashlib.sha256()
    digest.update(request_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update((repository_revision or "").encode("utf-8"))
    for artifact in artifacts:
        digest.update(b"\0")
        digest.update(artifact.repository_relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(artifact.sha256.encode("ascii"))
    return f"inventory-{digest.hexdigest()[:24]}"


def inventory_source_tree_sha256(
    inventory: SourceInventory,
    *,
    source_locator: str,
) -> str | None:
    """Return a deterministic digest for Python sources beneath one local locator."""

    if not isinstance(inventory, SourceInventory):
        raise TypeError("inventory must be a validated SourceInventory")
    relative = PurePosixPath(_validate_relative_source(source_locator))
    artifacts = tuple(
        artifact
        for artifact in inventory.artifacts
        if artifact.kind == "python_source"
        and (
            PurePosixPath(artifact.repository_relative_path) == relative
            or relative in PurePosixPath(artifact.repository_relative_path).parents
        )
    )
    if not artifacts:
        return None
    digest = hashlib.sha256()
    digest.update(b"ripple-python-source-tree-v1\0")
    for artifact in sorted(artifacts, key=lambda item: item.repository_relative_path):
        digest.update(artifact.repository_relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(artifact.byte_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(artifact.sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = [
    "DEFAULT_SOURCE_SUFFIXES",
    "EXPLICIT_CHECKPOINT_SUFFIXES",
    "EXPLICITLY_EXCLUDED_DOCUMENT_SUFFIXES",
    "InvalidInventoryPathError",
    "InventoryLimitError",
    "LocalInventoryPolicy",
    "SecretPathRejectedError",
    "SourceChangedError",
    "SourceInventoryError",
    "SymlinkRejectedError",
    "UnsupportedSourceTypeError",
    "inventory_local_sources",
    "inventory_source_tree_sha256",
]
