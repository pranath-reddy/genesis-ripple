"""Strict contracts for safe, read-only researcher repository intake."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN


_COMMIT_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
_REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}$"
_LANGUAGE_PATTERN = r"^[a-z][a-z0-9_+-]{0,31}$"


def _validate_repository_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("path must be a normalized, safe repository-relative path")
    return value


class GitHttpsRepositorySource(FrozenModel):
    kind: Literal["git_https"] = "git_https"
    url: str = Field(min_length=12, max_length=2048)
    revision: str = Field(default="HEAD", pattern=_REVISION_PATTERN)

    @field_validator("url")
    @classmethod
    def _safe_https_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            value != value.strip()
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError(
                "Git source must be a credential-free HTTPS URL without query or fragment"
            )
        try:
            parsed.hostname.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Git hostname must use an ASCII representation") from exc
        return value

    @field_validator("revision")
    @classmethod
    def _safe_revision(cls, value: str) -> str:
        if ".." in value or "@{" in value or "//" in value:
            raise ValueError("Git revision contains a disallowed sequence")
        return value


class LocalRepositorySource(FrozenModel):
    kind: Literal["local_path"] = "local_path"
    path: str = Field(min_length=1, max_length=4096)

    @field_validator("path")
    @classmethod
    def _absolute_local_path(cls, value: str) -> str:
        if (
            value != value.strip()
            or not value.startswith("/")
            or "\x00" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError(
                "local repository path must be an explicit absolute POSIX path"
            )
        return value


RepositorySource = Annotated[
    GitHttpsRepositorySource | LocalRepositorySource,
    Field(discriminator="kind"),
]


class RepositoryIntakeRequest(FrozenModel):
    schema_version: Literal["ripple.repository-intake-request.v1"] = (
        "ripple.repository-intake-request.v1"
    )
    intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    requested_at_utc: datetime
    source: RepositorySource

    @field_validator("requested_at_utc")
    @classmethod
    def _utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("requested_at_utc must use UTC")
        return value


class RepositoryIntakePolicy(FrozenModel):
    """Code-owned resource limits for one intake operation."""

    allowed_suffixes: tuple[str, ...] = (
        ".py",
        ".pyi",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
    )
    allowed_special_names: tuple[str, ...] = (
        "Dockerfile",
        "environment.txt",
        "requirements.txt",
        "constraints.txt",
    )
    max_entries: int = Field(default=10_000, ge=1, le=100_000)
    max_files: int = Field(default=256, ge=1, le=4096)
    max_depth: int = Field(default=16, ge=1, le=64)
    max_file_bytes: int = Field(default=2 * 1024 * 1024, ge=1, le=8 * 1024 * 1024)
    max_total_bytes: int = Field(default=32 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)
    max_git_transfer_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=16 * 1024 * 1024,
        le=2 * 1024 * 1024 * 1024,
    )
    git_timeout_seconds: int = Field(default=180, ge=10, le=1800)

    @field_validator("allowed_suffixes")
    @classmethod
    def _suffix_allowlist(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or len(value) != len(set(value))
            or any(not item.startswith(".") or item != item.lower() for item in value)
            or any(item in {".md", ".rst", ".ipynb"} for item in value)
        ):
            raise ValueError(
                "allowed suffixes must be a unique, lowercase code/config subset"
            )
        return value

    @model_validator(mode="after")
    def _coherent_limits(self) -> "RepositoryIntakePolicy":
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("per-file bound cannot exceed total byte bound")
        return self


class RepositoryFile(FrozenModel):
    path: str = Field(min_length=1, max_length=1024)
    role: Literal["source", "configuration"]
    language: str = Field(pattern=_LANGUAGE_PATTERN)
    byte_count: int = Field(gt=0, le=8 * 1024 * 1024)
    line_count: int = Field(gt=0, le=2_000_000)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


def repository_tree_sha256(files: tuple[RepositoryFile, ...]) -> str:
    payload = [item.model_dump(mode="json") for item in files]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RepositoryIntakeManifest(FrozenModel):
    schema_version: Literal["ripple.repository-intake.v1"] = (
        "ripple.repository-intake.v1"
    )
    intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    created_at_utc: datetime
    source_kind: Literal["git_https", "local_path"]
    source_locator: str = Field(min_length=1, max_length=4096)
    requested_revision: str | None = Field(default=None, max_length=256)
    resolved_commit: str | None = Field(default=None, pattern=_COMMIT_PATTERN)
    source_is_git: bool
    materialization: Literal["pinned_git_tree", "local_filesystem_snapshot"]
    snapshot_subdirectory: Literal["repository"] = "repository"
    files: tuple[RepositoryFile, ...] = Field(min_length=1)
    file_count: int = Field(gt=0, le=4096)
    total_bytes: int = Field(gt=0, le=256 * 1024 * 1024)
    excluded_entry_count: int = Field(ge=0)
    tree_sha256: str = Field(pattern=SHA256_PATTERN)
    source_code_executed: Literal[False] = False
    source_code_imported: Literal[False] = False
    git_hooks_executed: Literal[False] = False
    submodules_initialized: Literal[False] = False
    executable_permissions_preserved: Literal[False] = False
    credential_content_recorded: Literal[False] = False

    @field_validator("created_at_utc")
    @classmethod
    def _utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must use UTC")
        return value

    @model_validator(mode="after")
    def _consistent_manifest(self) -> "RepositoryIntakeManifest":
        if self.file_count != len(self.files):
            raise ValueError("file_count does not match files")
        if self.total_bytes != sum(item.byte_count for item in self.files):
            raise ValueError("total_bytes does not match files")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("repository files must be uniquely sorted by path")
        if repository_tree_sha256(self.files) != self.tree_sha256:
            raise ValueError("tree_sha256 does not match repository files")
        if self.source_kind == "git_https":
            if (
                not self.source_is_git
                or self.resolved_commit is None
                or self.requested_revision is None
                or self.materialization != "pinned_git_tree"
            ):
                raise ValueError(
                    "HTTPS Git intake must identify one pinned commit tree"
                )
        if (
            self.materialization == "local_filesystem_snapshot"
            and self.source_kind != "local_path"
        ):
            raise ValueError("local snapshot materialization requires a local source")
        return self


class RepositoryFileListing(FrozenModel):
    schema_version: Literal["ripple.repository-file-list.v1"] = (
        "ripple.repository-file-list.v1"
    )
    intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    total_matching: int = Field(ge=0)
    offset: int = Field(ge=0)
    files: tuple[RepositoryFile, ...]


class RepositorySearchRequest(FrozenModel):
    query: str = Field(min_length=1, max_length=256)
    case_sensitive: bool = False
    path_prefix: str | None = Field(default=None, max_length=1024)
    max_results: int = Field(default=40, ge=1, le=100)
    max_line_characters: int = Field(default=400, ge=40, le=1000)

    @field_validator("query")
    @classmethod
    def _safe_query(cls, value: str) -> str:
        if value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("search query must be trimmed printable literal text")
        return value

    @field_validator("path_prefix")
    @classmethod
    def _safe_prefix(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_repository_relative_path(value)


class RepositorySearchMatch(FrozenModel):
    path: str = Field(min_length=1, max_length=1024)
    line_number: int = Field(gt=0)
    line_text: str = Field(max_length=1001)
    line_truncated: bool

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class RepositorySearchResult(FrozenModel):
    schema_version: Literal["ripple.repository-search-result.v1"] = (
        "ripple.repository-search-result.v1"
    )
    intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    query: str
    matches: tuple[RepositorySearchMatch, ...]
    result_limit_reached: bool
    files_examined: int = Field(ge=0)


class RepositoryExcerptRequest(FrozenModel):
    path: str = Field(min_length=1, max_length=1024)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)

    @model_validator(mode="after")
    def _bounded_lines(self) -> "RepositoryExcerptRequest":
        if self.end_line < self.start_line or self.end_line - self.start_line + 1 > 200:
            raise ValueError("excerpt must request between 1 and 200 ordered lines")
        return self


class RepositoryExcerpt(FrozenModel):
    schema_version: Literal["ripple.repository-excerpt.v1"] = (
        "ripple.repository-excerpt.v1"
    )
    intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1, max_length=1024)
    full_file_sha256: str = Field(pattern=SHA256_PATTERN)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    numbered_text: str = Field(min_length=1, max_length=128 * 1024)
    excerpt_sha256: str = Field(pattern=SHA256_PATTERN)
    source_code_executed: Literal[False] = False
    source_code_imported: Literal[False] = False

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)

    @model_validator(mode="after")
    def _consistent_excerpt(self) -> "RepositoryExcerpt":
        if self.end_line < self.start_line or self.end_line - self.start_line + 1 > 200:
            raise ValueError("excerpt line bounds are invalid")
        if (
            hashlib.sha256(self.numbered_text.encode("utf-8")).hexdigest()
            != self.excerpt_sha256
        ):
            raise ValueError("excerpt_sha256 does not match numbered_text")
        rendered = self.numbered_text.splitlines()
        if len(rendered) != self.end_line - self.start_line + 1:
            raise ValueError("numbered excerpt does not match its line bounds")
        return self


__all__ = [
    "GitHttpsRepositorySource",
    "LocalRepositorySource",
    "RepositoryExcerpt",
    "RepositoryExcerptRequest",
    "RepositoryFile",
    "RepositoryFileListing",
    "RepositoryIntakeManifest",
    "RepositoryIntakePolicy",
    "RepositoryIntakeRequest",
    "RepositorySearchMatch",
    "RepositorySearchRequest",
    "RepositorySearchResult",
    "RepositorySource",
    "repository_tree_sha256",
]
