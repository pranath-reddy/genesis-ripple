"""Bounded, read-only excerpts from previously inventoried local source files.

This module never imports or executes inventoried source, performs no network
access, and returns content only after revalidating repository containment,
symlink absence, file identity, size, and the inventory's full-file SHA-256.
Binary, invalid UTF-8, and secret-like content is rejected without including the
detected value in errors.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .onboarding import SourceArtifact, SourceInventory


MAX_EXCERPT_LINES = 200
MAX_EXCERPT_BYTES = 64 * 1024
MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_SECRET_PATH_COMPONENT_PATTERN = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|credential|private[_-]?key|secret|token)(?:$|[._-])",
    flags=re.IGNORECASE,
)

_PRIVATE_KEY_MARKER = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    flags=re.IGNORECASE,
)
_BEARER_CREDENTIAL = re.compile(
    r"\bauthorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/=-]{12,}",
    flags=re.IGNORECASE,
)
_KNOWN_TOKEN_SHAPE = re.compile(
    r"(?:\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b)"
)
_SECRET_LITERAL_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?"
    r"[A-Za-z_][A-Za-z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|"
    r"PASSWORD|PASSWD|PRIVATE[_-]?KEY|SECRET)"
    r"\s*[:=]\s*([\"'])([^\r\n\"']+)\1\s*(?:[,;#].*)?$"
)
_SECRET_UNQUOTED_CONFIG = re.compile(
    r"(?im)^\s*(?:export\s+)?(?:"
    r"[A-Za-z][A-Za-z0-9_]*?(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|"
    r"PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY|SECRET)|"
    r"API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|PASSWORD|PASSWD|"
    r"CREDENTIAL|PRIVATE[_-]?KEY|SECRET)"
    r"\s*[:=]\s*([^\s#][^\r\n#]*)$"
)
_NAMED_SECRET_LITERAL = re.compile(
    r"(?im)^\s*(?:export\s+)?(?!_)"
    r"(?:[A-Za-z][A-Za-z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|"
    r"PASSWORD|PASSWD|CREDENTIAL|PRIVATE[_-]?KEY|SECRET)|TOKEN|CREDENTIAL)"
    r"\s*[:=]\s*([\"'])([^\r\n\"']+)\1\s*(?:[,;#].*)?$"
)
_MAPPING_SECRET_LITERAL = re.compile(
    r"(?i)[\"'](?:[A-Za-z0-9_]*?(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"token|password|passwd|credential|private[_-]?key|secret))[\"']"
    r"\s*:\s*([\"'])([^\r\n\"']+)\1"
)
_SAFE_PLACEHOLDER_VALUES = frozenset(
    {
        "changeme",
        "dummy",
        "example",
        "fake",
        "none",
        "null",
        "placeholder",
        "redacted",
        "test",
        "your-key-here",
        "your-token-here",
    }
)
_SAFE_ENV_REFERENCE = re.compile(
    r"(?:\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\}|"
    r"os\.getenv\(\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*\)|"
    r"os\.environ\[\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*\]|"
    r"env\(\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*\))",
    flags=re.IGNORECASE,
)


class SourceExcerptError(RuntimeError):
    """Base error for a refused or failed source excerpt operation."""


class ExcerptPathError(SourceExcerptError):
    """The artifact path was invalid, missing, or outside the repository root."""


class ExcerptSymlinkError(SourceExcerptError):
    """A symlink was present at the source-reading boundary."""


class ExcerptIntegrityError(SourceExcerptError):
    """The current file no longer matched its inventoried identity."""


class ExcerptBoundsError(SourceExcerptError):
    """The source or requested excerpt exceeded a fixed bound."""


class UnsafeSourceContentError(SourceExcerptError):
    """Source was binary, invalid UTF-8, or appeared to contain a secret."""


class SourceExcerpt(BaseModel):
    """Numbered UTF-8 source lines whose parent file still matches inventory."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        hide_input_in_errors=True,
    )

    schema_version: Literal["ripple.source-excerpt.v1"] = "ripple.source-excerpt.v1"
    inventory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    evidence_source_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    repository_relative_path: str = Field(min_length=1, max_length=1024)
    full_file_sha256: str = Field(pattern=_SHA256_PATTERN)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    line_count: int = Field(ge=1, le=MAX_EXCERPT_LINES)
    encoding: Literal["utf-8"] = "utf-8"
    line_number_format: Literal["<line>: <source>"] = "<line>: <source>"
    excerpt_text: str = Field(min_length=1, max_length=MAX_EXCERPT_BYTES)
    excerpt_byte_count: int = Field(ge=1, le=MAX_EXCERPT_BYTES)
    excerpt_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_executed: Literal[False] = False
    source_imported: Literal[False] = False
    network_accessed: Literal[False] = False
    execution_authorized: Literal[False] = False

    @field_validator("repository_relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != value
            or value in {"", "."}
        ):
            raise ValueError(
                "source excerpt path must be normalized and repository-relative"
            )
        return value

    @model_validator(mode="after")
    def _validate_excerpt_identity(self) -> "SourceExcerpt":
        expected_count = self.end_line - self.start_line + 1
        if self.end_line < self.start_line or self.line_count != expected_count:
            raise ValueError("source excerpt line bounds and count disagree")
        encoded = self.excerpt_text.encode("utf-8")
        if len(encoded) != self.excerpt_byte_count:
            raise ValueError("source excerpt byte count is inconsistent")
        if hashlib.sha256(encoded).hexdigest() != self.excerpt_sha256:
            raise ValueError("source excerpt digest is inconsistent")
        expected_prefixes = tuple(
            f"{line_number}: "
            for line_number in range(self.start_line, self.end_line + 1)
        )
        rendered_lines = self.excerpt_text.splitlines()
        if len(rendered_lines) != self.line_count or any(
            not rendered.startswith(prefix)
            for rendered, prefix in zip(rendered_lines, expected_prefixes)
        ):
            raise ValueError("source excerpt does not match its numbered-line contract")
        return self


def read_source_excerpt(
    inventory: SourceInventory,
    *,
    repository_root: Path,
    artifact_id: str,
    start_line: int,
    end_line: int,
) -> SourceExcerpt:
    """Return verified, numbered source lines from an inventoried artifact.

    The complete file is bounded, read once as bytes, and checked against the
    inventory digest before any requested text is returned. The decoded complete
    file is screened for unsafe control bytes and conservative secret patterns,
    so a safe-looking line range cannot expose part of a file containing detected
    credential material elsewhere.
    """

    if not isinstance(inventory, SourceInventory):
        raise TypeError("inventory must be a validated SourceInventory")
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    if not isinstance(artifact_id, str):
        raise TypeError("artifact_id must be a string")
    if isinstance(start_line, bool) or not isinstance(start_line, int):
        raise TypeError("start_line must be an integer")
    if isinstance(end_line, bool) or not isinstance(end_line, int):
        raise TypeError("end_line must be an integer")
    if start_line < 1 or end_line < start_line:
        raise ExcerptBoundsError("source line bounds must be positive and ordered")
    if end_line - start_line + 1 > MAX_EXCERPT_LINES:
        raise ExcerptBoundsError(
            f"source excerpt cannot exceed {MAX_EXCERPT_LINES} lines"
        )

    artifact = _find_artifact(inventory, artifact_id)
    root = _validate_repository_root(repository_root)
    target = _target_for_artifact(root, artifact)
    source_bytes = _read_and_verify_file(target, artifact)
    source_text = _decode_and_screen(source_bytes)
    source_lines = source_text.splitlines()
    if not source_lines:
        raise ExcerptBoundsError("inventoried source contained no text lines")
    if end_line > len(source_lines):
        raise ExcerptBoundsError("requested line range exceeds the source file")

    rendered = "\n".join(
        f"{line_number}: {source_lines[line_number - 1]}"
        for line_number in range(start_line, end_line + 1)
    )
    encoded_excerpt = rendered.encode("utf-8")
    if len(encoded_excerpt) > MAX_EXCERPT_BYTES:
        raise ExcerptBoundsError(
            f"rendered source excerpt cannot exceed {MAX_EXCERPT_BYTES} bytes"
        )

    return SourceExcerpt(
        inventory_id=inventory.inventory_id,
        artifact_id=artifact.artifact_id,
        evidence_source_id=artifact.evidence_source_id,
        repository_relative_path=artifact.repository_relative_path,
        full_file_sha256=artifact.sha256,
        start_line=start_line,
        end_line=end_line,
        line_count=end_line - start_line + 1,
        excerpt_text=rendered,
        excerpt_byte_count=len(encoded_excerpt),
        excerpt_sha256=hashlib.sha256(encoded_excerpt).hexdigest(),
    )


def _find_artifact(inventory: SourceInventory, artifact_id: str) -> SourceArtifact:
    matches = tuple(
        artifact
        for artifact in inventory.artifacts
        if artifact.artifact_id == artifact_id
    )
    if len(matches) != 1:
        raise ExcerptPathError(
            "artifact ID was not present exactly once in the inventory"
        )
    artifact = matches[0]
    if artifact.kind != "python_source" or artifact.inspection_mode not in {
        "hash_only",
        "source_parse",
    }:
        raise UnsafeSourceContentError(
            "only inventoried Python source may be excerpted"
        )
    if artifact.byte_count > MAX_SOURCE_FILE_BYTES:
        raise ExcerptBoundsError(
            f"inventoried source exceeds the {MAX_SOURCE_FILE_BYTES}-byte reader bound"
        )
    relative = PurePosixPath(artifact.repository_relative_path)
    if (
        relative.suffix.lower() not in {".py", ".pyi"}
        or artifact.media_type != "text/x-python"
        or artifact.language != "python"
        or any(part.startswith(".") for part in relative.parts)
        or any(_SECRET_PATH_COMPONENT_PATTERN.search(part) for part in relative.parts)
    ):
        raise UnsafeSourceContentError("inventoried Python source metadata is unsafe")

    linked_sources = tuple(
        source
        for source in inventory.evidence_sources
        if source.source_id == artifact.evidence_source_id
    )
    if len(linked_sources) != 1:
        raise ExcerptIntegrityError(
            "artifact evidence source was not present exactly once"
        )
    source = linked_sources[0]
    if (
        source.kind != "executable_source"
        or source.locator != artifact.repository_relative_path
        or source.sha256 != artifact.sha256
        or source.authority not in {"context_only", "direct_executable_evidence"}
    ):
        raise ExcerptIntegrityError("artifact and evidence-source identities disagree")
    return artifact


def _validate_repository_root(repository_root: Path) -> Path:
    if not repository_root.is_absolute():
        raise ExcerptPathError("repository_root must be an explicit absolute path")
    root = Path(os.path.abspath(os.fspath(repository_root)))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise ExcerptPathError("repository_root does not exist") from exc
        if stat.S_ISLNK(mode):
            raise ExcerptSymlinkError("repository_root contains a symlink component")
    if not stat.S_ISDIR(os.lstat(root).st_mode):
        raise ExcerptPathError("repository_root must be a directory")
    return root


def _target_for_artifact(root: Path, artifact: SourceArtifact) -> Path:
    relative = PurePosixPath(artifact.repository_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExcerptPathError("inventoried artifact path is not repository-relative")
    target = root.joinpath(*relative.parts)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ExcerptPathError(
            "inventoried artifact escaped the repository root"
        ) from exc

    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise ExcerptPathError("inventoried source no longer exists") from exc
        if stat.S_ISLNK(mode):
            raise ExcerptSymlinkError("symlink encountered in inventoried source path")
    return target


def _read_and_verify_file(target: Path, artifact: SourceArtifact) -> bytes:
    try:
        initial_stat = os.lstat(target)
    except FileNotFoundError as exc:
        raise ExcerptPathError("inventoried source no longer exists") from exc
    if stat.S_ISLNK(initial_stat.st_mode):
        raise ExcerptSymlinkError("inventoried source is now a symlink")
    if not stat.S_ISREG(initial_stat.st_mode):
        raise ExcerptPathError("inventoried source is not a regular file")
    if initial_stat.st_size != artifact.byte_count:
        raise ExcerptIntegrityError(
            "inventoried source size no longer matches inventory"
        )
    if initial_stat.st_size > MAX_SOURCE_FILE_BYTES:
        raise ExcerptBoundsError(
            f"source exceeds the {MAX_SOURCE_FILE_BYTES}-byte reader bound"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise ExcerptPathError(
            "source could not be opened with no-follow semantics"
        ) from exc

    chunks: list[bytes] = []
    digest = hashlib.sha256()
    bytes_read = 0
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ExcerptPathError("opened source is not a regular file")
        if not _same_file_state(initial_stat, opened_stat):
            raise ExcerptIntegrityError("source changed before it could be read")
        while True:
            chunk = os.read(descriptor, _HASH_CHUNK_BYTES)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > MAX_SOURCE_FILE_BYTES:
                raise ExcerptBoundsError("source exceeded the reader byte bound")
            digest.update(chunk)
            chunks.append(chunk)
        final_stat = os.fstat(descriptor)
        if (
            not _same_file_state(opened_stat, final_stat)
            or bytes_read != final_stat.st_size
        ):
            raise ExcerptIntegrityError("source changed while it was being read")
    finally:
        os.close(descriptor)

    if bytes_read != artifact.byte_count or digest.hexdigest() != artifact.sha256:
        raise ExcerptIntegrityError("source no longer matches its inventoried SHA-256")
    return b"".join(chunks)


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


def _decode_and_screen(source_bytes: bytes) -> str:
    if b"\x00" in source_bytes:
        raise UnsafeSourceContentError("binary source content was rejected")
    try:
        text = source_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise UnsafeSourceContentError("source is not valid UTF-8") from exc
    if any(
        (ord(character) < 32 and character not in {"\n", "\r", "\t"})
        or ord(character) == 127
        for character in text
    ):
        raise UnsafeSourceContentError("source contains unsafe control characters")
    if _contains_secret_like_content(text):
        raise UnsafeSourceContentError(
            "secret-like source content was detected; no excerpt was returned"
        )
    return text


def _contains_secret_like_content(text: str) -> bool:
    if (
        _PRIVATE_KEY_MARKER.search(text)
        or _BEARER_CREDENTIAL.search(text)
        or _KNOWN_TOKEN_SHAPE.search(text)
    ):
        return True
    literal_matches = (
        match.group(2).strip() for match in _SECRET_LITERAL_ASSIGNMENT.finditer(text)
    )
    unquoted_matches = (
        match.group(1).strip().strip("\"'")
        for match in _SECRET_UNQUOTED_CONFIG.finditer(text)
    )
    named_matches = (
        match.group(2).strip() for match in _NAMED_SECRET_LITERAL.finditer(text)
    )
    mapping_matches = (
        match.group(2).strip() for match in _MAPPING_SECRET_LITERAL.finditer(text)
    )
    return any(
        not _is_safe_placeholder(value)
        for value in (
            *literal_matches,
            *unquoted_matches,
            *named_matches,
            *mapping_matches,
        )
    )


def _is_safe_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in _SAFE_PLACEHOLDER_VALUES:
        return True
    if _SAFE_ENV_REFERENCE.fullmatch(value.strip()) is not None:
        return True
    if (normalized.startswith("<") and normalized.endswith(">")) or set(normalized) <= {
        ".",
        "*",
        "x",
        "-",
        "_",
    }:
        return True
    return False


__all__ = [
    "ExcerptBoundsError",
    "ExcerptIntegrityError",
    "ExcerptPathError",
    "ExcerptSymlinkError",
    "MAX_EXCERPT_BYTES",
    "MAX_EXCERPT_LINES",
    "MAX_SOURCE_FILE_BYTES",
    "SourceExcerpt",
    "SourceExcerptError",
    "UnsafeSourceContentError",
    "read_source_excerpt",
]
