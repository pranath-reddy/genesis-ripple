"""Safe acquisition and read-only inspection of researcher model repositories.

Repository content is treated as untrusted data.  This module never imports or
executes it, never checks out Git worktrees, never initializes submodules, and
never enables repository hooks.  Only a bounded UTF-8 code/config subset is
copied into an immutable, run-scoped snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ValidationError

from ..schemas.repository import (
    GitHttpsRepositorySource,
    LocalRepositorySource,
    RepositoryExcerpt,
    RepositoryExcerptRequest,
    RepositoryFile,
    RepositoryFileListing,
    RepositoryIntakeManifest,
    RepositoryIntakePolicy,
    RepositoryIntakeRequest,
    RepositorySearchMatch,
    RepositorySearchRequest,
    RepositorySearchResult,
    repository_tree_sha256,
)


class RepositoryIntakeError(RuntimeError):
    """Base class for a safely reportable repository-intake failure."""


class RepositoryPolicyError(RepositoryIntakeError):
    """Untrusted repository content violated the fixed intake policy."""


class RepositoryIntegrityError(RepositoryIntakeError):
    """A published snapshot no longer matches its manifest."""


class RepositoryAcquisitionError(RepositoryIntakeError):
    """A repository could not be acquired without relaxing safety controls."""


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
_SECRET_FILE_NAMES = {
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
_SECRET_SUFFIXES = {".jks", ".key", ".keystore", ".p12", ".pem", ".pfx"}
_SECRET_NAME_PATTERN = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|credentials?|private[_-]?key|secrets?)(?:$|[._-])",
    flags=re.IGNORECASE,
)
_PRIVATE_KEY_MARKER = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE
)
_BEARER_CREDENTIAL = re.compile(
    r"\bauthorization\s*[:=]\s*[\"']?bearer\s+[A-Za-z0-9._~+/=-]{12,}",
    re.IGNORECASE,
)
_KNOWN_TOKEN = re.compile(
    r"(?:\b(?:AKIA|ASIA)[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*"
    r"(?:API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|PASSWORD|PASSWD|"
    r"PRIVATE[_-]?KEY|SECRET)\s*[:=]\s*([\"'])([^\r\n\"']+)\1"
)
_SECRET_UNQUOTED = re.compile(
    r"(?im)^\s*(?:export\s+)?(?:[A-Za-z][A-Za-z0-9_]*?(?:API[_-]?KEY|"
    r"ACCESS[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|PASSWORD|PASSWD|CREDENTIAL|"
    r"PRIVATE[_-]?KEY|SECRET)|TOKEN|PASSWORD|SECRET)\s*[:=]\s*"
    r"([^\s#][^\r\n#]*)$"
)
_SECRET_MAPPING = re.compile(
    r"(?i)[\"'](?:[A-Za-z0-9_]*?(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"password|passwd|credential|private[_-]?key|secret))[\"']\s*:\s*"
    r"([\"'])([^\r\n\"']+)\1"
)
_SAFE_PLACEHOLDERS = {
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
_SAFE_ENV_REFERENCE = re.compile(
    r"(?:\$[A-Za-z_][A-Za-z0-9_]*|\$\{[A-Za-z_][A-Za-z0-9_]*\}|"
    r"os\.getenv\(\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*\)|"
    r"os\.environ\[\s*[\"'][A-Za-z_][A-Za-z0-9_]*[\"']\s*\])"
)


@dataclass(frozen=True)
class _Material:
    relative_path: str
    content: bytes
    role: Literal["source", "configuration"]
    language: str
    line_count: int


@dataclass(frozen=True)
class _Acquired:
    materials: tuple[_Material, ...]
    excluded_count: int
    source_locator: str
    requested_revision: str | None
    resolved_commit: str | None
    source_is_git: bool
    materialization: Literal["pinned_git_tree", "local_filesystem_snapshot"]


def create_repository_intake(
    request: RepositoryIntakeRequest,
    *,
    run_directory: Path,
    policy: RepositoryIntakePolicy | None = None,
) -> RepositoryIntakeManifest:
    """Acquire and publish one immutable repository snapshot.

    ``run_directory`` must already exist and must not overlap a local source.
    The final snapshot is published at ``repository-intake/repository`` only
    after every candidate file passes path, size, UTF-8, binary, and
    credential screening.
    """

    if not isinstance(request, RepositoryIntakeRequest):
        raise TypeError("request must be a RepositoryIntakeRequest")
    if not isinstance(run_directory, Path):
        raise TypeError("run_directory must be a pathlib.Path")
    effective_policy = policy or RepositoryIntakePolicy()
    run_root = _validate_existing_directory(run_directory, name="run_directory")
    final_root = run_root / "repository-intake"
    if final_root.exists() or final_root.is_symlink():
        raise RepositoryPolicyError("repository-intake output already exists")

    if isinstance(request.source, LocalRepositorySource):
        local_root = _validate_existing_directory(
            Path(request.source.path), name="local repository"
        )
        if _paths_overlap(run_root, local_root):
            raise RepositoryPolicyError(
                "run directory and local repository must not contain one another"
            )

    staging = Path(tempfile.mkdtemp(prefix=".repository-intake-", dir=run_root))
    try:
        if isinstance(request.source, GitHttpsRepositorySource):
            acquired = _acquire_https_git(request.source, staging, effective_policy)
            source_kind: Literal["git_https", "local_path"] = "git_https"
        else:
            acquired = _acquire_local(request.source, effective_policy)
            source_kind = "local_path"

        snapshot_root = staging / "repository"
        snapshot_root.mkdir(mode=0o700)
        files = _write_materials(snapshot_root, acquired.materials)
        manifest = RepositoryIntakeManifest(
            intake_id=request.intake_id,
            created_at_utc=datetime.now(timezone.utc),
            source_kind=source_kind,
            source_locator=acquired.source_locator,
            requested_revision=acquired.requested_revision,
            resolved_commit=acquired.resolved_commit,
            source_is_git=acquired.source_is_git,
            materialization=acquired.materialization,
            files=files,
            file_count=len(files),
            total_bytes=sum(item.byte_count for item in files),
            excluded_entry_count=acquired.excluded_count,
            tree_sha256=repository_tree_sha256(files),
        )
        _write_manifest(staging / "manifest.json", manifest)
        _publish_repository_tree(staging, final_root)
        shutil.rmtree(staging)
        staging = final_root
        loaded = load_repository_intake(final_root / "manifest.json")
        if loaded != manifest:
            raise RepositoryIntegrityError(
                "repository manifest changed during publication"
            )
        return loaded
    except Exception:
        if staging.exists() and staging != final_root:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_repository_tree(staging: Path, destination: Path) -> None:
    """Publish a verified snapshot without merging or replacing any path."""

    if os.path.lexists(destination):
        raise RepositoryPolicyError("repository-intake output already exists")
    try:
        destination.mkdir(mode=0o700)
        for directory, names, filenames in os.walk(
            staging,
            topdown=True,
            followlinks=False,
        ):
            names.sort()
            filenames.sort()
            source_directory = Path(directory)
            target_directory = destination / source_directory.relative_to(staging)
            for name in names:
                source = source_directory / name
                if not stat.S_ISDIR(os.lstat(source).st_mode):
                    raise RepositoryIntegrityError(
                        "repository staging tree contains an unsafe directory"
                    )
                (target_directory / name).mkdir(mode=0o700)
            for name in filenames:
                source = source_directory / name
                if not stat.S_ISREG(os.lstat(source).st_mode):
                    raise RepositoryIntegrityError(
                        "repository staging tree contains an unsafe file"
                    )
                os.link(
                    source,
                    target_directory / name,
                    follow_symlinks=False,
                )
        _freeze_tree(destination / "repository")
        (destination / "manifest.json").chmod(0o444)
        destination.chmod(0o555)
    except FileExistsError:
        raise RepositoryPolicyError("repository-intake output already exists") from None
    except Exception:
        if destination.exists() and not destination.is_symlink():
            for directory, names, _ in os.walk(destination, topdown=False):
                Path(directory).chmod(0o700)
                for name in names:
                    child = Path(directory) / name
                    if child.is_dir() and not child.is_symlink():
                        child.chmod(0o700)
            shutil.rmtree(destination, ignore_errors=True)
        raise


def load_repository_intake(path: Path) -> RepositoryIntakeManifest:
    """Load a bounded manifest from a regular, non-symlink file."""

    content = _read_regular_file(path, maximum_bytes=4 * 1024 * 1024)
    try:
        return RepositoryIntakeManifest.model_validate_json(content)
    except (ValueError, ValidationError) as exc:
        raise RepositoryIntegrityError(
            f"repository manifest is invalid ({type(exc).__name__})"
        ) from None


def list_repository_files(
    manifest: RepositoryIntakeManifest,
    *,
    offset: int = 0,
    limit: int = 50,
    role: Literal["source", "configuration"] | None = None,
) -> RepositoryFileListing:
    """List the immutable inventory without touching repository content."""

    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    matching = tuple(
        item for item in manifest.files if role is None or item.role == role
    )
    return RepositoryFileListing(
        intake_id=manifest.intake_id,
        total_matching=len(matching),
        offset=offset,
        files=matching[offset : offset + limit],
    )


def search_repository(
    manifest: RepositoryIntakeManifest,
    *,
    intake_directory: Path,
    request: RepositorySearchRequest,
) -> RepositorySearchResult:
    """Search literal text in verified snapshot files; regex execution is absent."""

    snapshot = _snapshot_root(intake_directory, manifest)
    needle = request.query if request.case_sensitive else request.query.casefold()
    matches: list[RepositorySearchMatch] = []
    examined = 0
    limit_reached = False
    for artifact in manifest.files:
        if request.path_prefix and not (
            artifact.path == request.path_prefix
            or artifact.path.startswith(request.path_prefix + "/")
        ):
            continue
        text = _verified_text(snapshot, artifact)
        examined += 1
        for line_number, line in enumerate(text.splitlines(), start=1):
            haystack = line if request.case_sensitive else line.casefold()
            if needle not in haystack:
                continue
            truncated = len(line) > request.max_line_characters
            matches.append(
                RepositorySearchMatch(
                    path=artifact.path,
                    line_number=line_number,
                    line_text=(
                        line[: request.max_line_characters] + ("…" if truncated else "")
                    ),
                    line_truncated=truncated,
                )
            )
            if len(matches) >= request.max_results:
                limit_reached = True
                break
        if limit_reached:
            break
    return RepositorySearchResult(
        intake_id=manifest.intake_id,
        query=request.query,
        matches=tuple(matches),
        result_limit_reached=limit_reached,
        files_examined=examined,
    )


def read_repository_excerpt(
    manifest: RepositoryIntakeManifest,
    *,
    intake_directory: Path,
    request: RepositoryExcerptRequest,
) -> RepositoryExcerpt:
    """Return at most 200 numbered lines from one digest-verified text file."""

    artifacts = tuple(item for item in manifest.files if item.path == request.path)
    if len(artifacts) != 1:
        raise RepositoryPolicyError("requested path is not present exactly once")
    text = _verified_text(_snapshot_root(intake_directory, manifest), artifacts[0])
    lines = text.splitlines()
    if request.end_line > len(lines):
        raise RepositoryPolicyError("requested excerpt exceeds the file line count")
    numbered = "\n".join(
        f"{line_number}: {lines[line_number - 1]}"
        for line_number in range(request.start_line, request.end_line + 1)
    )
    encoded = numbered.encode("utf-8")
    if len(encoded) > 128 * 1024:
        raise RepositoryPolicyError("numbered excerpt exceeds its byte bound")
    return RepositoryExcerpt(
        intake_id=manifest.intake_id,
        path=artifacts[0].path,
        full_file_sha256=artifacts[0].sha256,
        start_line=request.start_line,
        end_line=request.end_line,
        numbered_text=numbered,
        excerpt_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _acquire_local(
    source: LocalRepositorySource, policy: RepositoryIntakePolicy
) -> _Acquired:
    root = _validate_existing_directory(Path(source.path), name="local repository")
    materials, excluded = _scan_local_tree(root, policy)
    commit = _resolve_local_commit(root, policy)
    return _Acquired(
        materials=materials,
        excluded_count=excluded,
        source_locator=str(root),
        requested_revision=None,
        resolved_commit=commit,
        source_is_git=commit is not None,
        materialization="local_filesystem_snapshot",
    )


def _acquire_https_git(
    source: GitHttpsRepositorySource,
    staging: Path,
    policy: RepositoryIntakePolicy,
) -> _Acquired:
    bare = staging / ".git-acquisition"
    _run_git(
        ("init", "--bare", "--", str(bare)),
        timeout=policy.git_timeout_seconds,
        file_limit=policy.max_git_transfer_bytes,
    )
    _run_git(
        (
            "-C",
            str(bare),
            "fetch",
            "--no-tags",
            "--depth=1",
            "--no-recurse-submodules",
            "--",
            source.url,
            source.revision,
        ),
        timeout=policy.git_timeout_seconds,
        file_limit=policy.max_git_transfer_bytes,
        storage_root=bare,
        storage_limit=policy.max_git_transfer_bytes,
    )
    _enforce_directory_bytes(bare, policy.max_git_transfer_bytes)
    commit = (
        _run_git(
            ("-C", str(bare), "rev-parse", "--verify", "FETCH_HEAD^{commit}"),
            timeout=30,
            file_limit=1024 * 1024,
        )
        .decode("ascii")
        .strip()
    )
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None:
        raise RepositoryAcquisitionError("Git did not resolve a valid commit identity")
    tree = _run_git(
        ("-C", str(bare), "ls-tree", "-r", "-z", "-l", "--full-tree", commit),
        timeout=60,
        file_limit=16 * 1024 * 1024,
    )
    records, excluded = _parse_git_tree(tree, policy)
    materials: list[_Material] = []
    total = 0
    for path, object_id, declared_size in records:
        content = _run_git(
            ("-C", str(bare), "cat-file", "blob", object_id),
            timeout=60,
            file_limit=policy.max_file_bytes + 1,
        )
        if len(content) != declared_size:
            raise RepositoryIntegrityError("Git blob size disagrees with tree metadata")
        material = _screen_material(path, content)
        total += len(content)
        if total > policy.max_total_bytes:
            raise RepositoryPolicyError("repository text exceeds the total byte bound")
        materials.append(material)
    if not materials:
        raise RepositoryPolicyError(
            "repository contains no allowlisted code/config files"
        )
    shutil.rmtree(bare)
    return _Acquired(
        materials=tuple(materials),
        excluded_count=excluded,
        source_locator=source.url,
        requested_revision=source.revision,
        resolved_commit=commit,
        source_is_git=True,
        materialization="pinned_git_tree",
    )


def _scan_local_tree(
    root: Path, policy: RepositoryIntakePolicy
) -> tuple[tuple[_Material, ...], int]:
    materials: list[_Material] = []
    excluded = 0
    entries_seen = 0
    total = 0

    def visit(directory: Path, relative: PurePosixPath, depth: int) -> None:
        nonlocal excluded, entries_seen, total
        if depth > policy.max_depth:
            raise RepositoryPolicyError("repository exceeds the directory-depth bound")
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RepositoryAcquisitionError(
                "local repository could not be scanned"
            ) from exc
        for entry in entries:
            entries_seen += 1
            if entries_seen > policy.max_entries:
                raise RepositoryPolicyError("repository exceeds the entry-count bound")
            child_relative = relative / entry.name
            path_text = child_relative.as_posix()
            try:
                if entry.is_symlink():
                    raise RepositoryPolicyError("repository symlinks are not accepted")
                entry_stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError as exc:
                raise RepositoryIntegrityError(
                    "repository changed during intake"
                ) from exc
            if entry.name == ".git" and stat.S_ISDIR(entry_stat.st_mode):
                excluded += 1
                continue
            _validate_untrusted_path(path_text, policy)
            if _is_secret_path(child_relative):
                raise RepositoryPolicyError("repository contains a secret-shaped path")
            if any(part.startswith(".") for part in child_relative.parts):
                excluded += 1
                continue
            if stat.S_ISDIR(entry_stat.st_mode):
                visit(Path(entry.path), child_relative, depth + 1)
            elif stat.S_ISREG(entry_stat.st_mode):
                if not _is_allowlisted(child_relative, policy):
                    excluded += 1
                    continue
                # Empty package markers (most commonly ``__init__.py``) carry no
                # evidence for the research agent.  Exclude them instead of
                # rejecting an otherwise valid researcher repository.
                if entry_stat.st_size == 0:
                    excluded += 1
                    continue
                if entry_stat.st_size > policy.max_file_bytes:
                    raise RepositoryPolicyError(
                        "allowlisted source violates file-size bounds"
                    )
                content = _read_local_candidate(Path(entry.path), entry_stat, policy)
                material = _screen_material(path_text, content)
                total += len(content)
                if total > policy.max_total_bytes:
                    raise RepositoryPolicyError(
                        "repository text exceeds total byte bound"
                    )
                materials.append(material)
                if len(materials) > policy.max_files:
                    raise RepositoryPolicyError(
                        "repository exceeds the source-file bound"
                    )
            else:
                raise RepositoryPolicyError("repository contains a non-regular entry")

    visit(root, PurePosixPath(), 0)
    if not materials:
        raise RepositoryPolicyError(
            "repository contains no allowlisted code/config files"
        )
    return tuple(sorted(materials, key=lambda item: item.relative_path)), excluded


def _parse_git_tree(
    payload: bytes, policy: RepositoryIntakePolicy
) -> tuple[tuple[tuple[str, str, int], ...], int]:
    records: list[tuple[str, str, int]] = []
    excluded = 0
    entries = tuple(item for item in payload.split(b"\x00") if item)
    if len(entries) > policy.max_entries:
        raise RepositoryPolicyError("Git tree exceeds the entry-count bound")
    for raw in entries:
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, object_type, object_id, size_text = metadata.decode("ascii").split(
                " "
            )
            path = raw_path.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise RepositoryPolicyError(
                "Git tree contains invalid path metadata"
            ) from exc
        if mode in {"120000", "160000"} or object_type != "blob":
            raise RepositoryPolicyError("Git symlinks and submodules are not accepted")
        if mode not in {"100644", "100755"}:
            raise RepositoryPolicyError("Git tree contains an unsupported entry mode")
        relative = _validate_untrusted_path(path, policy)
        if _is_secret_path(relative):
            raise RepositoryPolicyError("repository contains a secret-shaped path")
        if any(part.startswith(".") for part in relative.parts):
            excluded += 1
            continue
        if not _is_allowlisted(relative, policy):
            excluded += 1
            continue
        try:
            size = int(size_text)
        except ValueError as exc:
            raise RepositoryPolicyError("Git blob lacks a bounded size") from exc
        if size == 0:
            excluded += 1
            continue
        if size < 0 or size > policy.max_file_bytes:
            raise RepositoryPolicyError(
                "allowlisted Git blob violates file-size bounds"
            )
        records.append((path, object_id, size))
        if len(records) > policy.max_files:
            raise RepositoryPolicyError("repository exceeds the source-file bound")
    return tuple(sorted(records)), excluded


def _screen_material(path: str, content: bytes) -> _Material:
    if b"\x00" in content:
        raise RepositoryPolicyError("binary content was rejected")
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RepositoryPolicyError("non-UTF-8 source content was rejected") from exc
    if any(
        (ord(character) < 32 and character not in {"\n", "\r", "\t"})
        or ord(character) == 127
        for character in text
    ):
        raise RepositoryPolicyError("source contains unsafe control characters")
    if _contains_secret(text):
        raise RepositoryPolicyError("secret-like source content was detected")
    lines = text.splitlines()
    if not lines:
        raise RepositoryPolicyError("allowlisted source must contain text")
    relative = PurePosixPath(path)
    suffix = relative.suffix.lower()
    language = _LANGUAGES.get(
        suffix, "dockerfile" if relative.name == "Dockerfile" else "text"
    )
    role: Literal["source", "configuration"] = (
        "source" if suffix in {".py", ".pyi"} else "configuration"
    )
    return _Material(path, content, role, language, len(lines))


def _contains_secret(text: str) -> bool:
    if (
        _PRIVATE_KEY_MARKER.search(text)
        or _BEARER_CREDENTIAL.search(text)
        or _KNOWN_TOKEN.search(text)
    ):
        return True
    values = [match.group(2).strip() for match in _SECRET_ASSIGNMENT.finditer(text)]
    values.extend(match.group(2).strip() for match in _SECRET_MAPPING.finditer(text))
    values.extend(
        match.group(1).strip().strip("\"'") for match in _SECRET_UNQUOTED.finditer(text)
    )
    return any(not _safe_placeholder(value) for value in values)


def _safe_placeholder(value: str) -> bool:
    normalized = value.lower().strip()
    return bool(
        not normalized
        or normalized in _SAFE_PLACEHOLDERS
        or _SAFE_ENV_REFERENCE.fullmatch(value.strip())
        or (normalized.startswith("<") and normalized.endswith(">"))
        or set(normalized) <= {".", "*", "x", "-", "_"}
    )


def _is_secret_path(path: PurePosixPath) -> bool:
    for part in path.parts:
        lowered = part.lower()
        if (
            lowered in _SECRET_FILE_NAMES
            or PurePosixPath(lowered).suffix in _SECRET_SUFFIXES
            or _SECRET_NAME_PATTERN.search(lowered)
        ):
            return True
    return False


def _is_allowlisted(path: PurePosixPath, policy: RepositoryIntakePolicy) -> bool:
    if path.suffix.lower() in policy.allowed_suffixes:
        return True
    name = path.name
    return name in policy.allowed_special_names or (
        name.startswith("requirements-") and name.endswith(".txt")
    )


def _validate_untrusted_path(
    path: str, policy: RepositoryIntakePolicy
) -> PurePosixPath:
    relative = PurePosixPath(path)
    if (
        not path
        or path != relative.as_posix()
        or relative.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or len(path) > 1024
        or len(relative.parts) > policy.max_depth + 1
    ):
        raise RepositoryPolicyError("repository path is unsafe or outside fixed bounds")
    return relative


def _write_materials(
    snapshot_root: Path, materials: tuple[_Material, ...]
) -> tuple[RepositoryFile, ...]:
    files: list[RepositoryFile] = []
    for material in sorted(materials, key=lambda item: item.relative_path):
        target = snapshot_root.joinpath(*PurePosixPath(material.relative_path).parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(material.content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        target.chmod(0o444)
        files.append(
            RepositoryFile(
                path=material.relative_path,
                role=material.role,
                language=material.language,
                byte_count=len(material.content),
                line_count=material.line_count,
                sha256=hashlib.sha256(material.content).hexdigest(),
            )
        )
    return tuple(files)


def _write_manifest(path: Path, manifest: RepositoryIntakeManifest) -> None:
    encoded = (
        json.dumps(
            manifest.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > 4 * 1024 * 1024:
        raise RepositoryPolicyError("repository manifest exceeds its byte bound")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directories.append(current)
        for name in names:
            candidate = current / name
            if candidate.is_symlink():
                raise RepositoryPolicyError("snapshot unexpectedly contains a symlink")
        for name in files:
            candidate = current / name
            if candidate.is_symlink() or not candidate.is_file():
                raise RepositoryPolicyError(
                    "snapshot unexpectedly contains an unsafe entry"
                )
            candidate.chmod(0o444)
    for directory in reversed(directories):
        directory.chmod(0o555)


def _snapshot_root(intake_directory: Path, manifest: RepositoryIntakeManifest) -> Path:
    root = _validate_existing_directory(intake_directory, name="intake directory")
    snapshot = root / manifest.snapshot_subdirectory
    return _validate_existing_directory(snapshot, name="repository snapshot")


def _verified_text(snapshot: Path, artifact: RepositoryFile) -> str:
    target = snapshot.joinpath(*PurePosixPath(artifact.path).parts)
    try:
        target.relative_to(snapshot)
    except ValueError as exc:
        raise RepositoryIntegrityError("snapshot path escaped its root") from exc
    content = _read_regular_file(target, maximum_bytes=8 * 1024 * 1024)
    if (
        len(content) != artifact.byte_count
        or hashlib.sha256(content).hexdigest() != artifact.sha256
    ):
        raise RepositoryIntegrityError("snapshot file no longer matches its manifest")
    screened = _screen_material(artifact.path, content)
    if screened.line_count != artifact.line_count:
        raise RepositoryIntegrityError(
            "snapshot line count no longer matches its manifest"
        )
    return content.decode("utf-8")


def _read_local_candidate(
    path: Path, initial: os.stat_result, policy: RepositoryIntakePolicy
) -> bytes:
    content = _read_regular_file(path, maximum_bytes=policy.max_file_bytes)
    final = os.lstat(path)
    if not _same_file(initial, final):
        raise RepositoryIntegrityError("local source changed during intake")
    return content


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        initial = os.lstat(path)
    except FileNotFoundError as exc:
        raise RepositoryIntegrityError(
            "required repository artifact is missing"
        ) from exc
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise RepositoryIntegrityError(
            "repository artifact must be a regular non-symlink file"
        )
    if initial.st_size <= 0 or initial.st_size > maximum_bytes:
        raise RepositoryPolicyError(
            "repository artifact violates its reader byte bound"
        )
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        opened = os.fstat(descriptor)
        if not _same_file(initial, opened):
            raise RepositoryIntegrityError("repository artifact changed before reading")
        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum_bytes:
                raise RepositoryPolicyError(
                    "repository artifact exceeded its reader bound"
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if not _same_file(opened, final) or count != final.st_size:
            raise RepositoryIntegrityError("repository artifact changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_existing_directory(path: Path, *, name: str) -> Path:
    if not path.is_absolute():
        raise RepositoryPolicyError(f"{name} must be an explicit absolute path")
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise RepositoryPolicyError(f"{name} does not exist") from exc
        if stat.S_ISLNK(mode):
            raise RepositoryPolicyError(f"{name} contains a symlink component")
    if not stat.S_ISDIR(os.lstat(normalized).st_mode):
        raise RepositoryPolicyError(f"{name} must be a directory")
    return normalized


def _resolve_local_commit(root: Path, policy: RepositoryIntakePolicy) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        value = (
            _run_git(
                ("-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"),
                timeout=min(policy.git_timeout_seconds, 30),
                file_limit=1024 * 1024,
            )
            .decode("ascii")
            .strip()
        )
    except (RepositoryAcquisitionError, UnicodeDecodeError):
        return None
    return value if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) else None


def _run_git(
    arguments: tuple[str, ...],
    *,
    timeout: int,
    file_limit: int,
    storage_root: Path | None = None,
    storage_limit: int | None = None,
) -> bytes:
    if (storage_root is None) != (storage_limit is None):
        raise ValueError("Git storage monitoring requires both root and limit")
    command = (
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "init.templateDir=",
        "-c",
        "credential.helper=",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "-c",
        "fetch.recurseSubmodules=false",
        "-c",
        "submodule.recurse=false",
        "-c",
        "http.followRedirects=false",
        *arguments,
    )
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/usr/bin/false",
        "SSH_ASKPASS": "/usr/bin/false",
    }
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=True,
        )
    except FileNotFoundError:
        raise RepositoryAcquisitionError("Git executable is unavailable") from None

    assert process.stdout is not None
    assert process.stderr is not None
    streams = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout
    try:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            streams.register(stream, selectors.EVENT_READ, data=name)
        while streams.get_map():
            if (
                storage_root is not None
                and storage_limit is not None
                and _directory_size_exceeds(storage_root, storage_limit)
            ):
                _terminate_process_group(process)
                raise RepositoryAcquisitionError(
                    "Git acquisition exceeded its storage budget"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                raise RepositoryAcquisitionError(
                    "Git operation exceeded its time bound"
                )
            for key, _ in streams.select(timeout=min(0.1, remaining)):
                name = key.data
                buffer = buffers[name]
                maximum_read = min(64 * 1024, file_limit - len(buffer) + 1)
                chunk = os.read(key.fileobj.fileno(), maximum_read)
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                buffer.extend(chunk)
                if len(buffer) > file_limit:
                    _terminate_process_group(process)
                    raise RepositoryAcquisitionError(
                        "Git output exceeded its fixed byte bound"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(process)
            raise RepositoryAcquisitionError("Git operation exceeded its time bound")
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            raise RepositoryAcquisitionError(
                "Git operation exceeded its time bound"
            ) from None
    finally:
        streams.close()
        process.stdout.close()
        process.stderr.close()

    if return_code != 0:
        raise RepositoryAcquisitionError(
            f"Git operation failed safely with exit status {return_code}"
        )
    return bytes(buffers["stdout"])


def _directory_size_exceeds(root: Path, maximum: int) -> bool:
    total = 0
    if not root.exists():
        return False
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            candidate = current / name
            if candidate.is_symlink():
                raise RepositoryAcquisitionError(
                    "Git acquisition created an unexpected symlink"
                )
        for name in files:
            candidate = current / name
            try:
                state = candidate.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
                raise RepositoryAcquisitionError(
                    "Git acquisition created an unsafe entry"
                )
            total += state.st_size
            if total > maximum:
                return True
    return False


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop a session-owned Git process safely from any calling thread."""

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (AttributeError, OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _enforce_directory_bytes(root: Path, maximum: int) -> None:
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            if (current / name).is_symlink():
                raise RepositoryPolicyError(
                    "Git acquisition produced an unexpected symlink"
                )
        for name in files:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise RepositoryPolicyError("Git acquisition produced an unsafe entry")
            total += path.stat().st_size
            if total > maximum:
                raise RepositoryPolicyError("Git acquisition exceeds its storage bound")


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
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
    "RepositoryAcquisitionError",
    "RepositoryIntegrityError",
    "RepositoryIntakeError",
    "RepositoryPolicyError",
    "create_repository_intake",
    "list_repository_files",
    "load_repository_intake",
    "read_repository_excerpt",
    "search_repository",
]
