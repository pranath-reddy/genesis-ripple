"""Bounded SSH/Rsync executor for the offline Python 3.12 worker."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Sequence

from ..schemas.campaign import SourceSyncEvidence
from ..schemas.remote import RemoteCommandResult, RemoteWorkerSettings
from ..paths import checked_real_directory, checked_real_file


class RemoteExecutionError(RuntimeError):
    pass


_OPERATIONS = {"environment", "simulate", "build-dataset", "train", "evaluate"}
_SOURCE_COMPONENTS = {"ripple_scientist", "slsim", "jaxtronomy"}


@dataclass(frozen=True)
class SourceTreeDigest:
    component: Literal["ripple_scientist", "slsim", "jaxtronomy"]
    remote_relative_path: str
    sha256: str
    regular_file_count: int


@dataclass(frozen=True)
class VerifiedRemoteSources:
    components: tuple[SourceTreeDigest, ...]
    package_initializer_sha256: str
    source_manifest_sha256: str
    source_root: str


_REMOTE_DIRECTORY_SETUP = r"""
import os, stat, sys

for raw in sys.argv[1:]:
    path = os.path.abspath(raw)
    current = os.path.sep
    for part in path.split(os.path.sep)[1:]:
        current = os.path.join(current, part)
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o700)
            details = os.lstat(current)
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise SystemExit(40)
"""


_REMOTE_SOURCE_VERIFIER = r"""
import base64, hashlib, json, os, stat, sys

root = os.path.abspath(sys.argv[1])
expected = json.loads(base64.b64decode(sys.argv[2]).decode("utf-8"))
quiet = sys.argv[3] == "1"

def check_chain(path):
    path = os.path.abspath(path)
    current = os.path.sep
    for part in path.split(os.path.sep)[1:]:
        current = os.path.join(current, part)
        details = os.lstat(current)
        if stat.S_ISLNK(details.st_mode):
            raise RuntimeError("symlinked source path")
    return path

def read_regular(path):
    path = check_chain(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise RuntimeError("non-regular source file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(details.st_size + 1)
        if len(payload) != details.st_size:
            raise RuntimeError("source changed while hashing")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)

def tree_digest(path):
    path = check_chain(path)
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        raise RuntimeError("source tree is not a directory")
    files = []
    for directory, names, filenames in os.walk(path, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        for name in names:
            candidate = os.path.join(directory, name)
            details = os.lstat(candidate)
            if name == "__pycache__" or stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise RuntimeError("unsafe or stale source directory")
        for name in filenames:
            if name.endswith(".pyc"):
                raise RuntimeError("stale bytecode in source tree")
            candidate = os.path.join(directory, name)
            if not stat.S_ISREG(os.lstat(candidate).st_mode):
                raise RuntimeError("unsafe source entry")
            files.append(candidate)
    files.sort(key=lambda value: os.path.relpath(value, path).replace(os.sep, "/"))
    digest = hashlib.sha256()
    for filename in files:
        relative = os.path.relpath(filename, path).replace(os.sep, "/").encode("utf-8")
        payload = read_regular(filename)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest(), len(files)

components = []
for item in sorted(expected["components"], key=lambda value: value["component"]):
    relative = item["remote_relative_path"]
    candidate = os.path.abspath(os.path.join(root, relative))
    if os.path.commonpath((root, candidate)) != root:
        raise RuntimeError("source component escaped remote root")
    digest, count = tree_digest(candidate)
    if digest != item["sha256"] or count != item["regular_file_count"]:
        raise RuntimeError("remote source tree differs from expected bytes")
    components.append({
        "component": item["component"],
        "remote_relative_path": relative,
        "sha256": digest,
        "regular_file_count": count,
    })

initializer = hashlib.sha256(read_regular(os.path.join(root, "ripple", "__init__.py"))).hexdigest()
if initializer != expected["package_initializer_sha256"]:
    raise RuntimeError("remote package initializer differs from expected bytes")
manifest = {"components": components, "package_initializer_sha256": initializer}
encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
manifest_sha256 = hashlib.sha256(encoded).hexdigest()
if manifest_sha256 != expected["source_manifest_sha256"]:
    raise RuntimeError("remote source manifest digest mismatch")
if not quiet:
    print(json.dumps({**manifest, "source_manifest_sha256": manifest_sha256}, sort_keys=True, separators=(",", ":")))
"""


_REMOTE_OUTPUT_SCANNER = r"""
import hashlib, json, os, stat, sys

root = os.path.abspath(sys.argv[1])
current = os.path.sep
for part in root.split(os.path.sep)[1:]:
    current = os.path.join(current, part)
    details = os.lstat(current)
    if stat.S_ISLNK(details.st_mode):
        raise SystemExit(41)

files = []
details = os.lstat(root)
if stat.S_ISREG(details.st_mode):
    kind = "file"
    files = [(root, ".")]
elif stat.S_ISDIR(details.st_mode):
    kind = "directory"
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        for name in names:
            child = os.path.join(directory, name)
            mode = os.lstat(child).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SystemExit(42)
        for name in filenames:
            child = os.path.join(directory, name)
            if not stat.S_ISREG(os.lstat(child).st_mode):
                raise SystemExit(43)
            files.append((child, os.path.relpath(child, root).replace(os.sep, "/")))
else:
    raise SystemExit(44)

files.sort(key=lambda value: value[1])
digest = hashlib.sha256()
byte_count = 0
file_evidence = []
for filename, relative_text in files:
    relative = relative_text.encode("utf-8")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(filename, flags)
    try:
        state = os.fstat(descriptor)
        if not stat.S_ISREG(state.st_mode):
            raise SystemExit(45)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(state.st_size + 1)
        if len(payload) != state.st_size:
            raise SystemExit(46)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    byte_count += len(payload)
    file_evidence.append({
        "relative_path": relative_text,
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
print(json.dumps({"kind": kind, "byte_count": byte_count, "regular_file_count": len(files), "sha256": digest.hexdigest(), "files": file_evidence}, sort_keys=True, separators=(",", ":")))
"""


_REMOTE_SOURCE_STATUS = r"""
import json, os, stat, sys

path = os.path.abspath(sys.argv[1])
current = os.path.sep
for part in path.split(os.path.sep)[1:-1]:
    current = os.path.join(current, part)
    details = os.lstat(current)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise SystemExit(55)
try:
    details = os.lstat(path)
except FileNotFoundError:
    print(json.dumps({"status": "missing"}, separators=(",", ":")))
    raise SystemExit(0)
if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
    raise SystemExit(56)
print(json.dumps({"status": "directory"}, separators=(",", ":")))
"""


_REMOTE_SOURCE_PUBLISH = r"""
import os, shutil, stat, sys

source = os.path.abspath(sys.argv[1])
target = os.path.abspath(sys.argv[2])
created = False
try:
    os.mkdir(target, 0o700)
    created = True
except FileExistsError:
    raise SystemExit(61)
try:
    for directory, names, filenames in os.walk(source, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        relative = os.path.relpath(directory, source)
        target_directory = target if relative == "." else os.path.join(target, relative)
        for name in names:
            child = os.path.join(directory, name)
            if not stat.S_ISDIR(os.lstat(child).st_mode):
                raise RuntimeError("unsafe source directory")
            os.mkdir(os.path.join(target_directory, name), 0o700)
        for name in filenames:
            child = os.path.join(directory, name)
            if not stat.S_ISREG(os.lstat(child).st_mode):
                raise RuntimeError("unsafe source file")
            published = os.path.join(target_directory, name)
            os.link(child, published, follow_symlinks=False)
            os.chmod(published, 0o400)
    for directory, names, _ in os.walk(target, topdown=False, followlinks=False):
        for name in names:
            os.chmod(os.path.join(directory, name), 0o500)
        os.chmod(directory, 0o500)
except Exception:
    if created:
        shutil.rmtree(target, ignore_errors=True)
    raise
"""


_REMOTE_SOURCE_CLEANUP = r"""
import os, shutil, stat, sys

staging_parent = os.path.abspath(sys.argv[1])
target = os.path.abspath(sys.argv[2])
if os.path.commonpath((staging_parent, target)) != staging_parent or target == staging_parent:
    raise SystemExit(65)
details = os.lstat(target)
if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
    raise SystemExit(66)
shutil.rmtree(target)
"""


_REMOTE_TARGET_CHECK = r"""
import os, stat, sys

target = os.path.abspath(sys.argv[1])
mode = sys.argv[2]
current = os.path.sep
parts = target.split(os.path.sep)[1:]
for part in parts[:-1]:
    current = os.path.join(current, part)
    details = os.lstat(current)
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise SystemExit(50)
try:
    details = os.lstat(target)
except FileNotFoundError:
    raise SystemExit(0)
if mode == "missing":
    raise SystemExit(51)
if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
    raise SystemExit(52)
"""


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 1800,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=os.environ.copy(),
    )


def _ssh_prefix(settings: RemoteWorkerSettings) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={settings.connect_timeout_seconds}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        settings.host,
    ]


def _source_manifest_payload(
    components: Sequence[SourceTreeDigest],
    package_initializer_sha256: str,
) -> dict[str, object]:
    return {
        "components": [
            {
                "component": item.component,
                "remote_relative_path": item.remote_relative_path,
                "sha256": item.sha256,
                "regular_file_count": item.regular_file_count,
            }
            for item in sorted(components, key=lambda value: value.component)
        ],
        "package_initializer_sha256": package_initializer_sha256,
    }


def _canonical_sha256(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_expectation_bytes(
    components: Sequence[SourceTreeDigest],
    package_initializer_sha256: str,
) -> tuple[bytes, str]:
    manifest = _source_manifest_payload(components, package_initializer_sha256)
    manifest_sha256 = _canonical_sha256(manifest)
    expectation = {**manifest, "source_manifest_sha256": manifest_sha256}
    encoded = json.dumps(
        expectation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return encoded, manifest_sha256


def _ensure_remote_directories(
    settings: RemoteWorkerSettings,
    directories: Sequence[str],
) -> None:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_DIRECTORY_SETUP,
        *directories,
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError(
            "remote worker directories are unavailable or contain symlinks"
        )


def _check_remote_file_target(
    *,
    settings: RemoteWorkerSettings,
    target: str,
    require_missing: bool,
) -> None:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_TARGET_CHECK,
        target,
        "missing" if require_missing else "regular-or-missing",
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError("remote file target is unsafe or already exists")


def _source_verifier_command(
    *,
    settings: RemoteWorkerSettings,
    source_root: str,
    components: Sequence[SourceTreeDigest],
    package_initializer_sha256: str,
    quiet: bool,
) -> tuple[list[str], str]:
    encoded, manifest_sha256 = _source_expectation_bytes(
        components,
        package_initializer_sha256,
    )
    return (
        [
            settings.python,
            "-I",
            "-S",
            "-B",
            "-c",
            _REMOTE_SOURCE_VERIFIER,
            source_root,
            base64.b64encode(encoded).decode("ascii"),
            "1" if quiet else "0",
        ],
        manifest_sha256,
    )


def _verify_remote_sources(
    *,
    settings: RemoteWorkerSettings,
    source_root: str,
    components: Sequence[SourceTreeDigest],
    package_initializer_sha256: str,
) -> VerifiedRemoteSources:
    command, manifest_sha256 = _source_verifier_command(
        settings=settings,
        source_root=source_root,
        components=components,
        package_initializer_sha256=package_initializer_sha256,
        quiet=False,
    )
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError(
            "remote worker source differs from the locally verified source"
        )
    try:
        parsed = json.loads(completed.stdout)
        observed_components = tuple(
            SourceTreeDigest(
                component=item["component"],
                remote_relative_path=item["remote_relative_path"],
                sha256=item["sha256"],
                regular_file_count=item["regular_file_count"],
            )
            for item in parsed["components"]
        )
        result = VerifiedRemoteSources(
            components=observed_components,
            package_initializer_sha256=parsed["package_initializer_sha256"],
            source_manifest_sha256=parsed["source_manifest_sha256"],
            source_root=source_root,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RemoteExecutionError(
            "remote source verifier returned an invalid result"
        ) from None
    expected = tuple(sorted(components, key=lambda value: value.component))
    observed = tuple(sorted(result.components, key=lambda value: value.component))
    if (
        observed != expected
        or result.package_initializer_sha256 != package_initializer_sha256
        or result.source_manifest_sha256 != manifest_sha256
    ):
        raise RemoteExecutionError(
            "remote source verification did not match expectations"
        )
    return result


def _remote_source_status(
    *, settings: RemoteWorkerSettings, source_root: str
) -> Literal["missing", "directory"]:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_SOURCE_STATUS,
        source_root,
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError("remote source deployment path is unsafe")
    try:
        status = json.loads(completed.stdout)["status"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise RemoteExecutionError("remote source status was invalid") from None
    if status not in {"missing", "directory"}:
        raise RemoteExecutionError("remote source status was invalid")
    return status


def _publish_remote_source(
    *,
    settings: RemoteWorkerSettings,
    staging_root: str,
    source_root: str,
) -> bool:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_SOURCE_PUBLISH,
        staging_root,
        source_root,
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode == 61:
        return False
    if completed.returncode != 0:
        raise RemoteExecutionError("remote source snapshot could not be published")
    return True


def _cleanup_remote_source_staging(
    *,
    settings: RemoteWorkerSettings,
    staging_parent: str,
    staging_root: str,
) -> None:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_SOURCE_CLEANUP,
        staging_parent,
        staging_root,
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError("remote source staging could not be removed safely")


def _source_components_from_evidence(
    evidence: SourceSyncEvidence,
) -> tuple[SourceTreeDigest, ...]:
    return tuple(
        SourceTreeDigest(
            component=item.component,
            remote_relative_path=item.remote_relative_path,
            sha256=item.remote_tree_sha256,
            regular_file_count=item.remote_regular_file_count,
        )
        for item in evidence.components
    )


def _scan_regular_tree(root: Path) -> dict[str, object]:
    """Measure a local regular file or tree without following symlinks."""

    root = Path(root)
    state = os.lstat(root)
    if stat.S_ISREG(state.st_mode):
        kind = "file"
        files = ((root, "."),)
    elif stat.S_ISDIR(state.st_mode):
        kind = "directory"
        collected: list[tuple[Path, str]] = []
        for directory, names, filenames in os.walk(
            root, topdown=True, followlinks=False
        ):
            names.sort()
            filenames.sort()
            directory_path = Path(directory)
            for name in names:
                child = directory_path / name
                child_state = os.lstat(child)
                if stat.S_ISLNK(child_state.st_mode) or not stat.S_ISDIR(
                    child_state.st_mode
                ):
                    raise RemoteExecutionError(
                        "copied output contains an unsafe directory"
                    )
            for name in filenames:
                child = directory_path / name
                if not stat.S_ISREG(os.lstat(child).st_mode):
                    raise RemoteExecutionError("copied output contains an unsafe file")
                collected.append((child, child.relative_to(root).as_posix()))
        files = tuple(sorted(collected, key=lambda value: value[1]))
    else:
        raise RemoteExecutionError("copied output is not a regular file or directory")

    digest = hashlib.sha256()
    byte_count = 0
    file_evidence: list[dict[str, object]] = []
    for path, relative_text in files:
        relative = relative_text.encode("utf-8")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise RemoteExecutionError("copied output changed type while hashing")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                payload = handle.read(details.st_size + 1)
            if len(payload) != details.st_size:
                raise RemoteExecutionError("copied output changed while hashing")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        byte_count += len(payload)
        file_evidence.append(
            {
                "relative_path": relative_text,
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return {
        "kind": kind,
        "byte_count": byte_count,
        "regular_file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": file_evidence,
    }


def _publish_verified_tree(source: Path, target: Path) -> None:
    """Publish verified regular files without replacing an existing path."""

    source_state = os.lstat(source)
    try:
        if stat.S_ISREG(source_state.st_mode):
            os.link(source, target, follow_symlinks=False)
            target.chmod(0o400)
            return
        if not stat.S_ISDIR(source_state.st_mode):
            raise RemoteExecutionError("staged output changed type before publication")
        target.mkdir(mode=0o700)
        for directory, names, filenames in os.walk(
            source, topdown=True, followlinks=False
        ):
            names.sort()
            filenames.sort()
            source_directory = Path(directory)
            relative_directory = source_directory.relative_to(source)
            target_directory = target / relative_directory
            for name in names:
                child = source_directory / name
                if not stat.S_ISDIR(os.lstat(child).st_mode):
                    raise RemoteExecutionError(
                        "staged output changed type before publication"
                    )
                (target_directory / name).mkdir(mode=0o700)
            for name in filenames:
                child = source_directory / name
                if not stat.S_ISREG(os.lstat(child).st_mode):
                    raise RemoteExecutionError(
                        "staged output changed type before publication"
                    )
                published = target_directory / name
                os.link(child, published, follow_symlinks=False)
                published.chmod(0o400)
    except FileExistsError:
        raise RemoteExecutionError("remote output destination already exists") from None
    except Exception:
        if os.path.lexists(target):
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        raise


def _remote_output_measurement(
    *,
    settings: RemoteWorkerSettings,
    remote_path: str,
) -> dict[str, object]:
    command = [
        settings.python,
        "-I",
        "-S",
        "-B",
        "-c",
        _REMOTE_OUTPUT_SCANNER,
        remote_path,
    ]
    completed = _run(_ssh_prefix(settings) + [shlex.join(command)])
    if completed.returncode != 0:
        raise RemoteExecutionError("remote output contains an unsafe filesystem entry")
    try:
        parsed = json.loads(completed.stdout)
        kind = parsed["kind"]
        byte_count = parsed["byte_count"]
        regular_file_count = parsed["regular_file_count"]
        sha256 = parsed["sha256"]
        files = parsed["files"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise RemoteExecutionError(
            "remote output scanner returned invalid evidence"
        ) from None
    if (
        kind not in {"file", "directory"}
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
        or isinstance(regular_file_count, bool)
        or not isinstance(regular_file_count, int)
        or regular_file_count < 1
        or regular_file_count > 10_000
        or not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or not isinstance(files, list)
        or len(files) != regular_file_count
    ):
        raise RemoteExecutionError("remote output scanner returned invalid evidence")
    normalized_files: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    observed_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise RemoteExecutionError(
                "remote output scanner returned invalid evidence"
            )
        relative_path = item.get("relative_path")
        file_bytes = item.get("byte_count")
        file_sha256 = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or ".." in PurePosixPath(relative_path).parts
            or PurePosixPath(relative_path).as_posix() != relative_path
            or relative_path in observed_paths
            or any(
                character
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-/"
                for character in relative_path
            )
            or isinstance(file_bytes, bool)
            or not isinstance(file_bytes, int)
            or file_bytes < 0
            or not isinstance(file_sha256, str)
            or len(file_sha256) != 64
            or any(character not in "0123456789abcdef" for character in file_sha256)
        ):
            raise RemoteExecutionError(
                "remote output scanner returned invalid evidence"
            )
        if kind == "file" and relative_path != ".":
            raise RemoteExecutionError(
                "remote output scanner returned invalid evidence"
            )
        if kind == "directory" and relative_path == ".":
            raise RemoteExecutionError(
                "remote output scanner returned invalid evidence"
            )
        observed_paths.add(relative_path)
        observed_bytes += file_bytes
        normalized_files.append(
            {
                "relative_path": relative_path,
                "byte_count": file_bytes,
                "sha256": file_sha256,
            }
        )
    if observed_bytes != byte_count:
        raise RemoteExecutionError("remote output scanner returned invalid evidence")
    return {
        "kind": kind,
        "byte_count": byte_count,
        "regular_file_count": regular_file_count,
        "sha256": sha256,
        "files": normalized_files,
    }


def _transfer_measured_remote_output(
    *,
    settings: RemoteWorkerSettings,
    remote_path: str,
    staged_target: Path,
    measurement: dict[str, object],
) -> None:
    """Transfer each measured file with a per-file cap whose sum is bounded."""

    kind = measurement["kind"]
    files = measurement["files"]
    if not isinstance(files, list):
        raise RemoteExecutionError("remote output measurement lost its file manifest")
    if kind == "directory":
        staged_target.mkdir(mode=0o700)
    for item in files:
        if not isinstance(item, dict):
            raise RemoteExecutionError("remote output measurement is malformed")
        relative_path = item["relative_path"]
        byte_count = item["byte_count"]
        if not isinstance(relative_path, str) or not isinstance(byte_count, int):
            raise RemoteExecutionError("remote output measurement is malformed")
        if kind == "file":
            remote_file = remote_path
            local_file = staged_target
        else:
            remote_file = f"{remote_path}/{relative_path}"
            local_file = staged_target.joinpath(*PurePosixPath(relative_path).parts)
            local_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if byte_count == 0:
            descriptor = os.open(
                local_file,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            os.close(descriptor)
            continue
        transfer = _run(
            [
                "rsync",
                "-tzc",
                f"--max-size={byte_count}",
                f"{settings.host}:{remote_file}",
                str(local_file),
            ]
        )
        if transfer.returncode != 0:
            raise RemoteExecutionError("remote output transfer failed")


def sync_worker_source(
    *,
    local_repository: str | os.PathLike[str],
    settings: RemoteWorkerSettings,
    expected_components: Sequence[SourceTreeDigest],
    expected_package_initializer_sha256: str,
) -> VerifiedRemoteSources:
    """Copy only executable worker sources; credentials and local outputs are excluded."""

    repository = checked_real_directory(local_repository)
    try:
        checked_real_directory(repository / "ripple" / "scientist")
        checked_real_file(repository / "ripple" / "__init__.py")
        checked_real_directory(
            repository / "ripple" / "scientist" / "vendor" / "slsim" / "slsim"
        )
        checked_real_directory(
            repository / "ripple" / "scientist" / "vendor" / "JAXtronomy" / "jaxtronomy"
        )
    except ValueError:
        raise RemoteExecutionError("worker source allowlist is incomplete or unsafe")
    if {item.component for item in expected_components} != _SOURCE_COMPONENTS or len(
        expected_components
    ) != len(_SOURCE_COMPONENTS):
        raise RemoteExecutionError("worker source expectation is incomplete")
    _, source_manifest_sha256 = _source_expectation_bytes(
        expected_components,
        expected_package_initializer_sha256,
    )
    sources_parent = f"{settings.remote_root}/sources"
    staging_parent = f"{settings.remote_root}/source-staging"
    source_root = f"{sources_parent}/{source_manifest_sha256}"
    _ensure_remote_directories(
        settings,
        (
            settings.remote_root,
            sources_parent,
            staging_parent,
            f"{settings.remote_root}/runs",
        ),
    )
    if _remote_source_status(settings=settings, source_root=source_root) == "directory":
        return _verify_remote_sources(
            settings=settings,
            source_root=source_root,
            components=expected_components,
            package_initializer_sha256=expected_package_initializer_sha256,
        )

    staging_root = f"{staging_parent}/{source_manifest_sha256}-{uuid.uuid4().hex}"
    _ensure_remote_directories(
        settings,
        (
            staging_root,
            f"{staging_root}/ripple",
            f"{staging_root}/ripple/scientist",
        ),
    )
    try:
        source = repository / "ripple" / "scientist"
        transfer = _run(
            [
                "rsync",
                "-rtzc",
                "--delete",
                "--delete-delay",
                "--delete-excluded",
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                f"{source}/",
                f"{settings.host}:{staging_root}/ripple/scientist/",
            ]
        )
        if transfer.returncode != 0:
            raise RemoteExecutionError("worker source transfer failed")

        remote_initializer = f"{staging_root}/ripple/__init__.py"
        _check_remote_file_target(
            settings=settings,
            target=remote_initializer,
            require_missing=True,
        )
        package_init = _run(
            [
                "rsync",
                "-tzc",
                str(repository / "ripple" / "__init__.py"),
                f"{settings.host}:{remote_initializer}",
            ]
        )
        if package_init.returncode != 0:
            raise RemoteExecutionError("worker package initializer transfer failed")
        _verify_remote_sources(
            settings=settings,
            source_root=staging_root,
            components=expected_components,
            package_initializer_sha256=expected_package_initializer_sha256,
        )
        _publish_remote_source(
            settings=settings,
            staging_root=staging_root,
            source_root=source_root,
        )
    finally:
        _cleanup_remote_source_staging(
            settings=settings,
            staging_parent=staging_parent,
            staging_root=staging_root,
        )
    return _verify_remote_sources(
        settings=settings,
        source_root=source_root,
        components=expected_components,
        package_initializer_sha256=expected_package_initializer_sha256,
    )


def run_remote_worker(
    *,
    settings: RemoteWorkerSettings,
    operation: Literal["environment", "simulate", "build-dataset", "train", "evaluate"],
    arguments: Sequence[str] = (),
    execution_timeout_seconds: int | None = None,
    source_sync: SourceSyncEvidence,
) -> RemoteCommandResult:
    if operation not in _OPERATIONS:
        raise RemoteExecutionError("remote worker operation is not allowlisted")
    if any("\x00" in item or "\n" in item or "\r" in item for item in arguments):
        raise RemoteExecutionError("remote worker arguments contain control characters")
    if (
        source_sync.host != settings.host
        or source_sync.remote_root != settings.remote_root
    ):
        raise RemoteExecutionError("source-sync evidence belongs to another worker")
    if execution_timeout_seconds is not None:
        if (
            isinstance(execution_timeout_seconds, bool)
            or not isinstance(execution_timeout_seconds, int)
            or not 1 <= execution_timeout_seconds <= 31_536_000
        ):
            raise RemoteExecutionError("remote execution timeout is outside its bound")
        if operation not in {"simulate", "train", "evaluate"}:
            raise RemoteExecutionError(
                "execution timeouts are only supported for bounded long operations"
            )
    source_root = source_sync.remote_source_root
    python_path = (
        f"{source_root}:"
        f"{source_root}/ripple/scientist/vendor/slsim:"
        f"{source_root}/ripple/scientist/vendor/JAXtronomy"
    )
    command = [
        "env",
        f"PYTHONPATH={python_path}",
        "PYTHONDONTWRITEBYTECODE=1",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        f"RIPPLE_VERIFIED_SOURCE_MANIFEST_SHA256={source_sync.source_manifest_sha256}",
        settings.python,
        "-P",
        "-B",
        "-m",
        "ripple.scientist.worker",
        operation,
        *arguments,
    ]
    if execution_timeout_seconds is not None:
        command = [
            "timeout",
            "--signal=KILL",
            "--kill-after=5s",
            f"{execution_timeout_seconds}s",
            *command,
        ]
    local_timeout = (
        1800
        if execution_timeout_seconds is None
        else execution_timeout_seconds + settings.connect_timeout_seconds + 30
    )
    verifier, manifest_sha256 = _source_verifier_command(
        settings=settings,
        source_root=source_root,
        components=_source_components_from_evidence(source_sync),
        package_initializer_sha256=source_sync.remote_package_initializer_sha256,
        quiet=True,
    )
    if manifest_sha256 != source_sync.source_manifest_sha256:
        raise RemoteExecutionError(
            "source-sync evidence has an invalid manifest digest"
        )
    verification_marker = f"RIPPLE_SOURCE_VERIFIED:{manifest_sha256}"
    verified_worker_command = (
        f"cd {shlex.quote(source_root)}"
        f" && {shlex.join(verifier)} >/dev/null"
        f" && printf '%s\\n' {shlex.quote(verification_marker)} >&2"
        f" && exec {shlex.join(command)}"
    )
    started = time.monotonic()
    completed = _run(
        _ssh_prefix(settings) + [shlex.join(["sh", "-c", verified_worker_command])],
        timeout_seconds=local_timeout,
    )
    elapsed_seconds = time.monotonic() - started
    source_verification_succeeded = verification_marker in completed.stderr
    stdout = completed.stdout[-50_000:]
    stderr = completed.stderr[-50_000:]
    return RemoteCommandResult(
        operation=operation,
        exit_code=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_sha256=hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        elapsed_seconds=elapsed_seconds,
        expected_source_manifest_sha256=source_sync.source_manifest_sha256,
        verified_source_manifest_sha256=(
            source_sync.source_manifest_sha256
            if source_verification_succeeded
            else None
        ),
        source_verification_succeeded=source_verification_succeeded,
        succeeded=completed.returncode == 0 and source_verification_succeeded,
    )


def copy_to_remote_run(
    *,
    local_path: str | os.PathLike[str],
    remote_relative_path: str,
    settings: RemoteWorkerSettings,
) -> str:
    try:
        source = checked_real_file(local_path)
    except ValueError:
        raise RemoteExecutionError("local job input must be a real file") from None
    relative = Path(remote_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RemoteExecutionError("remote job path must be normalized and relative")
    destination = f"{settings.remote_root}/runs/{relative.as_posix()}"
    parent = str(Path(destination).parent)
    _ensure_remote_directories(settings, (parent,))
    _check_remote_file_target(
        settings=settings,
        target=destination,
        require_missing=True,
    )
    transfer = _run(
        [
            "rsync",
            "-tzc",
            str(source),
            f"{settings.host}:{destination}",
        ]
    )
    if transfer.returncode != 0:
        raise RemoteExecutionError("remote job input transfer failed")
    local_measurement = _scan_regular_tree(source)
    remote_measurement = _remote_output_measurement(
        settings=settings,
        remote_path=destination,
    )
    if remote_measurement != local_measurement:
        raise RemoteExecutionError("remote job input failed integrity verification")
    return destination


def copy_remote_output(
    *,
    remote_path: str,
    local_parent: str | os.PathLike[str],
    settings: RemoteWorkerSettings,
    maximum_bytes: int,
) -> Path:
    allowed_prefix = f"{settings.remote_root}/runs/"
    normalized_remote = PurePosixPath(remote_path)
    if (
        not remote_path.startswith(allowed_prefix)
        or not normalized_remote.is_absolute()
        or normalized_remote.as_posix() != remote_path
        or ".." in normalized_remote.parts
    ):
        raise RemoteExecutionError("remote output is outside the dedicated run tree")
    if (
        isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or maximum_bytes < 1
    ):
        raise RemoteExecutionError(
            "remote output has no remaining local storage budget"
        )
    destination = checked_real_directory(local_parent, create=True)
    target = destination / normalized_remote.name
    if os.path.lexists(target):
        raise RemoteExecutionError("remote output destination already exists")
    remote_measurement = _remote_output_measurement(
        settings=settings,
        remote_path=remote_path,
    )
    if int(remote_measurement["byte_count"]) > maximum_bytes:
        raise RemoteExecutionError("remote output exceeds the remaining storage budget")
    staging_root = Path(tempfile.mkdtemp(prefix=".remote-transfer-", dir=destination))
    staging_root.chmod(0o700)
    staged_target = staging_root / normalized_remote.name
    try:
        _transfer_measured_remote_output(
            settings=settings,
            remote_path=remote_path,
            staged_target=staged_target,
            measurement=remote_measurement,
        )
        staged_entries = tuple(staging_root.iterdir())
        if staged_entries != (staged_target,):
            raise RemoteExecutionError(
                "remote output transfer produced an invalid tree"
            )
        local_measurement = _scan_regular_tree(staged_target)
        if local_measurement != remote_measurement:
            raise RemoteExecutionError(
                "copied output does not match its remote integrity measurement"
            )
        if int(local_measurement["byte_count"]) > maximum_bytes:
            raise RemoteExecutionError(
                "copied output exceeded the remaining storage budget"
            )
        if (
            _remote_output_measurement(
                settings=settings,
                remote_path=remote_path,
            )
            != remote_measurement
        ):
            raise RemoteExecutionError("remote output changed during transfer")
        _publish_verified_tree(staged_target, target)
        try:
            if _scan_regular_tree(target) != local_measurement:
                raise RemoteExecutionError(
                    "published output failed integrity verification"
                )
        except Exception:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            elif os.path.lexists(target):
                target.unlink()
            raise
        return target
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
