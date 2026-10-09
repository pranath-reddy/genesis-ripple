"""Adapter from a RIPPLe repository intake to the open research-agent protocol."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from ripple.modeling.research_contracts import (
    IDENTIFIER_PATTERN,
    ResearchEvidence,
)
from ripple.modeling.source_reader import (
    MAX_EXCERPT_BYTES,
    MAX_EXCERPT_LINES,
    ExcerptBoundsError,
    ExcerptIntegrityError,
    ExcerptPathError,
    UnsafeSourceContentError,
)

from ..schemas.repository import (
    RepositoryExcerptRequest,
    RepositoryFile,
    RepositoryIntakeManifest,
    RepositorySearchRequest,
)
from ..schemas.common import FrozenModel, SHA256_PATTERN, canonical_json_sha256
from .repository_intake import (
    RepositoryIntegrityError,
    RepositoryIntakeError,
    RepositoryPolicyError,
    load_repository_intake,
    read_repository_excerpt,
    search_repository,
)


@dataclass(frozen=True)
class _Artifact:
    artifact_id: str
    evidence_source_id: str
    kind: Literal["python_source", "training_configuration"]
    file: RepositoryFile


class RepositorySnapshotBinding(FrozenModel):
    """Explicit link between one research request and one immutable intake tree."""

    schema_version: Literal["ripple.repository-snapshot-binding.v1"] = (
        "ripple.repository-snapshot-binding.v1"
    )
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    inventory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    tree_sha256: str = Field(pattern=SHA256_PATTERN)
    manifest_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    materialization: Literal["pinned_git_tree", "local_filesystem_snapshot"]
    resolved_commit: str | None = None
    commit_pins_snapshot: bool
    snapshot_identity_authority: Literal["tree_sha256"] = "tree_sha256"

    @model_validator(mode="after")
    def _commit_authority(self) -> "RepositorySnapshotBinding":
        expected = self.materialization == "pinned_git_tree"
        if self.commit_pins_snapshot != expected:
            raise ValueError("commit authority disagrees with snapshot materialization")
        if expected and self.resolved_commit is None:
            raise ValueError("a pinned Git tree requires its resolved commit")
        return self


class IntakeRepositorySnapshot:
    """Read-only ``SafeRepositorySnapshot`` over one published intake.

    The adapter exposes stable artifact IDs instead of filesystem paths.  Every
    search and read delegates to the intake layer, which rechecks containment,
    symlinks, size, full-file SHA-256, UTF-8 validity, and credential screening.
    """

    def __init__(
        self,
        *,
        manifest: RepositoryIntakeManifest,
        intake_directory: Path,
        request_id: str,
    ) -> None:
        if not isinstance(manifest, RepositoryIntakeManifest):
            raise TypeError("manifest must be a RepositoryIntakeManifest")
        if not isinstance(intake_directory, Path) or not intake_directory.is_absolute():
            raise ValueError(
                "intake_directory must be an explicit absolute pathlib.Path"
            )
        if (
            not isinstance(request_id, str)
            or re.fullmatch(IDENTIFIER_PATTERN, request_id) is None
        ):
            raise ValueError("request_id must be a valid research-request identifier")

        root = _validate_intake_directory(intake_directory)
        persisted = load_repository_intake(root / "manifest.json")
        if persisted != manifest:
            raise ExcerptIntegrityError(
                "provided intake manifest does not match the published manifest"
            )

        artifacts = tuple(
            _artifact_for(manifest.intake_id, item) for item in manifest.files
        )
        artifact_ids = tuple(item.artifact_id for item in artifacts)
        source_ids = tuple(item.evidence_source_id for item in artifacts)
        if len(artifact_ids) != len(set(artifact_ids)) or len(source_ids) != len(
            set(source_ids)
        ):
            raise ExcerptIntegrityError(
                "derived repository artifact identities collided"
            )

        self._manifest = manifest
        self._root = root
        self._binding = RepositorySnapshotBinding(
            request_id=request_id,
            inventory_id=manifest.intake_id,
            tree_sha256=manifest.tree_sha256,
            manifest_payload_sha256=canonical_json_sha256(manifest),
            materialization=manifest.materialization,
            resolved_commit=manifest.resolved_commit,
            commit_pins_snapshot=manifest.materialization == "pinned_git_tree",
        )
        self._artifacts = artifacts
        self._by_id = {item.artifact_id: item for item in artifacts}

    @property
    def inventory_id(self) -> str:
        return self._manifest.intake_id

    @property
    def request_id(self) -> str:
        return self._binding.request_id

    @property
    def binding(self) -> RepositorySnapshotBinding:
        return self._binding

    def inspect(self) -> dict[str, object]:
        return {
            "inventory_id": self.inventory_id,
            "request_id": self.request_id,
            "repository_revision": (
                self._manifest.resolved_commit
                if self._binding.commit_pins_snapshot
                else None
            ),
            "contextual_head_commit": (
                self._manifest.resolved_commit
                if not self._binding.commit_pins_snapshot
                else None
            ),
            "repository_tree_sha256": self._manifest.tree_sha256,
            "snapshot_binding": self._binding.model_dump(mode="json"),
            "source_kind": self._manifest.source_kind,
            "materialization": self._manifest.materialization,
            "artifact_count": len(self._artifacts),
            "readable_text_artifact_count": len(self._artifacts),
            "excluded_path_count": self._manifest.excluded_entry_count,
            "inventory_complete_for_declared_scope": True,
            "source_execution_allowed": False,
            "repository_mutation_allowed": False,
            "checkpoint_loading_allowed": False,
        }

    def list_artifacts(
        self, *, offset: int, limit: int, kind: str | None = None
    ) -> dict[str, object]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("artifact offset must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("artifact page limit must be between 1 and 100")
        allowed_kinds = {
            "python_source",
            "training_configuration",
            "checkpoint_metadata",
            "checkpoint",
        }
        if kind is not None and kind not in allowed_kinds:
            raise ValueError("artifact kind is outside the snapshot allowlist")
        selected = tuple(
            artifact
            for artifact in self._artifacts
            if kind is None or artifact.kind == kind
        )
        page = selected[offset : offset + limit]
        return {
            "inventory_id": self.inventory_id,
            "offset": offset,
            "limit": limit,
            "total_matching": len(selected),
            "has_more": offset + len(page) < len(selected),
            "artifacts": [
                {
                    "artifact_id": artifact.artifact_id,
                    "evidence_source_id": artifact.evidence_source_id,
                    "kind": artifact.kind,
                    "repository_relative_path": artifact.file.path,
                    "byte_count": artifact.file.byte_count,
                    "sha256": artifact.file.sha256,
                    "readable_as_text": True,
                }
                for artifact in page
            ],
        }

    def search(
        self,
        *,
        query: str,
        artifact_ids: tuple[str, ...] | None,
        maximum_results: int,
    ) -> dict[str, object]:
        if (
            not isinstance(query, str)
            or query != query.strip()
            or not 2 <= len(query) <= 128
            or any(ord(character) < 32 or ord(character) == 127 for character in query)
        ):
            raise ValueError(
                "search query must be trimmed safe text from 2 to 128 characters"
            )
        if (
            isinstance(maximum_results, bool)
            or not isinstance(maximum_results, int)
            or not 1 <= maximum_results <= 50
        ):
            raise ValueError("maximum_results must be between 1 and 50")
        selected = self._select_artifacts(artifact_ids)

        matches: list[dict[str, object]] = []
        files_searched = 0
        for artifact in selected:
            remaining = maximum_results - len(matches)
            if remaining <= 0:
                break
            try:
                result = search_repository(
                    self._manifest,
                    intake_directory=self._root,
                    request=RepositorySearchRequest(
                        query=query,
                        path_prefix=artifact.file.path,
                        max_results=remaining,
                        max_line_characters=400,
                    ),
                )
            except RepositoryIntakeError as exc:
                _raise_snapshot_error(exc)
            files_searched += result.files_examined
            matches.extend(
                {
                    "artifact_id": artifact.artifact_id,
                    "evidence_source_id": artifact.evidence_source_id,
                    "repository_relative_path": match.path,
                    "line_number": match.line_number,
                    "preview": match.line_text.strip(),
                }
                for match in result.matches
            )
        return {
            "inventory_id": self.inventory_id,
            "query": query,
            "files_searched": files_searched,
            "maximum_results": maximum_results,
            "result_limit_reached": len(matches) >= maximum_results,
            "matches": matches,
            "search_results_are_navigation_only": True,
            "citation_rule": "read a bounded excerpt before citing a search match",
        }

    def read(
        self, *, artifact_id: str, start_line: int, end_line: int
    ) -> ResearchEvidence:
        if not isinstance(artifact_id, str):
            raise TypeError("artifact_id must be a string")
        if (
            isinstance(start_line, bool)
            or not isinstance(start_line, int)
            or isinstance(end_line, bool)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
        ):
            raise ExcerptBoundsError("source line bounds must be positive and ordered")
        if end_line - start_line + 1 > MAX_EXCERPT_LINES:
            raise ExcerptBoundsError(
                f"source excerpt cannot exceed {MAX_EXCERPT_LINES} lines"
            )
        artifact = self._by_id.get(artifact_id)
        if artifact is None:
            raise ExcerptPathError(
                "artifact ID is not present in the repository snapshot"
            )
        try:
            excerpt = read_repository_excerpt(
                self._manifest,
                intake_directory=self._root,
                request=RepositoryExcerptRequest(
                    path=artifact.file.path,
                    start_line=start_line,
                    end_line=end_line,
                ),
            )
        except RepositoryIntakeError as exc:
            _raise_snapshot_error(exc)
        encoded = excerpt.numbered_text.encode("utf-8")
        if len(encoded) > MAX_EXCERPT_BYTES:
            raise ExcerptBoundsError(
                f"rendered excerpt cannot exceed {MAX_EXCERPT_BYTES} bytes"
            )
        identity = (
            f"{self.inventory_id}\0{artifact.artifact_id}\0{start_line}\0"
            f"{end_line}\0{excerpt.excerpt_sha256}"
        ).encode("utf-8")
        return ResearchEvidence(
            evidence_id=f"evidence-{hashlib.sha256(identity).hexdigest()[:24]}",
            inventory_id=self.inventory_id,
            artifact_id=artifact.artifact_id,
            evidence_source_id=artifact.evidence_source_id,
            repository_relative_path=artifact.file.path,
            source_sha256=artifact.file.sha256,
            start_line=start_line,
            end_line=end_line,
            excerpt_sha256=excerpt.excerpt_sha256,
            excerpt_text=excerpt.numbered_text,
        )

    def _select_artifacts(
        self, artifact_ids: tuple[str, ...] | None
    ) -> tuple[_Artifact, ...]:
        if artifact_ids is None:
            return self._artifacts
        if not isinstance(artifact_ids, tuple):
            raise TypeError("artifact_ids must be a tuple or None")
        if len(artifact_ids) > 64 or len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError(
                "artifact search filter must contain at most 64 unique IDs"
            )
        if any(not isinstance(item, str) for item in artifact_ids):
            raise TypeError("artifact search filter IDs must be strings")
        unknown = set(artifact_ids) - set(self._by_id)
        if unknown:
            raise ValueError("artifact search filter references an unknown artifact ID")
        selected_ids = set(artifact_ids)
        return tuple(
            item for item in self._artifacts if item.artifact_id in selected_ids
        )


def build_intake_repository_snapshot(
    *,
    manifest: RepositoryIntakeManifest,
    intake_directory: Path,
    request_id: str,
) -> IntakeRepositorySnapshot:
    """Build the verified protocol adapter used by the open research agent."""

    return IntakeRepositorySnapshot(
        manifest=manifest,
        intake_directory=intake_directory,
        request_id=request_id,
    )


def _artifact_for(inventory_id: str, file: RepositoryFile) -> _Artifact:
    material = f"{inventory_id}\0{file.path}\0{file.sha256}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    kind: Literal["python_source", "training_configuration"] = (
        "python_source" if file.role == "source" else "training_configuration"
    )
    return _Artifact(
        artifact_id=f"artifact-{digest[:24]}",
        evidence_source_id=f"source-{digest[24:48]}",
        kind=kind,
        file=file,
    )


def _validate_intake_directory(path: Path) -> Path:
    normalized = Path(os.path.abspath(os.fspath(path)))
    current = Path(normalized.anchor)
    for part in normalized.parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise ExcerptPathError("intake directory does not exist") from exc
        if stat.S_ISLNK(mode):
            raise ExcerptPathError("intake directory contains a symlink component")
    if not stat.S_ISDIR(os.lstat(normalized).st_mode):
        raise ExcerptPathError("intake directory must be a directory")
    return normalized


def _raise_snapshot_error(error: RepositoryIntakeError) -> None:
    if isinstance(error, RepositoryIntegrityError):
        raise ExcerptIntegrityError(
            "repository snapshot integrity validation failed"
        ) from None
    if isinstance(error, RepositoryPolicyError):
        raise UnsafeSourceContentError(
            "repository snapshot read was rejected safely"
        ) from None
    raise ExcerptPathError("repository snapshot could not be read safely") from None


__all__ = [
    "IntakeRepositorySnapshot",
    "RepositorySnapshotBinding",
    "build_intake_repository_snapshot",
]
