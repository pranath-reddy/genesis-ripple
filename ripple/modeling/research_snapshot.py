"""Read-only repository snapshot interface for open-ended model research.

The language model never receives a filesystem path.  It may address only
artifact IDs already present in a :class:`SourceInventory`.  Every search and
read rechecks containment, symlinks, file identity, size, full-file SHA-256,
UTF-8 validity, and conservative credential screening before returning text.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

from .onboarding import SourceArtifact, SourceInventory
from .research_contracts import ResearchEvidence
from .source_reader import (
    MAX_EXCERPT_BYTES,
    MAX_EXCERPT_LINES,
    MAX_SOURCE_FILE_BYTES,
    ExcerptBoundsError,
    ExcerptIntegrityError,
    ExcerptPathError,
    SourceExcerptError,
    UnsafeSourceContentError,
    _decode_and_screen,
    _read_and_verify_file,
    _target_for_artifact,
    _validate_repository_root,
)


_READABLE_SUFFIXES = frozenset(
    {".py", ".pyi", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
)
_READABLE_KINDS = frozenset(
    {"python_source", "training_configuration", "checkpoint_metadata"}
)


@runtime_checkable
class SafeRepositorySnapshot(Protocol):
    """Minimal adapter expected by the research agent.

    A future clone/intake implementation can provide this protocol without
    changing the planner.  All returned text must already be integrity checked
    and credential screened by the adapter.
    """

    @property
    def inventory_id(self) -> str: ...

    @property
    def request_id(self) -> str: ...

    def inspect(self) -> dict[str, object]: ...

    def list_artifacts(
        self, *, offset: int, limit: int, kind: str | None = None
    ) -> dict[str, object]: ...

    def search(
        self,
        *,
        query: str,
        artifact_ids: tuple[str, ...] | None,
        maximum_results: int,
    ) -> dict[str, object]: ...

    def read(
        self, *, artifact_id: str, start_line: int, end_line: int
    ) -> ResearchEvidence: ...


class InventoriedRepositorySnapshot:
    """Safe snapshot adapter over the existing digest-bound source inventory."""

    def __init__(
        self,
        *,
        inventory: SourceInventory,
        repository_root: Path,
    ) -> None:
        if not isinstance(inventory, SourceInventory):
            raise TypeError("inventory must be a validated SourceInventory")
        if not isinstance(repository_root, Path) or not repository_root.is_absolute():
            raise ValueError(
                "repository_root must be an explicit absolute pathlib.Path"
            )
        self._inventory = inventory
        self._root = _validate_repository_root(repository_root)
        self._artifacts = {
            artifact.artifact_id: artifact for artifact in inventory.artifacts
        }

    @property
    def inventory_id(self) -> str:
        return self._inventory.inventory_id

    @property
    def request_id(self) -> str:
        return self._inventory.request_id

    def inspect(self) -> dict[str, object]:
        readable = tuple(
            artifact
            for artifact in self._inventory.artifacts
            if self._is_readable(artifact)
        )
        return {
            "inventory_id": self.inventory_id,
            "request_id": self.request_id,
            "repository_revision": self._inventory.repository_revision,
            "artifact_count": len(self._inventory.artifacts),
            "readable_text_artifact_count": len(readable),
            "excluded_path_count": self._inventory.excluded_path_count,
            "inventory_complete_for_declared_scope": (
                self._inventory.inventory_complete_for_declared_scope
            ),
            "source_execution_allowed": False,
            "repository_mutation_allowed": False,
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
        if kind is not None and kind not in {
            "python_source",
            "training_configuration",
            "checkpoint_metadata",
            "checkpoint",
        }:
            raise ValueError("artifact kind is outside the snapshot allowlist")
        selected = tuple(
            artifact
            for artifact in self._inventory.artifacts
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
                    "repository_relative_path": artifact.repository_relative_path,
                    "byte_count": artifact.byte_count,
                    "sha256": artifact.sha256,
                    "readable_as_text": self._is_readable(artifact),
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
            or not query
            or query != query.strip()
            or not 2 <= len(query) <= 128
            or any(ord(character) < 32 for character in query)
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
        if artifact_ids is not None:
            if len(artifact_ids) > 64 or len(artifact_ids) != len(set(artifact_ids)):
                raise ValueError(
                    "artifact search filter must contain at most 64 unique IDs"
                )
            unknown = set(artifact_ids) - set(self._artifacts)
            if unknown:
                raise ValueError(
                    "artifact search filter references an unknown artifact ID"
                )
            selected_ids = set(artifact_ids)
        else:
            selected_ids = set(self._artifacts)

        needle = query.casefold()
        matches: list[dict[str, object]] = []
        files_searched = 0
        for artifact in sorted(
            self._inventory.artifacts,
            key=lambda item: item.repository_relative_path,
        ):
            if artifact.artifact_id not in selected_ids or not self._is_readable(
                artifact
            ):
                continue
            text = self._verified_text(artifact)
            files_searched += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                if needle not in line.casefold():
                    continue
                preview = line.strip()
                if len(preview) > 400:
                    preview = preview[:397] + "..."
                matches.append(
                    {
                        "artifact_id": artifact.artifact_id,
                        "evidence_source_id": artifact.evidence_source_id,
                        "repository_relative_path": artifact.repository_relative_path,
                        "line_number": line_number,
                        "preview": preview,
                    }
                )
                if len(matches) >= maximum_results:
                    break
            if len(matches) >= maximum_results:
                break
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
        artifact = self._artifact(artifact_id)
        text = self._verified_text(artifact)
        lines = text.splitlines()
        if not lines:
            raise ExcerptBoundsError("inventoried text artifact contained no lines")
        if end_line > len(lines):
            raise ExcerptBoundsError("requested line range exceeds the text artifact")
        rendered = "\n".join(
            f"{line_number}: {lines[line_number - 1]}"
            for line_number in range(start_line, end_line + 1)
        )
        encoded = rendered.encode("utf-8")
        if len(encoded) > MAX_EXCERPT_BYTES:
            raise ExcerptBoundsError(
                f"rendered excerpt cannot exceed {MAX_EXCERPT_BYTES} bytes"
            )
        excerpt_sha256 = hashlib.sha256(encoded).hexdigest()
        identity_material = (
            f"{self.inventory_id}\0{artifact.artifact_id}\0{start_line}\0"
            f"{end_line}\0{excerpt_sha256}"
        ).encode("utf-8")
        evidence_id = f"evidence-{hashlib.sha256(identity_material).hexdigest()[:24]}"
        return ResearchEvidence(
            evidence_id=evidence_id,
            inventory_id=self.inventory_id,
            artifact_id=artifact.artifact_id,
            evidence_source_id=artifact.evidence_source_id,
            repository_relative_path=artifact.repository_relative_path,
            source_sha256=artifact.sha256,
            start_line=start_line,
            end_line=end_line,
            excerpt_sha256=excerpt_sha256,
            excerpt_text=rendered,
        )

    def _artifact(self, artifact_id: str) -> SourceArtifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ExcerptPathError(
                "artifact ID is not present in the repository snapshot"
            )
        if not self._is_readable(artifact):
            raise UnsafeSourceContentError(
                "artifact is not an allowlisted inventoried text source"
            )
        linked = tuple(
            source
            for source in self._inventory.evidence_sources
            if source.source_id == artifact.evidence_source_id
        )
        if len(linked) != 1:
            raise ExcerptIntegrityError(
                "artifact evidence source is not uniquely present"
            )
        source = linked[0]
        if (
            source.locator != artifact.repository_relative_path
            or source.sha256 != artifact.sha256
        ):
            raise ExcerptIntegrityError(
                "artifact and evidence-source identities disagree"
            )
        return artifact

    def _verified_text(self, artifact: SourceArtifact) -> str:
        verified = self._artifact(artifact.artifact_id)
        target = _target_for_artifact(self._root, verified)
        source_bytes = _read_and_verify_file(target, verified)
        return _decode_and_screen(source_bytes)

    @staticmethod
    def _is_readable(artifact: SourceArtifact) -> bool:
        relative = PurePosixPath(artifact.repository_relative_path)
        return (
            artifact.kind in _READABLE_KINDS
            and artifact.inspection_mode
            in {
                "hash_only",
                "source_parse",
                "structured_metadata_parse",
            }
            and relative.suffix.lower() in _READABLE_SUFFIXES
            and artifact.byte_count <= MAX_SOURCE_FILE_BYTES
            and not any(part.startswith(".") for part in relative.parts)
        )


__all__ = [
    "InventoriedRepositorySnapshot",
    "SafeRepositorySnapshot",
    "SourceExcerptError",
]
