"""Strict contracts for evidence-backed model onboarding.

This module contains data models only.  It performs no network access, model
inference, preprocessing, or LLM calls.  Every artifact produced here is a
proposal for human review; none of these contracts can authorize execution or
scientific use of a model.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .contracts import (
    EvidenceSource,
    ModelManifest,
    _safe_reference_locator,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_FIELD_PATH_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$"
_MEDIA_TYPE_PATTERN = r"^[a-z0-9.+-]+/[a-z0-9.+-]+$"
_SECRET_PATH_COMPONENT_PATTERN = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|credential|private[_-]?key|secret|token)(?:$|[._-])",
    flags=re.IGNORECASE,
)


class _ImmutableOnboardingModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


def _bounded_text(value: str, *, field_name: str, maximum: int) -> str:
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(
            f"{field_name} must be non-empty, trimmed, and at most {maximum} characters"
        )
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field_name} contains a control character")
    return value


def _safe_locator(value: str) -> str:
    return _safe_reference_locator(value, field="onboarding locator")


def _require_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must use UTC")
    return value


def _unique_identifiers(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique values")
    if any(re.fullmatch(_IDENTIFIER_PATTERN, value) is None for value in values):
        raise ValueError(f"{field_name} contains an invalid identifier")
    return values


class ModelOnboardingRequest(_ImmutableOnboardingModel):
    """Researcher request to investigate one concrete model artifact.

    Submitting this request starts evidence collection only.  It does not permit
    preprocessing, model execution, or scientific interpretation.
    """

    schema_version: Literal["ripple.model-onboarding-request.v1"] = (
        "ripple.model-onboarding-request.v1"
    )
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    requested_at_utc: datetime
    requested_by: str = Field(min_length=1, max_length=256)
    model_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    candidate_model_version: str = Field(pattern=_IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    scientific_task: Literal[
        "binary_lens_classification",
        "multiclass_lens_classification",
        "lens_reconstruction",
        "parameter_regression",
    ]
    target_observation_domain: str = Field(min_length=1, max_length=512)
    source_locators: tuple[str, ...] = Field(min_length=1)
    checkpoint_locators: tuple[str, ...] = ()
    primary_literature_locators: tuple[str, ...] = ()
    research_question: str = Field(min_length=1, max_length=2048)
    requested_scope: Literal["evidence_and_model_contract_proposal"] = (
        "evidence_and_model_contract_proposal"
    )
    proposal_only: Literal[True] = True
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @field_validator("requested_at_utc")
    @classmethod
    def _valid_requested_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="requested_at_utc")

    @field_validator("requested_by", "display_name", "target_observation_domain")
    @classmethod
    def _valid_short_text(cls, value: str, info: object) -> str:
        maximum = (
            512
            if getattr(info, "field_name", "") == "target_observation_domain"
            else 256
        )
        return _bounded_text(
            value,
            field_name=getattr(info, "field_name", "text"),
            maximum=maximum,
        )

    @field_validator("research_question")
    @classmethod
    def _valid_question(cls, value: str) -> str:
        return _bounded_text(value, field_name="research_question", maximum=2048)

    @field_validator(
        "source_locators", "checkpoint_locators", "primary_literature_locators"
    )
    @classmethod
    def _valid_locators(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_safe_locator(locator) for locator in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "onboarding locators must be unique within each collection"
            )
        return normalized


class SourceArtifact(_ImmutableOnboardingModel):
    """One immutable local artifact discovered during source inventory."""

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    evidence_source_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    kind: Literal[
        "python_source",
        "training_configuration",
        "checkpoint",
        "checkpoint_metadata",
        "primary_literature",
        "dataset_metadata",
        "researcher_supplied_record",
        "other_context",
    ]
    repository_relative_path: str = Field(min_length=1, max_length=1024)
    media_type: str = Field(pattern=_MEDIA_TYPE_PATTERN)
    byte_count: int = Field(ge=0, le=100 * 1024 * 1024 * 1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    inspection_mode: Literal[
        "source_parse",
        "structured_metadata_parse",
        "text_extract",
        "hash_only",
        "context_only",
    ]
    language: str | None = Field(default=None, min_length=1, max_length=64)
    executable_content_was_run: Literal[False] = False
    credential_content_recorded: Literal[False] = False

    @field_validator("repository_relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or value != parsed.as_posix()
            or parsed.name in {"", "."}
            or any(part.startswith(".") for part in parsed.parts)
            or any(_SECRET_PATH_COMPONENT_PATTERN.search(part) for part in parsed.parts)
        ):
            raise ValueError("artifact path must be normalized and repository-relative")
        return value

    @field_validator("language")
    @classmethod
    def _valid_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, field_name="language", maximum=64)

    @model_validator(mode="after")
    def _safe_inspection_mode(self) -> "SourceArtifact":
        permitted_modes = {
            "python_source": {"source_parse", "hash_only"},
            "training_configuration": {
                "structured_metadata_parse",
                "hash_only",
                "context_only",
            },
            "checkpoint": {"hash_only"},
            "checkpoint_metadata": {"structured_metadata_parse", "hash_only"},
            "primary_literature": {"text_extract", "hash_only", "context_only"},
            "dataset_metadata": {
                "structured_metadata_parse",
                "hash_only",
                "context_only",
            },
            "researcher_supplied_record": {
                "structured_metadata_parse",
                "hash_only",
                "context_only",
            },
            "other_context": {"context_only", "hash_only"},
        }
        if self.inspection_mode not in permitted_modes[self.kind]:
            raise ValueError("artifact kind and inspection mode are incompatible")
        suffix = PurePosixPath(self.repository_relative_path).suffix.lower()
        if self.kind == "python_source" and (
            suffix not in {".py", ".pyi"}
            or self.media_type != "text/x-python"
            or self.language != "python"
        ):
            raise ValueError(
                "Python source identity must match its path, media type, and language"
            )
        return self


class SourceInventory(_ImmutableOnboardingModel):
    """Hashed source inventory; it records evidence but approves nothing."""

    schema_version: Literal["ripple.model-source-inventory.v1"] = (
        "ripple.model-source-inventory.v1"
    )
    inventory_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    created_at_utc: datetime
    repository_revision: str | None = Field(default=None, max_length=256)
    artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    evidence_sources: tuple[EvidenceSource, ...] = Field(min_length=1)
    excluded_path_count: int = Field(default=0, ge=0)
    declared_scope: Literal[
        "explicit allowlisted executable, configuration, metadata, and checkpoint artifacts only"
    ] = "explicit allowlisted executable, configuration, metadata, and checkpoint artifacts only"
    inventory_complete_for_declared_scope: bool
    proposal_only: Literal[True] = True
    execution_authorized: Literal[False] = False

    @field_validator("created_at_utc")
    @classmethod
    def _valid_created_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="created_at_utc")

    @field_validator("repository_revision")
    @classmethod
    def _valid_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, field_name="repository_revision", maximum=256)

    @model_validator(mode="after")
    def _validate_inventory_links(self) -> "SourceInventory":
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        source_ids = [source.source_id for source in self.evidence_sources]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source artifact IDs must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("inventory evidence source IDs must be unique")
        known_sources = set(source_ids)
        if any(
            artifact.evidence_source_id not in known_sources
            for artifact in self.artifacts
        ):
            raise ValueError("a source artifact references an unknown evidence source")
        artifact_source_ids = [
            artifact.evidence_source_id for artifact in self.artifacts
        ]
        if len(artifact_source_ids) != len(set(artifact_source_ids)):
            raise ValueError(
                "each inventory evidence source must identify exactly one artifact"
            )
        if set(artifact_source_ids) != known_sources:
            raise ValueError(
                "inventory evidence sources and artifacts must have one-to-one coverage"
            )

        sources = {source.source_id: source for source in self.evidence_sources}
        expected_source_kinds = {
            "python_source": "executable_source",
            "training_configuration": "training_configuration",
            "checkpoint": "checkpoint_metadata",
            "checkpoint_metadata": "checkpoint_metadata",
            "primary_literature": "primary_literature",
            "dataset_metadata": "dataset_documentation",
            "researcher_supplied_record": "researcher_declaration",
            "other_context": "dataset_documentation",
        }
        for artifact in self.artifacts:
            source = sources[artifact.evidence_source_id]
            if (
                source.locator != artifact.repository_relative_path
                or source.sha256 != artifact.sha256
                or source.kind != expected_source_kinds[artifact.kind]
            ):
                raise ValueError(
                    "an inventory artifact and its evidence source disagree"
                )
            if (
                artifact.inspection_mode == "hash_only"
                and source.authority != "context_only"
            ):
                raise ValueError(
                    "hash-only inventory evidence must remain context-only"
                )
        return self


class EvidenceLocation(_ImmutableOnboardingModel):
    """Exact location supporting a requirement finding."""

    source_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    location_kind: Literal[
        "source_lines",
        "configuration_key",
        "checkpoint_field",
        "literature_pages",
        "literature_section",
        "researcher_declaration",
        "whole_artifact",
    ]
    locator: str = Field(min_length=1, max_length=512)
    excerpt_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("locator")
    @classmethod
    def _valid_locator(cls, value: str) -> str:
        return _bounded_text(value, field_name="evidence location", maximum=512)


class RequirementFinding(_ImmutableOnboardingModel):
    """A proposed model-contract value derived from cited evidence."""

    finding_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    field_path: str = Field(pattern=_FIELD_PATH_PATTERN)
    proposed_value: JsonValue
    status: Literal["directly_supported", "inferred", "unresolved"]
    evidence_claim_ids: tuple[str, ...] = Field(min_length=1)
    evidence_locations: tuple[EvidenceLocation, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1, max_length=2048)
    human_review_required: bool
    blocks_model_execution: bool
    proposal_only: Literal[True] = True
    execution_authorized: Literal[False] = False

    @field_validator("evidence_claim_ids")
    @classmethod
    def _valid_claim_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, field_name="evidence_claim_ids")

    @field_validator("rationale")
    @classmethod
    def _valid_rationale(cls, value: str) -> str:
        return _bounded_text(value, field_name="rationale", maximum=2048)

    @model_validator(mode="after")
    def _validate_review_boundary(self) -> "RequirementFinding":
        if self.status == "directly_supported" and self.blocks_model_execution:
            raise ValueError("a directly supported finding must not block execution")
        if self.status in {"inferred", "unresolved"} and not self.human_review_required:
            raise ValueError("inferred or unresolved findings require human review")
        if self.status == "unresolved" and not self.blocks_model_execution:
            raise ValueError("an unresolved finding must block model execution")
        return self


class RequirementConflict(_ImmutableOnboardingModel):
    """Contradictory evidence that must be resolved in a later draft."""

    conflict_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    field_path: str = Field(pattern=_FIELD_PATH_PATTERN)
    evidence_claim_ids: tuple[str, ...] = Field(min_length=2)
    description: str = Field(min_length=1, max_length=2048)
    state: Literal["unresolved", "resolution_proposed"] = "unresolved"
    proposed_resolution: str | None = Field(default=None, max_length=2048)
    human_review_required: Literal[True] = True
    blocks_model_execution: Literal[True] = True
    proposal_only: Literal[True] = True
    execution_authorized: Literal[False] = False

    @field_validator("evidence_claim_ids")
    @classmethod
    def _valid_claim_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, field_name="evidence_claim_ids")

    @field_validator("description")
    @classmethod
    def _valid_description(cls, value: str) -> str:
        return _bounded_text(value, field_name="description", maximum=2048)

    @field_validator("proposed_resolution")
    @classmethod
    def _valid_resolution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, field_name="proposed_resolution", maximum=2048)

    @model_validator(mode="after")
    def _validate_resolution_state(self) -> "RequirementConflict":
        if (self.state == "resolution_proposed") != (
            self.proposed_resolution is not None
        ):
            raise ValueError("conflict state and proposed resolution disagree")
        return self


class ModelContractDraft(_ImmutableOnboardingModel):
    """Evidence-linked manifest proposal with every execution gate closed."""

    schema_version: Literal["ripple.model-contract-draft.v1"] = (
        "ripple.model-contract-draft.v1"
    )
    draft_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    revision: int = Field(ge=1)
    created_at_utc: datetime
    onboarding_request: ModelOnboardingRequest
    source_inventory: SourceInventory
    proposed_manifest: ModelManifest
    requirement_findings: tuple[RequirementFinding, ...] = Field(min_length=1)
    conflicts: tuple[RequirementConflict, ...] = ()
    unresolved_field_paths: tuple[str, ...] = ()
    status: Literal["proposal_pending_qualification"] = "proposal_pending_qualification"
    proposal_only: Literal[True] = True
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False
    proof_boundary: Literal[
        "Evidence-backed contract proposal only; no preprocessing, model execution, or scientific use is authorized."
    ] = "Evidence-backed contract proposal only; no preprocessing, model execution, or scientific use is authorized."

    @field_validator("created_at_utc")
    @classmethod
    def _valid_created_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="created_at_utc")

    @field_validator("unresolved_field_paths")
    @classmethod
    def _valid_unresolved_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("unresolved field paths must be unique")
        if any(re.fullmatch(_FIELD_PATH_PATTERN, path) is None for path in value):
            raise ValueError("unresolved field paths contain an invalid path")
        return value

    @model_validator(mode="after")
    def _cross_validate_draft(self) -> "ModelContractDraft":
        if self.source_inventory.request_id != self.onboarding_request.request_id:
            raise ValueError(
                "source inventory belongs to a different onboarding request"
            )
        if self.proposed_manifest.model_id != self.onboarding_request.model_id:
            raise ValueError("proposed manifest model ID does not match the request")
        if (
            self.proposed_manifest.model_version
            != self.onboarding_request.candidate_model_version
        ):
            raise ValueError("proposed manifest version does not match the request")

        qualification = self.proposed_manifest.qualification
        if qualification.state != "draft" or any(
            (
                qualification.preprocessing_execution_allowed,
                qualification.model_execution_allowed,
                qualification.scientific_use_allowed,
            )
        ):
            raise ValueError(
                "a contract draft must embed a manifest with every gate closed"
            )

        finding_ids = [finding.finding_id for finding in self.requirement_findings]
        conflict_ids = [conflict.conflict_id for conflict in self.conflicts]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("requirement finding IDs must be unique")
        if len(conflict_ids) != len(set(conflict_ids)):
            raise ValueError("requirement conflict IDs must be unique")

        manifest_sources = {
            source.source_id: source
            for source in self.proposed_manifest.evidence_sources
        }
        inventory_sources = {
            source.source_id: source
            for source in self.source_inventory.evidence_sources
        }
        manifest_source_ids = set(manifest_sources)
        inventory_source_ids = set(inventory_sources)
        if not manifest_source_ids <= inventory_source_ids:
            raise ValueError(
                "the manifest uses evidence absent from the source inventory"
            )
        if any(
            manifest_sources[source_id] != inventory_sources[source_id]
            for source_id in manifest_source_ids
        ):
            raise ValueError(
                "manifest evidence must exactly match its inventoried source"
            )

        claims = {
            claim.claim_id: claim for claim in self.proposed_manifest.evidence_claims
        }
        known_claim_ids = set(claims)
        referenced_claim_ids = {
            claim_id
            for finding in self.requirement_findings
            for claim_id in finding.evidence_claim_ids
        } | {
            claim_id
            for conflict in self.conflicts
            for claim_id in conflict.evidence_claim_ids
        }
        if not referenced_claim_ids <= known_claim_ids:
            raise ValueError(
                "a finding or conflict references an unknown evidence claim"
            )

        artifacts = {
            artifact.artifact_id: artifact
            for artifact in self.source_inventory.artifacts
        }
        known_artifact_ids = set(artifacts)
        known_inventory_sources = inventory_source_ids
        for finding in self.requirement_findings:
            finding_claims = tuple(
                claims[claim_id] for claim_id in finding.evidence_claim_ids
            )
            if any(claim.field_path != finding.field_path for claim in finding_claims):
                raise ValueError(
                    "a finding and its evidence claims must address the same field"
                )
            if finding.status == "directly_supported" and any(
                claim.support != "direct" for claim in finding_claims
            ):
                raise ValueError(
                    "a directly supported finding requires direct evidence claims"
                )
            claim_source_ids = {
                source_id for claim in finding_claims for source_id in claim.source_ids
            }
            for location in finding.evidence_locations:
                if location.source_id not in known_inventory_sources:
                    raise ValueError(
                        "an evidence location references an unknown source"
                    )
                if location.source_id not in claim_source_ids:
                    raise ValueError(
                        "an evidence location is not cited by the finding's claims"
                    )
                if (
                    location.artifact_id is not None
                    and location.artifact_id not in known_artifact_ids
                ):
                    raise ValueError(
                        "an evidence location references an unknown artifact"
                    )
                if location.artifact_id is not None and (
                    artifacts[location.artifact_id].evidence_source_id
                    != location.source_id
                ):
                    raise ValueError(
                        "an evidence location mismatches its artifact source"
                    )
                if location.location_kind == "source_lines":
                    if location.artifact_id is None or location.excerpt_sha256 is None:
                        raise ValueError(
                            "source-line evidence requires an artifact and excerpt digest"
                        )
                    if artifacts[location.artifact_id].kind != "python_source":
                        raise ValueError(
                            "source-line evidence requires a Python source artifact"
                        )
            if finding.status == "directly_supported" and not any(
                location.location_kind
                in {"source_lines", "configuration_key", "checkpoint_field"}
                and location.excerpt_sha256 is not None
                for location in finding.evidence_locations
            ):
                raise ValueError(
                    "directly supported findings require digest-bound evidence"
                )

        expected_unresolved = {
            finding.field_path
            for finding in self.requirement_findings
            if finding.status == "unresolved"
        } | {conflict.field_path for conflict in self.conflicts}
        if set(self.unresolved_field_paths) != expected_unresolved:
            raise ValueError(
                "unresolved field paths must exactly cover unresolved findings and conflicts"
            )
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", exclude_none=False, by_alias=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QualificationFinding(_ImmutableOnboardingModel):
    """Result of one deterministic, non-LLM draft qualification check."""

    finding_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    category: Literal[
        "schema",
        "source_integrity",
        "evidence_links",
        "requirement_completeness",
        "conflict_check",
        "checkpoint_identity",
        "adapter_allowlist",
        "execution_gate",
    ]
    code: str = Field(pattern=_IDENTIFIER_PATTERN)
    outcome: Literal["passed", "failed"]
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1, max_length=1024)
    related_field_paths: tuple[str, ...] = ()
    deterministic: Literal[True] = True
    blocks_approval_freeze_request: bool

    @field_validator("message")
    @classmethod
    def _valid_message(cls, value: str) -> str:
        return _bounded_text(value, field_name="message", maximum=1024)

    @field_validator("related_field_paths")
    @classmethod
    def _valid_related_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("related field paths must be unique")
        if any(re.fullmatch(_FIELD_PATH_PATTERN, path) is None for path in value):
            raise ValueError("related field paths contain an invalid path")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> "QualificationFinding":
        if self.outcome == "passed" and self.blocks_approval_freeze_request:
            raise ValueError("a passed finding cannot block an approval/freeze request")
        if self.outcome == "failed" and self.severity == "error":
            if not self.blocks_approval_freeze_request:
                raise ValueError("a failed error must block an approval/freeze request")
        return self


class DeterministicQualificationReport(_ImmutableOnboardingModel):
    """Code-computed consistency report, not a scientific qualification."""

    schema_version: Literal["ripple.model-contract-qualification.v1"] = (
        "ripple.model-contract-qualification.v1"
    )
    report_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    draft_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    completed_at_utc: datetime
    validator_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    validator_version: str = Field(pattern=_IDENTIFIER_PATTERN)
    findings: tuple[QualificationFinding, ...] = Field(min_length=1)
    outcome: Literal["passed", "failed"]
    approval_freeze_request_allowed: bool
    scientific_qualification_established: Literal[False] = False
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False
    proof_boundary: Literal[
        "Deterministic schema and evidence-consistency checks only; this report cannot approve, freeze, execute, or scientifically qualify a model."
    ] = "Deterministic schema and evidence-consistency checks only; this report cannot approve, freeze, execute, or scientifically qualify a model."

    @field_validator("completed_at_utc")
    @classmethod
    def _valid_completed_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="completed_at_utc")

    @model_validator(mode="after")
    def _validate_report_outcome(self) -> "DeterministicQualificationReport":
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("qualification finding IDs must be unique")
        has_blocker = any(
            finding.outcome == "failed" and finding.blocks_approval_freeze_request
            for finding in self.findings
        )
        expected_outcome = "failed" if has_blocker else "passed"
        if self.outcome != expected_outcome:
            raise ValueError("qualification outcome disagrees with blocking findings")
        if self.approval_freeze_request_allowed != (self.outcome == "passed"):
            raise ValueError(
                "approval/freeze request gate disagrees with qualification outcome"
            )
        return self


class HumanApprovalFreezeRequest(_ImmutableOnboardingModel):
    """Request for a human decision; this object is not that decision."""

    schema_version: Literal["ripple.human-approval-freeze-request.v1"] = (
        "ripple.human-approval-freeze-request.v1"
    )
    approval_request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    submitted_at_utc: datetime
    submitted_by: str = Field(min_length=1, max_length=256)
    requested_action: Literal["human_review_and_freeze_if_approved"] = (
        "human_review_and_freeze_if_approved"
    )
    draft: ModelContractDraft
    qualification_report: DeterministicQualificationReport
    reviewer_instructions: str = Field(min_length=1, max_length=2048)
    decision_state: Literal["pending_human_decision"] = "pending_human_decision"
    proposal_only: Literal[True] = True
    manifest_frozen: Literal[False] = False
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False
    proof_boundary: Literal[
        "Pending human review only; submission does not approve or freeze the manifest and authorizes no execution or scientific use."
    ] = "Pending human review only; submission does not approve or freeze the manifest and authorizes no execution or scientific use."

    @field_validator("submitted_at_utc")
    @classmethod
    def _valid_submitted_at(cls, value: datetime) -> datetime:
        return _require_utc(value, field_name="submitted_at_utc")

    @field_validator("submitted_by")
    @classmethod
    def _valid_submitted_by(cls, value: str) -> str:
        return _bounded_text(value, field_name="submitted_by", maximum=256)

    @field_validator("reviewer_instructions")
    @classmethod
    def _valid_reviewer_instructions(cls, value: str) -> str:
        return _bounded_text(value, field_name="reviewer_instructions", maximum=2048)

    @model_validator(mode="after")
    def _validate_qualification_link(self) -> "HumanApprovalFreezeRequest":
        report = self.qualification_report
        if report.draft_id != self.draft.draft_id:
            raise ValueError("qualification report belongs to a different draft")
        if report.draft_sha256 != self.draft.canonical_sha256():
            raise ValueError("qualification report does not match the supplied draft")
        if report.outcome != "passed" or not report.approval_freeze_request_allowed:
            raise ValueError(
                "only a deterministically passing draft may request approval"
            )
        return self


__all__ = [
    "DeterministicQualificationReport",
    "EvidenceLocation",
    "HumanApprovalFreezeRequest",
    "ModelContractDraft",
    "ModelOnboardingRequest",
    "QualificationFinding",
    "RequirementConflict",
    "RequirementFinding",
    "SourceArtifact",
    "SourceInventory",
]
