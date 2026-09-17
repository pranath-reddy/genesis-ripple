"""Typed, proposal-only contracts for the optional PydanticAI control plane."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import EvidenceClaim, ModelManifest
from .literature import LiteratureSearchResult
from .onboarding import (
    DeterministicQualificationReport,
    ModelContractDraft,
    ModelOnboardingRequest,
    RequirementConflict,
    RequirementFinding,
    SourceInventory,
)
from .source_reader import SourceExcerpt


_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_RUNTIME_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EXECUTION_CRITICAL_FIELD_PATHS = frozenset(
    {
        "implementation.source_locator",
        "implementation.source_revision",
        "implementation.source_sha256",
        "implementation.architecture_entrypoint",
        "implementation.checkpoint_locator",
        "implementation.checkpoint_sha256",
        "observation",
        "preprocessing",
        "tensor",
        "output",
    }
)


class _ImmutableAgentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class OnboardingAgentLimits(_ImmutableAgentModel):
    """Hard budgets applied by ordinary Python, not by prompt convention."""

    maximum_excerpt_calls: int = Field(default=24, ge=1, le=128)
    maximum_excerpt_lines_total: int = Field(default=1200, ge=1, le=5000)
    maximum_literature_searches: int = Field(default=3, ge=0, le=10)
    maximum_model_requests_per_phase: int = Field(default=32, ge=2, le=128)
    maximum_tool_calls_per_phase: int = Field(default=32, ge=2, le=128)
    maximum_input_tokens_per_phase: int = Field(
        default=120_000,
        ge=1_000,
        le=1_000_000,
    )
    maximum_output_tokens_per_phase: int = Field(
        default=20_000,
        ge=1_000,
        le=200_000,
    )
    maximum_total_tokens_per_phase: int = Field(
        default=135_000,
        ge=2_000,
        le=1_200_000,
    )

    @model_validator(mode="after")
    def _budgets_are_consistent(self) -> "OnboardingAgentLimits":
        if self.maximum_tool_calls_per_phase < self.maximum_excerpt_calls + 1:
            raise ValueError(
                "the per-phase tool-call budget must cover source listing and every "
                "permitted excerpt call"
            )
        if self.maximum_total_tokens_per_phase > (
            self.maximum_input_tokens_per_phase + self.maximum_output_tokens_per_phase
        ):
            raise ValueError(
                "the total-token budget cannot exceed the sum of its input/output budgets"
            )
        return self


class AgentRuntimeIdentity(_ImmutableAgentModel):
    """Non-secret identity of the reasoning runtime used for an onboarding run."""

    framework: Literal["pydantic-ai"] = "pydantic-ai"
    framework_version: str = Field(pattern=_RUNTIME_ID_PATTERN)
    provider: str = Field(pattern=_RUNTIME_ID_PATTERN)
    model_id: str = Field(pattern=_RUNTIME_ID_PATTERN)
    model_implementation: str = Field(pattern=_RUNTIME_ID_PATTERN)
    credential_recorded: Literal[False] = False


class AgentUsageRecord(_ImmutableAgentModel):
    """Provider-reported aggregate usage with no prompts or response content."""

    requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    cache_read_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class AgentToolActivity(_ImmutableAgentModel):
    """Sanitized counter only; tool arguments and returned source are not logged."""

    tool_name: Literal[
        "list_source_artifacts",
        "read_inventoried_source",
        "list_allowlisted_adapters",
        "search_literature_metadata",
    ]
    attempts: int = Field(ge=0)
    successes: int = Field(ge=0)

    @model_validator(mode="after")
    def _successes_do_not_exceed_attempts(self) -> "AgentToolActivity":
        if self.successes > self.attempts:
            raise ValueError("successful tool calls cannot exceed attempted tool calls")
        return self


class AgentPhaseTrace(_ImmutableAgentModel):
    """Content-free audit record for one bounded PydanticAI phase."""

    phase: Literal["source_analysis", "contract_proposal"]
    run_id: str = Field(min_length=1, max_length=256)
    system_prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    user_prompt_sha256: str = Field(pattern=_SHA256_PATTERN)
    structured_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    retries: int = Field(ge=0, le=5)
    usage: AgentUsageRecord
    tool_activity: tuple[AgentToolActivity, ...] = Field(min_length=1)
    raw_messages_recorded: Literal[False] = False
    reasoning_content_recorded: Literal[False] = False
    tool_arguments_recorded: Literal[False] = False

    @field_validator("run_id")
    @classmethod
    def _safe_run_id(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("agent run ID must be trimmed printable text")
        return value

    @model_validator(mode="after")
    def _unique_tools(self) -> "AgentPhaseTrace":
        names = tuple(item.tool_name for item in self.tool_activity)
        if len(names) != len(set(names)):
            raise ValueError("agent phase tool counters must have unique names")
        return self


class SourceAnalysisProposal(_ImmutableAgentModel):
    """Phase-A source findings; source code is evidence, never instructions."""

    schema_version: Literal["ripple.source-analysis-proposal.v1"] = (
        "ripple.source-analysis-proposal.v1"
    )
    model_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    model_version: str = Field(pattern=_IDENTIFIER_PATTERN)
    evidence_claims: tuple[EvidenceClaim, ...] = Field(min_length=1)
    requirement_findings: tuple[RequirementFinding, ...] = Field(min_length=1)
    conflicts: tuple[RequirementConflict, ...] = ()
    unresolved_field_paths: tuple[str, ...] = ()
    source_execution_performed: Literal[False] = False
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False
    proof_boundary: Literal[
        "Source-analysis proposal only; inventoried files were read as bounded text, never imported or executed, and no preprocessing or model execution is authorized."
    ] = (
        "Source-analysis proposal only; inventoried files were read as bounded text, "
        "never imported or executed, and no preprocessing or model execution is authorized."
    )

    @model_validator(mode="after")
    def _cross_validate(self) -> "SourceAnalysisProposal":
        claims = {claim.claim_id: claim for claim in self.evidence_claims}
        if len(claims) != len(self.evidence_claims):
            raise ValueError("source-analysis evidence claim IDs must be unique")
        referenced = {
            claim_id
            for finding in self.requirement_findings
            for claim_id in finding.evidence_claim_ids
        } | {
            claim_id
            for conflict in self.conflicts
            for claim_id in conflict.evidence_claim_ids
        }
        if not referenced <= set(claims):
            raise ValueError("a source-analysis finding references an unknown claim")
        expected_unresolved = {
            finding.field_path
            for finding in self.requirement_findings
            if finding.status == "unresolved"
        } | {conflict.field_path for conflict in self.conflicts}
        if set(self.unresolved_field_paths) != expected_unresolved:
            raise ValueError(
                "source-analysis unresolved paths do not match its findings"
            )
        findings_by_path: dict[str, list[RequirementFinding]] = {}
        for finding in self.requirement_findings:
            findings_by_path.setdefault(finding.field_path, []).append(finding)
        if not _EXECUTION_CRITICAL_FIELD_PATHS <= set(findings_by_path):
            raise ValueError("source analysis omitted an execution-critical field")
        for path in _EXECUTION_CRITICAL_FIELD_PATHS:
            matching = findings_by_path[path]
            if len(matching) != 1:
                raise ValueError(
                    "source analysis requires exactly one finding per critical field"
                )
            if (
                matching[0].status != "directly_supported"
                and not matching[0].blocks_model_execution
            ):
                raise ValueError(
                    "a non-direct critical finding must block model execution"
                )
        return self


class OnboardingSemanticProposal(_ImmutableAgentModel):
    """Phase-B semantic proposal; trusted request/inventory data is injected later."""

    schema_version: Literal["ripple.onboarding-semantic-proposal.v1"] = (
        "ripple.onboarding-semantic-proposal.v1"
    )
    proposed_manifest: ModelManifest
    requirement_findings: tuple[RequirementFinding, ...] = Field(min_length=1)
    conflicts: tuple[RequirementConflict, ...] = ()
    unresolved_field_paths: tuple[str, ...] = ()
    proposal_only: Literal[True] = True
    preprocessing_execution_authorized: Literal[False] = False
    model_execution_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _draft_manifest_only(self) -> "OnboardingSemanticProposal":
        qualification = self.proposed_manifest.qualification
        if qualification.state != "draft" or any(
            (
                qualification.preprocessing_execution_allowed,
                qualification.model_execution_allowed,
                qualification.scientific_use_allowed,
            )
        ):
            raise ValueError(
                "an agent semantic proposal must keep every manifest gate closed"
            )
        expected_unresolved = {
            finding.field_path
            for finding in self.requirement_findings
            if finding.status == "unresolved"
        } | {conflict.field_path for conflict in self.conflicts}
        if set(self.unresolved_field_paths) != expected_unresolved:
            raise ValueError(
                "semantic proposal unresolved paths do not match its findings"
            )
        return self


class ModelOnboardingAgentOutcome(_ImmutableAgentModel):
    """Complete trace returned after both agent phases and deterministic review."""

    schema_version: Literal["ripple.model-onboarding-agent-outcome.v1"] = (
        "ripple.model-onboarding-agent-outcome.v1"
    )
    completed_at_utc: datetime
    runtime_identity: AgentRuntimeIdentity
    limits: OnboardingAgentLimits
    source_phase_trace: AgentPhaseTrace
    proposal_phase_trace: AgentPhaseTrace
    source_analysis: SourceAnalysisProposal
    source_excerpts: tuple[SourceExcerpt, ...] = Field(min_length=1)
    literature_searches: tuple[LiteratureSearchResult, ...] = ()
    semantic_proposal: OnboardingSemanticProposal
    draft: ModelContractDraft
    deterministic_qualification: DeterministicQualificationReport
    preprocessing_execution_performed: Literal[False] = False
    model_execution_performed: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @field_validator("completed_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("agent outcome timestamp must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("agent outcome timestamp must use UTC")
        return value

    @model_validator(mode="after")
    def _link_trace(self) -> "ModelOnboardingAgentOutcome":
        if self.source_phase_trace.phase != "source_analysis":
            raise ValueError("source trace has the wrong phase identity")
        if self.proposal_phase_trace.phase != "contract_proposal":
            raise ValueError("proposal trace has the wrong phase identity")
        if (
            self.source_phase_trace.structured_output_sha256
            != canonical_agent_payload_sha256(self.source_analysis)
        ):
            raise ValueError("source trace does not identify the returned analysis")
        if (
            self.proposal_phase_trace.structured_output_sha256
            != canonical_agent_payload_sha256(self.semantic_proposal)
        ):
            raise ValueError("proposal trace does not identify the returned proposal")
        if self.source_analysis.model_id != self.draft.proposed_manifest.model_id:
            raise ValueError("source analysis and draft identify different models")
        if (
            self.source_analysis.model_version
            != self.draft.proposed_manifest.model_version
        ):
            raise ValueError("source analysis and draft identify different versions")
        if self.deterministic_qualification.draft_id != self.draft.draft_id:
            raise ValueError("qualification belongs to a different draft")
        if (
            self.deterministic_qualification.draft_sha256
            != self.draft.canonical_sha256()
        ):
            raise ValueError("qualification does not identify the returned draft")
        if (
            self.semantic_proposal.proposed_manifest != self.draft.proposed_manifest
            or self.semantic_proposal.requirement_findings
            != self.draft.requirement_findings
            or self.semantic_proposal.conflicts != self.draft.conflicts
            or self.semantic_proposal.unresolved_field_paths
            != self.draft.unresolved_field_paths
        ):
            raise ValueError("semantic proposal and assembled draft disagree")

        proposal_claims = {
            claim.claim_id: claim
            for claim in self.semantic_proposal.proposed_manifest.evidence_claims
        }
        proposal_findings = {
            finding.finding_id: finding
            for finding in self.semantic_proposal.requirement_findings
        }
        proposal_conflicts = {
            conflict.conflict_id: conflict
            for conflict in self.semantic_proposal.conflicts
        }
        if any(
            proposal_claims.get(claim.claim_id) != claim
            for claim in self.source_analysis.evidence_claims
        ):
            raise ValueError("outcome proposal omitted or altered a source claim")
        if any(
            proposal_findings.get(finding.finding_id) != finding
            for finding in self.source_analysis.requirement_findings
        ):
            raise ValueError("outcome proposal omitted or altered a source finding")
        if any(
            proposal_conflicts.get(conflict.conflict_id) != conflict
            for conflict in self.source_analysis.conflicts
        ):
            raise ValueError("outcome proposal omitted or altered a source conflict")
        if not set(self.source_analysis.unresolved_field_paths) <= set(
            self.semantic_proposal.unresolved_field_paths
        ):
            raise ValueError("outcome proposal removed a source unresolved field")

        artifacts = {
            artifact.artifact_id: artifact
            for artifact in self.draft.source_inventory.artifacts
        }
        excerpt_identities: set[tuple[str, str]] = set()
        for excerpt in self.source_excerpts:
            artifact = artifacts.get(excerpt.artifact_id)
            identity = (excerpt.artifact_id, excerpt.excerpt_sha256)
            if identity in excerpt_identities:
                raise ValueError("outcome contains a duplicate source excerpt identity")
            excerpt_identities.add(identity)
            if (
                excerpt.inventory_id != self.draft.source_inventory.inventory_id
                or artifact is None
                or excerpt.evidence_source_id != artifact.evidence_source_id
                or excerpt.repository_relative_path != artifact.repository_relative_path
                or excerpt.full_file_sha256 != artifact.sha256
            ):
                raise ValueError(
                    "outcome excerpt does not match the returned inventory"
                )

        source_claims = {
            claim.claim_id: claim for claim in self.source_analysis.evidence_claims
        }
        excerpts_by_identity = {
            (excerpt.artifact_id, excerpt.excerpt_sha256): excerpt
            for excerpt in self.source_excerpts
        }
        for finding in self.source_analysis.requirement_findings:
            if finding.status != "directly_supported":
                continue
            verified_source_ids: set[str] = set()
            for location in finding.evidence_locations:
                excerpt = excerpts_by_identity.get(
                    (location.artifact_id or "", location.excerpt_sha256 or "")
                )
                if (
                    excerpt is None
                    or excerpt.evidence_source_id != location.source_id
                    or location.locator
                    != f"{excerpt.artifact_id}:{excerpt.start_line}-{excerpt.end_line}"
                ):
                    raise ValueError(
                        "outcome direct finding lacks its captured excerpt"
                    )
                verified_source_ids.add(location.source_id)
            required_source_ids = {
                source_id
                for claim_id in finding.evidence_claim_ids
                for source_id in source_claims[claim_id].source_ids
            }
            if not required_source_ids <= verified_source_ids:
                raise ValueError(
                    "outcome direct finding lacks evidence for a claimed source"
                )

        literature_successes = sum(
            activity.successes
            for activity in self.proposal_phase_trace.tool_activity
            if activity.tool_name == "search_literature_metadata"
        )
        if literature_successes != len(self.literature_searches):
            raise ValueError(
                "literature results disagree with the sanitized tool trace"
            )
        return self


def assemble_model_contract_draft(
    *,
    request: ModelOnboardingRequest,
    inventory: SourceInventory,
    proposal: OnboardingSemanticProposal,
    created_at_utc: datetime,
    revision: int = 1,
) -> ModelContractDraft:
    """Inject trusted state around an agent's smaller semantic proposal."""

    if created_at_utc.tzinfo is None or created_at_utc.utcoffset() is None:
        raise ValueError("created_at_utc must be timezone-aware")
    if created_at_utc.utcoffset() != timedelta(0):
        raise ValueError("created_at_utc must use UTC")
    identity_payload = {
        "request": request.model_dump(mode="json", exclude_none=False),
        "inventory": inventory.model_dump(mode="json", exclude_none=False),
        "proposal": proposal.model_dump(mode="json", exclude_none=False),
        "revision": revision,
    }
    identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()[:24]
    return ModelContractDraft(
        draft_id=f"draft-{identity}",
        revision=revision,
        created_at_utc=created_at_utc,
        onboarding_request=request,
        source_inventory=inventory,
        proposed_manifest=proposal.proposed_manifest,
        requirement_findings=proposal.requirement_findings,
        conflicts=proposal.conflicts,
        unresolved_field_paths=proposal.unresolved_field_paths,
    )


def canonical_agent_payload_sha256(model: BaseModel) -> str:
    encoded = json.dumps(
        model.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AgentPhaseTrace",
    "AgentRuntimeIdentity",
    "AgentToolActivity",
    "AgentUsageRecord",
    "ModelOnboardingAgentOutcome",
    "OnboardingAgentLimits",
    "OnboardingSemanticProposal",
    "SourceAnalysisProposal",
    "assemble_model_contract_draft",
    "canonical_agent_payload_sha256",
]
