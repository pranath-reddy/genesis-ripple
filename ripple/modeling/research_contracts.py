"""Small typed contracts for open-ended, source-first model research.

The contracts intentionally describe analysis, not execution.  A completed
result means that the requested *research analysis* was completed from the
available evidence.  It never authorizes importing researcher code, loading a
checkpoint, preprocessing observations, or running inference.
"""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

ResearchActionKind = Literal[
    "inspect_source",
    "analyze_architecture",
    "trace_data_flow",
    "analyze_preprocessing",
    "analyze_checkpoint_contract",
    "analyze_output_contract",
    "assess_domain_compatibility",
    "design_integration",
    "request_missing_evidence",
    "human_review",
]

ALLOWLISTED_RESEARCH_ACTIONS: tuple[ResearchActionKind, ...] = (
    "inspect_source",
    "analyze_architecture",
    "trace_data_flow",
    "analyze_preprocessing",
    "analyze_checkpoint_contract",
    "analyze_output_contract",
    "assess_domain_compatibility",
    "design_integration",
    "request_missing_evidence",
    "human_review",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


def _bounded_text(value: str, *, name: str, maximum: int) -> str:
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(
            f"{name} must be non-empty, trimmed, and at most {maximum} characters"
        )
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")
    return value


def _unique_identifiers(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique identifiers")
    if any(re.fullmatch(IDENTIFIER_PATTERN, value) is None for value in values):
        raise ValueError(f"{name} contains an invalid identifier")
    return values


class OpenResearchRequest(_FrozenModel):
    """One explicit analysis goal over an already-created safe source snapshot."""

    schema_version: Literal["ripple.open-research-request.v1"] = (
        "ripple.open-research-request.v1"
    )
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    end_goal: str = Field(min_length=1, max_length=2048)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=12)
    target_observation_domain: str | None = Field(default=None, max_length=512)
    allowed_action_kinds: tuple[ResearchActionKind, ...] = ALLOWLISTED_RESEARCH_ACTIONS
    proposal_only: Literal[True] = True
    network_access_authorized: Literal[False] = False
    repository_mutation_authorized: Literal[False] = False
    researcher_code_execution_authorized: Literal[False] = False
    checkpoint_loading_authorized: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @field_validator("end_goal")
    @classmethod
    def _valid_goal(cls, value: str) -> str:
        return _bounded_text(value, name="end_goal", maximum=2048)

    @field_validator("target_observation_domain")
    @classmethod
    def _valid_domain(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, name="target_observation_domain", maximum=512)

    @field_validator("success_criteria")
    @classmethod
    def _valid_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            _bounded_text(item, name="success criterion", maximum=512) for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("success criteria must be unique")
        return normalized

    @field_validator("allowed_action_kinds")
    @classmethod
    def _valid_actions(
        cls, value: tuple[ResearchActionKind, ...]
    ) -> tuple[ResearchActionKind, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError(
                "allowed research action kinds must be non-empty and unique"
            )
        if not set(value) <= set(ALLOWLISTED_RESEARCH_ACTIONS):
            raise ValueError(
                "research request contains an action outside the code allowlist"
            )
        return value


class ResearchAgentLimits(_FrozenModel):
    """Code-owned filesystem, tool, and model budgets for one research run."""

    maximum_artifact_list_calls: int = Field(default=3, ge=1, le=10)
    maximum_search_calls: int = Field(default=10, ge=1, le=50)
    maximum_read_calls: int = Field(default=16, ge=1, le=80)
    maximum_read_lines_total: int = Field(default=1200, ge=1, le=8000)
    maximum_search_results_per_call: int = Field(default=20, ge=1, le=50)
    maximum_evidence_items: int = Field(default=40, ge=1, le=200)
    maximum_tool_calls_per_phase: int = Field(default=32, ge=4, le=160)
    maximum_model_requests_per_phase: int = Field(default=12, ge=2, le=50)
    maximum_input_tokens_per_phase: int = Field(default=32_768, ge=1024, le=200_000)
    maximum_output_tokens_per_phase: int = Field(default=4096, ge=256, le=32_768)
    maximum_total_tokens_per_phase: int = Field(default=36_864, ge=1280, le=220_000)

    @model_validator(mode="after")
    def _token_budget_is_consistent(self) -> "ResearchAgentLimits":
        if self.maximum_total_tokens_per_phase < (
            self.maximum_input_tokens_per_phase + self.maximum_output_tokens_per_phase
        ):
            raise ValueError(
                "total token limit must cover the input and output token limits"
            )
        return self


class ResearchEvidence(_FrozenModel):
    """A digest-bound excerpt collected by a read-only repository tool."""

    evidence_id: str = Field(pattern=IDENTIFIER_PATTERN)
    inventory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    evidence_source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    repository_relative_path: str = Field(min_length=1, max_length=1024)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt_sha256: str = Field(pattern=SHA256_PATTERN)
    excerpt_text: str = Field(min_length=1, max_length=65_536)
    acquisition: Literal["read_inventoried_text"] = "read_inventoried_text"
    source_executed: Literal[False] = False
    source_imported: Literal[False] = False

    @model_validator(mode="after")
    def _ordered_lines(self) -> "ResearchEvidence":
        if self.end_line < self.start_line:
            raise ValueError("evidence line bounds are reversed")
        observed_digest = hashlib.sha256(self.excerpt_text.encode("utf-8")).hexdigest()
        if observed_digest != self.excerpt_sha256:
            raise ValueError("evidence excerpt SHA-256 does not match excerpt_text")
        return self


class ResearchPlanStepDraft(_FrozenModel):
    """One semantic plan step proposed by the model; code assigns its identity."""

    action_kind: ResearchActionKind
    objective: str = Field(min_length=1, max_length=1024)
    evidence_ids: tuple[str, ...] = ()
    completion_criteria: str = Field(min_length=1, max_length=768)

    @field_validator("evidence_ids")
    @classmethod
    def _valid_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="plan evidence_ids")


class ResearchPlanDraft(_FrozenModel):
    """Compact agent output after iterative repository inspection."""

    goal_summary: str = Field(min_length=1, max_length=1600)
    evidence_summary: str = Field(min_length=1, max_length=2400)
    steps: tuple[ResearchPlanStepDraft, ...] = Field(min_length=1, max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=20)
    sufficient_evidence_to_answer: bool


class ResearchPlanStep(_FrozenModel):
    step_id: str = Field(pattern=IDENTIFIER_PATTERN)
    sequence: int = Field(ge=1, le=12)
    action_kind: ResearchActionKind
    objective: str = Field(min_length=1, max_length=1024)
    evidence_ids: tuple[str, ...] = ()
    completion_criteria: str = Field(min_length=1, max_length=768)

    @field_validator("evidence_ids")
    @classmethod
    def _valid_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="plan evidence_ids")


class ResearchPlan(_FrozenModel):
    schema_version: Literal["ripple.open-research-plan.v1"] = (
        "ripple.open-research-plan.v1"
    )
    plan_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    inventory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    end_goal: str = Field(min_length=1, max_length=2048)
    goal_summary: str = Field(min_length=1, max_length=1600)
    evidence_summary: str = Field(min_length=1, max_length=2400)
    steps: tuple[ResearchPlanStep, ...] = Field(min_length=1, max_length=12)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=20)
    sufficient_evidence_to_answer: bool
    allowed_action_kinds: tuple[ResearchActionKind, ...] = Field(min_length=1)
    proposal_only: Literal[True] = True
    execution_authorized: Literal[False] = False

    @model_validator(mode="after")
    def _validate_steps(self) -> "ResearchPlan":
        expected = tuple(range(1, len(self.steps) + 1))
        if tuple(step.sequence for step in self.steps) != expected:
            raise ValueError("research-plan sequences must be contiguous and ordered")
        if len({step.step_id for step in self.steps}) != len(self.steps):
            raise ValueError("research-plan step IDs must be unique")
        allowed = set(self.allowed_action_kinds)
        if any(step.action_kind not in allowed for step in self.steps):
            raise ValueError(
                "research plan contains an action outside the request allowlist"
            )
        return self


class ResearchFindingDraft(_FrozenModel):
    statement: str = Field(min_length=1, max_length=1800)
    support: Literal["direct", "inferred", "unresolved"]
    evidence_ids: tuple[str, ...] = ()
    implication: str = Field(min_length=1, max_length=1200)

    @field_validator("evidence_ids")
    @classmethod
    def _valid_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="finding evidence_ids")

    @model_validator(mode="after")
    def _supported_findings_need_evidence(self) -> "ResearchFindingDraft":
        if self.support in {"direct", "inferred"} and not self.evidence_ids:
            raise ValueError(
                "direct and inferred findings must cite collected evidence"
            )
        return self


class SuccessCriterionAssessmentDraft(_FrozenModel):
    """Evidence-linked assessment of one exact researcher success criterion."""

    criterion: str = Field(min_length=1, max_length=512)
    satisfied: bool
    evidence_ids: tuple[str, ...] = ()
    assessment: str = Field(min_length=1, max_length=768)

    @field_validator("criterion")
    @classmethod
    def _valid_criterion(cls, value: str) -> str:
        return _bounded_text(value, name="success criterion", maximum=512)

    @field_validator("evidence_ids")
    @classmethod
    def _valid_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="criterion evidence_ids")

    @model_validator(mode="after")
    def _satisfied_criteria_need_evidence(self) -> "SuccessCriterionAssessmentDraft":
        if self.satisfied and not self.evidence_ids:
            raise ValueError(
                "a satisfied success criterion must cite collected evidence"
            )
        return self


class ResearchResultDraft(_FrozenModel):
    """Small second-phase model output; code determines final result identity."""

    goal_satisfied: bool
    answer_summary: str = Field(min_length=1, max_length=2400)
    criterion_assessments: tuple[SuccessCriterionAssessmentDraft, ...] = Field(
        min_length=1,
        max_length=12,
    )
    findings: tuple[ResearchFindingDraft, ...] = Field(min_length=1, max_length=20)
    blockers: tuple[str, ...] = Field(default=(), max_length=20)
    recommended_next_actions: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)

    @model_validator(mode="after")
    def _goal_and_blockers_agree(self) -> "ResearchResultDraft":
        if self.goal_satisfied and self.blockers:
            raise ValueError("a satisfied goal cannot retain blocking conditions")
        if not self.goal_satisfied and not self.blockers:
            raise ValueError("an unsatisfied goal must identify at least one blocker")
        return self


class ResearchFinding(_FrozenModel):
    finding_id: str = Field(pattern=IDENTIFIER_PATTERN)
    statement: str = Field(min_length=1, max_length=1800)
    support: Literal["direct", "inferred", "unresolved"]
    evidence_ids: tuple[str, ...] = ()
    implication: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def _supported_findings_need_evidence(self) -> "ResearchFinding":
        if self.support in {"direct", "inferred"} and not self.evidence_ids:
            raise ValueError(
                "direct and inferred findings must cite collected evidence"
            )
        return self


class SuccessCriterionAssessment(_FrozenModel):
    criterion_id: str = Field(pattern=IDENTIFIER_PATTERN)
    criterion: str = Field(min_length=1, max_length=512)
    satisfied: bool
    evidence_ids: tuple[str, ...] = ()
    assessment: str = Field(min_length=1, max_length=768)

    @field_validator("criterion")
    @classmethod
    def _valid_criterion(cls, value: str) -> str:
        return _bounded_text(value, name="success criterion", maximum=512)

    @field_validator("evidence_ids")
    @classmethod
    def _valid_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="criterion evidence_ids")

    @model_validator(mode="after")
    def _satisfied_criteria_need_evidence(self) -> "SuccessCriterionAssessment":
        if self.satisfied and not self.evidence_ids:
            raise ValueError(
                "a satisfied success criterion must cite collected evidence"
            )
        return self


class FinalResearchResult(_FrozenModel):
    """Terminal analysis outcome; ``completed`` never means code was executed."""

    schema_version: Literal["ripple.final-research-result.v2"] = (
        "ripple.final-research-result.v2"
    )
    result_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    plan_id: str = Field(pattern=IDENTIFIER_PATTERN)
    inventory_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["completed", "blocked"]
    end_goal: str = Field(min_length=1, max_length=2048)
    answer_summary: str = Field(min_length=1, max_length=2400)
    criterion_assessments: tuple[SuccessCriterionAssessment, ...] = Field(
        min_length=1,
        max_length=12,
    )
    findings: tuple[ResearchFinding, ...] = Field(min_length=1, max_length=20)
    blockers: tuple[str, ...] = Field(default=(), max_length=20)
    recommended_next_actions: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    cited_evidence_ids: tuple[str, ...] = ()
    analysis_completed: bool
    repository_cloned_by_agent: Literal[False] = False
    source_executed: Literal[False] = False
    checkpoint_loaded: Literal[False] = False
    preprocessing_executed: Literal[False] = False
    model_inference_executed: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @field_validator("cited_evidence_ids")
    @classmethod
    def _valid_citations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_identifiers(value, name="cited_evidence_ids")

    @model_validator(mode="after")
    def _terminal_shape(self) -> "FinalResearchResult":
        if self.status == "completed":
            if not self.analysis_completed or self.blockers:
                raise ValueError("completed analysis cannot retain blockers")
            if not all(item.satisfied for item in self.criterion_assessments):
                raise ValueError("completed analysis requires every success criterion")
        elif self.analysis_completed or not self.blockers:
            raise ValueError("blocked analysis must be incomplete and name blockers")
        result_citations = {
            evidence_id
            for finding in self.findings
            for evidence_id in finding.evidence_ids
        }
        result_citations.update(
            evidence_id
            for assessment in self.criterion_assessments
            for evidence_id in assessment.evidence_ids
        )
        if result_citations != set(self.cited_evidence_ids):
            raise ValueError(
                "result citation index must exactly cover result citations"
            )
        criterion_ids = tuple(item.criterion_id for item in self.criterion_assessments)
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("success-criterion assessment IDs must be unique")
        return self


class ResearchRunUsage(_FrozenModel):
    planner_requests: int = Field(ge=0)
    planner_tool_calls: int = Field(ge=0)
    result_requests: int = Field(ge=0)
    result_tool_calls: int = Field(ge=0)
    artifact_list_calls: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    read_calls: int = Field(ge=0)
    read_lines: int = Field(ge=0)


class OpenResearchOutcome(_FrozenModel):
    schema_version: Literal["ripple.open-research-outcome.v2"] = (
        "ripple.open-research-outcome.v2"
    )
    request: OpenResearchRequest
    plan: ResearchPlan
    evidence: tuple[ResearchEvidence, ...] = Field(min_length=1)
    result: FinalResearchResult
    called_tools: tuple[str, ...] = Field(min_length=1)
    usage: ResearchRunUsage

    @model_validator(mode="after")
    def _linked_identity(self) -> "OpenResearchOutcome":
        if not (
            self.request.request_id == self.plan.request_id == self.result.request_id
        ):
            raise ValueError(
                "open-research request, plan, and result identities disagree"
            )
        if self.plan.plan_id != self.result.plan_id:
            raise ValueError("result references a different research plan")
        if self.plan.inventory_id != self.result.inventory_id:
            raise ValueError("research plan and result inventories disagree")
        if (
            self.request.end_goal != self.plan.end_goal
            or self.plan.end_goal != self.result.end_goal
        ):
            raise ValueError("research request, plan, and result end goals disagree")
        if self.request.allowed_action_kinds != self.plan.allowed_action_kinds:
            raise ValueError("research plan action allowlist differs from its request")
        if (
            tuple(item.criterion for item in self.result.criterion_assessments)
            != self.request.success_criteria
        ):
            raise ValueError(
                "research result must assess every success criterion exactly once and in order"
            )

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("open-research evidence IDs must be unique")
        if any(item.inventory_id != self.plan.inventory_id for item in self.evidence):
            raise ValueError("open-research evidence belongs to a different inventory")
        known_evidence_ids = set(evidence_ids)
        plan_citations = {
            evidence_id for step in self.plan.steps for evidence_id in step.evidence_ids
        }
        result_citations = set(self.result.cited_evidence_ids)
        if not plan_citations <= known_evidence_ids:
            raise ValueError("research plan cites evidence absent from the outcome")
        if not result_citations <= known_evidence_ids:
            raise ValueError("research result cites evidence absent from the outcome")

        finding_ids = tuple(finding.finding_id for finding in self.result.findings)
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("research-result finding IDs must be unique")
        return self


__all__ = [
    "ALLOWLISTED_RESEARCH_ACTIONS",
    "FinalResearchResult",
    "OpenResearchOutcome",
    "OpenResearchRequest",
    "ResearchActionKind",
    "ResearchAgentLimits",
    "ResearchEvidence",
    "ResearchFinding",
    "ResearchFindingDraft",
    "ResearchPlan",
    "ResearchPlanDraft",
    "ResearchPlanStep",
    "ResearchPlanStepDraft",
    "ResearchResultDraft",
    "ResearchRunUsage",
    "SuccessCriterionAssessment",
    "SuccessCriterionAssessmentDraft",
]
