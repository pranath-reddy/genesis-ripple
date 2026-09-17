"""Three-route run requests and append-only orchestrator state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .common import (
    ArtifactRef,
    BudgetUsage,
    ComputeBudget,
    FrozenModel,
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    ScientificGate,
    SkyCoordinate,
)
from .repository import RepositorySource
from .remote import RemoteWorkerSettings


class MrigankaDp2Request(FrozenModel):
    branch: Literal["mriganka_dp2"] = "mriganka_dp2"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    observation_package: str = Field(min_length=1, max_length=1024)
    model_manifest: str = Field(min_length=1, max_length=1024)
    target: SkyCoordinate
    budget: ComputeBudget = ComputeBudget()


class ResearcherModelRequest(FrozenModel):
    branch: Literal["researcher_model"] = "researcher_model"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    repository_source: RepositorySource
    end_goal: str = Field(min_length=1, max_length=2048)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=12)
    target_observation_domain: str | None = Field(default=None, max_length=512)
    budget: ComputeBudget = ComputeBudget(
        max_llm_requests=12,
        max_tool_calls=32,
    )

    @field_validator("end_goal", "target_observation_domain")
    @classmethod
    def _safe_research_text(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        maximum = 2048 if getattr(info, "field_name", "") == "end_goal" else 512
        if (
            not value
            or value != value.strip()
            or len(value) > maximum
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("research text must be non-empty, trimmed, and printable")
        return value

    @field_validator("success_criteria")
    @classmethod
    def _safe_success_criteria(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("success criteria must be unique")
        if any(
            not item
            or item != item.strip()
            or len(item) > 512
            or any(ord(character) < 32 for character in item)
            for item in value
        ):
            raise ValueError(
                "success criteria must be trimmed printable text up to 512 characters"
            )
        return value


class SimulationTrainingRequest(FrozenModel):
    branch: Literal["simulation_training"] = "simulation_training"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    simulation_spec: str = Field(min_length=1, max_length=1024)
    output_root: str = Field(min_length=1, max_length=1024)
    gpu_host: str = Field(min_length=1, max_length=256)
    gpu_python: str = Field(min_length=1, max_length=1024)
    gpu_remote_root: str = Field(min_length=1, max_length=1024)
    budget: ComputeBudget = ComputeBudget()

    @model_validator(mode="after")
    def _safe_remote_worker(self) -> "SimulationTrainingRequest":
        RemoteWorkerSettings(
            host=self.gpu_host,
            python=self.gpu_python,
            remote_root=self.gpu_remote_root,
        )
        return self


PipelineRunRequest = Annotated[
    Union[MrigankaDp2Request, ResearcherModelRequest, SimulationTrainingRequest],
    Field(discriminator="branch"),
]
PIPELINE_REQUEST_ADAPTER = TypeAdapter(PipelineRunRequest)


class DecisionRecord(FrozenModel):
    decision_id: str = Field(pattern=IDENTIFIER_PATTERN)
    agent_name: str = Field(pattern=IDENTIFIER_PATTERN)
    phase: str = Field(pattern=IDENTIFIER_PATTERN)
    observed_evidence_ids: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = Field(min_length=1)
    selected_action: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=2048)
    expected_cost: str = Field(min_length=1, max_length=512)
    created_at_utc: datetime

    @field_validator("created_at_utc")
    @classmethod
    def _utc_only(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("decision time must be UTC")
        return value

    @model_validator(mode="after")
    def _selected_is_allowed(self) -> "DecisionRecord":
        if self.selected_action not in self.allowed_actions:
            raise ValueError("agent selected an action outside the allowlist")
        return self


class RunState(FrozenModel):
    schema_version: Literal["ripple.agentic-run-state.v1"] = (
        "ripple.agentic-run-state.v1"
    )
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    branch: Literal["mriganka_dp2", "researcher_model", "simulation_training"]
    phase: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal[
        "created",
        "running",
        "awaiting_input",
        "awaiting_human_approval",
        "blocked",
        "failed",
        "complete",
    ]
    sequence: int = Field(ge=0)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    previous_state_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    artifacts: tuple[ArtifactRef, ...] = ()
    decisions: tuple[DecisionRecord, ...] = ()
    budget: ComputeBudget
    usage: BudgetUsage = BudgetUsage()
    gates: tuple[ScientificGate, ...] = ()
    blockers: tuple[str, ...] = ()
    next_allowed_actions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_state(self) -> "RunState":
        if not self.usage.fits(self.budget):
            raise ValueError("recorded usage exceeds the run budget")
        if self.status == "blocked" and not self.blockers:
            raise ValueError("blocked state must identify at least one blocker")
        if (
            self.status in {"complete", "failed", "blocked"}
            and self.next_allowed_actions
        ):
            raise ValueError("terminal state cannot advertise another action")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique in a run state")
        return self


class CoordinatorDecision(FrozenModel):
    """Typed output of the coordinator after it has called state tools."""

    selected_action: str = Field(min_length=1, max_length=128)
    evidence_ids: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=1600)
    expected_cost: str = Field(min_length=1, max_length=512)
    unresolved_risks: tuple[str, ...] = ()


class ToolOutcome(FrozenModel):
    """The only shape an approved deterministic workflow tool may return."""

    phase: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal[
        "running",
        "awaiting_input",
        "awaiting_human_approval",
        "blocked",
        "failed",
        "complete",
    ]
    artifacts: tuple[ArtifactRef, ...] = ()
    gates: tuple[ScientificGate, ...] = ()
    blockers: tuple[str, ...] = ()
    next_allowed_actions: tuple[str, ...] = ()
    usage_delta: BudgetUsage = BudgetUsage()

    @model_validator(mode="after")
    def _terminal_shape(self) -> "ToolOutcome":
        if self.status == "blocked" and not self.blockers:
            raise ValueError("a blocked tool outcome must record a blocker")
        if (
            self.status in {"blocked", "failed", "complete"}
            and self.next_allowed_actions
        ):
            raise ValueError("a terminal tool outcome cannot expose another action")
        return self
