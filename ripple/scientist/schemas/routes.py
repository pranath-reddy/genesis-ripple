"""Typed outcomes for the bounded top-level RIPPLe route dispatcher."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, TypeAdapter, model_validator

from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN


class RoutePlan(FrozenModel):
    """Read-only dispatch plan; constructing it performs no external action."""

    schema_version: Literal["ripple.route-plan.v1"] = "ripple.route-plan.v1"
    branch: Literal["mriganka_dp2", "researcher_model", "simulation_training"]
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    can_execute_with_supplied_inputs: bool
    required_runtime_inputs: tuple[str, ...]
    missing_runtime_inputs: tuple[str, ...]
    stages: tuple[str, ...] = Field(min_length=1)
    terminal_boundary: str = Field(min_length=1, max_length=512)
    lenscat_policy: Literal[
        "final_only_after_real_candidate_evidence",
        "not_applicable_synthetic_without_coordinates",
    ]
    arbitrary_researcher_code_execution_allowed: Literal[False] = False
    network_or_remote_action_performed: Literal[False] = False

    @model_validator(mode="after")
    def _consistent_readiness(self) -> "RoutePlan":
        if self.can_execute_with_supplied_inputs == bool(self.missing_runtime_inputs):
            raise ValueError("route readiness and missing-input status disagree")
        if len(self.required_runtime_inputs) != len(set(self.required_runtime_inputs)):
            raise ValueError("route plan contains duplicate required inputs")
        if len(self.missing_runtime_inputs) != len(set(self.missing_runtime_inputs)):
            raise ValueError("route plan contains duplicate missing inputs")
        return self


class MrigankaDp2RouteResult(FrozenModel):
    """Verified real observation plus deterministic preprocessing, never inference."""

    schema_version: Literal["ripple.mriganka-dp2-route-result.v1"] = (
        "ripple.mriganka-dp2-route-result.v1"
    )
    branch: Literal["mriganka_dp2"] = "mriganka_dp2"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["blocked", "awaiting_implementation"]
    dataset_id: str = Field(min_length=1, max_length=1024)
    observation_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_manifest_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    preprocessing_run_directory: str = Field(min_length=1, max_length=4096)
    preprocessing_completion_sha256: str = Field(pattern=SHA256_PATTERN)
    qualification_state: str = Field(min_length=1, max_length=128)
    preprocessing_execution_allowed: bool
    preprocessing_execution_performed: Literal[True] = True
    model_execution_allowed: bool
    scientific_use_allowed: bool
    classifier_execution_performed: Literal[False] = False
    classifier_block_reason: Literal[
        "model_execution_gate_closed",
        "classifier_executor_not_registered",
    ]
    candidate_report_generated: Literal[False] = False
    lenscat_invoked: Literal[False] = False
    lenscat_reason: Literal["no_classifier_candidate_exists"] = (
        "no_classifier_candidate_exists"
    )

    @model_validator(mode="after")
    def _gate_controls_terminal_status(self) -> "MrigankaDp2RouteResult":
        if not self.preprocessing_execution_allowed:
            raise ValueError(
                "a completed preprocessing route needs an open preprocessing gate"
            )
        expected = (
            ("blocked", "model_execution_gate_closed")
            if not self.model_execution_allowed
            else ("awaiting_implementation", "classifier_executor_not_registered")
        )
        if (self.status, self.classifier_block_reason) != expected:
            raise ValueError("classifier gate and route status disagree")
        return self


class ResearcherModelRouteResult(FrozenModel):
    """Evidence-linked source research; no researcher code is ever executed."""

    schema_version: Literal["ripple.researcher-model-route-result.v2"] = (
        "ripple.researcher-model-route-result.v2"
    )
    branch: Literal["researcher_model"] = "researcher_model"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    run_directory: str = Field(min_length=1, max_length=4096)
    status: Literal["completed", "blocked"]
    repository_acquisition: Literal["local_snapshot", "https_git_clone"]
    repository_intake_id: str = Field(pattern=IDENTIFIER_PATTERN)
    repository_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    pipeline_request_path: str = Field(min_length=1, max_length=4096)
    pipeline_request_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_intake_manifest_path: str = Field(min_length=1, max_length=4096)
    repository_intake_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    repository_snapshot_binding_path: str = Field(min_length=1, max_length=4096)
    repository_snapshot_binding_sha256: str = Field(pattern=SHA256_PATTERN)
    open_research_request_path: str = Field(min_length=1, max_length=4096)
    open_research_request_sha256: str = Field(pattern=SHA256_PATTERN)
    effective_limits_path: str = Field(min_length=1, max_length=4096)
    effective_limits_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_runtime_identity_path: str = Field(min_length=1, max_length=4096)
    provider_runtime_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    research_plan_path: str = Field(min_length=1, max_length=4096)
    research_plan_sha256: str = Field(pattern=SHA256_PATTERN)
    final_result_path: str = Field(min_length=1, max_length=4096)
    final_result_sha256: str = Field(pattern=SHA256_PATTERN)
    research_outcome_path: str = Field(min_length=1, max_length=4096)
    research_outcome_sha256: str = Field(pattern=SHA256_PATTERN)
    research_plan_id: str = Field(pattern=IDENTIFIER_PATTERN)
    final_result_id: str = Field(pattern=IDENTIFIER_PATTERN)
    cited_evidence_count: int = Field(ge=0, le=200)
    analysis_completed: bool
    terminal_reason: Literal[
        "research_goal_answered_from_cited_source_evidence",
        "research_goal_not_satisfied_from_available_source_evidence",
    ]
    repository_cloned_by_agent: Literal[False] = False
    source_execution_performed: Literal[False] = False
    preprocessing_execution_performed: Literal[False] = False
    model_execution_performed: Literal[False] = False
    scientific_use_authorized: Literal[False] = False
    observation_package_opened: Literal[False] = False
    arbitrary_researcher_code_execution_performed: Literal[False] = False
    candidate_report_generated: Literal[False] = False
    lenscat_invoked: Literal[False] = False

    @model_validator(mode="after")
    def _analysis_boundary_is_consistent(self) -> "ResearcherModelRouteResult":
        expected = (
            (
                "completed",
                "research_goal_answered_from_cited_source_evidence",
            )
            if self.analysis_completed
            else (
                "blocked",
                "research_goal_not_satisfied_from_available_source_evidence",
            )
        )
        if (self.status, self.terminal_reason) != expected:
            raise ValueError("research status and terminal reason disagree")
        if self.analysis_completed and self.cited_evidence_count < 1:
            raise ValueError("completed source research must cite evidence")
        return self


class SimulationTrainingRouteResult(FrozenModel):
    """Pointer set returned after the existing simulation campaign completes."""

    schema_version: Literal["ripple.simulation-training-route-result.v1"] = (
        "ripple.simulation-training-route-result.v1"
    )
    branch: Literal["simulation_training"] = "simulation_training"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["complete"] = "complete"
    campaign_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    campaign_run_directory: str = Field(min_length=1, max_length=4096)
    completion_path: str = Field(min_length=1, max_length=4096)
    completion_sha256: str = Field(pattern=SHA256_PATTERN)
    technical_report_path: str = Field(min_length=1, max_length=4096)
    final_state_path: str = Field(min_length=1, max_length=4096)
    smoke_only: Literal[True] = True
    scientific_performance_claim_allowed: Literal[False] = False
    candidate_report_generated: Literal[False] = False
    lenscat_invoked: Literal[False] = False
    lenscat_reason: Literal["synthetic_samples_have_no_sky_coordinates"] = (
        "synthetic_samples_have_no_sky_coordinates"
    )


PipelineRouteResult = Annotated[
    Union[
        MrigankaDp2RouteResult,
        ResearcherModelRouteResult,
        SimulationTrainingRouteResult,
    ],
    Field(discriminator="branch"),
]
PIPELINE_ROUTE_RESULT_ADAPTER = TypeAdapter(PipelineRouteResult)


__all__ = [
    "MrigankaDp2RouteResult",
    "PIPELINE_ROUTE_RESULT_ADAPTER",
    "PipelineRouteResult",
    "ResearcherModelRouteResult",
    "RoutePlan",
    "SimulationTrainingRouteResult",
]
