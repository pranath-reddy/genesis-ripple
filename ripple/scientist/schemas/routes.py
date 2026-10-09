"""Typed outcomes for the bounded top-level RIPPLe route dispatcher."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import PurePath
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, field_validator, model_validator

from .common import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    BudgetUsage,
    FrozenModel,
    SkyCoordinate,
)
from .report import (
    MRIGANKA_DP2_SCIENTIFIC_BLOCKERS,
    MrigankaDp2ScientificBlocker,
)


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
    def _consistent_readiness(self) -> RoutePlan:
        if self.can_execute_with_supplied_inputs == bool(self.missing_runtime_inputs):
            raise ValueError("route readiness and missing-input status disagree")
        if len(self.required_runtime_inputs) != len(set(self.required_runtime_inputs)):
            raise ValueError("route plan contains duplicate required inputs")
        if len(self.missing_runtime_inputs) != len(set(self.missing_runtime_inputs)):
            raise ValueError("route plan contains duplicate missing inputs")
        return self


class MrigankaRouteArtifactRef(FrozenModel):
    """Absolute, content-addressed reference emitted by the known-model route."""

    path: str = Field(min_length=1, max_length=4096)
    byte_count: int = Field(gt=0, le=2 * 1024**3)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def _absolute_path_without_traversal(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("route artifact paths must be absolute without traversal")
        return value


class MrigankaDp2BandResult(FrozenModel):
    band: Literal["g", "r", "i"]
    dataset_id: str = Field(min_length=1, max_length=1024)
    run_directory: str = Field(min_length=1, max_length=4096)
    package_manifest: MrigankaRouteArtifactRef
    fits_artifact: MrigankaRouteArtifactRef
    image_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    mask_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    variance_decoded_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("run_directory")
    @classmethod
    def _absolute_run_directory(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("M2 run directory must be absolute without traversal")
        return value

    @model_validator(mode="after")
    def _fixed_m2_artifacts(self) -> MrigankaDp2BandResult:
        root = PurePath(self.run_directory)
        if (
            PurePath(self.package_manifest.path).parent != root
            or PurePath(self.package_manifest.path).name != "package.json"
            or PurePath(self.fits_artifact.path).parent != root
            or PurePath(self.fits_artifact.path).name != "cutout.fits"
        ):
            raise ValueError(
                "M2 artifact references must belong to their run directory"
            )
        return self


class MrigankaM3Result(FrozenModel):
    run_directory: str = Field(min_length=1, max_length=4096)
    model_manifest_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    package_manifest: MrigankaRouteArtifactRef
    completion: MrigankaRouteArtifactRef
    model_input_bchw: MrigankaRouteArtifactRef
    qa_preview: MrigankaRouteArtifactRef
    cross_band_wcs_maximum_separation_arcsec: float = Field(ge=0.0, le=1e-7)
    cross_band_wcs_passed: Literal[True] = True

    @field_validator("run_directory")
    @classmethod
    def _absolute_run_directory(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("M3 run directory must be absolute without traversal")
        return value

    @model_validator(mode="after")
    def _fixed_m3_artifacts(self) -> MrigankaM3Result:
        root = PurePath(self.run_directory)
        observed = (
            (PurePath(self.package_manifest.path), "manifest.json"),
            (PurePath(self.completion.path), "completion.json"),
            (PurePath(self.model_input_bchw.path), "model_input_bchw.npy"),
            (PurePath(self.qa_preview.path), "preprocessing_preview.png"),
        )
        if any(path.parent != root or path.name != name for path, name in observed):
            raise ValueError(
                "M3 artifact references must belong to their run directory"
            )
        return self


class MrigankaBridgeResult(FrozenModel):
    run_directory: str = Field(min_length=1, max_length=4096)
    bridge_manifest: MrigankaRouteArtifactRef
    completion: MrigankaRouteArtifactRef
    model_input_chw: MrigankaRouteArtifactRef
    payload_identical: Literal[True] = True
    physical_channel_mapping_verified: Literal[False] = False

    @field_validator("run_directory")
    @classmethod
    def _absolute_run_directory(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("bridge run directory must be absolute without traversal")
        return value

    @model_validator(mode="after")
    def _fixed_bridge_artifacts(self) -> MrigankaBridgeResult:
        root = PurePath(self.run_directory)
        observed = (
            (PurePath(self.bridge_manifest.path), "bridge.json"),
            (PurePath(self.completion.path), "completion.json"),
            (PurePath(self.model_input_chw.path), "model_input_chw.npy"),
        )
        if any(path.parent != root or path.name != name for path, name in observed):
            raise ValueError(
                "bridge artifact references must belong to their run directory"
            )
        return self


class MrigankaM4Result(FrozenModel):
    run_directory: str = Field(min_length=1, max_length=4096)
    bundle_id: str = Field(pattern=IDENTIFIER_PATTERN)
    bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    inference_result: MrigankaRouteArtifactRef
    completion: MrigankaRouteArtifactRef
    embedding: MrigankaRouteArtifactRef
    raw_logits: tuple[float, float]
    uncalibrated_softmax_components: tuple[float, float]
    execution_scope: Literal["unqualified_dp2_technical_integration"] = (
        "unqualified_dp2_technical_integration"
    )
    score_is_calibrated_probability: Literal[False] = False
    decision_threshold_applied: Literal[False] = False
    candidate_decision_made: Literal[False] = False

    @field_validator("run_directory")
    @classmethod
    def _absolute_run_directory(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("M4 run directory must be absolute without traversal")
        return value

    @model_validator(mode="after")
    def _uncalibrated_softmax_is_well_formed(self) -> MrigankaM4Result:
        if any(
            value < 0.0 or value > 1.0 for value in self.uncalibrated_softmax_components
        ):
            raise ValueError("uncalibrated softmax components must lie in [0,1]")
        if not math.isclose(
            sum(self.uncalibrated_softmax_components),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("uncalibrated softmax components must sum to one")
        maximum_logit = max(self.raw_logits)
        exponentials = tuple(
            math.exp(value - maximum_logit) for value in self.raw_logits
        )
        total = sum(exponentials)
        expected = tuple(value / total for value in exponentials)
        if any(
            not math.isclose(observed, wanted, rel_tol=0.0, abs_tol=1e-6)
            for observed, wanted in zip(
                self.uncalibrated_softmax_components,
                expected,
                strict=True,
            )
        ):
            raise ValueError("uncalibrated components do not match logits softmax")
        root = PurePath(self.run_directory)
        observed_artifacts = (
            (PurePath(self.inference_result.path), "inference.json"),
            (PurePath(self.completion.path), "completion.json"),
            (PurePath(self.embedding.path), "embedding.npy"),
        )
        if any(
            path.parent != root or path.name != name
            for path, name in observed_artifacts
        ):
            raise ValueError(
                "M4 artifact references must belong to their run directory"
            )
        return self


class MrigankaDp2RouteCompletion(FrozenModel):
    """Last-written hash binding for one complete known-model technical run."""

    schema_version: Literal["ripple.mriganka-dp2-route-completion.v2"] = (
        "ripple.mriganka-dp2-route-completion.v2"
    )
    status: Literal["complete"] = "complete"
    route_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    completed_at_utc: datetime
    bands: tuple[Literal["g"], Literal["r"], Literal["i"]] = ("g", "r", "i")
    m2_package_manifest_sha256: tuple[str, str, str]
    m3_model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    m3_completion_sha256: str = Field(pattern=SHA256_PATTERN)
    bridge_completion_sha256: str = Field(pattern=SHA256_PATTERN)
    m4_bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    m4_inference_result_sha256: str = Field(pattern=SHA256_PATTERN)
    m4_completion_sha256: str = Field(pattern=SHA256_PATTERN)
    technical_report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    technical_report_sha256: str = Field(pattern=SHA256_PATTERN)
    usage: BudgetUsage
    unresolved_scientific_blockers: tuple[MrigankaDp2ScientificBlocker, ...] = (
        MRIGANKA_DP2_SCIENTIFIC_BLOCKERS
    )
    completion_written_last: Literal[True] = True
    technical_integration_completed: Literal[True] = True
    score_is_calibrated_probability: Literal[False] = False
    decision_threshold_applied: Literal[False] = False
    candidate_decision_made: Literal[False] = False
    candidate_report_generated: Literal[False] = False
    scientific_use_allowed: Literal[False] = False
    lenscat_invoked: Literal[False] = False

    @field_validator("completed_at_utc")
    @classmethod
    def _utc_completion_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("route completion time must use UTC")
        return value

    @field_validator("m2_package_manifest_sha256")
    @classmethod
    def _three_valid_m2_digests(
        cls, value: tuple[str, str, str]
    ) -> tuple[str, str, str]:
        if any(re.fullmatch(SHA256_PATTERN, digest) is None for digest in value):
            raise ValueError("M2 package digests must be SHA-256 values")
        return value

    @field_validator("unresolved_scientific_blockers")
    @classmethod
    def _safe_unique_completion_blockers(
        cls, value: tuple[MrigankaDp2ScientificBlocker, ...]
    ) -> tuple[MrigankaDp2ScientificBlocker, ...]:
        if value != MRIGANKA_DP2_SCIENTIFIC_BLOCKERS:
            raise ValueError("route completion must retain the exact six blockers")
        return value

    @model_validator(mode="after")
    def _exact_durable_usage(self) -> MrigankaDp2RouteCompletion:
        if (
            self.usage.tool_calls != 7
            or self.usage.storage_bytes < 1
            or self.usage.llm_requests
            or self.usage.simulations
            or self.usage.training_runs
            or self.usage.gpu_seconds
        ):
            raise ValueError("route completion usage must describe the seven CPU tools")
        return self


class MrigankaDp2RouteResult(FrozenModel):
    """Completed g/r/i technical integration, never a candidate decision."""

    schema_version: Literal["ripple.mriganka-dp2-route-result.v2"] = (
        "ripple.mriganka-dp2-route-result.v2"
    )
    branch: Literal["mriganka_dp2"] = "mriganka_dp2"
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["technical_integration_complete"] = "technical_integration_complete"
    route_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    route_run_directory: str = Field(min_length=1, max_length=4096)
    target: SkyCoordinate
    bands: tuple[Literal["g"], Literal["r"], Literal["i"]] = ("g", "r", "i")
    m2_packages: tuple[
        MrigankaDp2BandResult,
        MrigankaDp2BandResult,
        MrigankaDp2BandResult,
    ]
    m3: MrigankaM3Result
    bridge: MrigankaBridgeResult
    m4: MrigankaM4Result
    technical_report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    technical_report: MrigankaRouteArtifactRef
    route_completion: MrigankaRouteArtifactRef
    usage: BudgetUsage
    unresolved_scientific_blockers: tuple[MrigankaDp2ScientificBlocker, ...] = (
        MRIGANKA_DP2_SCIENTIFIC_BLOCKERS
    )
    technical_integration_completed: Literal[True] = True
    preprocessing_execution_performed: Literal[True] = True
    classifier_execution_performed: Literal[True] = True
    softmax_is_calibrated_probability: Literal[False] = False
    decision_threshold_applied: Literal[False] = False
    candidate_decision_made: Literal[False] = False
    candidate_report_generated: Literal[False] = False
    scientific_use_allowed: Literal[False] = False
    lenscat_invoked: Literal[False] = False
    lenscat_reason: Literal["unqualified_technical_integration_has_no_candidate"] = (
        "unqualified_technical_integration_has_no_candidate"
    )

    @field_validator("route_run_directory")
    @classmethod
    def _absolute_route_directory(cls, value: str) -> str:
        candidate = PurePath(value)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("route run directory must be absolute without traversal")
        return value

    @field_validator("unresolved_scientific_blockers")
    @classmethod
    def _exact_blockers(
        cls, value: tuple[MrigankaDp2ScientificBlocker, ...]
    ) -> tuple[MrigankaDp2ScientificBlocker, ...]:
        if value != MRIGANKA_DP2_SCIENTIFIC_BLOCKERS:
            raise ValueError("route result must retain the exact six blockers")
        return value

    @field_validator("usage")
    @classmethod
    def _exact_route_usage(cls, value: BudgetUsage) -> BudgetUsage:
        if (
            value.tool_calls != 7
            or value.storage_bytes < 1
            or value.llm_requests
            or value.simulations
            or value.training_runs
            or value.gpu_seconds
        ):
            raise ValueError("route result usage must describe the seven CPU tools")
        return value

    @model_validator(mode="after")
    def _technical_only_completion_is_consistent(self) -> MrigankaDp2RouteResult:
        if tuple(item.band for item in self.m2_packages) != self.bands:
            raise ValueError("M2 package results must be ordered g, r, i")
        root = PurePath(self.route_run_directory)
        for item in self.m2_packages:
            run_directory = PurePath(item.run_directory)
            if (
                run_directory.parent.parent != root / "m2"
                or run_directory.parent.name != item.band
            ):
                raise ValueError("M2 run directories must belong to the route and band")
        stage_directories = (
            (PurePath(self.m3.run_directory), "m3"),
            (PurePath(self.bridge.run_directory), "bridge"),
            (PurePath(self.m4.run_directory), "m4"),
        )
        if any(path.parent != root / stage for path, stage in stage_directories):
            raise ValueError("M3, bridge, and M4 runs must belong to the route")
        if (
            PurePath(self.technical_report.path)
            != root / "report" / "technical-report.json"
            or PurePath(self.route_completion.path) != root / "completion.json"
        ):
            raise ValueError("route report and completion references are misplaced")
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
    def _analysis_boundary_is_consistent(self) -> ResearcherModelRouteResult:
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
    MrigankaDp2RouteResult | ResearcherModelRouteResult | SimulationTrainingRouteResult,
    Field(discriminator="branch"),
]
PIPELINE_ROUTE_RESULT_ADAPTER = TypeAdapter(PipelineRouteResult)


__all__ = [
    "PIPELINE_ROUTE_RESULT_ADAPTER",
    "MrigankaBridgeResult",
    "MrigankaDp2BandResult",
    "MrigankaDp2RouteCompletion",
    "MrigankaDp2RouteResult",
    "MrigankaM3Result",
    "MrigankaM4Result",
    "MrigankaRouteArtifactRef",
    "PipelineRouteResult",
    "ResearcherModelRouteResult",
    "RoutePlan",
    "SimulationTrainingRouteResult",
]
