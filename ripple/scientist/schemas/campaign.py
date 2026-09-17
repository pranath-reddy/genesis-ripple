"""Top-level contracts for the executable simulation-trained route."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .architecture import TrainingConfiguration
from .common import (
    FrozenModel,
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    canonical_json_sha256,
)
from .orchestration import SimulationTrainingRequest
from .remote import RemoteCommandResult


class SimulationCampaignConfiguration(FrozenModel):
    schema_version: Literal["ripple.simulation-campaign.v1"] = (
        "ripple.simulation-campaign.v1"
    )
    request: SimulationTrainingRequest
    architecture_shortlist_size: int = Field(default=2, ge=1, le=4)
    training: TrainingConfiguration = TrainingConfiguration()
    architecture_implementation: Literal["ripple-native-model-builders"] = (
        "ripple-native-model-builders"
    )
    execute_remote: Literal[True] = True
    smoke_only: Literal[True] = True

    @model_validator(mode="after")
    def _budget_covers_campaign(self) -> "SimulationCampaignConfiguration":
        if self.architecture_shortlist_size > self.request.budget.max_training_runs:
            raise ValueError("shortlist size exceeds the training-run budget")
        if self.request.budget.max_simulations < 10:
            raise ValueError("campaign budget cannot cover the ten-image smoke")
        if self.request.budget.max_llm_requests < 21:
            raise ValueError("campaign budget must allow its bounded agent stages")
        minimum_tool_calls = 21 + 3 * self.architecture_shortlist_size
        if self.request.budget.max_tool_calls < minimum_tool_calls:
            raise ValueError("campaign budget cannot cover every approved tool action")
        if self.request.budget.max_gpu_seconds < self.architecture_shortlist_size + 1:
            raise ValueError("campaign GPU budget cannot cover training and evaluation")
        if self.request.budget.max_storage_bytes < 64 * 1024 * 1024:
            raise ValueError("campaign storage budget must be at least 64 MiB")
        return self


class RuntimeSourceRevision(FrozenModel):
    name: Literal["slsim", "JAXtronomy"]
    repository: str = Field(min_length=1, max_length=512)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    local_path: str = Field(min_length=1, max_length=512)
    runtime_role: str = Field(min_length=1, max_length=512)
    transfer_tree_sha256: str = Field(pattern=SHA256_PATTERN)


class SourcePrecedentRevision(FrozenModel):
    name: Literal["DeepLense-AI-Scientist"]
    repository: str = Field(min_length=1, max_length=512)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_dependency: Literal[False] = False
    port_path: Literal["ripple/scientist/tools/model_builders.py"]
    local_revision: Literal["ripple-model-builders-v1"]
    port_sha256: str = Field(pattern=SHA256_PATTERN)


class TutorialPrecedentRevision(FrozenModel):
    name: Literal["slsim-tutorials"]
    repository: str = Field(min_length=1, max_length=512)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_dependency: Literal[False] = False
    evidence_role: str = Field(min_length=1, max_length=512)


class ScientistSourceRevisions(FrozenModel):
    """Pinned runtime sources and non-runtime architecture attribution."""

    schema_version: Literal["ripple.scientist.source-revisions.v1"] = (
        "ripple.scientist.source-revisions.v1"
    )
    runtime_sources: tuple[RuntimeSourceRevision, ...] = Field(
        min_length=2, max_length=2
    )
    source_precedent_only: tuple[
        SourcePrecedentRevision | TutorialPrecedentRevision,
        ...,
    ] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def _complete_sources(self) -> "ScientistSourceRevisions":
        if {item.name for item in self.runtime_sources} != {"slsim", "JAXtronomy"}:
            raise ValueError("source lock must identify SLSim and JAXtronomy")
        if {item.name for item in self.source_precedent_only} != {
            "DeepLense-AI-Scientist",
            "slsim-tutorials",
        }:
            raise ValueError("source lock must identify both design precedents")
        return self


class SyncedSourceComponent(FrozenModel):
    """One locally hashed regular-file tree copied to the deterministic worker."""

    component: Literal["ripple_scientist", "slsim", "jaxtronomy"]
    local_relative_path: str = Field(min_length=1, max_length=512)
    remote_relative_path: str = Field(min_length=1, max_length=512)
    source_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    regular_file_count: int = Field(ge=1)
    remote_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    remote_regular_file_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _remote_matches_local(self) -> "SyncedSourceComponent":
        if (
            self.remote_tree_sha256 != self.source_tree_sha256
            or self.remote_regular_file_count != self.regular_file_count
        ):
            raise ValueError("remote source tree does not match its local source tree")
        return self


class SourceSyncEvidence(FrozenModel):
    """Credential-free evidence for the bounded source synchronization step."""

    schema_version: Literal["ripple.campaign-source-sync.v2"] = (
        "ripple.campaign-source-sync.v2"
    )
    host: str = Field(min_length=1, max_length=256)
    remote_root: str = Field(min_length=1, max_length=1024)
    components: tuple[SyncedSourceComponent, ...] = Field(min_length=3, max_length=3)
    package_initializer_sha256: str = Field(pattern=SHA256_PATTERN)
    remote_package_initializer_sha256: str = Field(pattern=SHA256_PATTERN)
    source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    remote_source_root: str = Field(min_length=1, max_length=1024)
    verification_method: Literal["ssh-isolated-python-sha256-v1"] = (
        "ssh-isolated-python-sha256-v1"
    )
    excluded_transient_patterns: tuple[str, str] = (
        "**/__pycache__/**",
        "**/*.pyc",
    )
    synchronized_at_utc: datetime
    verified_at_utc: datetime

    @field_validator("synchronized_at_utc", "verified_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("source synchronization time must be UTC")
        return value

    @model_validator(mode="after")
    def _complete_component_set(self) -> "SourceSyncEvidence":
        names = tuple(item.component for item in self.components)
        if len(names) != len(set(names)) or set(names) != {
            "ripple_scientist",
            "slsim",
            "jaxtronomy",
        }:
            raise ValueError(
                "source sync must cover each required source tree exactly once"
            )
        expected_paths = {
            "ripple_scientist": "ripple/scientist",
            "slsim": "ripple/scientist/vendor/slsim/slsim",
            "jaxtronomy": "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
        }
        if any(
            item.remote_relative_path != expected_paths[item.component]
            for item in self.components
        ):
            raise ValueError("source sync component used a non-canonical remote path")
        if self.remote_package_initializer_sha256 != self.package_initializer_sha256:
            raise ValueError(
                "remote package initializer does not match the local source"
            )
        manifest_payload = {
            "components": [
                {
                    "component": item.component,
                    "remote_relative_path": item.remote_relative_path,
                    "sha256": item.remote_tree_sha256,
                    "regular_file_count": item.remote_regular_file_count,
                }
                for item in sorted(self.components, key=lambda value: value.component)
            ],
            "package_initializer_sha256": self.remote_package_initializer_sha256,
        }
        if canonical_json_sha256(manifest_payload) != self.source_manifest_sha256:
            raise ValueError("source sync manifest digest is invalid")
        expected_source_root = (
            f"{self.remote_root.rstrip('/')}/sources/{self.source_manifest_sha256}"
        )
        if self.remote_source_root != expected_source_root:
            raise ValueError("remote source root is not manifest-addressed")
        return self


class AgentStageEvidence(FrozenModel):
    """Bounded local-agent usage attached to its persisted typed output."""

    schema_version: Literal["ripple.campaign-agent-stage.v1"] = (
        "ripple.campaign-agent-stage.v1"
    )
    stage: Literal[
        "simulation_planner",
        "architecture_generator",
        "architecture_judge",
    ]
    output_artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    output_sha256: str = Field(pattern=SHA256_PATTERN)
    called_tools: tuple[str, ...] = Field(min_length=1)
    request_count: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class RemoteStageEvidence(FrozenModel):
    """One deterministic worker invocation and its parsed final JSON object."""

    schema_version: Literal["ripple.campaign-remote-stage.v1"] = (
        "ripple.campaign-remote-stage.v1"
    )
    result: RemoteCommandResult
    parsed_payload: dict[str, JsonValue]
    completed_at_utc: datetime

    @field_validator("completed_at_utc")
    @classmethod
    def _remote_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("remote-stage completion time must be UTC")
        return value

    @model_validator(mode="after")
    def _successful_result_only(self) -> "RemoteStageEvidence":
        if not self.result.succeeded or self.result.exit_code != 0:
            raise ValueError(
                "successful remote-stage evidence requires a zero exit code"
            )
        if not self.parsed_payload:
            raise ValueError("remote-stage evidence requires the worker JSON payload")
        return self


class SimulationCampaignCompletion(FrozenModel):
    """Last-written completion marker for a fully collected smoke campaign."""

    schema_version: Literal["ripple.simulation-campaign-completion.v2"] = (
        "ripple.simulation-campaign-completion.v2"
    )
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    campaign_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    simulation_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    architecture_plan_sha256: str = Field(pattern=SHA256_PATTERN)
    selected_candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_training_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    final_evaluation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    final_evaluation_sha256: str = Field(pattern=SHA256_PATTERN)
    technical_report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    technical_report_sha256: str = Field(pattern=SHA256_PATTERN)
    final_state_sequence: int = Field(ge=1)
    final_state_sha256: str = Field(pattern=SHA256_PATTERN)
    completed_at_utc: datetime
    smoke_only: Literal[True] = True
    scientific_performance_claim_allowed: Literal[False] = False

    @field_validator("completed_at_utc")
    @classmethod
    def _completion_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("campaign completion time must be UTC")
        return value
