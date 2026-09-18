"""Executable configuration for a controlled architecture/model study."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .architecture import ArchitectureCandidate, TrainingConfiguration
from .common import FrozenModel, IDENTIFIER_PATTERN
from .remote import RemoteWorkerSettings


class StudyBedrockModel(FrozenModel):
    model_id: str = Field(min_length=1, max_length=512)
    pricing_model_id: str = Field(min_length=1, max_length=512)
    inference_profile_id: str | None = Field(default=None, max_length=512)
    display_name: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    category: Literal["paid_api", "free_open_source", "external_api"] = (
        "paid_api"
    )


class StudyArchitectureArm(FrozenModel):
    arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    role: Literal["searched", "trained_baseline"]
    candidate: ArchitectureCandidate

    @model_validator(mode="after")
    def _candidate_matches_arm(self) -> "StudyArchitectureArm":
        if self.candidate.candidate_id != self.arm_id:
            raise ValueError("architecture arm and candidate identifiers must match")
        return self


class ArchitectureModelStudyConfiguration(FrozenModel):
    schema_version: Literal["ripple.architecture-model-study-config.v1"] = (
        "ripple.architecture-model-study-config.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    output_root: str = Field(min_length=1, max_length=1024)
    simulation_spec: str = Field(min_length=1, max_length=1024)
    pricing_snapshot: str = Field(min_length=1, max_length=1024)
    remote: RemoteWorkerSettings
    bedrock_profile_name: str = Field(
        default="ripple-bedrock",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    bedrock_region: str = Field(
        default="ap-south-1",
        pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d+$",
    )
    bedrock_models: tuple[StudyBedrockModel, ...] = Field(min_length=1, max_length=32)
    primary_planner_model_id: str = Field(min_length=1, max_length=512)
    tuning_model_id: str = Field(min_length=1, max_length=512)
    arms: tuple[StudyArchitectureArm, ...] = Field(min_length=2, max_length=32)
    baseline_arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    base_training: TrainingConfiguration
    minimum_tuning_iterations: int = Field(default=2, ge=1, le=100)
    maximum_tuning_iterations: int = Field(default=2, ge=1, le=100)
    minimum_validation_balanced_accuracy_for_early_stop: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )
    maximum_generalization_gap_for_early_stop: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
    )
    minimum_validation_improvement: float = Field(default=0.005, ge=0.0, le=1.0)
    random_seeds: tuple[int, ...] = Field(
        default=(20260917,), min_length=1, max_length=1
    )
    architecture_shortlist_size: int = Field(default=2, ge=1, le=4)
    remote_simulation_timeout_seconds: int = Field(default=7200, ge=30, le=86400)
    remote_training_timeout_seconds: int = Field(default=1800, ge=30, le=86400)
    remote_evaluation_timeout_seconds: int = Field(default=600, ge=30, le=86400)
    maximum_provider_requests: int = Field(default=256, ge=1, le=10000)
    maximum_measured_tokens: int = Field(default=2_000_000, ge=1)
    maximum_rate_based_cost_usd: float = Field(default=5.0, gt=0.0, le=10000.0)
    maximum_training_runs: int = Field(default=12, ge=1, le=10000)
    maximum_remote_wall_seconds: float = Field(default=21_600.0, gt=0.0)
    maximum_local_storage_bytes: int = Field(
        default=12 * 1024 * 1024 * 1024,
        ge=64 * 1024 * 1024,
    )
    maximum_transfer_bytes: int = Field(
        default=8 * 1024 * 1024 * 1024,
        ge=64 * 1024 * 1024,
    )
    execute_remote: Literal[True] = True
    scientific_performance_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _valid_study(self) -> "ArchitectureModelStudyConfiguration":
        model_ids = [item.model_id for item in self.bedrock_models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Bedrock model IDs must be unique")
        if self.primary_planner_model_id not in model_ids:
            raise ValueError("primary planner model is absent from the model matrix")
        if self.tuning_model_id not in model_ids:
            raise ValueError("tuning model is absent from the model matrix")
        arm_ids = [item.arm_id for item in self.arms]
        if len(arm_ids) != len(set(arm_ids)):
            raise ValueError("architecture arm IDs must be unique")
        if self.baseline_arm_id not in arm_ids:
            raise ValueError("baseline arm is absent from the architecture plan")
        if sum(item.role == "trained_baseline" for item in self.arms) != 1:
            raise ValueError("study v1 requires exactly one trained baseline")
        baseline = next(item for item in self.arms if item.arm_id == self.baseline_arm_id)
        if baseline.role != "trained_baseline":
            raise ValueError("baseline arm must use the trained_baseline role")
        if self.minimum_tuning_iterations > self.maximum_tuning_iterations:
            raise ValueError("minimum tuning iterations exceed the maximum")
        if len(self.random_seeds) != len(set(self.random_seeds)):
            raise ValueError("study seeds must be unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.random_seeds):
            raise ValueError("study seeds must fit unsigned 32-bit integers")
        if self.base_training.seed != self.random_seeds[0]:
            raise ValueError("base training seed must match the declared study seed")
        if not self.base_training.require_cuda:
            raise ValueError("the offline study requires CUDA training")
        required_training_runs = len(self.arms) * self.maximum_tuning_iterations
        if self.maximum_training_runs < required_training_runs:
            raise ValueError("training-run budget cannot cover the declared hard loop")
        return self


__all__ = [
    "ArchitectureModelStudyConfiguration",
    "StudyArchitectureArm",
    "StudyBedrockModel",
]
