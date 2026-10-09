"""Typed architecture-search contracts derived from DeepLense AI Scientist."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN


class ArchitectureFamily(str, Enum):
    CNN = "cnn"
    RESNET = "resnet"
    VIT = "vit"
    EQUIVARIANT = "equivariant"
    MLPMIXER = "mlpmixer"
    HYBRID = "hybrid"


class DatasetSummary(FrozenModel):
    """Only aggregate metadata exposed to architecture agents."""

    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    image_shape: tuple[int, int]
    channels: int = Field(ge=1, le=32)
    class_names: tuple[str, ...] = Field(min_length=2, max_length=32)
    split_counts: dict[Literal["train", "validation", "test"], int]
    purpose: Literal["integration_smoke", "scientific_training"]
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_summary(self) -> "DatasetSummary":
        if any(axis < 16 or axis > 4096 for axis in self.image_shape):
            raise ValueError("image dimensions must be between 16 and 4096 pixels")
        if len(self.class_names) != len(set(self.class_names)):
            raise ValueError("class names must be unique")
        if any(count <= 0 for count in self.split_counts.values()):
            raise ValueError("every frozen split must be non-empty")
        return self


class ArchitectureCandidate(FrozenModel):
    """LLM-proposed structure; dataset-bound dimensions are deliberately absent."""

    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(pattern=IDENTIFIER_PATTERN)
    family: ArchitectureFamily
    depths: tuple[int, ...] = Field(min_length=2, max_length=4)
    widths: tuple[int, ...] = Field(min_length=2, max_length=4)
    physics_informed: bool = False
    rationale: str = Field(min_length=1, max_length=1200)

    @model_validator(mode="after")
    def _validate_structure(self) -> "ArchitectureCandidate":
        if len(self.depths) != len(self.widths):
            raise ValueError("depth and width stages must have equal length")
        if any(depth < 1 or depth > 6 for depth in self.depths):
            raise ValueError("each stage depth must be between 1 and 6")
        if any(width < 8 or width > 1024 for width in self.widths):
            raise ValueError("each stage width must be between 8 and 1024")
        return self


class ArchitectureSearchPlan(FrozenModel):
    """Typed result of the architecture-planning agent."""

    schema_version: Literal["ripple.architecture-search-plan.v1"] = (
        "ripple.architecture-search-plan.v1"
    )
    objective: Literal["binary_strong_lens_classification"]
    candidates: tuple[ArchitectureCandidate, ...] = Field(min_length=2, max_length=4)
    comparison_rationale: str = Field(min_length=1, max_length=1600)
    smoke_only: bool

    @field_validator("candidates")
    @classmethod
    def _unique_candidates(
        cls, value: tuple[ArchitectureCandidate, ...]
    ) -> tuple[ArchitectureCandidate, ...]:
        ids = [candidate.candidate_id for candidate in value]
        names = [candidate.name for candidate in value]
        structures = [
            (candidate.family, candidate.depths, candidate.widths)
            for candidate in value
        ]
        if len(ids) != len(set(ids)) or len(names) != len(set(names)):
            raise ValueError("candidate IDs and names must be unique")
        if len(structures) != len(set(structures)):
            raise ValueError("candidate structures must be distinct")
        return value


class ArchitectureJudgeVerdict(FrozenModel):
    schema_version: Literal["ripple.architecture-judge-verdict.v1"] = (
        "ripple.architecture-judge-verdict.v1"
    )
    ranking: tuple[str, ...] = Field(min_length=2, max_length=4)
    shortlist: tuple[str, ...] = Field(min_length=1, max_length=4)
    reasoning: str = Field(min_length=1, max_length=1600)
    smoke_only: bool

    @model_validator(mode="after")
    def _valid_ranking(self) -> "ArchitectureJudgeVerdict":
        if len(self.ranking) != len(set(self.ranking)):
            raise ValueError("architecture ranking contains duplicates")
        if len(self.shortlist) != len(set(self.shortlist)):
            raise ValueError("architecture shortlist contains duplicates")
        if not set(self.shortlist) <= set(self.ranking):
            raise ValueError("shortlist must be selected from the ranking")
        if tuple(self.ranking[: len(self.shortlist)]) != self.shortlist:
            raise ValueError("shortlist must be the leading segment of the ranking")
        return self


class BoundArchitecture(FrozenModel):
    """Code-bound candidate ready for deterministic construction."""

    candidate: ArchitectureCandidate
    input_shape: tuple[int, int]
    channels: int = Field(ge=1, le=32)
    num_classes: int = Field(ge=2, le=32)
    source_backend: Literal["ripple-native-model-builders"] = (
        "ripple-native-model-builders"
    )
    source_revision: str = Field(min_length=7, max_length=128)


class TrainingConfiguration(FrozenModel):
    loss: Literal["cross_entropy"] = "cross_entropy"
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=3e-4, gt=0.0, le=0.1)
    batch_size: int = Field(default=2, ge=1, le=4096)
    epochs: int = Field(default=2, ge=1, le=1000)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    lr_scheduler: Literal["none", "cosine"] = "cosine"
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    augment_d4: bool = False
    seed: int = Field(default=20260917, ge=0, le=2**32 - 1)
    require_cuda: bool = True


class ArchitectureEvaluation(FrozenModel):
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_loss: float = Field(ge=0.0)
    validation_accuracy: float = Field(ge=0.0, le=1.0)
    validation_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    elapsed_gpu_seconds: float = Field(ge=0.0)
    epochs_completed: int = Field(ge=1)
    smoke_only: bool


class ArchitectureSearchResult(FrozenModel):
    schema_version: Literal["ripple.architecture-search-result.v1"] = (
        "ripple.architecture-search-result.v1"
    )
    plan_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluations: tuple[ArchitectureEvaluation, ...] = Field(min_length=1)
    selected_candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selection_metric: Literal["validation_balanced_accuracy"] = (
        "validation_balanced_accuracy"
    )
    scientific_selection_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _selected_candidate_exists(self) -> "ArchitectureSearchResult":
        if self.selected_candidate_id not in {
            item.candidate_id for item in self.evaluations
        }:
            raise ValueError("selected candidate has no measured evaluation")
        return self
