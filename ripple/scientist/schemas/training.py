"""Typed records for safe CUDA training and held-out evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .architecture import BoundArchitecture, TrainingConfiguration
from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN


class BinaryMetrics(FrozenModel):
    loss: float = Field(ge=0.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    sample_count: int = Field(ge=1)


class EpochRecord(FrozenModel):
    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0.0)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    validation: BinaryMetrics


class NormalizationRecord(FrozenModel):
    method: Literal["per_channel_train_mean_std"] = "per_channel_train_mean_std"
    mean: tuple[float, ...] = Field(min_length=1)
    standard_deviation: tuple[float, ...] = Field(min_length=1)
    derived_from_split: Literal["train"] = "train"

    @model_validator(mode="after")
    def _matching_channels(self) -> "NormalizationRecord":
        if len(self.mean) != len(self.standard_deviation):
            raise ValueError("normalization mean/std channel counts differ")
        if any(value <= 0.0 for value in self.standard_deviation):
            raise ValueError("normalization standard deviations must be positive")
        return self


class TrainingRunRecord(FrozenModel):
    schema_version: Literal["ripple.training-run.v1"] = "ripple.training-run.v1"
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    architecture: BoundArchitecture
    configuration: TrainingConfiguration
    normalization: NormalizationRecord
    epochs: tuple[EpochRecord, ...] = Field(min_length=1)
    best_epoch: int = Field(ge=1)
    best_validation: BinaryMetrics
    best_checkpoint_train: BinaryMetrics | None = None
    weights_relative_path: str = Field(min_length=1, max_length=1024)
    weights_sha256: str = Field(pattern=SHA256_PATTERN)
    checkpoint_metadata_relative_path: str = Field(min_length=1, max_length=1024)
    checkpoint_metadata_sha256: str = Field(pattern=SHA256_PATTERN)
    parameter_count: int = Field(gt=0)
    device: str = Field(min_length=1, max_length=256)
    python_version: str = Field(min_length=1, max_length=64)
    torch_version: str = Field(min_length=1, max_length=128)
    cuda_version: str | None = Field(default=None, max_length=128)
    elapsed_seconds: float = Field(ge=0.0)
    smoke_only: bool
    scientific_performance_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _best_epoch_exists(self) -> "TrainingRunRecord":
        matching = [record for record in self.epochs if record.epoch == self.best_epoch]
        if len(matching) != 1 or matching[0].validation != self.best_validation:
            raise ValueError("best epoch and validation record disagree")
        if (
            self.best_checkpoint_train is not None
            and self.best_checkpoint_train.sample_count < 1
        ):
            raise ValueError("checkpoint train evaluation must be non-empty")
        return self


class PredictionRecord(FrozenModel):
    sample_id: str = Field(pattern=IDENTIFIER_PATTERN)
    true_label: Literal[0, 1]
    predicted_label: Literal[0, 1]
    non_lens_score: float = Field(ge=0.0, le=1.0)
    lens_score: float = Field(ge=0.0, le=1.0)


class FinalEvaluationRecord(FrozenModel):
    schema_version: Literal["ripple.final-evaluation.v1"] = "ripple.final-evaluation.v1"
    evaluation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_training_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    dataset_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    split: Literal["test"] = "test"
    metrics: BinaryMetrics
    predictions: tuple[PredictionRecord, ...] = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    smoke_only: bool
    scientific_performance_claim_allowed: Literal[False] = False
