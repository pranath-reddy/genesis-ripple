"""Content-addressed dataset and split contracts."""

from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN, SourceIdentity


class SampleRecord(FrozenModel):
    sample_id: str = Field(pattern=IDENTIFIER_PATTERN)
    label_name: Literal["non_lens", "lens"]
    label_index: Literal[0, 1]
    seed: int = Field(ge=0, le=2**32 - 1)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(pattern=SHA256_PATTERN)
    shape: tuple[int, int, int]
    dtype: Literal["float32"] = "float32"
    finite: Literal[True] = True
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0.0)
    simulator_parameters_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _consistent_label_and_shape(self) -> "SampleRecord":
        expected = 1 if self.label_name == "lens" else 0
        if self.label_index != expected:
            raise ValueError("label name and index disagree")
        if any(size <= 0 for size in self.shape):
            raise ValueError("sample shape dimensions must be positive")
        if self.minimum > self.maximum:
            raise ValueError("sample minimum exceeds maximum")
        return self


class SplitConfiguration(FrozenModel):
    strategy: Literal["stratified_seeded"] = "stratified_seeded"
    train_fraction: float = Field(default=0.6, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = Field(default=20260917, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _sums_to_one(self) -> "SplitConfiguration":
        if (
            abs(
                self.train_fraction
                + self.validation_fraction
                + self.test_fraction
                - 1.0
            )
            > 1e-9
        ):
            raise ValueError("split fractions must sum to one")
        return self


class FrozenSplits(FrozenModel):
    strategy: Literal["stratified_seeded"] = "stratified_seeded"
    seed: int = Field(ge=0, le=2**32 - 1)
    train_sample_ids: tuple[str, ...] = Field(min_length=2)
    validation_sample_ids: tuple[str, ...] = Field(min_length=2)
    test_sample_ids: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _disjoint(self) -> "FrozenSplits":
        collections = [
            set(self.train_sample_ids),
            set(self.validation_sample_ids),
            set(self.test_sample_ids),
        ]
        if any(
            len(items) != len(original)
            for items, original in zip(
                collections,
                (
                    self.train_sample_ids,
                    self.validation_sample_ids,
                    self.test_sample_ids,
                ),
                strict=True,
            )
        ):
            raise ValueError("sample IDs must be unique inside each split")
        if (
            collections[0] & collections[1]
            or collections[0] & collections[2]
            or collections[1] & collections[2]
        ):
            raise ValueError("dataset splits must be disjoint")
        return self


class DatasetManifest(FrozenModel):
    schema_version: Literal["ripple.binary-lens-dataset.v1"] = (
        "ripple.binary-lens-dataset.v1"
    )
    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    purpose: Literal["integration_smoke", "scientific_training"]
    simulator: SourceIdentity
    simulation_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    class_names: tuple[Literal["non_lens", "lens"], Literal["non_lens", "lens"]] = (
        "non_lens",
        "lens",
    )
    bands: tuple[str, ...] = Field(min_length=1)
    samples: tuple[SampleRecord, ...] = Field(min_length=6)
    splits: FrozenSplits
    scientific_use_allowed: bool = False
    qualification_notes: tuple[str, ...] = ()

    @field_validator("bands")
    @classmethod
    def _unique_bands(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(not band.strip() for band in value):
            raise ValueError("bands must be unique and non-empty")
        return value

    @model_validator(mode="after")
    def _validate_manifest(self) -> "DatasetManifest":
        if self.class_names != ("non_lens", "lens"):
            raise ValueError("binary label order must be non_lens=0, lens=1")
        ids = [sample.sample_id for sample in self.samples]
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs must be unique")
        expected = set(ids)
        split_ids = (
            set(self.splits.train_sample_ids)
            | set(self.splits.validation_sample_ids)
            | set(self.splits.test_sample_ids)
        )
        if split_ids != expected:
            raise ValueError("frozen splits must cover every sample exactly once")
        sample_by_id = {sample.sample_id: sample for sample in self.samples}
        for split in (
            self.splits.train_sample_ids,
            self.splits.validation_sample_ids,
            self.splits.test_sample_ids,
        ):
            labels = Counter(sample_by_id[item].label_name for item in split)
            if labels["lens"] == 0 or labels["non_lens"] == 0:
                raise ValueError("each split must contain both binary classes")
        if self.purpose == "integration_smoke" and self.scientific_use_allowed:
            raise ValueError(
                "an integration smoke dataset cannot authorize scientific use"
            )
        first_shape = self.samples[0].shape
        if any(sample.shape != first_shape for sample in self.samples):
            raise ValueError("all dataset samples must have one tensor shape")
        if first_shape[0] != len(self.bands):
            raise ValueError("sample channels must equal the declared band count")
        return self
