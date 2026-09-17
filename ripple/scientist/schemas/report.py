"""Scientific-candidate and synthetic-smoke report contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import FrozenModel, IDENTIFIER_PATTERN, SHA256_PATTERN, SkyCoordinate
from .lenscat import LensCatAttempt, LensCatAttemptStatus


class CandidateEvidence(FrozenModel):
    """Classifier evidence available before final catalog association."""

    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    coordinate: SkyCoordinate
    classifier_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    lens_score: float = Field(ge=0.0, le=1.0)
    decision_threshold: float = Field(ge=0.0, le=1.0)
    input_artifact_ids: tuple[str, ...] = Field(min_length=1)
    quality_flags: tuple[str, ...] = ()


class CandidateScientificReport(FrozenModel):
    """Candidate report whose final evidence step is a LensCat attempt."""

    schema_version: Literal["ripple.candidate-scientific-report.v1"] = (
        "ripple.candidate-scientific-report.v1"
    )
    report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    candidate: CandidateEvidence
    lenscat_attempt: LensCatAttempt
    catalog_disposition: LensCatAttemptStatus
    final_pre_report_step: Literal["lenscat_attempt"] = "lenscat_attempt"
    final_pre_report_attempt_id: str = Field(pattern=IDENTIFIER_PATTERN)
    summary: str = Field(min_length=1, max_length=4096)
    limitations: tuple[str, ...] = Field(min_length=1)
    assembled_at_utc: datetime
    catalog_association_is_classifier_label: Literal[False] = False
    automatic_confirmation_claim_allowed: Literal[False] = False

    @field_validator("assembled_at_utc")
    @classmethod
    def _assembled_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("report assembly time must be UTC")
        return value

    @model_validator(mode="after")
    def _bind_final_attempt(self) -> "CandidateScientificReport":
        if self.candidate.candidate_id != self.lenscat_attempt.query.candidate_id:
            raise ValueError("LensCat attempt belongs to a different candidate")
        if self.candidate.coordinate != self.lenscat_attempt.query.coordinate:
            raise ValueError("LensCat attempt used a different sky coordinate")
        if self.catalog_disposition != self.lenscat_attempt.status:
            raise ValueError("catalog disposition does not match LensCat attempt")
        if self.final_pre_report_attempt_id != self.lenscat_attempt.attempt_id:
            raise ValueError("final pre-report step does not reference LensCat attempt")
        if self.assembled_at_utc < self.lenscat_attempt.completed_at_utc:
            raise ValueError("report cannot predate its mandatory LensCat attempt")
        return self


class SyntheticTrainingSmokeEvidence(FrozenModel):
    """Minimal evidence for a wiring-only synthetic training run."""

    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_training_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    evaluation_artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    total_samples: int = Field(ge=2)
    lens_samples: int = Field(ge=1)
    non_lens_samples: int = Field(ge=1)
    has_sky_coordinates: Literal[False] = False

    @model_validator(mode="after")
    def _class_counts_sum(self) -> "SyntheticTrainingSmokeEvidence":
        if self.lens_samples + self.non_lens_samples != self.total_samples:
            raise ValueError("synthetic class counts must sum to total samples")
        return self


class SyntheticTrainingTechnicalReport(FrozenModel):
    """Technical smoke report, deliberately separate from candidate reporting."""

    schema_version: Literal["ripple.synthetic-training-technical-report.v1"] = (
        "ripple.synthetic-training-technical-report.v1"
    )
    report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    evidence: SyntheticTrainingSmokeEvidence
    summary: str = Field(min_length=1, max_length=4096)
    lenscat_status: Literal["not_applicable_no_sky_coordinates"] = (
        "not_applicable_no_sky_coordinates"
    )
    lenscat_attempt_performed: Literal[False] = False
    smoke_only: Literal[True] = True
    scientific_performance_claim_allowed: Literal[False] = False
    assembled_at_utc: datetime

    @field_validator("assembled_at_utc")
    @classmethod
    def _assembled_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("report assembly time must be UTC")
        return value
