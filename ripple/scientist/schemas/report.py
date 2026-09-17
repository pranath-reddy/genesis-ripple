"""Scientific-candidate and technical-integration report contracts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ripple.inference.contracts import (
    M3ToM4BridgeCompletionRecord,
    M3ToM4BridgeRecord,
    M4CompletionRecord,
    M4InferenceResult,
)
from ripple.modeling.service import PreprocessingCompletionRecord
from ripple.preprocessing.mriganka_enn.contracts import (
    MrigankaEnnThreeBandModelInputPackage,
)

from .common import IDENTIFIER_PATTERN, SHA256_PATTERN, FrozenModel, SkyCoordinate
from .lenscat import LensCatAttempt, LensCatAttemptStatus

MrigankaDp2ScientificBlocker = Literal[
    "physical_channel_order_authority",
    "angular_field_and_resampling_policy",
    "psf_matching_policy",
    "hsc_to_rubin_domain_compatibility",
    "score_calibration",
    "candidate_threshold",
]

MRIGANKA_DP2_SCIENTIFIC_BLOCKERS: tuple[MrigankaDp2ScientificBlocker, ...] = (
    "physical_channel_order_authority",
    "angular_field_and_resampling_policy",
    "psf_matching_policy",
    "hsc_to_rubin_domain_compatibility",
    "score_calibration",
    "candidate_threshold",
)


class MrigankaDp2M2BandEvidence(FrozenModel):
    """Content identities from one independently verified live DP2 package."""

    band: Literal["g", "r", "i"]
    package_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    fits_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_id: str = Field(min_length=1, max_length=1024)
    obs_id: str = Field(min_length=1, max_length=256)
    target: SkyCoordinate
    image_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    mask_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    variance_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    celestial_wcs_sha256: str = Field(pattern=SHA256_PATTERN)
    psf_state: Literal["present", "absent_from_package", "unknown"]
    authenticated_live_rubin_rsp: Literal[True] = True
    byte_preserved_before_m3: Literal[True] = True


class MrigankaDp2M3Evidence(FrozenModel):
    """Verified three-band M3 package and its last-written completion record."""

    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    completion_sha256: str = Field(pattern=SHA256_PATTERN)
    package: MrigankaEnnThreeBandModelInputPackage
    completion: PreprocessingCompletionRecord

    @model_validator(mode="after")
    def _completion_binds_package(self) -> MrigankaDp2M3Evidence:
        if self.completion.adapter_package_manifest.file_sha256 != self.manifest_sha256:
            raise ValueError(
                "M3 completion does not bind the reported package manifest"
            )
        if self.completion.recipe_id != self.package.recipe.recipe_id:
            raise ValueError("M3 completion and package recipe identifiers disagree")
        if (
            self.completion.manifest.manifest_id
            != "mriganka-enn-three-band-dp2-provisional-v1"
            or self.completion.manifest.model_id != "deeplense.mriganka.enn-sda"
            or self.completion.adapter.adapter_id != "mriganka-enn-native64-three-band"
            or self.completion.adapter.adapter_version != "v1"
        ):
            raise ValueError("M3 completion is not the audited three-band registration")
        return self


class MrigankaDp2BridgeEvidence(FrozenModel):
    """Verified, byte-preserving singleton-batch M3-to-M4 handoff."""

    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    completion_sha256: str = Field(pattern=SHA256_PATTERN)
    record: M3ToM4BridgeRecord
    completion: M3ToM4BridgeCompletionRecord

    @model_validator(mode="after")
    def _completion_binds_bridge(self) -> MrigankaDp2BridgeEvidence:
        if self.completion.run_id != self.record.run_id:
            raise ValueError("bridge record and completion run identifiers disagree")
        if self.completion.bridge_manifest.sha256 != self.manifest_sha256:
            raise ValueError("bridge completion does not bind the reported manifest")
        if self.completion.model_input_chw != self.record.model_input_chw.file:
            raise ValueError("bridge completion does not bind the reported CHW tensor")
        return self


class MrigankaDp2M4Evidence(FrozenModel):
    """Checkpoint-bound M4 technical result and immutable publication record."""

    result_sha256: str = Field(pattern=SHA256_PATTERN)
    completion_sha256: str = Field(pattern=SHA256_PATTERN)
    result: M4InferenceResult
    completion: M4CompletionRecord

    @model_validator(mode="after")
    def _completion_binds_result(self) -> MrigankaDp2M4Evidence:
        if self.completion.run_id != self.result.run_id:
            raise ValueError("M4 result and completion run identifiers disagree")
        if self.completion.inference_result.sha256 != self.result_sha256:
            raise ValueError("M4 completion does not bind the reported result")
        if self.completion.embedding.sha256 != self.result.embedding.file_sha256:
            raise ValueError("M4 embedding identities disagree")
        return self


class MrigankaDp2TechnicalReport(FrozenModel):
    """Audited Rubin-to-Mriganka wiring report with no scientific claim."""

    schema_version: Literal["ripple.mriganka-dp2-technical-report.v1"] = (
        "ripple.mriganka-dp2-technical-report.v1"
    )
    report_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    target: SkyCoordinate
    m2_bands: tuple[
        MrigankaDp2M2BandEvidence,
        MrigankaDp2M2BandEvidence,
        MrigankaDp2M2BandEvidence,
    ]
    m3: MrigankaDp2M3Evidence
    bridge: MrigankaDp2BridgeEvidence
    m4: MrigankaDp2M4Evidence
    raw_logits: tuple[float, float]
    uncalibrated_softmax_components: tuple[float, float]
    unresolved_scientific_blockers: tuple[MrigankaDp2ScientificBlocker, ...] = (
        MRIGANKA_DP2_SCIENTIFIC_BLOCKERS
    )
    summary: str = Field(min_length=1, max_length=4096)
    assembled_at_utc: datetime
    score_is_calibrated_probability: Literal[False] = False
    probability_claim_allowed: Literal[False] = False
    candidate_threshold_selected: Literal[False] = False
    decision_threshold_applied: Literal[False] = False
    candidate_decision_made: Literal[False] = False
    candidate_report_generated: Literal[False] = False
    scientific_use_allowed: Literal[False] = False
    dp2_scientific_inference_allowed: Literal[False] = False
    lenscat_invoked: Literal[False] = False
    lenscat_association_performed: Literal[False] = False

    @field_validator("assembled_at_utc")
    @classmethod
    def _technical_report_time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("technical report assembly time must be UTC")
        return value

    @model_validator(mode="after")
    def _bind_full_technical_chain(self) -> MrigankaDp2TechnicalReport:
        if tuple(item.band for item in self.m2_bands) != ("g", "r", "i"):
            raise ValueError("technical report M2 evidence must be ordered g, r, i")
        if any(item.target != self.target for item in self.m2_bands):
            raise ValueError("an M2 package was retrieved for a different target")

        sources = self.m3.package.sources
        evidence_by_band = {item.band: item for item in self.m2_bands}
        for source in sources:
            evidence = evidence_by_band[source.band]
            if (
                source.manifest_sha256 != evidence.package_manifest_sha256
                or source.fits_sha256 != evidence.fits_sha256
                or source.dataset_id != evidence.dataset_id
                or source.obs_id != evidence.obs_id
                or not math.isclose(
                    source.ra_deg,
                    evidence.target.ra_deg,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                or not math.isclose(
                    source.dec_deg,
                    evidence.target.dec_deg,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                or source.psf_state != evidence.psf_state
            ):
                raise ValueError("M3 source provenance does not bind the M2 evidence")

        expected_blockers = set(MRIGANKA_DP2_SCIENTIFIC_BLOCKERS)
        if (
            len(self.unresolved_scientific_blockers) != len(expected_blockers)
            or set(self.unresolved_scientific_blockers) != expected_blockers
            or set(self.m3.package.compatibility.unresolved_requirements)
            != expected_blockers
        ):
            raise ValueError("technical report must retain all six scientific blockers")

        source_m3 = self.bridge.record.source_m3
        if (
            source_m3.completion_record.sha256 != self.m3.completion_sha256
            or source_m3.adapter_package_manifest.sha256 != self.m3.manifest_sha256
            or source_m3.registry_manifest_id != self.m3.completion.manifest.manifest_id
            or source_m3.registry_manifest_sha256 != self.m3.completion.manifest.sha256
            or source_m3.model_id != self.m3.completion.manifest.model_id
            or source_m3.model_version != self.m3.completion.manifest.model_version
            or source_m3.adapter_id != self.m3.completion.adapter.adapter_id
            or source_m3.adapter_version != self.m3.completion.adapter.adapter_version
            or source_m3.recipe_id != self.m3.completion.recipe_id
            or source_m3.model_input_bchw.file.sha256
            != self.m3.package.model_input.file_sha256
            or source_m3.model_input_bchw.decoded_sha256
            != self.m3.package.model_input.decoded_array_sha256
            or self.bridge.record.channel_mapping.configured_bands
            != self.m3.package.recipe.channel_bands
        ):
            raise ValueError("bridge provenance does not bind the reported M3 run")

        bridge_provenance = self.m4.result.bridge_provenance
        if (
            self.m4.result.execution_scope_used
            != "unqualified_dp2_technical_integration"
            or bridge_provenance is None
            or bridge_provenance.record != self.bridge.record
            or bridge_provenance.bridge_manifest.sha256 != self.bridge.manifest_sha256
            or bridge_provenance.bridge_completion.sha256
            != self.bridge.completion_sha256
        ):
            raise ValueError("M4 result does not bind the reported audited bridge")
        if self.raw_logits != self.m4.result.logits:
            raise ValueError("reported raw logits do not match the M4 result")
        if self.uncalibrated_softmax_components != self.m4.result.scores:
            raise ValueError("reported softmax components do not match the M4 result")
        maximum_logit = max(self.raw_logits)
        exponentials = tuple(
            math.exp(value - maximum_logit) for value in self.raw_logits
        )
        total = sum(exponentials)
        expected_softmax = tuple(value / total for value in exponentials)
        if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-6)
            for observed, expected in zip(
                self.uncalibrated_softmax_components,
                expected_softmax,
                strict=True,
            )
        ):
            raise ValueError("reported softmax components do not match raw logits")
        if self.assembled_at_utc < self.m4.result.completed_at_utc:
            raise ValueError("technical report cannot predate M4 completion")
        return self


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
    def _bind_final_attempt(self) -> CandidateScientificReport:
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
    def _class_counts_sum(self) -> SyntheticTrainingSmokeEvidence:
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
