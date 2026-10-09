"""Strict records for checkpoint-bound M4 inference."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class CheckpointSpec(FrozenModel):
    role: Literal["encoder", "classifier"]
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    byte_count: int = Field(gt=0, le=256 * 1024 * 1024)
    sha256: str = Field(pattern=SHA256_PATTERN)
    serialization: Literal["pytorch_weights_only_state_dict"] = (
        "pytorch_weights_only_state_dict"
    )

    @field_validator("filename")
    @classmethod
    def _basename_only(cls, value: str) -> str:
        if PurePath(value).name != value or value in {".", ".."}:
            raise ValueError("checkpoint filename must be a plain basename")
        return value


class UpstreamSourceFile(FrozenModel):
    filename: Literal[
        "read_module.py",
        "preprocess_module.py",
        "model.py",
        "main.py",
    ]
    sha256: str = Field(pattern=SHA256_PATTERN)


class UpstreamExecutionLogEvidence(FrozenModel):
    filename: Literal["execution_log_20260917_130238.txt"]
    sha256: Literal["9c5b89428141edf9ecc7015a84c8706da3e64d7802ef6324a37f9c72bd3fe428"]
    selected_preprocess_mode: Literal["none"]
    evidence_purpose: Literal["provenance_only"] = "provenance_only"
    runtime_verified: Literal[False] = False


class UpstreamPreprocessingEvidence(FrozenModel):
    evidence_scope: Literal["evaluation_loader_from_serialized_npy"]
    source_repository: Literal["https://github.com/astrofyz/sda_deeplense"]
    source_commit: Literal["d5f245c7e169436b83c4ab83e451ad29e52689cb"]
    source_files: tuple[UpstreamSourceFile, ...] = Field(
        min_length=4,
        max_length=4,
    )
    evaluation_order: tuple[
        Literal["load_npy_as_float32"],
        Literal["center_crop_64"],
        Literal["independent_channel_minmax_over_y_x"],
        Literal["nan_to_num_zero"],
        Literal["optional_preprocess_none_identity"],
    ]
    execution_log: UpstreamExecutionLogEvidence

    @model_validator(mode="after")
    def _exact_source_snapshot(self) -> UpstreamPreprocessingEvidence:
        expected = {
            "read_module.py": (
                "4e7ea32fb187e58dcbf59946351ce9e27b51c64e90a76dc151fb2603cec62d2d"
            ),
            "preprocess_module.py": (
                "377335dceec171b7d24da03247c5af8edce04437e660d542c6b1e16cbeb53291"
            ),
            "model.py": (
                "c904ba2171467ba6c93a56688e5a92a274b3ebbf1f5832258966208a0c485ce8"
            ),
            "main.py": (
                "18ef79fc09d403cd8a89491d083c81de90d127ef1fedd9a6f974828cf08ca0a7"
            ),
        }
        observed = {item.filename: item.sha256 for item in self.source_files}
        if observed != expected or len(observed) != len(self.source_files):
            raise ValueError(
                "upstream source identities do not match the audited snapshot"
            )
        return self


class HscObservationContext(FrozenModel):
    survey_release: Literal["HSC-SSP PDR2 Wide"]
    reported_physical_bands: tuple[Literal["g"], Literal["r"], Literal["i"]]
    source_cutout_shape_yx: tuple[Literal[72], Literal[72]]
    source_cutout_fov_arcsec_approx_yx: tuple[Literal[11.5], Literal[11.5]]
    model_image_shape_yx: tuple[Literal[64], Literal[64]]
    model_image_fov_arcsec_approx_yx: tuple[Literal[10.2], Literal[10.2]]
    evidence_source: Literal["arXiv:2410.01203v1"]
    serialized_channel_order_proven: Literal[False] = False


class InputTensorSpec(FrozenModel):
    artifact_axes: tuple[Literal["channel", "y", "x"], ...]
    artifact_shape: tuple[int, ...]
    runtime_axes: tuple[Literal["batch", "channel", "y", "x"], ...]
    runtime_shape: tuple[int, ...]
    dtype: Literal["float32"]
    value_range: tuple[float, float]
    channel_semantics: tuple[str, ...]
    physical_band_mapping: Literal["unresolved"]
    upstream_preprocessing: UpstreamPreprocessingEvidence
    observation_context: HscObservationContext

    @model_validator(mode="after")
    def _exact_checkpoint_boundary(self) -> InputTensorSpec:
        if self.artifact_axes != ("channel", "y", "x"):
            raise ValueError("M4 artifact axes must be channel,y,x")
        if self.artifact_shape != (3, 64, 64):
            raise ValueError("M4 artifact shape must be 3x64x64")
        if self.runtime_axes != ("batch", "channel", "y", "x"):
            raise ValueError("M4 runtime axes must be batch,channel,y,x")
        if self.runtime_shape != (1, 3, 64, 64):
            raise ValueError("M4 runtime shape must be 1x3x64x64")
        if self.value_range != (0.0, 1.0):
            raise ValueError("M4 values must lie in the closed interval [0,1]")
        if len(self.channel_semantics) != 3 or len(set(self.channel_semantics)) != 3:
            raise ValueError("M4 requires three distinct channel placeholders")
        return self


class OutputSpec(FrozenModel):
    raw_output: Literal["two_logits"] = "two_logits"
    score_transform: Literal["softmax"] = "softmax"
    labels_by_index: tuple[Literal["non_lens", "lens"], ...]
    lens_score_index: Literal[1] = 1
    score_is_calibrated_probability: Literal[False] = False
    decision_threshold: None = None

    @model_validator(mode="after")
    def _exact_label_contract(self) -> OutputSpec:
        if self.labels_by_index != ("non_lens", "lens"):
            raise ValueError("label index 0 must be non_lens and index 1 lens")
        return self


class ReferenceCaseSpec(FrozenModel):
    case_id: str = Field(pattern=IDENTIFIER_PATTERN)
    input_filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
    input_file_sha256: str = Field(pattern=SHA256_PATTERN)
    input_decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_scores: tuple[float, float]
    maximum_absolute_error: float = Field(gt=0.0, le=1e-3)
    evidence_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_artifact_purpose: Literal["provenance_only"] = "provenance_only"
    evidence_artifact_runtime_verified: Literal[False] = False

    @model_validator(mode="after")
    def _scores_are_valid_softmax_values(self) -> ReferenceCaseSpec:
        if any(value < 0.0 or value > 1.0 for value in self.expected_scores):
            raise ValueError("reference scores must lie in [0,1]")
        if abs(sum(self.expected_scores) - 1.0) > 1e-6:
            raise ValueError("reference scores must sum to one")
        return self


class MrigankaEnnBundleManifest(FrozenModel):
    schema_version: Literal["ripple.mriganka-enn-bundle.v1"] = (
        "ripple.mriganka-enn-bundle.v1"
    )
    bundle_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: Literal["deeplense.mriganka.enn-sda"] = "deeplense.mriganka.enn-sda"
    model_version: Literal["sda-epoch-20-iteration-0"] = "sda-epoch-20-iteration-0"
    architecture_id: Literal["mriganka-enn-d4-concrete-v1"] = (
        "mriganka-enn-d4-concrete-v1"
    )
    encoder: CheckpointSpec
    classifier: CheckpointSpec
    input: InputTensorSpec
    output: OutputSpec
    reference_cases: tuple[ReferenceCaseSpec, ...] = ()
    execution_scope: Literal[
        "hsc_reference_or_unqualified_dp2_technical_integration"
    ] = "hsc_reference_or_unqualified_dp2_technical_integration"
    scientific_use_allowed: Literal[False] = False
    dp2_inference_allowed: Literal[False] = False
    unresolved_requirements: tuple[str, ...] = Field(min_length=1)
    provenance_notes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent_bundle(self) -> MrigankaEnnBundleManifest:
        if self.encoder.role != "encoder" or self.classifier.role != "classifier":
            raise ValueError("checkpoint roles are reversed or missing")
        if self.encoder.filename == self.classifier.filename:
            raise ValueError("encoder and classifier must be separate files")
        case_ids = tuple(case.case_id for case in self.reference_cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("reference case IDs must be unique")
        if len(self.unresolved_requirements) != len(set(self.unresolved_requirements)):
            raise ValueError("unresolved requirements must be unique")
        return self

    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class FileIdentity(FrozenModel):
    filename: str = Field(min_length=1, max_length=256)
    byte_count: int = Field(gt=0, le=256 * 1024 * 1024)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M3BchwArrayEvidence(FrozenModel):
    file: FileIdentity
    decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: Literal["float32"]
    shape: tuple[Literal[1], Literal[3], Literal[64], Literal[64]]
    axes: tuple[
        Literal["batch"],
        Literal["channel"],
        Literal["y"],
        Literal["x"],
    ]
    minimum: float = Field(ge=0.0, le=1.0)
    maximum: float = Field(ge=0.0, le=1.0)
    all_finite: Literal[True] = True
    native_float32: Literal[True] = True

    @model_validator(mode="after")
    def _fixed_m3_array(self) -> M3BchwArrayEvidence:
        if self.file.filename != "model_input_bchw.npy":
            raise ValueError("M3 bridge source must be model_input_bchw.npy")
        if self.minimum > self.maximum:
            raise ValueError("M3 bridge source minimum exceeds maximum")
        return self


class M4ChwArrayEvidence(FrozenModel):
    file: FileIdentity
    decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: Literal["float32"]
    shape: tuple[Literal[3], Literal[64], Literal[64]]
    axes: tuple[Literal["channel"], Literal["y"], Literal["x"]]
    minimum: float = Field(ge=0.0, le=1.0)
    maximum: float = Field(ge=0.0, le=1.0)
    all_finite: Literal[True] = True
    native_float32: Literal[True] = True

    @model_validator(mode="after")
    def _fixed_m4_array(self) -> M4ChwArrayEvidence:
        if self.file.filename != "model_input_chw.npy":
            raise ValueError("M4 bridge output must be model_input_chw.npy")
        if self.minimum > self.maximum:
            raise ValueError("M4 bridge output minimum exceeds maximum")
        return self


class ConfiguredChannelMapping(FrozenModel):
    status: Literal["configured_unverified"] = "configured_unverified"
    source_channel_axis: Literal[1] = 1
    output_channel_axis: Literal[0] = 0
    channel_indices: tuple[Literal[0], Literal[1], Literal[2]] = (0, 1, 2)
    configured_bands: tuple[Literal["g", "r", "i"], ...]
    checkpoint_physical_mapping_verified: Literal[False] = False

    @field_validator("configured_bands")
    @classmethod
    def _band_permutation(
        cls, value: tuple[Literal["g", "r", "i"], ...]
    ) -> tuple[Literal["g", "r", "i"], ...]:
        if len(value) != 3 or set(value) != {"g", "r", "i"}:
            raise ValueError("configured bands must be one unique g/r/i permutation")
        return value


class COrderPayloadIntegrity(FrozenModel):
    operation: Literal["select-singleton-batch-index-0"] = (
        "select-singleton-batch-index-0"
    )
    source_c_order_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    output_c_order_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    payload_identical: Literal[True] = True

    @model_validator(mode="after")
    def _same_payload(self) -> COrderPayloadIntegrity:
        if self.source_c_order_payload_sha256 != self.output_c_order_payload_sha256:
            raise ValueError("bridge operation changed the C-order numeric payload")
        return self


class M3BridgeSourceProvenance(FrozenModel):
    run_directory: str = Field(min_length=1, max_length=4096)
    completion_record: FileIdentity
    run_envelope: FileIdentity
    selected_model_manifest: FileIdentity
    adapter_package_manifest: FileIdentity
    registry_manifest_id: Literal["mriganka-enn-three-band-dp2-provisional-v1"]
    registry_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    model_id: Literal["deeplense.mriganka.enn-sda"]
    model_version: Literal["sda-epoch-20-iteration-0-dp2-provisional-v1"]
    adapter_id: Literal["mriganka-enn-native64-three-band"]
    adapter_version: Literal["v1"]
    recipe_id: Literal["mriganka-enn-dp2-native64-three-band-minmax-v1"]
    package_schema_version: Literal[
        "ripple.preprocessing.mriganka-enn-three-band-model-input.v1"
    ]
    model_input_bchw: M3BchwArrayEvidence

    @field_validator("run_directory")
    @classmethod
    def _absolute_run_directory(cls, value: str) -> str:
        parsed = PurePath(value)
        if not parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(
                "source M3 run directory must be absolute without traversal"
            )
        return value

    @model_validator(mode="after")
    def _fixed_source_filenames(self) -> M3BridgeSourceProvenance:
        observed = (
            self.completion_record.filename,
            self.run_envelope.filename,
            self.selected_model_manifest.filename,
            self.adapter_package_manifest.filename,
        )
        if observed != (
            "completion.json",
            "run-envelope.json",
            "selected-model-manifest.json",
            "manifest.json",
        ):
            raise ValueError("M3 source record filenames do not match the contract")
        return self


class M3ToM4BridgeRecord(FrozenModel):
    schema_version: Literal["ripple.m3-to-m4-bridge.v1"] = "ripple.m3-to-m4-bridge.v1"
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    created_at_utc: datetime
    source_m3: M3BridgeSourceProvenance
    channel_mapping: ConfiguredChannelMapping
    model_input_chw: M4ChwArrayEvidence
    payload_integrity: COrderPayloadIntegrity
    implementation_source_sha256: dict[str, str]
    scientific_use_allowed: Literal[False] = False
    dp2_inference_allowed: Literal[False] = False
    candidate_decision_made: Literal[False] = False
    proof_boundary: Literal[
        "Mechanical M3-to-M4 tensor handoff only. The configured g/r/i order is not verified as the checkpoint's physical channel mapping, and Rubin DP2 scientific compatibility remains unresolved."
    ] = (
        "Mechanical M3-to-M4 tensor handoff only. The configured g/r/i order is not "
        "verified as the checkpoint's physical channel mapping, and Rubin DP2 scientific "
        "compatibility remains unresolved."
    )

    @field_validator("created_at_utc")
    @classmethod
    def _aware_bridge_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bridge creation time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _bridge_consistency(self) -> M3ToM4BridgeRecord:
        expected_sources = {"contracts.py", "m3_bridge.py"}
        if set(self.implementation_source_sha256) != expected_sources or any(
            re.fullmatch(SHA256_PATTERN, value) is None
            for value in self.implementation_source_sha256.values()
        ):
            raise ValueError("bridge implementation source identity is invalid")
        return self


class M4BridgeProvenance(FrozenModel):
    bridge_manifest: FileIdentity
    bridge_completion: FileIdentity
    record: M3ToM4BridgeRecord

    @model_validator(mode="after")
    def _bridge_record_names(self) -> M4BridgeProvenance:
        if (
            self.bridge_manifest.filename != "bridge.json"
            or self.bridge_completion.filename != "completion.json"
        ):
            raise ValueError("verified bridge records have unexpected filenames")
        return self


class InputArrayEvidence(FrozenModel):
    file: FileIdentity
    decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: Literal["float32"]
    shape: tuple[int, int, int]
    minimum: float
    maximum: float
    all_finite: Literal[True] = True

    @model_validator(mode="after")
    def _array_contract(self) -> InputArrayEvidence:
        if self.shape != (3, 64, 64):
            raise ValueError("inference evidence requires a 3x64x64 array")
        if self.minimum < 0.0 or self.maximum > 1.0 or self.minimum > self.maximum:
            raise ValueError("input values fall outside the checkpoint contract")
        return self


class ArrayArtifact(FrozenModel):
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    byte_count: int = Field(gt=0, le=4 * 1024 * 1024)
    file_sha256: str = Field(pattern=SHA256_PATTERN)
    decoded_sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: Literal["float32"]
    shape: tuple[int, ...]


class ReferenceVerification(FrozenModel):
    status: Literal["reproduced", "not_applicable"]
    case_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    maximum_absolute_error: float | None = Field(default=None, ge=0.0)
    tolerance: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _status_fields(self) -> ReferenceVerification:
        populated = (
            self.case_id is not None
            and self.maximum_absolute_error is not None
            and self.tolerance is not None
        )
        if populated != (self.status == "reproduced"):
            raise ValueError("reference verification fields disagree with status")
        if (
            self.status == "reproduced"
            and self.maximum_absolute_error is not None
            and self.tolerance is not None
            and self.maximum_absolute_error > self.tolerance
        ):
            raise ValueError("reference output exceeded its tolerance")
        return self


class RuntimeIdentity(FrozenModel):
    python: str = Field(min_length=1, max_length=64)
    numpy: str = Field(min_length=1, max_length=64)
    pydantic: str = Field(min_length=1, max_length=64)
    torch: str = Field(min_length=1, max_length=64)
    device: Literal["cpu"] = "cpu"
    weights_only_loading: Literal[True] = True
    deterministic_algorithms: Literal[True] = True


class M4InferenceResult(FrozenModel):
    schema_version: Literal["ripple.m4-inference-result.v1"] = (
        "ripple.m4-inference-result.v1"
    )
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    completed_at_utc: datetime
    bundle_id: str = Field(pattern=IDENTIFIER_PATTERN)
    bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    architecture_id: str = Field(pattern=IDENTIFIER_PATTERN)
    input: InputArrayEvidence
    encoder_checkpoint: FileIdentity
    classifier_checkpoint: FileIdentity
    embedding: ArrayArtifact
    logits: tuple[float, float]
    scores: tuple[float, float]
    lens_score: float = Field(ge=0.0, le=1.0)
    reference_verification: ReferenceVerification
    runtime: RuntimeIdentity
    implementation_source_sha256: dict[str, str]
    execution_scope_used: Literal[
        "hsc_reference_path",
        "unqualified_dp2_technical_integration",
    ] = "hsc_reference_path"
    bridge_provenance: M4BridgeProvenance | None = None
    score_is_calibrated_probability: Literal[False] = False
    decision_threshold_applied: Literal[False] = False
    candidate_decision_made: Literal[False] = False
    scientific_use_allowed: Literal[False] = False
    dp2_inference_allowed: Literal[False] = False
    proof_boundary: Literal[
        "Checkpoint integration result only. The physical three-band mapping and Rubin DP2 compatibility are unresolved; this score is not a scientifically qualified lens decision."
    ] = (
        "Checkpoint integration result only. The physical three-band mapping and Rubin "
        "DP2 compatibility are unresolved; this score is not a scientifically qualified "
        "lens decision."
    )

    @field_validator("completed_at_utc")
    @classmethod
    def _aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completion time must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _output_consistency(self) -> M4InferenceResult:
        if abs(sum(self.scores) - 1.0) > 1e-6:
            raise ValueError("softmax scores must sum to one")
        if abs(self.lens_score - self.scores[1]) > 1e-7:
            raise ValueError("lens score must be class-index one")
        expected_sources = {
            "checkpoint_io.py",
            "contracts.py",
            "mriganka_enn.py",
            "service.py",
        }
        if set(self.implementation_source_sha256) != expected_sources or any(
            key not in expected_sources
            or not isinstance(value, str)
            or re.fullmatch(SHA256_PATTERN, value) is None
            for key, value in self.implementation_source_sha256.items()
        ):
            raise ValueError(
                "implementation source identities are incomplete or invalid"
            )
        if self.bridge_provenance is None:
            if self.execution_scope_used != "hsc_reference_path":
                raise ValueError(
                    "bridge-free inference must use the HSC reference path"
                )
            if self.reference_verification.status != "reproduced":
                raise ValueError("the HSC reference path must reproduce a pinned case")
        else:
            if self.execution_scope_used != "unqualified_dp2_technical_integration":
                raise ValueError(
                    "bridge inference must remain an unqualified integration"
                )
            if self.reference_verification.status != "not_applicable":
                raise ValueError(
                    "bridge inference cannot claim HSC reference reproduction"
                )
            bridged = self.bridge_provenance.record.model_input_chw
            if (
                self.input.file != bridged.file
                or self.input.decoded_sha256 != bridged.decoded_sha256
                or self.input.dtype != bridged.dtype
                or self.input.shape != bridged.shape
                or self.input.minimum != bridged.minimum
                or self.input.maximum != bridged.maximum
            ):
                raise ValueError(
                    "M4 input evidence does not match the verified bridge output"
                )
        return self


class PublishedFile(FrozenModel):
    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    byte_count: int = Field(gt=0, le=4 * 1024 * 1024)
    sha256: str = Field(pattern=SHA256_PATTERN)


class M3ToM4BridgeCompletionRecord(FrozenModel):
    schema_version: Literal["ripple.m3-to-m4-bridge-completion.v1"] = (
        "ripple.m3-to-m4-bridge-completion.v1"
    )
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["complete"] = "complete"
    model_input_chw: FileIdentity
    bridge_manifest: FileIdentity
    completion_written_last: Literal[True] = True
    scientific_use_allowed: Literal[False] = False
    dp2_inference_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _fixed_bridge_outputs(self) -> M3ToM4BridgeCompletionRecord:
        if (
            self.model_input_chw.filename != "model_input_chw.npy"
            or self.bridge_manifest.filename != "bridge.json"
        ):
            raise ValueError("bridge completion references unexpected output filenames")
        return self


class M4CompletionRecord(FrozenModel):
    schema_version: Literal["ripple.m4-completion.v1"] = "ripple.m4-completion.v1"
    run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["complete"] = "complete"
    bundle_manifest: PublishedFile
    embedding: PublishedFile
    inference_result: PublishedFile
    completion_written_last: Literal[True] = True
    scientific_use_allowed: Literal[False] = False


__all__ = [
    "ArrayArtifact",
    "COrderPayloadIntegrity",
    "CheckpointSpec",
    "ConfiguredChannelMapping",
    "FileIdentity",
    "FrozenModel",
    "HscObservationContext",
    "InputArrayEvidence",
    "M3BchwArrayEvidence",
    "M3BridgeSourceProvenance",
    "M3ToM4BridgeCompletionRecord",
    "M3ToM4BridgeRecord",
    "M4BridgeProvenance",
    "M4ChwArrayEvidence",
    "M4CompletionRecord",
    "M4InferenceResult",
    "MrigankaEnnBundleManifest",
    "PublishedFile",
    "ReferenceCaseSpec",
    "ReferenceVerification",
    "RuntimeIdentity",
    "UpstreamExecutionLogEvidence",
    "UpstreamPreprocessingEvidence",
    "UpstreamSourceFile",
]
