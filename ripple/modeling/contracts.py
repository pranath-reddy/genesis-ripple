"""Typed contracts for model onboarding and model-specific preprocessing.

These models describe *what* a scientific model expects and the evidence for
that description.  They intentionally do not execute transformations.  Pixel
operations remain in reviewed, allowlisted preprocessing adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_VERSION_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_FIELD_PATH_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$"
_CREDENTIAL_TEXT_PATTERN = re.compile(
    r"(?:access[_-]?token|api[_-]?key|apikey|authorization|bearer|credential|"
    r"password|private[_-]?key|secret|signature|token)\s*[:=]",
    flags=re.IGNORECASE,
)
_CREDENTIAL_VALUE_PATTERN = re.compile(
    r"(?:\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bsk-[A-Za-z0-9_-]{20,}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b)"
)


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


def _bounded_text(value: str, *, field: str, maximum: int) -> str:
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(
            f"{field} must be non-empty, trimmed, and at most {maximum} characters"
        )
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} contains a control character")
    return value


def _safe_reference_locator(value: str, *, field: str) -> str:
    value = _bounded_text(value, field=field, maximum=1024)
    if _CREDENTIAL_TEXT_PATTERN.search(value) or _CREDENTIAL_VALUE_PATTERN.search(
        value
    ):
        raise ValueError(f"{field} must not contain credential-shaped content")

    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https"}:
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{field} must be a public URL without credentials, query, or fragment"
            )
        return value
    if scheme in {"doi", "arxiv"}:
        if parsed.netloc or parsed.query or parsed.fragment or not parsed.path:
            raise ValueError(f"{field} contains an invalid public identifier")
        return value
    if scheme:
        raise ValueError(f"{field} uses an unsupported locator scheme")

    if "?" in value or "#" in value or value.startswith("//") or "\\" in value:
        raise ValueError(f"{field} must be a normalized local locator")
    local = PurePosixPath(value)
    if (
        local.is_absolute()
        or ".." in local.parts
        or local.as_posix() != value
        or value in {"", "."}
    ):
        raise ValueError(f"{field} must be a normalized repository-relative locator")
    return value


class EvidenceSource(_ImmutableModel):
    """A pinned source used to derive one or more manifest requirements."""

    source_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    kind: Literal[
        "executable_source",
        "training_configuration",
        "checkpoint_metadata",
        "primary_literature",
        "researcher_declaration",
        "dataset_documentation",
    ]
    authority: Literal[
        "direct_executable_evidence",
        "direct_metadata_evidence",
        "published_methods_evidence",
        "researcher_supplied_evidence",
        "context_only",
    ]
    title: str = Field(min_length=1, max_length=256)
    locator: str = Field(min_length=1, max_length=1024)
    revision: str | None = Field(default=None, max_length=256)
    sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    accessed_at_utc: datetime | None = None

    @field_validator("title")
    @classmethod
    def _valid_title(cls, value: str) -> str:
        return _bounded_text(value, field="title", maximum=256)

    @field_validator("locator")
    @classmethod
    def _safe_locator(cls, value: str) -> str:
        return _safe_reference_locator(value, field="public evidence locator")

    @field_validator("revision")
    @classmethod
    def _valid_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, field="revision", maximum=256)

    @field_validator("accessed_at_utc")
    @classmethod
    def _utc_access_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("accessed_at_utc must be timezone-aware")
        return value


class EvidenceClaim(_ImmutableModel):
    """One proposed contract value and its traceable evidentiary basis."""

    claim_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    field_path: str = Field(pattern=_FIELD_PATH_PATTERN)
    value_summary: str = Field(min_length=1, max_length=1024)
    source_ids: tuple[str, ...] = Field(min_length=1)
    support: Literal["direct", "inferred", "conflicting", "unresolved"]
    review_required: bool
    notes: str | None = Field(default=None, max_length=2048)

    @field_validator("source_ids")
    @classmethod
    def _unique_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("evidence claim source IDs must be unique")
        if any(re.fullmatch(_IDENTIFIER_PATTERN, item) is None for item in value):
            raise ValueError("evidence claim contains an invalid source ID")
        return value

    @field_validator("value_summary")
    @classmethod
    def _valid_value_summary(cls, value: str) -> str:
        return _bounded_text(value, field="value_summary", maximum=1024)

    @field_validator("notes")
    @classmethod
    def _valid_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, field="notes", maximum=2048)


class ComponentRequirements(_ImmutableModel):
    image: Literal[True] = True
    mask: Literal["required", "optional", "unused"]
    variance: Literal["required", "optional", "unused"]
    celestial_wcs: Literal["required", "optional", "unused"]
    psf: Literal["required", "optional", "unused", "must_be_modeled_separately"]
    photometric_calibration: Literal["required", "optional", "unused"]


class NumericRange(_ImmutableModel):
    minimum: float = Field(gt=0)
    maximum: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> "NumericRange":
        if self.minimum > self.maximum:
            raise ValueError("numeric range minimum exceeds maximum")
        return self


class ObservationRequirements(_ImmutableModel):
    """Survey-facing requirements before model-specific transformations."""

    accepted_product_kinds: tuple[str, ...] = Field(min_length=1)
    required_bands: tuple[str, ...] = Field(min_length=1)
    optional_bands: tuple[str, ...] = ()
    required_components: ComponentRequirements
    accepted_image_units: tuple[str, ...] = Field(min_length=1)
    pixel_scale_arcsec: NumericRange | None = None
    field_of_view_arcsec: NumericRange | None = None
    band_alignment: Literal[
        "not_applicable",
        "common_pixel_grid_required",
        "adapter_must_reproject",
        "unresolved",
    ]
    missing_band_policy: Literal["reject", "approved_subset_only", "unresolved"]
    domain_notes: tuple[str, ...] = ()

    @field_validator(
        "accepted_product_kinds",
        "required_bands",
        "optional_bands",
        "accepted_image_units",
    )
    @classmethod
    def _unique_nonempty_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or any(
            not item or item != item.strip() for item in value
        ):
            raise ValueError("contract values must be unique, non-empty, and trimmed")
        return value

    @model_validator(mode="after")
    def _disjoint_bands(self) -> "ObservationRequirements":
        if set(self.required_bands) & set(self.optional_bands):
            raise ValueError("required and optional bands must be disjoint")
        return self


class TensorContract(_ImmutableModel):
    """Exact tensor boundary consumed by a model implementation."""

    axes: tuple[Literal["batch", "channel", "y", "x", "feature"], ...]
    shape: tuple[int, ...]
    dtype: Literal["float16", "float32", "float64", "int32", "int64"]
    unit: str = Field(min_length=1, max_length=64)
    channel_semantics: tuple[str, ...]
    value_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _validate_tensor(self) -> "TensorContract":
        if not self.axes or len(self.axes) != len(self.shape):
            raise ValueError("tensor axes and shape must have the same non-zero length")
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("tensor axes must be unique")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if "channel" in self.axes:
            channel_count = self.shape[self.axes.index("channel")]
            if channel_count != len(self.channel_semantics):
                raise ValueError("channel semantics do not match the channel dimension")
        elif self.channel_semantics:
            raise ValueError("channel semantics require a channel axis")
        if self.value_range is not None and self.value_range[0] >= self.value_range[1]:
            raise ValueError("tensor value range must be increasing")
        return self


class TransformStep(_ImmutableModel):
    """Declarative description of one reviewed adapter operation.

    ``implementation_id`` is resolved only through the local allowlist.  It is
    not a Python import path and cannot cause arbitrary code execution.
    """

    order: int = Field(ge=0, le=255)
    operation: str = Field(pattern=_IDENTIFIER_PATTERN)
    implementation_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_claim_ids: tuple[str, ...] = ()

    @field_validator("evidence_claim_ids")
    @classmethod
    def _unique_claim_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("transform evidence claim IDs must be unique")
        return value


class PreprocessingContract(_ImmutableModel):
    adapter_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    adapter_version: str = Field(pattern=_VERSION_PATTERN)
    recipe_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    steps: tuple[TransformStep, ...] = Field(min_length=1)
    inference_augmentation: Literal["none", "deterministic_ensemble", "unresolved"]

    @model_validator(mode="after")
    def _ordered_steps(self) -> "PreprocessingContract":
        orders = tuple(step.order for step in self.steps)
        if orders != tuple(sorted(orders)) or len(orders) != len(set(orders)):
            raise ValueError(
                "preprocessing steps must have unique ascending order values"
            )
        return self


class ImplementationIdentity(_ImmutableModel):
    framework: str = Field(min_length=1, max_length=64)
    source_locator: str = Field(min_length=1, max_length=1024)
    source_revision: str | None = Field(default=None, max_length=256)
    source_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    architecture_entrypoint: str | None = Field(default=None, max_length=512)
    checkpoint_locator: str | None = Field(default=None, max_length=1024)
    checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("source_locator")
    @classmethod
    def _safe_source_locator(cls, value: str) -> str:
        return _safe_reference_locator(value, field="source locator")

    @field_validator("checkpoint_locator")
    @classmethod
    def _safe_checkpoint_locator(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _safe_reference_locator(value, field="checkpoint locator")

    @model_validator(mode="after")
    def _checkpoint_is_pinned(self) -> "ImplementationIdentity":
        if (self.checkpoint_locator is None) != (self.checkpoint_sha256 is None):
            raise ValueError("checkpoint locator and digest must be supplied together")
        return self


class OutputContract(_ImmutableModel):
    task: Literal[
        "binary_lens_classification",
        "multiclass_lens_classification",
        "lens_reconstruction",
        "parameter_regression",
    ]
    raw_output: Literal[
        "logit", "logits", "probability", "embedding", "image", "parameters"
    ]
    labels: tuple[str, ...] = ()
    score_is_calibrated_probability: bool
    decision_threshold: float | None = None

    @model_validator(mode="after")
    def _validate_output(self) -> "OutputContract":
        if self.task == "binary_lens_classification" and len(self.labels) != 2:
            raise ValueError("binary classification requires exactly two labels")
        if self.raw_output != "probability" and self.score_is_calibrated_probability:
            raise ValueError("only probability output can be declared calibrated")
        if self.decision_threshold is not None:
            if not self.score_is_calibrated_probability:
                raise ValueError(
                    "a decision threshold requires a calibrated probability"
                )
            if not 0.0 <= self.decision_threshold <= 1.0:
                raise ValueError("decision threshold must lie in [0, 1]")
        return self


class QualificationRecord(_ImmutableModel):
    """Execution gates separating a draft from validated scientific use."""

    state: Literal[
        "draft",
        "preprocessing_preview_only",
        "integration_qualified",
        "scientifically_qualified",
        "rejected",
    ]
    preprocessing_execution_allowed: bool
    model_execution_allowed: bool
    scientific_use_allowed: bool
    unresolved_requirements: tuple[str, ...] = ()
    approval_record: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def _validate_gates(self) -> "QualificationRecord":
        if self.scientific_use_allowed and not self.model_execution_allowed:
            raise ValueError("scientific use requires model execution")
        if self.model_execution_allowed and not self.preprocessing_execution_allowed:
            raise ValueError("model execution requires preprocessing execution")
        expected = {
            "draft": (False, False, False),
            "preprocessing_preview_only": (True, False, False),
            "integration_qualified": (True, True, False),
            "scientifically_qualified": (True, True, True),
            "rejected": (False, False, False),
        }[self.state]
        observed = (
            self.preprocessing_execution_allowed,
            self.model_execution_allowed,
            self.scientific_use_allowed,
        )
        if observed != expected:
            raise ValueError("qualification state and execution gates disagree")
        if self.state == "scientifically_qualified":
            if self.unresolved_requirements or self.approval_record is None:
                raise ValueError(
                    "scientific qualification requires approval and no unresolved items"
                )
        return self


class ModelManifest(_ImmutableModel):
    """Versioned, evidence-backed contract for one concrete model artifact."""

    schema_version: Literal["ripple.model-manifest.v1"] = "ripple.model-manifest.v1"
    manifest_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    model_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    model_version: str = Field(pattern=_IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    created_at_utc: datetime
    implementation: ImplementationIdentity
    observation: ObservationRequirements
    preprocessing: PreprocessingContract
    tensor: TensorContract
    output: OutputContract
    evidence_sources: tuple[EvidenceSource, ...] = ()
    evidence_claims: tuple[EvidenceClaim, ...] = ()
    qualification: QualificationRecord

    @field_validator("created_at_utc")
    @classmethod
    def _utc_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _cross_validate_manifest(self) -> "ModelManifest":
        source_ids = [source.source_id for source in self.evidence_sources]
        claim_ids = [claim.claim_id for claim in self.evidence_claims]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("manifest evidence source IDs must be unique")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("manifest evidence claim IDs must be unique")
        known_sources = set(source_ids)
        if any(
            not set(claim.source_ids) <= known_sources for claim in self.evidence_claims
        ):
            raise ValueError("an evidence claim references an unknown source")
        for claim in self.evidence_claims:
            current: object = self
            for part in claim.field_path.split("."):
                if (
                    not isinstance(current, BaseModel)
                    or part not in type(current).model_fields
                ):
                    raise ValueError(
                        "an evidence claim references a nonexistent manifest field"
                    )
                current = getattr(current, part)
        known_claims = set(claim_ids)
        if any(
            not set(step.evidence_claim_ids) <= known_claims
            for step in self.preprocessing.steps
        ):
            raise ValueError("a transform step references an unknown evidence claim")
        if self.qualification.model_execution_allowed:
            if self.implementation.checkpoint_sha256 is None:
                raise ValueError("model execution requires a pinned checkpoint")
            if any(
                claim.support in {"conflicting", "unresolved"}
                for claim in self.evidence_claims
            ):
                raise ValueError(
                    "model execution is blocked by conflicting or unresolved claims"
                )
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", exclude_none=False, by_alias=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelManifestRef(_ImmutableModel):
    manifest_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    model_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    model_version: str = Field(pattern=_IDENTIFIER_PATTERN)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ModelManifestRef":
        return cls(
            manifest_id=manifest.manifest_id,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            sha256=manifest.canonical_sha256(),
        )
