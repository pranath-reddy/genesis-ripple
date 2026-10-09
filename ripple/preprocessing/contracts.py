"""Immutable metadata contracts for the provisional Mriganka M3 adapter."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class MaskPolicy(_ImmutableModel):
    """Frozen DP2 mask interpretation for this morphology-classifier preview."""

    fatal_bits: tuple[Literal["NO_DATA", "SATURATED"], ...] = (
        "NO_DATA",
        "SATURATED",
    )
    caution_bits: tuple[
        Literal[
            "INTERPOLATED",
            "COSMIC_RAY",
            "DETECTION_EDGE",
            "CLIPPED",
            "REJECTED",
            "INEXACT_PSF",
        ],
        ...,
    ] = (
        "INTERPOLATED",
        "COSMIC_RAY",
        "DETECTION_EDGE",
        "CLIPPED",
        "REJECTED",
        "INEXACT_PSF",
    )
    retained_bits: tuple[Literal["DETECTED"], ...] = ("DETECTED",)
    maximum_fatal_fraction: Literal[0.0] = 0.0

    @model_validator(mode="after")
    def _validate_partition(self) -> "MaskPolicy":
        groups = [set(self.fatal_bits), set(self.caution_bits), set(self.retained_bits)]
        if any(
            groups[index] & groups[other]
            for index in range(3)
            for other in range(index + 1, 3)
        ):
            raise ValueError("mask-policy groups must not overlap")
        required = {
            "NO_DATA",
            "INTERPOLATED",
            "COSMIC_RAY",
            "SATURATED",
            "DETECTION_EDGE",
            "CLIPPED",
            "REJECTED",
            "DETECTED",
            "INEXACT_PSF",
        }
        if set.union(*groups) != required:
            raise ValueError(
                "mask policy must cover the complete DP2 deep-coadd schema"
            )
        return self


class Mriganka64Recipe(_ImmutableModel):
    """Versioned preview recipe; scientific compatibility is deliberately unresolved."""

    schema_version: Literal["ripple.preprocessing.mriganka64.recipe.v1"] = (
        "ripple.preprocessing.mriganka64.recipe.v1"
    )
    recipe_id: Literal["mriganka-domain-adaptation-native64-minmax-v1"] = (
        "mriganka-domain-adaptation-native64-minmax-v1"
    )
    required_band: Literal["r"] = "r"
    crop_shape_yx: tuple[Literal[64], Literal[64]] = (64, 64)
    centering: Literal[
        "nearest-native-pixel-window-around-requested-wcs-coordinate"
    ] = "nearest-native-pixel-window-around-requested-wcs-coordinate"
    resampling: Literal["none"] = "none"
    orientation_change: Literal["none"] = "none"
    background_operation: Literal["none"] = "none"
    normalization: Literal["per-crop-minmax"] = "per-crop-minmax"
    output_dtype: Literal["float32"] = "float32"
    output_axes: tuple[
        Literal["batch"], Literal["channel"], Literal["y"], Literal["x"]
    ] = ("batch", "channel", "y", "x")
    output_shape_bchw: tuple[Literal[1], Literal[1], Literal[64], Literal[64]] = (
        1,
        1,
        64,
        64,
    )
    inference_augmentation: Literal["none"] = "none"
    mask_policy: MaskPolicy = Field(default_factory=MaskPolicy)
    scientific_status: Literal["provisional_unqualified"] = "provisional_unqualified"


class SourcePackageRef(_ImmutableModel):
    manifest_filename: Literal["package.json"] = "package.json"
    run_relative_manifest_path: str = Field(min_length=1, max_length=512)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fits_sha256: str = Field(pattern=_SHA256_PATTERN)
    m2_schema_version: Literal["ripple.dp2.cutout-package.v1"]
    dataset_id: str = Field(min_length=1, max_length=1024)
    obs_id: str = Field(min_length=1, max_length=256)
    band: Literal["r"]
    ra_deg: float = Field(ge=0.0, lt=360.0)
    dec_deg: float = Field(ge=-90.0, le=90.0)

    @field_validator("run_relative_manifest_path")
    @classmethod
    def _safe_run_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.name != "package.json"
            or value != parsed.as_posix()
            or not parsed.parts
            or parsed.parts[0] != "inputs"
        ):
            raise ValueError(
                "source manifest must be a normalized run-relative path beneath inputs"
            )
        return value


class CropGeometry(_ImmutableModel):
    source_shape_yx: tuple[int, int]
    crop_bounds_xyxy: tuple[int, int, int, int]
    crop_shape_yx: tuple[Literal[64], Literal[64]]
    target_source_xy: tuple[float, float]
    target_crop_xy: tuple[float, float]
    crop_geometric_center_xy: tuple[float, float]
    target_offset_from_center_xy: tuple[float, float]
    pixel_scale_arcsec_xy: tuple[float, float]
    field_of_view_arcsec_xy: tuple[float, float]
    padding_applied: Literal[False] = False
    resampling_applied: Literal[False] = False

    @model_validator(mode="after")
    def _validate_geometry(self) -> "CropGeometry":
        source_y, source_x = self.source_shape_yx
        x0, y0, x1, y1 = self.crop_bounds_xyxy
        crop_y, crop_x = self.crop_shape_yx
        if source_y <= 0 or source_x <= 0:
            raise ValueError("source dimensions must be positive")
        if not (0 <= x0 < x1 <= source_x and 0 <= y0 < y1 <= source_y):
            raise ValueError("crop bounds are outside the source image")
        if (y1 - y0, x1 - x0) != self.crop_shape_yx:
            raise ValueError("crop bounds do not match the declared crop shape")
        tx, ty = self.target_crop_xy
        if not (0.0 <= tx < crop_x and 0.0 <= ty < crop_y):
            raise ValueError("target does not lie inside the crop")
        expected_center = ((crop_x - 1) / 2.0, (crop_y - 1) / 2.0)
        if any(
            not math.isclose(observed, expected, abs_tol=1e-12)
            for observed, expected in zip(
                self.crop_geometric_center_xy, expected_center
            )
        ):
            raise ValueError("crop geometric center is inconsistent")
        expected_fov = (
            crop_x * self.pixel_scale_arcsec_xy[0],
            crop_y * self.pixel_scale_arcsec_xy[1],
        )
        if any(
            not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-9)
            for observed, expected in zip(self.field_of_view_arcsec_xy, expected_fov)
        ):
            raise ValueError(
                "field of view is inconsistent with crop size and pixel scale"
            )
        return self


class MaskBitSummary(_ImmutableModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    value: int = Field(gt=0)
    category: Literal["fatal", "caution", "retained"]
    set_pixel_count: int = Field(ge=0, le=4096)
    set_pixel_fraction: float = Field(ge=0.0, le=1.0)


class NumericSummary(_ImmutableModel):
    image_min_njy: float
    image_median_njy: float
    image_max_njy: float
    normalization_denominator_njy: float = Field(gt=0.0)
    normalized_min: float = Field(ge=0.0, le=1.0)
    normalized_median: float = Field(ge=0.0, le=1.0)
    normalized_max: float = Field(ge=0.0, le=1.0)
    inverse_normalization_max_abs_error_njy: float = Field(ge=0.0)
    variance_min_njy2: float = Field(gt=0.0)
    variance_median_njy2: float = Field(gt=0.0)
    variance_max_njy2: float = Field(gt=0.0)
    noise_sigma_median_njy: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> "NumericSummary":
        if not self.image_min_njy < self.image_max_njy:
            raise ValueError("image dynamic range must be positive")
        if (
            not self.variance_min_njy2
            <= self.variance_median_njy2
            <= self.variance_max_njy2
        ):
            raise ValueError("variance statistics are not ordered")
        if not math.isclose(self.normalized_min, 0.0, abs_tol=1e-7):
            raise ValueError("min-max output does not begin at zero")
        if not math.isclose(self.normalized_max, 1.0, abs_tol=1e-7):
            raise ValueError("min-max output does not end at one")
        return self


class QualitySummary(_ImmutableModel):
    crop_pixel_count: Literal[4096] = 4096
    image_finite_fraction: Literal[1.0] = 1.0
    variance_positive_finite_fraction: Literal[1.0] = 1.0
    fatal_pixel_count: int = Field(ge=0, le=4096)
    fatal_pixel_fraction: float = Field(ge=0.0, le=1.0)
    caution_pixel_count: int = Field(ge=0, le=4096)
    caution_pixel_fraction: float = Field(ge=0.0, le=1.0)
    detected_pixel_count: int = Field(ge=0, le=4096)
    detected_pixel_fraction: float = Field(ge=0.0, le=1.0)
    mask_bits: tuple[MaskBitSummary, ...]
    psf_state: Literal["present", "absent_from_package", "unknown"]
    numeric: NumericSummary

    @model_validator(mode="after")
    def _validate_quality(self) -> "QualitySummary":
        names = [item.name for item in self.mask_bits]
        required_names = {
            "NO_DATA",
            "INTERPOLATED",
            "COSMIC_RAY",
            "SATURATED",
            "DETECTION_EDGE",
            "CLIPPED",
            "REJECTED",
            "DETECTED",
            "INEXACT_PSF",
        }
        if len(names) != 9 or set(names) != required_names:
            raise ValueError("quality summary requires the nine named DP2 mask bits")
        values = [item.value for item in self.mask_bits]
        if len(values) != len(set(values)) or any(
            value & (value - 1) for value in values
        ):
            raise ValueError("quality mask values must be unique powers of two")
        expected = self.crop_pixel_count
        for count, fraction in (
            (self.fatal_pixel_count, self.fatal_pixel_fraction),
            (self.caution_pixel_count, self.caution_pixel_fraction),
            (self.detected_pixel_count, self.detected_pixel_fraction),
        ):
            if not math.isclose(fraction, count / expected, abs_tol=1e-12):
                raise ValueError("mask union count and fraction disagree")
        return self


class ArrayArtifactRef(_ImmutableModel):
    filename: Literal[
        "model_input.npy",
        "native_crop_njy.npy",
        "mask_crop.npy",
        "variance_crop_njy2.npy",
    ]
    role: Literal["model_input", "native_image", "mask", "variance"]
    media_type: Literal["application/x-npy"] = "application/x-npy"
    dtype: Literal["float32", "int32"]
    shape: tuple[int, ...]
    axes: tuple[str, ...]
    unit: Literal["dimensionless", "nJy", "bitfield", "nJy2"]
    byte_count: int = Field(gt=0, le=1024 * 1024)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)
    decoded_array_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("filename")
    @classmethod
    def _filename_only(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("artifact filename must not contain a path")
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> "ArrayArtifactRef":
        expected = {
            "model_input.npy": (
                "model_input",
                "float32",
                (1, 1, 64, 64),
                ("batch", "channel", "y", "x"),
                "dimensionless",
            ),
            "native_crop_njy.npy": (
                "native_image",
                "float32",
                (64, 64),
                ("y", "x"),
                "nJy",
            ),
            "mask_crop.npy": ("mask", "int32", (64, 64), ("y", "x"), "bitfield"),
            "variance_crop_njy2.npy": (
                "variance",
                "float32",
                (64, 64),
                ("y", "x"),
                "nJy2",
            ),
        }
        observed = (self.role, self.dtype, self.shape, self.axes, self.unit)
        if observed != expected[self.filename]:
            raise ValueError(
                "array artifact semantics do not match its fixed filename contract"
            )
        return self


class PreviewArtifactRef(_ImmutableModel):
    filename: Literal["preprocessing_preview.png"] = "preprocessing_preview.png"
    role: Literal["human_qa_only"] = "human_qa_only"
    media_type: Literal["image/png"] = "image/png"
    pixel_size_xy: tuple[int, int]
    byte_count: int = Field(gt=0, le=20 * 1024 * 1024)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)
    used_as_model_input: Literal[False] = False

    @field_validator("pixel_size_xy")
    @classmethod
    def _bounded_pixel_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if any(axis <= 0 or axis > 5000 for axis in value):
            raise ValueError("preview dimensions are outside the bounded QA contract")
        return value


class CompatibilityStatus(_ImmutableModel):
    state: Literal["provisional_unqualified"] = "provisional_unqualified"
    classifier_execution_allowed: Literal[False] = False
    classifier_score_scientific_use_allowed: Literal[False] = False
    unresolved_requirements: tuple[
        Literal[
            "encoder_checkpoint",
            "classifier_checkpoint",
            "algorithm_and_backbone",
            "training_band",
            "training_pixel_scale_or_angular_field",
            "training_psf_noise_and_preparation_contract",
        ],
        ...,
    ] = (
        "encoder_checkpoint",
        "classifier_checkpoint",
        "algorithm_and_backbone",
        "training_band",
        "training_pixel_scale_or_angular_field",
        "training_psf_noise_and_preparation_contract",
    )

    @model_validator(mode="after")
    def _require_complete_unresolved_set(self) -> "CompatibilityStatus":
        required = {
            "encoder_checkpoint",
            "classifier_checkpoint",
            "algorithm_and_backbone",
            "training_band",
            "training_pixel_scale_or_angular_field",
            "training_psf_noise_and_preparation_contract",
        }
        if (
            set(self.unresolved_requirements) != required
            or len(self.unresolved_requirements) != 6
        ):
            raise ValueError(
                "the provisional compatibility record must list every blocker"
            )
        return self


class NamedVersion(_ImmutableModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    version: str = Field(min_length=1, max_length=128)


class SourceDigest(_ImmutableModel):
    relative_path: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^ripple/preprocessing(?:/[A-Za-z0-9_]+)*\.py$",
    )
    sha256: str = Field(pattern=_SHA256_PATTERN)


class MrigankaModelInputPackage(_ImmutableModel):
    schema_version: Literal["ripple.preprocessing.mriganka-model-input.v2"] = (
        "ripple.preprocessing.mriganka-model-input.v2"
    )
    status: Literal["success"] = "success"
    created_at_utc: datetime
    source: SourcePackageRef
    recipe: Mriganka64Recipe
    crop: CropGeometry
    quality: QualitySummary
    compatibility: CompatibilityStatus = Field(default_factory=CompatibilityStatus)
    transformations: tuple[
        Literal[
            "reverify-m2-package-and-fits",
            "native-wcs-centered-64x64-crop-no-resampling",
            "per-crop-minmax-float32",
            "add-channel-and-batch-axes",
        ],
        ...,
    ] = (
        "reverify-m2-package-and-fits",
        "native-wcs-centered-64x64-crop-no-resampling",
        "per-crop-minmax-float32",
        "add-channel-and-batch-axes",
    )
    model_input: ArrayArtifactRef
    native_crop: ArrayArtifactRef
    mask_crop: ArrayArtifactRef
    variance_crop: ArrayArtifactRef
    preview: PreviewArtifactRef
    implementation_versions: tuple[NamedVersion, ...]
    implementation_sources: tuple[SourceDigest, ...]
    proof_boundary: Literal[
        "Deterministic preview preprocessing of one verified Rubin DP2 r-band cutout only; the native 64-pixel angular field is not matched to a recovered training contract, no classifier checkpoint was loaded, and no lens score, probability, candidate decision, or agent decision is validated."
    ] = (
        "Deterministic preview preprocessing of one verified Rubin DP2 r-band cutout only; "
        "the native 64-pixel angular field is not matched to a recovered training contract, "
        "no classifier checkpoint was loaded, and no lens score, probability, candidate "
        "decision, or agent decision is validated."
    )

    @field_validator("created_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _cross_validate(self) -> "MrigankaModelInputPackage":
        if self.source.band != self.recipe.required_band:
            raise ValueError("source band does not match recipe")
        if self.crop.crop_shape_yx != self.recipe.crop_shape_yx:
            raise ValueError("crop geometry does not match recipe")
        if self.model_input.shape != self.recipe.output_shape_bchw:
            raise ValueError("model-input artifact shape does not match recipe")
        if self.model_input.axes != self.recipe.output_axes:
            raise ValueError("model-input axes do not match recipe")
        expected_artifact_fields = (
            (self.model_input, "model_input.npy", "model_input"),
            (self.native_crop, "native_crop_njy.npy", "native_image"),
            (self.mask_crop, "mask_crop.npy", "mask"),
            (self.variance_crop, "variance_crop_njy2.npy", "variance"),
        )
        if any(
            artifact.filename != filename or artifact.role != role
            for artifact, filename, role in expected_artifact_fields
        ):
            raise ValueError(
                "package artifact fields do not match their fixed semantic roles"
            )
        if self.transformations != (
            "reverify-m2-package-and-fits",
            "native-wcs-centered-64x64-crop-no-resampling",
            "per-crop-minmax-float32",
            "add-channel-and-batch-axes",
        ):
            raise ValueError("transformation record is incomplete or out of order")
        if (
            self.quality.fatal_pixel_fraction
            > self.recipe.mask_policy.maximum_fatal_fraction
        ):
            raise ValueError("fatal mask fraction exceeds recipe")
        categories = {
            **{name: "fatal" for name in self.recipe.mask_policy.fatal_bits},
            **{name: "caution" for name in self.recipe.mask_policy.caution_bits},
            **{name: "retained" for name in self.recipe.mask_policy.retained_bits},
        }
        if any(
            item.category != categories[item.name] for item in self.quality.mask_bits
        ):
            raise ValueError("mask summary categories do not match the frozen recipe")
        expected_versions = {
            "python",
            "numpy",
            "astropy",
            "pydantic",
            "matplotlib",
            "pillow",
        }
        version_names = [item.name for item in self.implementation_versions]
        if len(version_names) != 6 or set(version_names) != expected_versions:
            raise ValueError("implementation version inventory is incomplete")
        expected_sources = {
            "ripple/preprocessing/__init__.py",
            "ripple/preprocessing/errors.py",
            "ripple/preprocessing/contracts.py",
            "ripple/preprocessing/mriganka.py",
            "ripple/preprocessing/artifact_io.py",
            "ripple/preprocessing/visualization.py",
            "ripple/preprocessing/service.py",
            "ripple/preprocessing/cli.py",
        }
        source_paths = [item.relative_path for item in self.implementation_sources]
        if len(source_paths) != 8 or set(source_paths) != expected_sources:
            raise ValueError("implementation source inventory is incomplete")
        return self


class M3FailureEvidence(_ImmutableModel):
    schema_version: Literal["ripple.preprocessing.failure.v1"] = (
        "ripple.preprocessing.failure.v1"
    )
    status: Literal["failure"] = "failure"
    created_at_utc: datetime
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")
    safe_message: str = Field(min_length=1, max_length=512)
    recipe: Mriganka64Recipe = Field(default_factory=Mriganka64Recipe)

    @field_validator("created_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value
