"""Immutable evidence contracts for the three-band Mriganka ENN adapter."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ripple.preprocessing.contracts import (
    CropGeometry,
    MaskPolicy,
    NamedVersion,
    PreviewArtifactRef,
    QualitySummary,
)

BandName = Literal["g", "r", "i"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_EXPECTED_SOURCE_PATHS = {
    "ripple/preprocessing/mriganka_enn/__init__.py",
    "ripple/preprocessing/mriganka_enn/contracts.py",
    "ripple/preprocessing/mriganka_enn/transform.py",
    "ripple/preprocessing/mriganka_enn/artifact_io.py",
    "ripple/preprocessing/mriganka_enn/service.py",
    "ripple/preprocessing/mriganka_enn/visualization.py",
    "ripple/modeling/mriganka_enn_adapter.py",
}


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class MrigankaEnnThreeBandRecipe(_ImmutableModel):
    """Code-owned native-grid recipe for an unqualified technical input."""

    schema_version: Literal[
        "ripple.preprocessing.mriganka-enn-three-band.recipe.v1"
    ] = "ripple.preprocessing.mriganka-enn-three-band.recipe.v1"
    recipe_id: Literal["mriganka-enn-dp2-native64-three-band-minmax-v1"] = (
        "mriganka-enn-dp2-native64-three-band-minmax-v1"
    )
    channel_bands: tuple[BandName, BandName, BandName] = ("g", "r", "i")
    physical_band_mapping_status: Literal["configured_unverified"] = (
        "configured_unverified"
    )
    crop_shape_yx: tuple[Literal[64], Literal[64]] = (64, 64)
    centering: Literal[
        "nearest-native-pixel-window-around-requested-wcs-coordinate"
    ] = "nearest-native-pixel-window-around-requested-wcs-coordinate"
    resampling: Literal["none"] = "none"
    orientation_change: Literal["none"] = "none"
    background_operation: Literal["none"] = "none"
    field_of_view_policy: Literal["native-64px-no-resampling-technical-only"] = (
        "native-64px-no-resampling-technical-only"
    )
    normalization: Literal["per-channel-minmax-then-nan-to-num-zero"] = (
        "per-channel-minmax-then-nan-to-num-zero"
    )
    positive_dynamic_range_required: Literal[True] = True
    output_dtype: Literal["float32"] = "float32"
    output_axes: tuple[
        Literal["batch"], Literal["channel"], Literal["y"], Literal["x"]
    ] = ("batch", "channel", "y", "x")
    output_shape_bchw: tuple[Literal[1], Literal[3], Literal[64], Literal[64]] = (
        1,
        3,
        64,
        64,
    )
    wcs_alignment_tolerance_arcsec: Literal[1e-7] = 1e-7
    wcs_sample_grid_xy: tuple[float, float, float] = (0.0, 31.5, 63.0)
    inference_augmentation: Literal["none"] = "none"
    mask_policy: MaskPolicy = Field(default_factory=MaskPolicy)
    scientific_status: Literal["provisional_unqualified"] = "provisional_unqualified"

    @model_validator(mode="after")
    def _validate_channel_mapping(self) -> MrigankaEnnThreeBandRecipe:
        if set(self.channel_bands) != {"g", "r", "i"}:
            raise ValueError(
                "channel_bands must be a unique permutation of g, r, and i"
            )
        if self.wcs_sample_grid_xy != (0.0, 31.5, 63.0):
            raise ValueError("the native-grid WCS sample grid is fixed for recipe v1")
        return self


class SourcePackageRef(_ImmutableModel):
    manifest_filename: Literal["package.json"] = "package.json"
    run_relative_manifest_path: str = Field(min_length=1, max_length=512)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fits_sha256: str = Field(pattern=_SHA256_PATTERN)
    m2_schema_version: Literal["ripple.dp2.cutout-package.v1"]
    product_kind: Literal["rubin_dp2_deep_coadd_masked_image"]
    release: Literal["DP2"]
    observation_collection: Literal["LSST.DP2"]
    product_subtype: Literal["lsst.deep_coadd"]
    dataset_id: str = Field(min_length=1, max_length=1024)
    obs_id: str = Field(min_length=1, max_length=256)
    band: BandName
    tract: int = Field(ge=0)
    patch: int = Field(ge=0)
    ra_deg: float = Field(ge=0.0, lt=360.0)
    dec_deg: float = Field(ge=-90.0, le=90.0)
    image_unit: Literal["nJy"]
    variance_unit: Literal["nJy2"]
    psf_state: Literal["present", "absent_from_package", "unknown"]

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


class BandArrayArtifactRef(_ImmutableModel):
    filename: str = Field(
        pattern=r"^(?:native_crop_[gri]_njy|mask_crop_[gri]|variance_crop_[gri]_njy2)\.npy$"
    )
    band: BandName
    role: Literal["native_image", "mask", "variance"]
    media_type: Literal["application/x-npy"] = "application/x-npy"
    dtype: Literal["float32", "int32"]
    shape: tuple[Literal[64], Literal[64]] = (64, 64)
    axes: tuple[Literal["y"], Literal["x"]] = ("y", "x")
    unit: Literal["nJy", "bitfield", "nJy2"]
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
    def _validate_semantics(self) -> BandArrayArtifactRef:
        expected = {
            "native_image": (
                f"native_crop_{self.band}_njy.npy",
                "float32",
                "nJy",
            ),
            "mask": (f"mask_crop_{self.band}.npy", "int32", "bitfield"),
            "variance": (
                f"variance_crop_{self.band}_njy2.npy",
                "float32",
                "nJy2",
            ),
        }[self.role]
        if (self.filename, self.dtype, self.unit) != expected:
            raise ValueError("band-array filename, dtype, unit, and role disagree")
        return self


class ModelInputArtifactRef(_ImmutableModel):
    filename: Literal["model_input_bchw.npy"] = "model_input_bchw.npy"
    role: Literal["model_input"] = "model_input"
    media_type: Literal["application/x-npy"] = "application/x-npy"
    dtype: Literal["float32"] = "float32"
    shape: tuple[Literal[1], Literal[3], Literal[64], Literal[64]] = (
        1,
        3,
        64,
        64,
    )
    axes: tuple[Literal["batch"], Literal["channel"], Literal["y"], Literal["x"]] = (
        "batch",
        "channel",
        "y",
        "x",
    )
    unit: Literal["dimensionless"] = "dimensionless"
    byte_count: int = Field(gt=0, le=1024 * 1024)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)
    decoded_array_sha256: str = Field(pattern=_SHA256_PATTERN)


class BandChannelEvidence(_ImmutableModel):
    channel_index: int = Field(ge=0, le=2)
    band: BandName
    source_manifest_path: str = Field(min_length=1, max_length=512)
    crop: CropGeometry
    quality: QualitySummary
    normalization_nonfinite_replacement_count: Literal[0] = 0
    native_crop: BandArrayArtifactRef
    mask_crop: BandArrayArtifactRef
    variance_crop: BandArrayArtifactRef

    @field_validator("source_manifest_path")
    @classmethod
    def _safe_source_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.name != "package.json"
            or parsed.parts[0] != "inputs"
            or parsed.as_posix() != value
        ):
            raise ValueError("channel source path must identify a bundled M2 package")
        return value

    @model_validator(mode="after")
    def _validate_artifacts(self) -> BandChannelEvidence:
        expected = (
            (self.native_crop, "native_image"),
            (self.mask_crop, "mask"),
            (self.variance_crop, "variance"),
        )
        if any(
            artifact.band != self.band or artifact.role != role
            for artifact, role in expected
        ):
            raise ValueError("channel artifacts do not match their band and roles")
        return self


class PairwiseWcsAlignment(_ImmutableModel):
    bands: tuple[BandName, BandName]
    sample_count: Literal[13] = 13
    maximum_separation_arcsec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _unique_pair(self) -> PairwiseWcsAlignment:
        if self.bands[0] == self.bands[1]:
            raise ValueError("WCS alignment pair must contain two different bands")
        return self


class CrossBandWcsEvidence(_ImmutableModel):
    state: Literal["valid_common_native_grid"] = "valid_common_native_grid"
    comparison: Literal["pairwise-sky-separation-at-common-crop-relative-pixels"] = (
        "pairwise-sky-separation-at-common-crop-relative-pixels"
    )
    pixel_convention: Literal[
        "astropy_zero_based_pixel_centers_with_half_pixel_edges"
    ] = "astropy_zero_based_pixel_centers_with_half_pixel_edges"
    pixel_center_grid_xy: tuple[float, float, float] = (0.0, 31.5, 63.0)
    footprint_corners_xy: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((-0.5, -0.5), (63.5, -0.5), (63.5, 63.5), (-0.5, 63.5))
    pairwise: tuple[
        PairwiseWcsAlignment,
        PairwiseWcsAlignment,
        PairwiseWcsAlignment,
    ]
    maximum_separation_arcsec: float = Field(ge=0.0)
    tolerance_arcsec: Literal[1e-7] = 1e-7
    passed: Literal[True] = True
    reprojection_applied: Literal[False] = False

    @model_validator(mode="after")
    def _validate_alignment(self) -> CrossBandWcsEvidence:
        pairs = {frozenset(item.bands) for item in self.pairwise}
        required = {
            frozenset(("g", "r")),
            frozenset(("g", "i")),
            frozenset(("r", "i")),
        }
        if pairs != required:
            raise ValueError("cross-band WCS evidence must cover all three band pairs")
        observed_maximum = max(item.maximum_separation_arcsec for item in self.pairwise)
        if not math.isclose(
            self.maximum_separation_arcsec,
            observed_maximum,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("global WCS residual does not match pairwise evidence")
        if self.maximum_separation_arcsec > self.tolerance_arcsec:
            raise ValueError(
                "cross-band WCS residual exceeds the native-grid tolerance"
            )
        return self


class CompatibilityStatus(_ImmutableModel):
    state: Literal["provisional_unqualified"] = "provisional_unqualified"
    classifier_execution_allowed: Literal[False] = False
    classifier_score_scientific_use_allowed: Literal[False] = False
    candidate_decision_allowed: Literal[False] = False
    unresolved_requirements: tuple[
        Literal[
            "physical_channel_order_authority",
            "angular_field_and_resampling_policy",
            "psf_matching_policy",
            "hsc_to_rubin_domain_compatibility",
            "score_calibration",
            "candidate_threshold",
        ],
        ...,
    ] = (
        "physical_channel_order_authority",
        "angular_field_and_resampling_policy",
        "psf_matching_policy",
        "hsc_to_rubin_domain_compatibility",
        "score_calibration",
        "candidate_threshold",
    )

    @model_validator(mode="after")
    def _complete_blocker_set(self) -> CompatibilityStatus:
        required = {
            "physical_channel_order_authority",
            "angular_field_and_resampling_policy",
            "psf_matching_policy",
            "hsc_to_rubin_domain_compatibility",
            "score_calibration",
            "candidate_threshold",
        }
        if set(self.unresolved_requirements) != required:
            raise ValueError(
                "the provisional package must retain every scientific blocker"
            )
        return self


class ImplementationSourceRef(_ImmutableModel):
    relative_path: str = Field(
        min_length=1,
        max_length=256,
        pattern=(
            r"^(?:ripple/preprocessing/mriganka_enn/[A-Za-z0-9_]+\.py|"
            r"ripple/modeling/mriganka_enn_adapter\.py)$"
        ),
    )
    sha256: str = Field(pattern=_SHA256_PATTERN)


class MrigankaEnnThreeBandModelInputPackage(_ImmutableModel):
    schema_version: Literal[
        "ripple.preprocessing.mriganka-enn-three-band-model-input.v1"
    ] = "ripple.preprocessing.mriganka-enn-three-band-model-input.v1"
    status: Literal["success"] = "success"
    created_at_utc: datetime
    sources: tuple[SourcePackageRef, SourcePackageRef, SourcePackageRef]
    recipe: MrigankaEnnThreeBandRecipe
    channels: tuple[
        BandChannelEvidence,
        BandChannelEvidence,
        BandChannelEvidence,
    ]
    cross_band_wcs: CrossBandWcsEvidence
    compatibility: CompatibilityStatus = Field(default_factory=CompatibilityStatus)
    transformations: tuple[str, ...] = (
        "reverify-three-m2-packages-and-fits",
        "select-channels-by-configured-band-mapping",
        "independent-native-wcs-centered-64x64-crops-no-resampling",
        "validate-pairwise-common-native-wcs-grid",
        "per-channel-minmax-then-nan-to-num-zero-float32",
        "stack-batch-channel-y-x",
    )
    model_input: ModelInputArtifactRef
    preview: PreviewArtifactRef
    implementation_versions: tuple[NamedVersion, ...]
    implementation_sources: tuple[ImplementationSourceRef, ...]
    proof_boundary: Literal[
        "Technical three-band preprocessing of independently verified Rubin DP2 g/r/i cutouts only. The configured channel-to-band order is not authoritatively tied to the serialized HSC training arrays; the native Rubin angular field is not matched to the HSC field; no reprojection or PSF matching is applied; HSC-to-Rubin domain compatibility, score calibration, and a candidate threshold are unresolved. No probability, candidate decision, or scientific classification is authorized."
    ] = (
        "Technical three-band preprocessing of independently verified Rubin DP2 g/r/i "
        "cutouts only. The configured channel-to-band order is not authoritatively tied "
        "to the serialized HSC training arrays; the native Rubin angular field is not "
        "matched to the HSC field; no reprojection or PSF matching is applied; "
        "HSC-to-Rubin domain compatibility, score calibration, and a candidate threshold "
        "are unresolved. No probability, candidate decision, or scientific classification "
        "is authorized."
    )

    @field_validator("created_at_utc")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _cross_validate(self) -> MrigankaEnnThreeBandModelInputPackage:
        source_bands = tuple(source.band for source in self.sources)
        if len(set(source_bands)) != 3 or set(source_bands) != {"g", "r", "i"}:
            raise ValueError("sources must contain exactly one g, r, and i package")
        if source_bands != self.recipe.channel_bands:
            raise ValueError("sources must use the configured canonical channel order")
        reference = self.sources[0]
        if any(
            not (
                math.isclose(
                    source.ra_deg,
                    reference.ra_deg,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    source.dec_deg,
                    reference.dec_deg,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                and source.release == reference.release
                and source.observation_collection == reference.observation_collection
                and source.product_subtype == reference.product_subtype
                and source.product_kind == reference.product_kind
                and source.obs_id == reference.obs_id
                and source.tract == reference.tract
                and source.patch == reference.patch
                and source.image_unit == reference.image_unit
                and source.variance_unit == reference.variance_unit
            )
            for source in self.sources[1:]
        ):
            raise ValueError(
                "source package identities are not one compatible observation"
            )

        channel_bands = tuple(channel.band for channel in self.channels)
        if channel_bands != self.recipe.channel_bands:
            raise ValueError(
                "channel evidence order does not match the configured mapping"
            )
        if tuple(channel.channel_index for channel in self.channels) != (0, 1, 2):
            raise ValueError("channel indices must be exactly zero, one, and two")
        source_by_band = {source.band: source for source in self.sources}
        if any(
            channel.source_manifest_path
            != source_by_band[channel.band].run_relative_manifest_path
            for channel in self.channels
        ):
            raise ValueError(
                "channel evidence does not reference its verified M2 source"
            )
        if any(
            channel.crop.crop_shape_yx != self.recipe.crop_shape_yx
            for channel in self.channels
        ):
            raise ValueError("a channel crop does not match the recipe")
        if (
            self.model_input.shape != self.recipe.output_shape_bchw
            or self.model_input.axes != self.recipe.output_axes
            or self.model_input.dtype != self.recipe.output_dtype
        ):
            raise ValueError("model-input artifact does not match the recipe tensor")
        expected_transformations = (
            "reverify-three-m2-packages-and-fits",
            "select-channels-by-configured-band-mapping",
            "independent-native-wcs-centered-64x64-crops-no-resampling",
            "validate-pairwise-common-native-wcs-grid",
            "per-channel-minmax-then-nan-to-num-zero-float32",
            "stack-batch-channel-y-x",
        )
        if self.transformations != expected_transformations:
            raise ValueError("transformation record is incomplete or out of order")
        version_names = tuple(item.name for item in self.implementation_versions)
        expected_versions = {
            "python",
            "numpy",
            "astropy",
            "pydantic",
            "matplotlib",
            "pillow",
        }
        if len(version_names) != 6 or set(version_names) != expected_versions:
            raise ValueError("implementation version inventory is incomplete")
        source_paths = tuple(item.relative_path for item in self.implementation_sources)
        if (
            len(source_paths) != len(_EXPECTED_SOURCE_PATHS)
            or set(source_paths) != _EXPECTED_SOURCE_PATHS
        ):
            raise ValueError("implementation source inventory is incomplete")
        return self


__all__ = [
    "BandArrayArtifactRef",
    "BandChannelEvidence",
    "BandName",
    "CompatibilityStatus",
    "CrossBandWcsEvidence",
    "ImplementationSourceRef",
    "ModelInputArtifactRef",
    "MrigankaEnnThreeBandModelInputPackage",
    "MrigankaEnnThreeBandRecipe",
    "PairwiseWcsAlignment",
    "SourcePackageRef",
]
