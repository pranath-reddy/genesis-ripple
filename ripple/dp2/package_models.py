"""Immutable metadata contracts for a Rubin DP2 masked-image cutout package.

The JSON model deliberately contains references and scientific metadata only.
The colocated, checksummed FITS file remains the lossless pixel payload.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qs, urlparse

from astropy import units as u
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .models import (
    DatasetIdentity,
    Dp2CutoutRequest,
    FitsIdentityEvidence,
    SecurityEvidence,
    StageCheck,
    _validated_plain_text,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SAFE_NAME_PATTERN = r"^[A-Za-z0-9._:/+*() -]+$"


class _ImmutablePackageModel(BaseModel):
    """Deeply JSON-oriented base: child collections are tuples, not dicts/lists."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class NamedVersion(_ImmutablePackageModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    version: str = Field(min_length=1, max_length=128, pattern=_SAFE_NAME_PATTERN)


class SourceDigest(_ImmutablePackageModel):
    relative_path: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^ripple(?:/[A-Za-z0-9_]+)*\.py$",
    )
    sha256: str = Field(pattern=_SHA256_PATTERN)


class FitsArtifactRef(_ImmutablePackageModel):
    filename: Literal["cutout.fits"] = "cutout.fits"
    media_type: Literal["application/fits"] = "application/fits"
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    hdu_count: int = Field(ge=4, le=32)

    @field_validator("filename")
    @classmethod
    def _filename_only(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("artifact filename must not contain a path")
        return value


class FitsScaling(_ImmutablePackageModel):
    bscale: float | None = None
    bzero: float | None = None
    blank: int | None = None


class DecodedArrayDigest(_ImmutablePackageModel):
    algorithm: Literal["sha256"] = "sha256"
    encoding: Literal["dtype-shape-prefixed-little-endian-c-order-v1"] = (
        "dtype-shape-prefixed-little-endian-c-order-v1"
    )
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class _PlaneRef(_ImmutablePackageModel):
    hdu_index: int = Field(ge=1, le=31)
    extname: str = Field(min_length=1, max_length=68, pattern=r"^[A-Z][A-Z0-9_]*$")
    extver: int = Field(ge=1)
    extver_declared: bool
    shape_yx: tuple[int, int]
    decoded_dtype: str = Field(
        min_length=2,
        max_length=32,
        pattern=r"^(?:u?int(?:8|16|32|64)|float(?:32|64))$",
    )
    decoded_byteorder: Literal["little", "big", "not_applicable"]
    canonical_dtype: str = Field(
        min_length=2,
        max_length=16,
        pattern=r"^(?:[|<>][uif][1248])$",
    )
    serialization_dtype: str = Field(
        min_length=2,
        max_length=32,
        pattern=r"^(?:u?int(?:8|16|32|64)|float(?:32|64))$",
    )
    serialization_byteorder: Literal["little", "big"]
    fits_logical_bitpix: Literal[-64, -32, 8, 16, 32, 64]
    storage_kind: Literal["tile_compressed_image", "image_hdu"]
    compression_type: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        pattern=r"^[A-Z0-9_]+$",
    )
    scaling: FitsScaling
    digest: DecodedArrayDigest
    total_pixel_count: int = Field(gt=0)
    valid: Literal[True] = True

    @model_validator(mode="after")
    def _validate_geometry_and_storage(self) -> "_PlaneRef":
        if any(axis <= 0 or axis > 8192 for axis in self.shape_yx):
            raise ValueError("plane dimensions are outside the package safety bound")
        if self.total_pixel_count != self.shape_yx[0] * self.shape_yx[1]:
            raise ValueError("plane pixel count does not match its shape")
        if (
            self.storage_kind == "tile_compressed_image"
            and self.compression_type is None
        ):
            raise ValueError("compressed planes require a compression type")
        if self.storage_kind == "image_hdu" and self.compression_type is not None:
            raise ValueError("uncompressed planes cannot declare a compression type")
        return self


class ImagePlaneRef(_PlaneRef):
    role: Literal["science_image"] = "science_image"
    extname: Literal["IMAGE"]
    declared_bunit: str = Field(min_length=1, max_length=64, pattern=_SAFE_NAME_PATTERN)
    serialization_unit: str = Field(
        min_length=1, max_length=64, pattern=_SAFE_NAME_PATTERN
    )
    canonical_unit: Literal["nJy"]
    finite_pixel_count: int = Field(ge=0)
    nonfinite_pixel_count: int = Field(ge=0)
    zero_mask_nonfinite_pixel_count: int = Field(ge=0)
    finite_min: float
    finite_max: float

    @model_validator(mode="after")
    def _validate_image_counts(self) -> "ImagePlaneRef":
        if (
            self.finite_pixel_count + self.nonfinite_pixel_count
            != self.total_pixel_count
        ):
            raise ValueError("image finite/nonfinite counts are inconsistent")
        if self.finite_pixel_count < 1 or self.finite_min > self.finite_max:
            raise ValueError("image has no valid finite range")
        if self.nonfinite_pixel_count != 0:
            raise ValueError(
                "the fixed M2 image contract requires every pixel to be finite"
            )
        if self.zero_mask_nonfinite_pixel_count != 0:
            raise ValueError("nonfinite image pixels with zero mask are forbidden")
        return self


class MaskBitDefinition(_ImmutablePackageModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    bit_index: int = Field(ge=0, le=63)
    value: int = Field(gt=0)
    description: str = Field(min_length=1, max_length=1024)
    set_pixel_count: int = Field(ge=0)
    source: Literal["FITS mask cards cross-checked with Rubin JSON metadata"] = (
        "FITS mask cards cross-checked with Rubin JSON metadata"
    )

    @field_validator("description")
    @classmethod
    def _safe_description(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)

    @model_validator(mode="after")
    def _validate_bit_value(self) -> "MaskBitDefinition":
        if self.value != 1 << self.bit_index:
            raise ValueError("mask value must equal one shifted by bit_index")
        return self


class MaskPlaneRef(_PlaneRef):
    role: Literal["mask"] = "mask"
    extname: Literal["MASK"]
    semantics: Literal["integer_bitfield"] = "integer_bitfield"
    declared_bunit: None = None
    bit_width: Literal[8, 16, 32, 64]
    bits: tuple[MaskBitDefinition, ...]
    nonzero_pixel_count: int = Field(ge=0)
    unique_combination_count: int = Field(ge=1)
    unsigned_min: int = Field(ge=0)
    unsigned_max: int = Field(ge=0)
    undefined_set_bits: tuple[int, ...] = ()
    undefined_set_pixel_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_mask_semantics(self) -> "MaskPlaneRef":
        if not self.bits:
            raise ValueError("mask plane requires named bit definitions")
        names = [item.name for item in self.bits]
        indices = [item.bit_index for item in self.bits]
        values = [item.value for item in self.bits]
        if len(names) != len(set(names)) or len(indices) != len(set(indices)):
            raise ValueError("mask bit names and positions must be unique")
        if len(values) != len(set(values)) or any(
            bit >= self.bit_width for bit in indices
        ):
            raise ValueError(
                "mask bit definitions exceed or duplicate the integer width"
            )
        if self.nonzero_pixel_count > self.total_pixel_count:
            raise ValueError("mask nonzero count exceeds its pixel count")
        if self.unique_combination_count > self.total_pixel_count:
            raise ValueError("mask combination count exceeds its pixel count")
        if any(item.set_pixel_count > self.total_pixel_count for item in self.bits):
            raise ValueError("a mask-bit count exceeds the plane pixel count")
        if self.unsigned_min > self.unsigned_max:
            raise ValueError("mask unsigned range is invalid")
        if self.unsigned_max >= 1 << self.bit_width:
            raise ValueError("mask values exceed the declared integer width")
        if self.undefined_set_bits or self.undefined_set_pixel_count:
            raise ValueError("mask contains set bits without definitions")
        return self


class VariancePlaneRef(_PlaneRef):
    role: Literal["variance"] = "variance"
    extname: Literal["VARIANCE"]
    semantics: Literal["per_pixel_variance"] = "per_pixel_variance"
    declared_bunit: str = Field(min_length=1, max_length=64, pattern=_SAFE_NAME_PATTERN)
    serialization_unit: str = Field(
        min_length=1, max_length=64, pattern=_SAFE_NAME_PATTERN
    )
    canonical_unit: Literal["nJy2"]
    finite_pixel_count: int = Field(ge=0)
    nonfinite_pixel_count: int = Field(ge=0)
    positive_finite_pixel_count: int = Field(ge=0)
    zero_finite_pixel_count: int = Field(ge=0)
    negative_finite_pixel_count: int = Field(ge=0)
    zero_mask_nonpositive_or_nonfinite_pixel_count: int = Field(ge=0)
    finite_min: float
    finite_max: float

    @model_validator(mode="after")
    def _validate_variance_counts(self) -> "VariancePlaneRef":
        if (
            self.finite_pixel_count + self.nonfinite_pixel_count
            != self.total_pixel_count
        ):
            raise ValueError("variance finite/nonfinite counts are inconsistent")
        if (
            self.positive_finite_pixel_count
            + self.zero_finite_pixel_count
            + self.negative_finite_pixel_count
            != self.finite_pixel_count
        ):
            raise ValueError("variance sign counts are inconsistent")
        if self.positive_finite_pixel_count < 1 or self.finite_min > self.finite_max:
            raise ValueError("variance has no positive finite samples")
        if self.finite_min <= 0:
            raise ValueError("variance finite minimum must be positive")
        if self.negative_finite_pixel_count != 0:
            raise ValueError("finite negative variance is forbidden")
        if self.nonfinite_pixel_count != 0 or self.zero_finite_pixel_count != 0:
            raise ValueError(
                "the fixed M2 variance contract requires positive finite values"
            )
        if self.positive_finite_pixel_count != self.total_pixel_count:
            raise ValueError("every variance pixel must be positive and finite")
        if self.zero_mask_nonpositive_or_nonfinite_pixel_count != 0:
            raise ValueError(
                "nonpositive or nonfinite variance with zero mask is forbidden"
            )
        return self


class SkyCoordinate(_ImmutablePackageModel):
    ra_deg: float = Field(ge=0, lt=360)
    dec_deg: float = Field(ge=-90, le=90)


class PlaneWcsDigest(_ImmutablePackageModel):
    hdu_index: int = Field(ge=1, le=31)
    extname: Literal["IMAGE", "MASK", "VARIANCE"]
    sha256: str = Field(pattern=_SHA256_PATTERN)


class CelestialWcsRef(_ImmutablePackageModel):
    state: Literal["valid_aligned"] = "valid_aligned"
    transform_digest_algorithm: Literal["sha256"] = "sha256"
    transform_canonicalization: Literal["astropy-wcs-header-cards-json-v1"] = (
        "astropy-wcs-header-cards-json-v1"
    )
    plane_digests: tuple[PlaneWcsDigest, ...]
    frame: Literal["icrs"]
    ctype: tuple[str, str]
    axis_units: tuple[Literal["deg"], Literal["deg"]]
    projection: Literal["TAN"]
    pixel_axis_order: Literal["x_y"] = "x_y"
    array_axis_order: Literal["y_x"] = "y_x"
    pixel_convention: Literal[
        "astropy_zero_based_pixel_centers_with_half_pixel_edges"
    ] = "astropy_zero_based_pixel_centers_with_half_pixel_edges"
    target_pixel_xy: tuple[float, float]
    target_inside: Literal[True]
    pixel_scale_arcsec_xy: tuple[float, float]
    footprint_corners_icrs: tuple[
        SkyCoordinate, SkyCoordinate, SkyCoordinate, SkyCoordinate
    ]
    alignment_sample_count: int = Field(ge=5)
    max_cross_plane_alignment_error_arcsec: float = Field(ge=0)
    boundary_sample_count: int = Field(ge=360)
    sampled_requested_circle_boundary_inside_pixel_footprint: Literal[True]
    valid: Literal[True] = True

    @model_validator(mode="after")
    def _validate_wcs(self) -> "CelestialWcsRef":
        indices = [entry.hdu_index for entry in self.plane_digests]
        names = [entry.extname for entry in self.plane_digests]
        digests = [entry.sha256 for entry in self.plane_digests]
        if len(self.plane_digests) != 3 or len(indices) != len(set(indices)):
            raise ValueError("WCS evidence requires three unique plane references")
        if set(names) != {"IMAGE", "MASK", "VARIANCE"}:
            raise ValueError("WCS evidence must cover image, mask, and variance")
        if len(set(digests)) != 1:
            raise ValueError("plane WCS transforms are not identical")
        if any(scale <= 0 for scale in self.pixel_scale_arcsec_xy):
            raise ValueError("WCS pixel scales must be positive")
        if self.max_cross_plane_alignment_error_arcsec > 1e-7:
            raise ValueError("cross-plane WCS alignment exceeds tolerance")
        return self


class ArchiveMetadataRef(_ImmutablePackageModel):
    json_hdu_index: int = Field(ge=1, le=31)
    index_hdu_index: int = Field(ge=1, le=31)
    json_bytes_sha256: str = Field(pattern=_SHA256_PATTERN)
    root_schema_version: Literal["1.0.0"]
    component_schema_version: Literal["1.0.0"]
    min_read_version: Literal[1]
    image_origin_yx: tuple[int, int]
    mask_origin_yx: tuple[int, int]
    variance_origin_yx: tuple[int, int]
    index_entry_count: int = Field(ge=1, le=32)
    serialized_components: tuple[
        Literal["image", "mask", "variance", "sky_projection"], ...
    ]
    array_references_cross_checked: Literal[True]
    index_entries_cross_checked: Literal[True]
    sky_projection_validation: Literal["presence_only"]
    metadata_preserved_in_hashed_fits: Literal[True]

    @model_validator(mode="after")
    def _validate_archive_metadata(self) -> "ArchiveMetadataRef":
        if len({self.json_hdu_index, self.index_hdu_index}) != 2:
            raise ValueError("archive metadata HDU indices must be unique")
        if not (self.image_origin_yx == self.mask_origin_yx == self.variance_origin_yx):
            raise ValueError("Rubin plane origins do not match")
        if set(self.serialized_components) != {
            "image",
            "mask",
            "variance",
            "sky_projection",
        }:
            raise ValueError("archive metadata component inventory is incomplete")
        return self


class ComponentLocator(_ImmutablePackageModel):
    hdu_index: int = Field(ge=1, le=31)
    extname: str = Field(min_length=1, max_length=68, pattern=r"^[A-Z][A-Z0-9_]*$")
    extver: int = Field(ge=1)


class PresentComponent(_ImmutablePackageModel):
    state: Literal["present"]
    locator: ComponentLocator
    basis: str = Field(min_length=1, max_length=512)

    @field_validator("basis")
    @classmethod
    def _safe_basis(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)


class AbsentComponent(_ImmutablePackageModel):
    state: Literal["absent_from_package"]
    reason_code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")
    basis: str = Field(min_length=1, max_length=512)
    checked_locations: tuple[str, ...]

    @field_validator("basis")
    @classmethod
    def _safe_basis(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)

    @field_validator("checked_locations")
    @classmethod
    def _safe_locations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("absent components require checked locations")
        return tuple(_validated_plain_text(item, remote=True) for item in value)


class UnknownComponent(_ImmutablePackageModel):
    state: Literal["unknown"]
    reason_code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")
    basis: str = Field(min_length=1, max_length=512)
    checked_locations: tuple[str, ...]

    @field_validator("basis")
    @classmethod
    def _safe_basis(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)

    @field_validator("checked_locations")
    @classmethod
    def _safe_locations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("unknown components require checked locations")
        return tuple(_validated_plain_text(item, remote=True) for item in value)


OptionalComponent = Annotated[
    PresentComponent | AbsentComponent | UnknownComponent,
    Field(discriminator="state"),
]


class CalibrationMetadata(_ImmutablePackageModel):
    state: Literal["physical_units_declared"] = "physical_units_declared"
    image_unit: Literal["nJy"]
    variance_unit: Literal["nJy2"]
    image_unit_sources: tuple[Literal["IMAGE BUNIT", "Rubin JSON image unit"], ...]
    variance_unit_sources: tuple[
        Literal["VARIANCE BUNIT", "Rubin JSON variance unit"], ...
    ]
    separate_photometric_calibration: OptionalComponent
    valid: Literal[True] = True

    @model_validator(mode="after")
    def _validate_unit_sources(self) -> "CalibrationMetadata":
        if set(self.image_unit_sources) != {"IMAGE BUNIT", "Rubin JSON image unit"}:
            raise ValueError("image unit source inventory is incomplete")
        if set(self.variance_unit_sources) != {
            "VARIANCE BUNIT",
            "Rubin JSON variance unit",
        }:
            raise ValueError("variance unit source inventory is incomplete")
        return self


class RetrievalProvenance(_ImmutablePackageModel):
    proof_mode: Literal["live_rubin_rsp"]
    authenticated: Literal[True]
    release: Literal["DP2"] = "DP2"
    service_origin: Literal["https://data.lsst.cloud"]
    sia_path: Literal["/api/sia/dp2/query"]
    soda_service_type: Literal["cutout-sync-maskedimage"]
    total_match_count: int = Field(ge=1)
    eligible_match_count: int = Field(ge=1)
    selection_rule: Literal["exact obs_id=lsst_cells_v2-5063-34"]
    content_type: Literal["application/fits"]
    service_cutout_created_at: str = Field(min_length=1, max_length=64)
    service_cutout_time_source: Literal["FITS DATE-CUT"] = "FITS DATE-CUT"
    checks: tuple[StageCheck, ...]

    @field_validator("service_cutout_created_at")
    @classmethod
    def _safe_created_at(cls, value: str) -> str:
        value = _validated_plain_text(value, remote=True)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", value):
            raise ValueError("FITS DATE-CUT is not in the expected ISO-like form")
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("FITS DATE-CUT is not a valid calendar timestamp") from exc
        return value

    @model_validator(mode="after")
    def _validate_retrieval(self) -> "RetrievalProvenance":
        if self.eligible_match_count > self.total_match_count:
            raise ValueError("eligible match count exceeds total matches")
        if not self.checks or not all(check.passed for check in self.checks):
            raise ValueError("retrieval provenance requires passing checks")
        names = [check.name for check in self.checks]
        required = {
            "sia_query",
            "dataset_selection",
            "datalink_soda_resolution",
            "soda_download",
            "fits_container",
            "local_artifact_integrity",
            "fits_identity",
            "masked_image_planes",
            "mask_bit_definitions",
            "variance_semantics",
            "celestial_wcs_alignment",
            "requested_region_coverage",
            "archive_metadata_crosscheck",
            "explicit_component_availability",
        }
        if len(names) != len(set(names)) or not required.issubset(names):
            raise ValueError("retrieval provenance has missing or duplicate checks")
        return self


class PreservationProvenance(_ImmutablePackageModel):
    canonical_payload: Literal["cutout.fits"] = "cutout.fits"
    payload_preserved_byte_for_byte: Literal[True]
    scientific_pixel_transformations: tuple[str, ...] = ()
    normalization_applied: Literal[False] = False
    resampling_applied: Literal[False] = False
    ripple_background_operation_applied: Literal[False] = False
    mask_modified: Literal[False] = False
    variance_modified: Literal[False] = False
    derived_metadata_operations: tuple[
        Literal[
            "read-only FITS validation",
            "decoded-plane hashing",
            "WCS alignment validation",
            "Rubin array and INDEX metadata cross-check",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def _validate_preservation(self) -> "PreservationProvenance":
        if self.scientific_pixel_transformations:
            raise ValueError("M2 cannot contain scientific pixel transformations")
        required = {
            "read-only FITS validation",
            "decoded-plane hashing",
            "WCS alignment validation",
            "Rubin array and INDEX metadata cross-check",
        }
        if set(self.derived_metadata_operations) != required:
            raise ValueError("derived metadata operation inventory is incomplete")
        return self


class PackageCompleteness(_ImmutablePackageModel):
    image: Literal["valid"]
    mask: Literal["valid"]
    variance: Literal["valid"]
    celestial_wcs: Literal["valid_aligned"]
    retrieval_provenance: Literal["valid"]
    psf: Literal["present", "absent_from_package", "unknown"]
    coadd_input_provenance: Literal["present", "absent_from_package", "unknown"]
    required_m2_components_complete: Literal[True]


class Dp2CutoutPackage(_ImmutablePackageModel):
    """Validated M2 manifest paired with one immutable masked-image FITS."""

    schema_version: Literal["ripple.dp2.cutout-package.v1"] = (
        "ripple.dp2.cutout-package.v1"
    )
    status: Literal["success"] = "success"
    product_kind: Literal["rubin_dp2_deep_coadd_masked_image"] = (
        "rubin_dp2_deep_coadd_masked_image"
    )
    request: Dp2CutoutRequest
    dataset: DatasetIdentity
    artifact: FitsArtifactRef
    fits_identity: FitsIdentityEvidence
    image: ImagePlaneRef
    mask: MaskPlaneRef
    variance: VariancePlaneRef
    celestial_wcs: CelestialWcsRef
    archive_metadata: ArchiveMetadataRef
    calibration: CalibrationMetadata
    psf: OptionalComponent
    coadd_input_provenance: OptionalComponent
    retrieval: RetrievalProvenance
    preservation: PreservationProvenance
    completeness: PackageCompleteness
    security: SecurityEvidence
    implementation_versions: tuple[NamedVersion, ...]
    implementation_sources: tuple[SourceDigest, ...]
    invocation: tuple[str, ...] = ("python", "-m", "ripple.dp2.package_cli")
    proof_boundary: Literal[
        "Authenticated DP2 masked-image retrieval and byte-preserving deep-coadd cutout packaging only; no scientific preprocessing, PSF model, lens classification, lens-classifier or model uncertainty estimate, or agent decision is validated."
    ] = (
        "Authenticated DP2 masked-image retrieval and byte-preserving deep-coadd cutout "
        "packaging only; no scientific preprocessing, PSF model, lens classification, "
        "lens-classifier or model uncertainty estimate, or agent decision is validated."
    )

    @model_validator(mode="after")
    def _validate_package(self) -> "Dp2CutoutPackage":
        if self.request.soda_service_type != "cutout-sync-maskedimage":
            raise ValueError("M2 requires the masked-image SODA service")
        if self.retrieval.soda_service_type != self.request.soda_service_type:
            raise ValueError("retrieval service does not match the request")
        if self.dataset.dataset_id != self.dataset.obs_publisher_did:
            raise ValueError("dataset identifiers disagree")
        for coordinate in (self.dataset.central_ra_deg, self.dataset.central_dec_deg):
            if coordinate is not None and not math.isfinite(coordinate):
                raise ValueError("dataset center contains a nonfinite coordinate")
        if self.dataset.observation_collection != self.request.expected_collection:
            raise ValueError("dataset collection does not match the request")
        if self.dataset.product_subtype != self.request.product_subtype:
            raise ValueError("dataset product subtype does not match the request")
        if self.dataset.calibration_level != self.request.calibration_level:
            raise ValueError("dataset calibration level does not match the request")
        if self.dataset.obs_id != self.request.expected_obs_id:
            raise ValueError(
                "dataset observation identifier does not match the request"
            )
        if self.dataset.band_name != self.request.band_name:
            raise ValueError("dataset band does not match the request")
        if (
            self.dataset.tract != self.request.expected_tract
            or self.dataset.patch != self.request.expected_patch
        ):
            raise ValueError("dataset tract or patch does not match the request")
        if (
            self.dataset.wavelength_min_m is None
            or self.dataset.wavelength_max_m is None
            or not self.dataset.wavelength_min_m
            <= self.request.effective_wavelength_m
            <= self.dataset.wavelength_max_m
        ):
            raise ValueError("requested wavelength is outside the selected dataset")

        query_ids = parse_qs(urlparse(self.dataset.obs_publisher_did).query).get(
            "id", []
        )
        expected_uuid = (
            query_ids[0].replace("-", "").lower() if len(query_ids) == 1 else ""
        )
        if self.fits_identity.butler_uuid != expected_uuid:
            raise ValueError("FITS Butler UUID does not match the selected dataset")
        if (
            self.fits_identity.dataset_type != "deep_coadd"
            or self.fits_identity.band_name != self.dataset.band_name
            or self.fits_identity.tract != self.dataset.tract
            or self.fits_identity.patch != self.dataset.patch
            or self.fits_identity.skymap != self.request.expected_skymap
        ):
            raise ValueError("FITS identity does not match dataset dimensions")
        if (
            self.fits_identity.stencil_type != "CIRCLE"
            or abs(self.fits_identity.stencil_ra_deg - self.request.ra_deg) > 1e-10
            or abs(self.fits_identity.stencil_dec_deg - self.request.dec_deg) > 1e-10
            or abs(
                self.fits_identity.stencil_radius_deg - self.request.cutout_radius_deg
            )
            > 1e-10
        ):
            raise ValueError("FITS stencil does not match the cutout request")

        plane_indices = [
            self.image.hdu_index,
            self.mask.hdu_index,
            self.variance.hdu_index,
        ]
        if (
            len(set(plane_indices)) != 3
            or max(plane_indices) >= self.artifact.hdu_count
        ):
            raise ValueError("plane HDU references are duplicate or out of range")
        if not (self.image.shape_yx == self.mask.shape_yx == self.variance.shape_yx):
            raise ValueError("image, mask, and variance shapes do not match")
        if not self.image.decoded_dtype.startswith("float"):
            raise ValueError("science image must use a floating decoded dtype")
        if not self.variance.decoded_dtype.startswith("float"):
            raise ValueError("variance must use a floating decoded dtype")
        if "int" not in self.mask.decoded_dtype:
            raise ValueError("mask must use an integer decoded dtype")
        expected_bitpix = {
            "float32": -32,
            "float64": -64,
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
            "int64": 64,
            "uint64": 64,
        }
        for plane in (self.image, self.mask, self.variance):
            if plane.serialization_dtype != plane.decoded_dtype:
                raise ValueError("serialized and decoded plane dtypes disagree")
            if plane.serialization_byteorder != "big":
                raise ValueError(
                    "Rubin FITS serialization must declare big-endian arrays"
                )
            if expected_bitpix.get(plane.decoded_dtype) != plane.fits_logical_bitpix:
                raise ValueError("plane dtype and FITS logical BITPIX disagree")
            expected_canonical_dtype = {
                "float32": "<f4",
                "float64": "<f8",
                "int8": "|i1",
                "uint8": "|u1",
                "int16": "<i2",
                "uint16": "<u2",
                "int32": "<i4",
                "uint32": "<u4",
                "int64": "<i8",
                "uint64": "<u8",
            }[plane.decoded_dtype]
            if plane.canonical_dtype != expected_canonical_dtype:
                raise ValueError("decoded and canonical plane dtypes disagree")
            if plane.digest.byte_count != plane.total_pixel_count * int(
                expected_canonical_dtype[-1]
            ):
                raise ValueError(
                    "plane digest byte count does not match dtype and shape"
                )

        target_x, target_y = self.celestial_wcs.target_pixel_xy
        height, width = self.image.shape_yx
        if not (-0.5 <= target_x <= width - 0.5 and -0.5 <= target_y <= height - 0.5):
            raise ValueError("WCS target pixel is outside the declared image footprint")

        try:
            image_unit = u.Unit(self.image.canonical_unit)
            variance_unit = u.Unit(self.variance.canonical_unit)
            declared_image_unit = u.Unit(self.image.declared_bunit)
            serialized_image_unit = u.Unit(self.image.serialization_unit)
            declared_variance_unit = u.Unit(self.variance.declared_bunit)
            serialized_variance_unit = u.Unit(self.variance.serialization_unit)
        except Exception as exc:
            raise ValueError("package contains an unparseable physical unit") from exc
        if not (
            declared_image_unit.is_equivalent(image_unit)
            and serialized_image_unit.is_equivalent(image_unit)
            and declared_variance_unit.is_equivalent(image_unit**2)
            and serialized_variance_unit.is_equivalent(image_unit**2)
            and variance_unit.is_equivalent(image_unit**2)
            and declared_image_unit.to(image_unit) == 1.0
            and serialized_image_unit.to(image_unit) == 1.0
            and declared_variance_unit.to(image_unit**2) == 1.0
            and serialized_variance_unit.to(image_unit**2) == 1.0
            and variance_unit.to(image_unit**2) == 1.0
        ):
            raise ValueError("image and variance units are inconsistent")
        if (
            self.calibration.image_unit != self.image.canonical_unit
            or self.calibration.variance_unit != self.variance.canonical_unit
        ):
            raise ValueError("calibration unit summary does not match the planes")

        wcs_indices = {
            entry.extname: entry.hdu_index for entry in self.celestial_wcs.plane_digests
        }
        if wcs_indices != {
            "IMAGE": self.image.hdu_index,
            "MASK": self.mask.hdu_index,
            "VARIANCE": self.variance.hdu_index,
        }:
            raise ValueError("WCS references do not match plane references")
        metadata_indices = {
            self.archive_metadata.json_hdu_index,
            self.archive_metadata.index_hdu_index,
        }
        if (
            metadata_indices.intersection(plane_indices)
            or max(metadata_indices) >= self.artifact.hdu_count
        ):
            raise ValueError("archive metadata HDU references are invalid")

        if self.completeness.psf != self.psf.state:
            raise ValueError("PSF completeness state is inconsistent")
        if (
            self.completeness.coadd_input_provenance
            != self.coadd_input_provenance.state
        ):
            raise ValueError("input-provenance completeness state is inconsistent")
        for component in (
            self.psf,
            self.coadd_input_provenance,
            self.calibration.separate_photometric_calibration,
        ):
            if (
                component.state == "present"
                and component.locator.hdu_index >= self.artifact.hdu_count
            ):
                raise ValueError("optional component HDU locator is out of range")
        if self.security.credential_was_present is not True:
            raise ValueError("live M2 package requires a process-supplied RSP token")
        if self.invocation != ("python", "-m", "ripple.dp2.package_cli"):
            raise ValueError("unexpected M2 invocation descriptor")

        versions = [item.name for item in self.implementation_versions]
        if len(versions) != len(set(versions)) or set(versions) != {
            "python",
            "pydantic",
            "requests",
            "astropy",
            "pyvo",
            "numpy",
        }:
            raise ValueError(
                "implementation version inventory is incomplete or duplicated"
            )
        source_names = [item.relative_path for item in self.implementation_sources]
        required_sources = {
            "ripple/__init__.py",
            "ripple/dp2/__init__.py",
            "ripple/dp2/errors.py",
            "ripple/dp2/models.py",
            "ripple/dp2/client.py",
            "ripple/dp2/service.py",
            "ripple/dp2/package_models.py",
            "ripple/dp2/package_service.py",
            "ripple/dp2/package_cli.py",
        }
        if (
            len(source_names) != len(set(source_names))
            or set(source_names) != required_sources
        ):
            raise ValueError("M2 implementation source digest inventory is incomplete")
        return self


class Dp2PackageFailureEvidence(_ImmutablePackageModel):
    schema_version: Literal["ripple.dp2.cutout-package.v1"] = (
        "ripple.dp2.cutout-package.v1"
    )
    status: Literal["failure"] = "failure"
    completed_at_utc: datetime
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")
    safe_message: str = Field(min_length=1, max_length=512)
    http_status: int | None = Field(default=None, ge=100, le=599)
    request: Dp2CutoutRequest
    security: SecurityEvidence

    @field_validator("safe_message")
    @classmethod
    def _safe_failure_message(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)

    @field_validator("completed_at_utc")
    @classmethod
    def _aware_utc_timestamp(cls, value: datetime) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
            or value.utcoffset().total_seconds() != 0
        ):
            raise ValueError("failure timestamp must be timezone-aware UTC")
        return value
