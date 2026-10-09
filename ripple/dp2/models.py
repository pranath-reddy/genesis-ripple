"""Pydantic contracts for the bounded Rubin DP2 Stage-1 workflow."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

SodaServiceType = Literal[
    "cutout-sync",
    "cutout-sync-maskedimage",
    "cutout-sync-exposure",
]
ComponentState = Literal["present", "absent", "unknown"]
Dp2Band = Literal["g", "r", "i"]

# Rubin LSSTCam effective wavelengths, in metres.  SIA2 BAND accepts a scalar
# wavelength and returns datasets whose spectral coverage contains that point.
# Authority: https://lsstcam.lsst.io/ (LSSTCam filter table).
DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND: Final[Mapping[Dp2Band, float]] = MappingProxyType(
    {
        "g": 4.807e-7,
        "r": 6.221e-7,
        "i": 7.559e-7,
    }
)

# Rubin's DP2 SIA tutorial publishes these recommended LSST band edges.  They
# are retained as an executable guard against an accidental selector mismatch.
# Authority: https://dp2.lsst.io/tutorials/notebook/103/notebook-103-2.html
DP2_SIA_BAND_EDGES_M_BY_BAND: Final[Mapping[Dp2Band, tuple[float, float]]] = (
    MappingProxyType(
        {
            "g": (4.026e-7, 5.483e-7),
            "r": (5.510e-7, 6.891e-7),
            "i": (6.936e-7, 8.188e-7),
        }
    )
)

_SAFE_REMOTE_TEXT_PATTERN = re.compile(r"^[\x20-\x7e]+$")
_FORBIDDEN_EVIDENCE_FRAGMENTS = (
    "http://",
    "https://",
    "bearer ",
    "authorization:",
    "cookie:",
    "set-cookie:",
    "token=",
    "access_token",
    "x-amz-",
    "x-goog-signature",
)


def _validated_plain_text(value: str, *, remote: bool = False) -> str:
    """Reject control characters and credential/URL-shaped evidence text."""
    if not _SAFE_REMOTE_TEXT_PATTERN.fullmatch(value):
        raise ValueError("evidence text must contain printable ASCII only")
    lowered = value.lower()
    fragments = (
        _FORBIDDEN_EVIDENCE_FRAGMENTS if remote else _FORBIDDEN_EVIDENCE_FRAGMENTS[2:]
    )
    if any(fragment in lowered for fragment in fragments):
        raise ValueError(
            "evidence text contained a forbidden credential or URL fragment"
        )
    return value


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )


class Dp2Credentials(_ImmutableModel):
    """Transient credential container that cannot serialize or reveal its value."""

    rsp_token: SecretStr = Field(exclude=True, repr=False)
    source: Literal["environment"] = "environment"
    environment_name: Literal["RSP_TOKEN"] = "RSP_TOKEN"

    @classmethod
    def from_environment(cls) -> "Dp2Credentials":
        value = os.environ.get("RSP_TOKEN")
        if value is None or not value.strip():
            from .errors import Dp2ConfigurationError

            raise Dp2ConfigurationError(
                stage="credentials",
                code="missing_rsp_token",
                message="Required environment variable RSP_TOKEN is not set.",
            )
        return cls(rsp_token=SecretStr(value))


class Dp2ClientConfig(_ImmutableModel):
    """Non-secret network and safety configuration."""

    sia_url: Literal["https://data.lsst.cloud/api/sia/dp2/query"] = (
        "https://data.lsst.cloud/api/sia/dp2/query"
    )
    token_environment_name: Literal["RSP_TOKEN"] = "RSP_TOKEN"
    connect_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    read_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    retries_total: int = Field(default=2, ge=0, le=5)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, le=10)
    max_metadata_response_bytes: int = Field(
        default=8 * 1024 * 1024,
        ge=1024,
        le=64 * 1024 * 1024,
    )
    max_download_bytes: int = Field(default=64 * 1024 * 1024, ge=1024, le=1024**3)
    user_agent: str = Field(
        default="ripple-dp2-smoke/1.0",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._/-]+$",
    )


class Dp2CutoutRequest(_ImmutableModel):
    """Scientifically explicit SIA discovery and SODA cutout request."""

    ra_deg: float = Field(default=53.1246023, ge=0, lt=360)
    dec_deg: float = Field(default=-27.7404715, ge=-90, le=90)
    search_radius_deg: float = Field(default=0.01, gt=0, le=1)
    cutout_radius_deg: float = Field(default=0.01, gt=0, le=0.25)
    effective_wavelength_m: float = Field(default=6.221e-7, gt=0.0)
    time_start_mjd_tai: float | None = Field(default=None, gt=0)
    time_end_mjd_tai: float | None = Field(default=None, gt=0)
    calibration_level: Literal[3] = 3
    product_subtype: Literal["lsst.deep_coadd"] = "lsst.deep_coadd"
    band_name: Dp2Band = "r"
    expected_collection: Literal["LSST.DP2"] = "LSST.DP2"
    expected_obs_id: Literal["lsst_cells_v2-5063-34"] = "lsst_cells_v2-5063-34"
    expected_skymap: Literal["lsst_cells_v2"] = "lsst_cells_v2"
    expected_tract: Literal[5063] = 5063
    expected_patch: Literal[34] = 34
    soda_service_type: SodaServiceType = "cutout-sync"

    @model_validator(mode="before")
    @classmethod
    def _derive_authoritative_wavelength(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        fields = dict(value)
        band = fields.get("band_name", "r")
        if (
            "effective_wavelength_m" not in fields
            and isinstance(band, str)
            and band in DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND
        ):
            fields["effective_wavelength_m"] = DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND[band]
        return fields

    @model_validator(mode="after")
    def _validate_ranges(self) -> "Dp2CutoutRequest":
        expected_wavelength = DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND[self.band_name]
        if not math.isclose(
            self.effective_wavelength_m,
            expected_wavelength,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "effective_wavelength_m must equal the authoritative LSSTCam value "
                "for band_name"
            )
        band_min, band_max = DP2_SIA_BAND_EDGES_M_BY_BAND[self.band_name]
        if not band_min < self.effective_wavelength_m < band_max:
            raise ValueError(
                "the authoritative effective wavelength is outside the DP2 SIA band edges"
            )
        if (self.time_start_mjd_tai is None) != (self.time_end_mjd_tai is None):
            raise ValueError("both time bounds must be provided together")
        if (
            self.time_start_mjd_tai is not None
            and self.time_end_mjd_tai is not None
            and self.time_end_mjd_tai <= self.time_start_mjd_tai
        ):
            raise ValueError("time_end_mjd_tai must be greater than time_start_mjd_tai")
        if self.cutout_radius_deg > self.search_radius_deg:
            raise ValueError("cutout_radius_deg cannot exceed search_radius_deg")
        return self


class DatasetIdentity(_ImmutableModel):
    dataset_id: str = Field(
        min_length=1,
        max_length=512,
        pattern=(
            r"^ivo://org\.rubinobs/usdac/lsst-dp2\?repo=dp2&"
            r"id=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    )
    identifier_field: Literal["obs_publisher_did"]
    obs_id: Literal["lsst_cells_v2-5063-34"]
    obs_publisher_did: str = Field(
        max_length=512,
        pattern=(
            r"^ivo://org\.rubinobs/usdac/lsst-dp2\?repo=dp2&"
            r"id=[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        ),
    )
    observation_collection: Literal["LSST.DP2"]
    product_subtype: Literal["lsst.deep_coadd"]
    calibration_level: Literal[3]
    facility_name: str | None = Field(default=None, max_length=128)
    instrument_name: str | None = Field(default=None, max_length=128)
    central_ra_deg: float | None = None
    central_dec_deg: float | None = None
    wavelength_min_m: float | None = None
    wavelength_max_m: float | None = None
    tract: Literal[5063]
    patch: Literal[34]
    band_name: Dp2Band
    access_format: Literal["application/x-votable+xml;content=datalink"]

    @field_validator(
        "dataset_id",
        "obs_id",
        "obs_publisher_did",
        "observation_collection",
        "product_subtype",
        "facility_name",
        "instrument_name",
        "band_name",
        "access_format",
    )
    @classmethod
    def _reject_unsafe_remote_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_plain_text(value, remote=True)


class StageCheck(_ImmutableModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    passed: bool
    message: str = Field(min_length=1, max_length=512)

    @field_validator("message")
    @classmethod
    def _safe_message(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)


class ComponentEvidence(_ImmutableModel):
    state: ComponentState
    basis: str = Field(min_length=1, max_length=512)
    hdu_index: int | None = None

    @field_validator("basis")
    @classmethod
    def _safe_basis(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)


class HduSummary(_ImmutableModel):
    index: int
    name: str = Field(min_length=1, max_length=68)
    shape: tuple[int, ...] | None = None
    dtype: str | None = Field(default=None, max_length=256)
    unit: str | None = Field(default=None, max_length=128)
    numeric: bool = False
    finite_fraction: float | None = Field(default=None, ge=0, le=1)

    @field_validator("name", "dtype", "unit")
    @classmethod
    def _safe_header_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_plain_text(value, remote=True)


class WcsEvidence(_ImmutableModel):
    state: ComponentState
    basis: str
    hdu_index: int | None = None
    target_pixel_x: float | None = None
    target_pixel_y: float | None = None
    target_inside_image: bool | None = None
    pixel_scale_arcsec: float | None = Field(default=None, gt=0)

    @field_validator("basis")
    @classmethod
    def _safe_basis(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)


class FitsIdentityEvidence(_ImmutableModel):
    butler_uuid: str = Field(min_length=32, max_length=64, pattern=r"^[0-9a-f]+$")
    dataset_type: Literal["deep_coadd"]
    band_name: Dp2Band
    skymap: Literal["lsst_cells_v2"]
    tract: Literal[5063]
    patch: Literal[34]
    stencil_type: Literal["CIRCLE"]
    stencil_ra_deg: float = Field(ge=0, lt=360)
    stencil_dec_deg: float = Field(ge=-90, le=90)
    stencil_radius_deg: float = Field(gt=0)
    cutout_service_version: str | None = Field(default=None, max_length=128)
    cutout_created_at: str | None = Field(default=None, max_length=64)
    matches_sia_dataset: Literal[True]
    matches_request: Literal[True]

    @field_validator("cutout_service_version", "cutout_created_at")
    @classmethod
    def _safe_header_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validated_plain_text(value, remote=True)


class FitsEvidence(_ImmutableModel):
    opened: Literal[True]
    hdu_count: int = Field(ge=1)
    hdus: tuple[HduSummary, ...]
    science_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image: ComponentEvidence
    mask: ComponentEvidence
    variance: ComponentEvidence
    wcs: WcsEvidence
    identity: FitsIdentityEvidence
    psf: ComponentEvidence
    retrieval_provenance: ComponentEvidence
    input_provenance: ComponentEvidence


class DownloadEvidence(_ImmutableModel):
    filename: str = Field(min_length=1, max_length=128)
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str | None = Field(default=None, max_length=256)

    @field_validator("filename")
    @classmethod
    def _filename_only(cls, value: str) -> str:
        if Path(value).name != value:
            raise ValueError("download filename must not contain a path")
        return value

    @field_validator("content_type")
    @classmethod
    def _safe_content_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if "\r" in value or "\n" in value or "bearer " in lowered or "://" in value:
            raise ValueError("unsafe content type")
        return value


class SecurityEvidence(_ImmutableModel):
    credential_source: Literal["environment"]
    credential_name: Literal["RSP_TOKEN"]
    credential_was_present: bool
    credential_value_recorded: Literal[False]
    authorization_headers_recorded: Literal[False]
    access_urls_recorded: Literal[False]


class Dp2SmokeEvidence(_ImmutableModel):
    schema_version: Literal["ripple.dp2.smoke.v1"] = "ripple.dp2.smoke.v1"
    status: Literal["success"] = "success"
    completed_at_utc: datetime
    service_origin: Literal["https://data.lsst.cloud"] = "https://data.lsst.cloud"
    sia_path: Literal["/api/sia/dp2/query"] = "/api/sia/dp2/query"
    query: Dp2CutoutRequest
    total_match_count: int = Field(ge=1)
    eligible_match_count: int = Field(ge=1)
    selection_rule: Literal["exact obs_id=lsst_cells_v2-5063-34"]
    dataset: DatasetIdentity
    download: DownloadEvidence
    fits: FitsEvidence
    checks: tuple[StageCheck, ...]
    security: SecurityEvidence
    implementation_versions: dict[str, str]
    implementation_source_sha256: dict[str, str]
    invocation: tuple[Literal["python", "-m", "ripple.dp2.cli"], ...] = (
        "python",
        "-m",
        "ripple.dp2.cli",
    )
    proof_boundary: str = (
        "Authenticated DP2 discovery, cutout download, and FITS opening only; "
        "no scientific preprocessing or lens classification is validated."
    )

    @model_validator(mode="after")
    def _validate_success_semantics(self) -> "Dp2SmokeEvidence":
        if not self.checks or not all(check.passed for check in self.checks):
            raise ValueError("success evidence requires every recorded check to pass")
        check_names = [check.name for check in self.checks]
        required = {
            "sia_query",
            "dataset_selection",
            "datalink_soda_resolution",
            "soda_download",
            "fits_container",
            "fits_science_image",
            "fits_celestial_wcs",
            "fits_identity",
            "local_artifact_integrity",
        }
        if len(check_names) != len(set(check_names)) or not required.issubset(
            check_names
        ):
            raise ValueError(
                "success evidence has missing or duplicate required checks"
            )
        if self.eligible_match_count > self.total_match_count:
            raise ValueError("eligible match count cannot exceed total match count")
        if self.dataset.obs_id != self.query.expected_obs_id:
            raise ValueError("selected dataset does not match the expected obs_id")
        if self.dataset.observation_collection != self.query.expected_collection:
            raise ValueError("selected dataset does not match the expected collection")
        if self.dataset.band_name != self.query.band_name:
            raise ValueError("selected dataset does not match the requested band")
        if (
            self.dataset.tract != self.query.expected_tract
            or self.dataset.patch != self.query.expected_patch
        ):
            raise ValueError("selected dataset does not match the expected tract/patch")
        if not self.fits.opened or self.fits.image.state != "present":
            raise ValueError("success evidence requires an opened science image")
        if (
            self.fits.wcs.state != "present"
            or self.fits.wcs.target_inside_image is not True
        ):
            raise ValueError(
                "success evidence requires target-containing celestial WCS"
            )
        if (
            not self.fits.identity.matches_sia_dataset
            or not self.fits.identity.matches_request
        ):
            raise ValueError("FITS identity must match both SIA selection and request")
        if self.security.credential_was_present is not True:
            raise ValueError(
                "success evidence requires an RSP token supplied to the process"
            )
        required_sources = {
            "ripple/__init__.py",
            "ripple/dp2/__init__.py",
            "ripple/dp2/errors.py",
            "ripple/dp2/models.py",
            "ripple/dp2/client.py",
            "ripple/dp2/service.py",
            "ripple/dp2/cli.py",
        }
        if set(self.implementation_source_sha256) != required_sources:
            raise ValueError("implementation source hash manifest is incomplete")
        for name, digest in self.implementation_source_sha256.items():
            if (
                not name
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("invalid implementation source hash")
        if self.invocation != ("python", "-m", "ripple.dp2.cli"):
            raise ValueError("unexpected Stage-1 invocation descriptor")
        return self


class FailureEvidence(_ImmutableModel):
    schema_version: Literal["ripple.dp2.smoke.v1"] = "ripple.dp2.smoke.v1"
    status: Literal["failure"] = "failure"
    completed_at_utc: datetime
    stage: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    code: str = Field(min_length=1, max_length=96, pattern=r"^[a-z0-9_]+$")
    safe_message: str = Field(min_length=1, max_length=512)
    http_status: int | None = Field(default=None, ge=100, le=599)
    query: Dp2CutoutRequest
    security: SecurityEvidence

    @field_validator("safe_message")
    @classmethod
    def _safe_failure_message(cls, value: str) -> str:
        return _validated_plain_text(value, remote=True)


class RetrievalReceipt(_ImmutableModel):
    proof_mode: Literal["live_rubin_rsp"]
    authenticated: Literal[True]
    service_origin: Literal["https://data.lsst.cloud"]
    sia_path: Literal["/api/sia/dp2/query"]
    dataset: DatasetIdentity
    total_match_count: int = Field(ge=1)
    eligible_match_count: int = Field(ge=1)
    selection_rule: Literal["exact obs_id=lsst_cells_v2-5063-34"]
    artifact_path: Path
    byte_count: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: str | None = None
    checks: tuple[StageCheck, ...]
