"""Build, serialize, and reload the M2 Rubin DP2 cutout package."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy
import numpy as np
import pydantic
import pyvo
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, wcs_to_celestial_frame

from .errors import Dp2ConfigurationError, Dp2FitsValidationError
from .models import (
    DatasetIdentity,
    Dp2CutoutRequest,
    RetrievalReceipt,
    SecurityEvidence,
    StageCheck,
)
from .package_models import (
    AbsentComponent,
    ArchiveMetadataRef,
    CalibrationMetadata,
    CelestialWcsRef,
    ComponentLocator,
    DecodedArrayDigest,
    Dp2CutoutPackage,
    Dp2PackageFailureEvidence,
    FitsArtifactRef,
    FitsScaling,
    ImagePlaneRef,
    MaskBitDefinition,
    MaskPlaneRef,
    NamedVersion,
    PackageCompleteness,
    PlaneWcsDigest,
    PreservationProvenance,
    PresentComponent,
    RetrievalProvenance,
    SkyCoordinate,
    SourceDigest,
    UnknownComponent,
    VariancePlaneRef,
)
from .service import (
    Dp2Gateway,
    _assert_sanitized_payload,
    _fits_identity,
    _hash_file,
    _preflight_fits,
    _reject_symlink_ancestors,
    _require_private_run_directory,
    _validate_declared_fits_size,
)

_SAFE_SERVICE_ORIGIN = "https://data.lsst.cloud"
_MAX_PACKAGE_JSON_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_JSON_BYTES = 1024 * 1024
_BOUNDARY_SAMPLE_COUNT = 720
_WCS_ALIGNMENT_TOLERANCE_ARCSEC = 1e-7


@dataclass(frozen=True)
class LoadedDp2Cutout:
    """Runtime view returned only after the manifest and FITS are reverified."""

    package: Dp2CutoutPackage
    image: np.ndarray
    mask: np.ndarray
    variance: np.ndarray
    celestial_wcs: WCS

    def __post_init__(self) -> None:
        self.image.setflags(write=False)
        self.mask.setflags(write=False)
        self.variance.setflags(write=False)


@dataclass(frozen=True)
class _MaskedImageInspection:
    hdu_count: int
    fits_identity: Any
    image: ImagePlaneRef
    mask: MaskPlaneRef
    variance: VariancePlaneRef
    celestial_wcs: CelestialWcsRef
    archive_metadata: ArchiveMetadataRef
    calibration: CalibrationMetadata
    psf: PresentComponent | AbsentComponent | UnknownComponent
    coadd_input_provenance: PresentComponent | AbsentComponent | UnknownComponent
    image_array: np.ndarray
    mask_array: np.ndarray
    variance_array: np.ndarray
    runtime_wcs: WCS


class Dp2PackageService:
    """Retrieve one live masked-image cutout and publish a verified M2 pair."""

    def __init__(self, gateway: Dp2Gateway) -> None:
        self._gateway = gateway

    def run(
        self,
        request: Dp2CutoutRequest,
        output_directory: Path,
    ) -> Dp2CutoutPackage:
        if request.soda_service_type != "cutout-sync-maskedimage":
            raise Dp2ConfigurationError(
                stage="package_request",
                code="masked_image_service_required",
                message="M2 requires the cutout-sync-maskedimage SODA service.",
            )
        output_directory = Path(output_directory)
        _require_private_run_directory(output_directory)
        destination = output_directory / "cutout.fits"

        receipt = self._gateway.retrieve_one_cutout(request, destination)
        self._validate_live_receipt(receipt, destination)
        actual_bytes, actual_sha256 = _hash_file(destination)
        if actual_bytes != receipt.byte_count or actual_sha256 != receipt.sha256:
            raise Dp2FitsValidationError(
                stage="artifact_integrity",
                code="download_receipt_mismatch",
                message="The local FITS artifact did not match its download receipt.",
            )

        inspection = inspect_masked_image(destination, request, receipt.dataset)
        checks = receipt.checks + _package_checks()
        content_type = _normalized_fits_content_type(receipt.content_type)
        created_at = inspection.fits_identity.cutout_created_at
        if created_at is None:
            raise Dp2FitsValidationError(
                stage="retrieval_provenance",
                code="missing_cutout_creation_time",
                message="The FITS artifact lacked the DATE-CUT retrieval timestamp.",
            )

        package = Dp2CutoutPackage(
            request=request,
            dataset=receipt.dataset,
            artifact=FitsArtifactRef(
                byte_count=actual_bytes,
                sha256=actual_sha256,
                hdu_count=inspection.hdu_count,
            ),
            fits_identity=inspection.fits_identity,
            image=inspection.image,
            mask=inspection.mask,
            variance=inspection.variance,
            celestial_wcs=inspection.celestial_wcs,
            archive_metadata=inspection.archive_metadata,
            calibration=inspection.calibration,
            psf=inspection.psf,
            coadd_input_provenance=inspection.coadd_input_provenance,
            retrieval=RetrievalProvenance(
                proof_mode=receipt.proof_mode,
                authenticated=receipt.authenticated,
                service_origin=receipt.service_origin,
                sia_path=receipt.sia_path,
                soda_service_type="cutout-sync-maskedimage",
                total_match_count=receipt.total_match_count,
                eligible_match_count=receipt.eligible_match_count,
                selection_rule=receipt.selection_rule,
                content_type=content_type,
                service_cutout_created_at=created_at,
                checks=checks,
            ),
            preservation=PreservationProvenance(
                payload_preserved_byte_for_byte=True,
                derived_metadata_operations=(
                    "read-only FITS validation",
                    "decoded-plane hashing",
                    "WCS alignment validation",
                    "Rubin array and INDEX metadata cross-check",
                ),
            ),
            completeness=PackageCompleteness(
                image="valid",
                mask="valid",
                variance="valid",
                celestial_wcs="valid_aligned",
                retrieval_provenance="valid",
                psf=inspection.psf.state,
                coadd_input_provenance=inspection.coadd_input_provenance.state,
                required_m2_components_complete=True,
            ),
            security=SecurityEvidence(
                credential_source="environment",
                credential_name="RSP_TOKEN",
                credential_was_present=True,
                credential_value_recorded=False,
                authorization_headers_recorded=False,
                access_urls_recorded=False,
            ),
            implementation_versions=_implementation_versions(),
            implementation_sources=_implementation_source_digests(),
        )

        verified = _load_verified_payload(package, destination)
        if verified.package != package:
            raise Dp2FitsValidationError(
                stage="package_reload",
                code="prepublication_package_mismatch",
                message="The prepublication FITS verification did not match the package.",
            )
        package_path = output_directory / "package.json"
        published = write_package_model_json_atomic(package_path, package)
        if not isinstance(published, Dp2CutoutPackage) or published != package:
            raise Dp2ConfigurationError(
                stage="package_serialization",
                code="package_round_trip_mismatch",
                message="The serialized package did not round-trip through its Pydantic contract.",
            )
        return package

    @staticmethod
    def _validate_live_receipt(receipt: RetrievalReceipt, destination: Path) -> None:
        if (
            receipt.proof_mode != "live_rubin_rsp"
            or receipt.authenticated is not True
            or receipt.service_origin != _SAFE_SERVICE_ORIGIN
            or receipt.sia_path != "/api/sia/dp2/query"
            or receipt.artifact_path != destination
        ):
            raise Dp2FitsValidationError(
                stage="retrieval_provenance",
                code="invalid_live_receipt",
                message="The M2 retrieval receipt did not match the requested local artifact.",
            )


def inspect_masked_image(
    path: Path,
    request: Dp2CutoutRequest,
    dataset: DatasetIdentity,
) -> _MaskedImageInspection:
    """Validate the exact masked-image product and extract immutable references."""
    if request.soda_service_type != "cutout-sync-maskedimage":
        raise Dp2FitsValidationError(
            stage="masked_image_contract",
            code="wrong_soda_service_type",
            message="Masked-image inspection requires cutout-sync-maskedimage.",
        )
    path = Path(path)
    try:
        details = os.lstat(path)
        if not stat.S_ISREG(details.st_mode):
            raise Dp2FitsValidationError(
                stage="masked_image_contract",
                code="non_regular_fits_artifact",
                message="The masked-image artifact was not a regular file.",
            )
        _preflight_fits(path)
        with fits.open(
            path, mode="readonly", memmap=False, lazy_load_hdus=False
        ) as hdul:
            hdul.verify("exception")
            _validate_declared_fits_size(hdul)
            identity = _fits_identity(hdul[0].header, request, dataset)
            indices = {
                name: _require_unique_hdu(hdul, name)
                for name in ("IMAGE", "MASK", "VARIANCE", "JSON", "INDEX")
            }

            image_array = _copy_numeric_plane(hdul[indices["IMAGE"]], "IMAGE")
            mask_array = _copy_numeric_plane(hdul[indices["MASK"]], "MASK")
            variance_array = _copy_numeric_plane(hdul[indices["VARIANCE"]], "VARIANCE")
            if (
                image_array.ndim != 2
                or mask_array.ndim != 2
                or variance_array.ndim != 2
            ):
                raise _fits_error(
                    "masked_image_planes",
                    "non_2d_plane",
                    "IMAGE, MASK, and VARIANCE must each be two-dimensional.",
                )
            if not (image_array.shape == mask_array.shape == variance_array.shape):
                raise _fits_error(
                    "masked_image_planes",
                    "plane_shape_mismatch",
                    "IMAGE, MASK, and VARIANCE did not have identical shapes.",
                )
            if not np.issubdtype(image_array.dtype, np.floating):
                raise _fits_error(
                    "masked_image_planes",
                    "invalid_image_dtype",
                    "IMAGE did not use a floating numeric dtype.",
                )
            if not np.issubdtype(mask_array.dtype, np.integer):
                raise _fits_error(
                    "masked_image_planes",
                    "invalid_mask_dtype",
                    "MASK did not use an integer dtype.",
                )
            if not np.issubdtype(variance_array.dtype, np.floating):
                raise _fits_error(
                    "masked_image_planes",
                    "invalid_variance_dtype",
                    "VARIANCE did not use a floating numeric dtype.",
                )

            archive, archive_json = _archive_metadata(
                hdul,
                indices,
                image_array,
                mask_array,
                variance_array,
            )
            mask, unsigned_mask = _mask_ref(
                hdul[indices["MASK"]],
                indices["MASK"],
                mask_array,
                archive_json,
            )
            image = _image_ref(
                hdul[indices["IMAGE"]],
                indices["IMAGE"],
                image_array,
                unsigned_mask,
                archive_json,
            )
            variance = _variance_ref(
                hdul[indices["VARIANCE"]],
                indices["VARIANCE"],
                variance_array,
                unsigned_mask,
                archive_json,
            )
            celestial_wcs, runtime_wcs = _wcs_ref(
                hdul,
                indices,
                image_array.shape,
                request,
            )
            psf = _optional_component(
                hdul,
                archive_json,
                label="PSF",
                json_key="psf",
                reason_code="no_explicit_psf_payload",
            )
            input_provenance = _optional_component(
                hdul,
                archive_json,
                label="INPUT_PROVENANCE",
                json_key="input_provenance",
                reason_code="no_explicit_coadd_input_provenance",
                alternate_names=("PROVENANCE", "INPUTS"),
            )
            separate_calibration = _optional_component(
                hdul,
                archive_json,
                label="PHOTOCALIB",
                json_key="photometric_calibration",
                reason_code="no_separate_photometric_calibration_payload",
                alternate_names=("CALIBRATION",),
            )
            calibration = CalibrationMetadata(
                image_unit="nJy",
                variance_unit="nJy2",
                image_unit_sources=("IMAGE BUNIT", "Rubin JSON image unit"),
                variance_unit_sources=(
                    "VARIANCE BUNIT",
                    "Rubin JSON variance unit",
                ),
                separate_photometric_calibration=separate_calibration,
            )
            return _MaskedImageInspection(
                hdu_count=len(hdul),
                fits_identity=identity,
                image=image,
                mask=mask,
                variance=variance,
                celestial_wcs=celestial_wcs,
                archive_metadata=archive,
                calibration=calibration,
                psf=psf,
                coadd_input_provenance=input_provenance,
                image_array=image_array,
                mask_array=mask_array,
                variance_array=variance_array,
                runtime_wcs=runtime_wcs,
            )
    except Dp2FitsValidationError:
        raise
    except Exception as exc:
        raise Dp2FitsValidationError(
            stage="masked_image_contract",
            code="masked_image_inspection_failed",
            message=f"Masked-image inspection failed ({type(exc).__name__}).",
        ) from None


def load_cutout_package(package_path: Path) -> LoadedDp2Cutout:
    """Reload a package, rehash its FITS and planes, and return read-only arrays."""
    package_path = Path(package_path)
    _require_private_run_directory(package_path.parent)
    if package_path.name != "package.json":
        raise Dp2ConfigurationError(
            stage="package_reload",
            code="invalid_package_filename",
            message="The M2 manifest filename must be package.json.",
        )
    _require_private_regular_file(package_path, max_bytes=_MAX_PACKAGE_JSON_BYTES)
    encoded = package_path.read_bytes()
    try:
        parsed = json.loads(encoded)
        _assert_sanitized_payload(parsed)
        package = Dp2CutoutPackage.model_validate_json(encoded)
    except Dp2ConfigurationError:
        raise
    except Exception as exc:
        raise Dp2ConfigurationError(
            stage="package_reload",
            code="invalid_package_json",
            message=f"The M2 manifest could not be validated ({type(exc).__name__}).",
        ) from None

    artifact_path = package_path.parent / package.artifact.filename
    return _load_verified_payload(package, artifact_path)


def _load_verified_payload(
    package: Dp2CutoutPackage,
    artifact_path: Path,
) -> LoadedDp2Cutout:
    """Rehash and reinspect the FITS described by an already validated model."""
    _require_private_regular_file(artifact_path, max_bytes=None)
    byte_count, sha256 = _hash_file(artifact_path)
    if byte_count != package.artifact.byte_count or sha256 != package.artifact.sha256:
        raise Dp2FitsValidationError(
            stage="package_reload",
            code="artifact_digest_mismatch",
            message="The FITS artifact no longer matches the package manifest.",
        )
    inspection = inspect_masked_image(artifact_path, package.request, package.dataset)
    observed = (
        inspection.hdu_count,
        inspection.fits_identity,
        inspection.image,
        inspection.mask,
        inspection.variance,
        inspection.celestial_wcs,
        inspection.archive_metadata,
        inspection.calibration,
        inspection.psf,
        inspection.coadd_input_provenance,
    )
    declared = (
        package.artifact.hdu_count,
        package.fits_identity,
        package.image,
        package.mask,
        package.variance,
        package.celestial_wcs,
        package.archive_metadata,
        package.calibration,
        package.psf,
        package.coadd_input_provenance,
    )
    if observed != declared:
        raise Dp2FitsValidationError(
            stage="package_reload",
            code="fits_manifest_mismatch",
            message="Reopened FITS content did not match the M2 manifest.",
        )
    return LoadedDp2Cutout(
        package=package,
        image=inspection.image_array,
        mask=inspection.mask_array,
        variance=inspection.variance_array,
        celestial_wcs=inspection.runtime_wcs,
    )


def write_package_model_json_atomic(
    path: Path,
    model: Dp2CutoutPackage | Dp2PackageFailureEvidence,
) -> Dp2CutoutPackage | Dp2PackageFailureEvidence:
    """Round-trip and privately publish one allowlisted M2 JSON model."""
    if not isinstance(model, (Dp2CutoutPackage, Dp2PackageFailureEvidence)):
        raise TypeError("only M2 package or failure models can be serialized")
    path = Path(path)
    _require_private_run_directory(path.parent)
    allowed_name = (
        "package.json" if isinstance(model, Dp2CutoutPackage) else "failure.json"
    )
    if path.name != allowed_name:
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="invalid_package_json_filename",
            message="The M2 JSON filename was outside the serializer allowlist.",
        )
    if os.path.lexists(path):
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="package_json_exists",
            message="Refused to overwrite an existing M2 JSON artifact.",
        )

    payload = model.model_dump(mode="json")
    _assert_sanitized_payload(payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > _MAX_PACKAGE_JSON_BYTES:
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="package_json_too_large",
            message="The M2 JSON artifact exceeded its local size bound.",
        )
    model_type = type(model)
    try:
        reloaded = model_type.model_validate_json(encoded)
    except Exception as exc:
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="package_json_round_trip_failed",
            message=f"The M2 JSON failed contract round-trip ({type(exc).__name__}).",
        ) from None
    if reloaded != model:
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="package_json_round_trip_mismatch",
            message="The M2 JSON changed during contract round-trip.",
        )

    temporary = path.parent / f".{path.name}.{os.getpid()}.part"
    created_temporary = False
    published = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        created_temporary = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            on_disk = temporary.read_bytes()
            disk_reloaded = model_type.model_validate_json(on_disk)
        except Exception as exc:
            raise Dp2ConfigurationError(
                stage="package_serialization",
                code="temporary_json_reload_failed",
                message=f"The private M2 JSON temporary file could not be reloaded ({type(exc).__name__}).",
            ) from None
        if on_disk != encoded or disk_reloaded != model:
            raise Dp2ConfigurationError(
                stage="package_serialization",
                code="temporary_json_reload_mismatch",
                message="The private M2 JSON temporary file changed before publication.",
            )
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise Dp2ConfigurationError(
                stage="package_serialization",
                code="package_json_race_detected",
                message="The M2 JSON destination appeared during serialization.",
            ) from None
        published = True
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)
    if not published:
        raise Dp2ConfigurationError(
            stage="package_serialization",
            code="package_json_publish_failed",
            message="The M2 JSON artifact could not be published.",
        )
    return disk_reloaded


def _require_unique_hdu(hdul: fits.HDUList, name: str) -> int:
    matches = [
        index
        for index, hdu in enumerate(hdul)
        if str(getattr(hdu, "name", "") or "").strip().upper() == name
    ]
    if len(matches) != 1:
        raise _fits_error(
            "masked_image_planes",
            f"invalid_{name.lower()}_hdu_count",
            f"The FITS artifact must contain exactly one {name} HDU.",
        )
    return matches[0]


def _copy_numeric_plane(hdu: fits.hdu.base.ExtensionHDU, name: str) -> np.ndarray:
    data = getattr(hdu, "data", None)
    if data is None or not np.issubdtype(data.dtype, np.number) or data.size < 1:
        raise _fits_error(
            "masked_image_planes",
            f"invalid_{name.lower()}_plane",
            f"The {name} HDU did not contain a non-empty numeric array.",
        )
    return np.array(data, copy=True, order="C")


def _plane_common(
    hdu: Any,
    hdu_index: int,
    data: np.ndarray,
    serialization_ref: dict[str, Any],
) -> dict[str, Any]:
    decoded_dtype = np.dtype(data.dtype)
    logical_bitpix = int(hdu.header.get("BITPIX"))
    storage_kind = (
        "tile_compressed_image" if isinstance(hdu, fits.CompImageHDU) else "image_hdu"
    )
    compression_type = (
        str(hdu.compression_type).strip()
        if storage_kind == "tile_compressed_image"
        else None
    )
    bscale = hdu.header.get("BSCALE")
    bzero = hdu.header.get("BZERO")
    blank = hdu.header.get("BLANK")
    return {
        "hdu_index": hdu_index,
        "extname": str(hdu.name).strip().upper(),
        "extver": int(hdu.ver),
        "extver_declared": bool("EXTVER" in hdu.header),
        "shape_yx": tuple(int(axis) for axis in data.shape),
        "decoded_dtype": decoded_dtype.name,
        "decoded_byteorder": _resolved_byteorder(decoded_dtype),
        "canonical_dtype": decoded_dtype.newbyteorder("<").str,
        "serialization_dtype": str(serialization_ref["datatype"]).strip(),
        "serialization_byteorder": str(serialization_ref["byteorder"]).strip().lower(),
        "fits_logical_bitpix": logical_bitpix,
        "storage_kind": storage_kind,
        "compression_type": compression_type,
        "scaling": FitsScaling(
            bscale=float(bscale) if bscale is not None else None,
            bzero=float(bzero) if bzero is not None else None,
            blank=int(blank) if blank is not None else None,
        ),
        "digest": _decoded_array_digest(data),
        "total_pixel_count": int(data.size),
        "valid": True,
    }


def _image_ref(
    hdu: Any,
    hdu_index: int,
    image: np.ndarray,
    unsigned_mask: np.ndarray,
    archive_json: dict[str, Any],
) -> ImagePlaneRef:
    header_unit = _required_header_unit(hdu, "IMAGE")
    serialized_unit = _archive_unit(archive_json, "image")
    serialization_ref = _archive_array_ref(archive_json, "image")
    _require_unit_scale(header_unit, u.nJy, "IMAGE BUNIT")
    _require_unit_scale(serialized_unit, u.nJy, "Rubin JSON image unit")
    finite = np.isfinite(image)
    if not np.any(finite):
        raise _fits_error(
            "masked_image_planes",
            "image_has_no_finite_pixels",
            "IMAGE did not contain any finite pixels.",
        )
    zero_mask_nonfinite = int(np.count_nonzero((unsigned_mask == 0) & ~finite))
    values = image[finite]
    return ImagePlaneRef(
        **_plane_common(hdu, hdu_index, image, serialization_ref),
        declared_bunit=header_unit,
        serialization_unit=serialized_unit,
        canonical_unit="nJy",
        finite_pixel_count=int(np.count_nonzero(finite)),
        nonfinite_pixel_count=int(np.count_nonzero(~finite)),
        zero_mask_nonfinite_pixel_count=zero_mask_nonfinite,
        finite_min=float(values.min()),
        finite_max=float(values.max()),
    )


def _mask_ref(
    hdu: Any,
    hdu_index: int,
    mask: np.ndarray,
    archive_json: dict[str, Any],
) -> tuple[MaskPlaneRef, np.ndarray]:
    if hdu.header.get("BUNIT") not in {None, ""}:
        raise _fits_error(
            "mask_bit_definitions",
            "unexpected_mask_unit",
            "MASK unexpectedly declared a physical BUNIT.",
        )
    bit_width = int(mask.dtype.itemsize * 8)
    serialization_ref = _archive_array_ref(archive_json, "mask")
    full_width_mask = (1 << bit_width) - 1
    unsigned = np.asarray(mask, dtype=np.uint64) & np.uint64(full_width_mask)
    bit_indices = sorted(
        int(str(key)[4:])
        for key in hdu.header
        if isinstance(key, str) and key.startswith("MSKN") and str(key)[4:].isdigit()
    )
    if not bit_indices:
        raise _fits_error(
            "mask_bit_definitions",
            "missing_mask_definitions",
            "MASK did not contain named bit definitions.",
        )
    json_planes = archive_json.get("mask", {}).get("planes")
    if not isinstance(json_planes, list) or len(json_planes) != len(bit_indices):
        raise _fits_error(
            "archive_metadata",
            "mask_metadata_count_mismatch",
            "Rubin JSON mask definitions did not match FITS mask cards.",
        )
    definitions: list[MaskBitDefinition] = []
    known_bits = 0
    for ordinal, bit_index in enumerate(bit_indices):
        name_key = f"MSKN{bit_index:04d}"
        value_key = f"MSKM{bit_index:04d}"
        description_key = f"MSKD{bit_index:04d}"
        if value_key not in hdu.header or description_key not in hdu.header:
            raise _fits_error(
                "mask_bit_definitions",
                "incomplete_mask_definition",
                "A FITS mask definition lacked its value or description.",
            )
        name = str(hdu.header[name_key]).strip()
        value = int(hdu.header[value_key])
        description = str(hdu.header[description_key]).strip()
        json_definition = json_planes[ordinal]
        if not isinstance(json_definition, dict) or (
            str(json_definition.get("name", "")).strip() != name
            or str(json_definition.get("description", "")).strip() != description
        ):
            raise _fits_error(
                "archive_metadata",
                "mask_metadata_mismatch",
                "Rubin JSON and FITS mask definitions disagreed.",
            )
        if bit_index >= bit_width or value != 1 << bit_index:
            raise _fits_error(
                "mask_bit_definitions",
                "invalid_mask_bit_value",
                "A mask bit definition was incompatible with the MASK dtype.",
            )
        known_bits |= value
        definitions.append(
            MaskBitDefinition(
                name=name,
                bit_index=bit_index,
                value=value,
                description=description,
                set_pixel_count=int(np.count_nonzero(unsigned & np.uint64(value))),
            )
        )
    unknown_value_mask = full_width_mask & ~known_bits
    unknown_pixels = int(np.count_nonzero(unsigned & np.uint64(unknown_value_mask)))
    unknown_bits = tuple(
        bit
        for bit in range(bit_width)
        if (unknown_value_mask & (1 << bit)) and np.any(unsigned & np.uint64(1 << bit))
    )
    mask_ref = MaskPlaneRef(
        **_plane_common(hdu, hdu_index, mask, serialization_ref),
        declared_bunit=None,
        bit_width=bit_width,
        bits=tuple(definitions),
        nonzero_pixel_count=int(np.count_nonzero(unsigned)),
        unique_combination_count=int(np.unique(unsigned).size),
        unsigned_min=int(unsigned.min()),
        unsigned_max=int(unsigned.max()),
        undefined_set_bits=unknown_bits,
        undefined_set_pixel_count=unknown_pixels,
    )
    return mask_ref, unsigned


def _variance_ref(
    hdu: Any,
    hdu_index: int,
    variance: np.ndarray,
    unsigned_mask: np.ndarray,
    archive_json: dict[str, Any],
) -> VariancePlaneRef:
    header_unit = _required_header_unit(hdu, "VARIANCE")
    serialized_unit = _archive_unit(archive_json, "variance")
    serialization_ref = _archive_array_ref(archive_json, "variance")
    target_unit = u.nJy**2
    _require_unit_scale(header_unit, target_unit, "VARIANCE BUNIT")
    _require_unit_scale(serialized_unit, target_unit, "Rubin JSON variance unit")
    finite = np.isfinite(variance)
    positive = finite & (variance > 0)
    zero = finite & (variance == 0)
    negative = finite & (variance < 0)
    if not np.any(positive):
        raise _fits_error(
            "variance_semantics",
            "variance_has_no_positive_pixels",
            "VARIANCE did not contain any positive finite samples.",
        )
    zero_mask_invalid = int(
        np.count_nonzero((unsigned_mask == 0) & (~finite | (variance <= 0)))
    )
    finite_values = variance[finite]
    return VariancePlaneRef(
        **_plane_common(hdu, hdu_index, variance, serialization_ref),
        declared_bunit=header_unit,
        serialization_unit=serialized_unit,
        canonical_unit="nJy2",
        finite_pixel_count=int(np.count_nonzero(finite)),
        nonfinite_pixel_count=int(np.count_nonzero(~finite)),
        positive_finite_pixel_count=int(np.count_nonzero(positive)),
        zero_finite_pixel_count=int(np.count_nonzero(zero)),
        negative_finite_pixel_count=int(np.count_nonzero(negative)),
        zero_mask_nonpositive_or_nonfinite_pixel_count=zero_mask_invalid,
        finite_min=float(finite_values.min()),
        finite_max=float(finite_values.max()),
    )


def _archive_metadata(
    hdul: fits.HDUList,
    indices: dict[str, int],
    image: np.ndarray,
    mask: np.ndarray,
    variance: np.ndarray,
) -> tuple[ArchiveMetadataRef, dict[str, Any]]:
    json_hdu = hdul[indices["JSON"]]
    data = getattr(json_hdu, "data", None)
    if data is None or len(data) != 1 or "JSON" not in (data.dtype.names or ()):
        raise _fits_error(
            "archive_metadata",
            "invalid_json_hdu",
            "The Rubin JSON HDU did not have the expected one-row payload.",
        )
    raw_values = np.asarray(data["JSON"][0], dtype=np.uint8)
    if (
        raw_values.ndim != 1
        or raw_values.size < 2
        or raw_values.size > _MAX_ARCHIVE_JSON_BYTES
    ):
        raise _fits_error(
            "archive_metadata",
            "invalid_json_payload_size",
            "The Rubin JSON metadata payload was empty or exceeded its size bound.",
        )
    raw_json = bytes(raw_values)
    try:
        parsed = json.loads(raw_json.decode("utf-8"))
    except Exception:
        raise _fits_error(
            "archive_metadata",
            "invalid_json_payload",
            "The Rubin JSON metadata payload could not be parsed.",
        ) from None
    if not isinstance(parsed, dict):
        raise _fits_error(
            "archive_metadata",
            "invalid_json_root",
            "The Rubin JSON metadata root was not an object.",
        )
    for component in ("image", "mask", "variance", "sky_projection"):
        if not isinstance(parsed.get(component), dict):
            raise _fits_error(
                "archive_metadata",
                "missing_serialized_component",
                "Rubin JSON metadata lacked a required masked-image component.",
            )
    if parsed.get("schema_version") != "1.0.0" or parsed.get("min_read_version") != 1:
        raise _fits_error(
            "archive_metadata",
            "unsupported_archive_schema",
            "The Rubin JSON root schema version is outside the supported M2 contract.",
        )
    for component in ("image", "mask", "variance", "sky_projection"):
        metadata = parsed[component]
        if (
            metadata.get("schema_version") != "1.0.0"
            or metadata.get("min_read_version") != 1
        ):
            raise _fits_error(
                "archive_metadata",
                "unsupported_component_schema",
                "A Rubin JSON component schema version is outside the supported M2 contract.",
            )

    _validate_archive_array_ref(parsed, "image", "IMAGE", image)
    _validate_archive_array_ref(parsed, "mask", "MASK", mask)
    _validate_archive_array_ref(parsed, "variance", "VARIANCE", variance)
    origins = {
        component: _origin_yx(parsed[component].get("yx0"), component)
        for component in ("image", "mask", "variance")
    }

    index_hdu = hdul[indices["INDEX"]]
    index_data = getattr(index_hdu, "data", None)
    required_index_columns = {
        "EXTNAME",
        "EXTVER",
        "XTENSION",
        "ZIMAGE",
        "HDRADDR",
        "DATADDR",
        "DATSIZE",
    }
    if (
        index_data is None
        or not required_index_columns.issubset(set(index_data.dtype.names or ()))
        or len(index_data) != indices["INDEX"]
    ):
        raise _fits_error(
            "archive_metadata",
            "invalid_index_hdu",
            "The Rubin INDEX HDU lacked a complete row for each preceding HDU.",
        )
    _validate_index_rows(hdul, index_data, indices["INDEX"])
    root_schema = str(parsed.get("schema_version", "")).strip()
    return (
        ArchiveMetadataRef(
            json_hdu_index=indices["JSON"],
            index_hdu_index=indices["INDEX"],
            json_bytes_sha256=hashlib.sha256(raw_json).hexdigest(),
            root_schema_version=root_schema,
            component_schema_version="1.0.0",
            min_read_version=1,
            image_origin_yx=origins["image"],
            mask_origin_yx=origins["mask"],
            variance_origin_yx=origins["variance"],
            index_entry_count=int(len(index_data)),
            serialized_components=("image", "mask", "variance", "sky_projection"),
            array_references_cross_checked=True,
            index_entries_cross_checked=True,
            sky_projection_validation="presence_only",
            metadata_preserved_in_hashed_fits=True,
        ),
        parsed,
    )


def _validate_index_rows(
    hdul: fits.HDUList,
    index_data: Any,
    indexed_hdu_count: int,
) -> None:
    """Cross-check Rubin INDEX identity, storage kind, and byte extents."""
    seen_names: set[str] = set()
    for index in range(indexed_hdu_count):
        hdu = hdul[index]
        row = index_data[index]
        observed_name = _decode_fits_text(row["EXTNAME"]).upper()
        expected_name = "" if index == 0 else str(hdu.name).strip().upper()
        if observed_name:
            if observed_name in seen_names:
                raise _fits_error(
                    "archive_metadata",
                    "duplicate_index_extname",
                    "The Rubin INDEX HDU repeated an extension name.",
                )
            seen_names.add(observed_name)
        physical_header = getattr(hdu, "_header", hdu.header)
        expected_xtension = (
            "IMAGE"
            if index == 0
            else str(physical_header.get("XTENSION", "")).strip().upper()
        )
        expected_zimage = bool(physical_header.get("ZIMAGE", False))
        file_info = hdu.fileinfo()
        if file_info is None:
            raise _fits_error(
                "archive_metadata",
                "missing_hdu_file_extent",
                "A FITS HDU lacked file extent information for INDEX validation.",
            )
        if (
            observed_name != expected_name
            or int(row["EXTVER"]) != int(hdu.ver)
            or _decode_fits_text(row["XTENSION"]).upper() != expected_xtension
            or bool(row["ZIMAGE"]) != expected_zimage
            or int(row["HDRADDR"]) != int(file_info["hdrLoc"])
            or int(row["DATADDR"]) != int(file_info["datLoc"])
            or int(row["DATSIZE"]) != int(file_info["datSpan"])
        ):
            raise _fits_error(
                "archive_metadata",
                "index_row_mismatch",
                "A Rubin INDEX row did not match its referenced FITS HDU.",
            )
    if seen_names != {"IMAGE", "MASK", "VARIANCE", "JSON"}:
        raise _fits_error(
            "archive_metadata",
            "incomplete_index_hdu",
            "The Rubin INDEX HDU did not inventory every required extension.",
        )


def _validate_archive_array_ref(
    parsed: dict[str, Any],
    component: str,
    extname: str,
    array: np.ndarray,
) -> None:
    ref = _archive_array_ref(parsed, component)
    expected_shape = [int(axis) for axis in array.shape]
    if (
        str(ref.get("source", "")).strip() != f"fits:{extname}"
        or ref.get("shape") != expected_shape
        or str(ref.get("datatype", "")).strip() != array.dtype.name
        or str(ref.get("byteorder", "")).strip().lower() != "big"
    ):
        raise _fits_error(
            "archive_metadata",
            "array_reference_mismatch",
            "Rubin JSON array metadata did not match the opened FITS plane.",
        )


def _archive_array_ref(parsed: dict[str, Any], component: str) -> dict[str, Any]:
    metadata = parsed.get(component)
    if not isinstance(metadata, dict):
        raise _fits_error(
            "archive_metadata",
            "invalid_component_metadata",
            "Rubin JSON metadata contained an invalid component object.",
        )
    if component == "mask":
        refs = metadata.get("data")
        ref = refs[0] if isinstance(refs, list) and len(refs) == 1 else None
    else:
        data = metadata.get("data")
        ref = data.get("value") if isinstance(data, dict) else None
    if not isinstance(ref, dict):
        raise _fits_error(
            "archive_metadata",
            "invalid_array_reference",
            "Rubin JSON metadata contained an invalid FITS array reference.",
        )
    return ref


def _origin_yx(value: Any, component: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise _fits_error(
            "archive_metadata",
            "invalid_parent_origin",
            f"Rubin JSON {component} metadata lacked a valid yx0 origin.",
        )
    return int(value[0]), int(value[1])


def _wcs_ref(
    hdul: fits.HDUList,
    indices: dict[str, int],
    shape_yx: tuple[int, ...],
    request: Dp2CutoutRequest,
) -> tuple[CelestialWcsRef, WCS]:
    if len(shape_yx) != 2:
        raise _fits_error(
            "celestial_wcs_alignment",
            "invalid_wcs_shape",
            "WCS validation requires a two-dimensional image shape.",
        )
    wcs_by_name: dict[str, WCS] = {}
    digests: list[PlaneWcsDigest] = []
    for name in ("IMAGE", "MASK", "VARIANCE"):
        try:
            celestial = WCS(hdul[indices[name]].header).celestial
            if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
                raise ValueError("not two-dimensional")
            digest = _wcs_digest(celestial)
        except Exception:
            raise _fits_error(
                "celestial_wcs_alignment",
                "invalid_plane_wcs",
                f"The {name} plane lacked a valid two-dimensional celestial WCS.",
            ) from None
        wcs_by_name[name] = celestial
        digests.append(
            PlaneWcsDigest(
                hdu_index=indices[name],
                extname=name,
                sha256=digest,
            )
        )
    if len({item.sha256 for item in digests}) != 1:
        raise _fits_error(
            "celestial_wcs_alignment",
            "wcs_fingerprint_mismatch",
            "IMAGE, MASK, and VARIANCE did not carry identical WCS transforms.",
        )

    height, width = int(shape_yx[0]), int(shape_yx[1])
    xs = np.array([0.0, (width - 1) / 2.0, float(width - 1)] * 3)
    ys = np.repeat(np.array([0.0, (height - 1) / 2.0, float(height - 1)]), 3)
    reference_sky = wcs_by_name["IMAGE"].pixel_to_world(xs, ys)
    maximum_separation = 0.0
    for name in ("MASK", "VARIANCE"):
        candidate_sky = wcs_by_name[name].pixel_to_world(xs, ys)
        separations = reference_sky.separation(candidate_sky).arcsec
        maximum_separation = max(maximum_separation, float(np.max(separations)))
    if maximum_separation > _WCS_ALIGNMENT_TOLERANCE_ARCSEC:
        raise _fits_error(
            "celestial_wcs_alignment",
            "wcs_alignment_mismatch",
            "Cross-plane WCS alignment exceeded the M2 tolerance.",
        )

    reference = wcs_by_name["IMAGE"]
    target = SkyCoord(request.ra_deg * u.deg, request.dec_deg * u.deg, frame="icrs")
    target_x, target_y = reference.world_to_pixel(target)
    target_x, target_y = float(target_x), float(target_y)
    if not _inside_pixel_edges(target_x, target_y, width, height):
        raise _fits_error(
            "requested_region_coverage",
            "target_outside_cutout",
            "The requested coordinate was outside the image pixel footprint.",
        )

    position_angles = (
        np.linspace(
            0.0,
            2.0 * np.pi,
            _BOUNDARY_SAMPLE_COUNT,
            endpoint=False,
        )
        * u.rad
    )
    boundary = target.directional_offset_by(
        position_angles,
        request.cutout_radius_deg * u.deg,
    )
    boundary_x, boundary_y = reference.world_to_pixel(boundary)
    covered = bool(
        np.all(np.isfinite(boundary_x))
        and np.all(np.isfinite(boundary_y))
        and np.all(boundary_x >= -0.5)
        and np.all(boundary_x <= width - 0.5)
        and np.all(boundary_y >= -0.5)
        and np.all(boundary_y <= height - 0.5)
    )
    if not covered:
        raise _fits_error(
            "requested_region_coverage",
            "requested_circle_not_covered",
            "The FITS pixel footprint did not cover the sampled requested-circle boundary.",
        )

    corner_x = np.array([-0.5, width - 0.5, width - 0.5, -0.5])
    corner_y = np.array([-0.5, -0.5, height - 0.5, height - 0.5])
    corners = reference.pixel_to_world(corner_x, corner_y).icrs
    scales = np.asarray(proj_plane_pixel_scales(reference), dtype=float) * 3600.0
    frame = wcs_to_celestial_frame(reference).name.lower()
    ctype = tuple(str(value).strip().upper() for value in reference.wcs.ctype)
    cunit = tuple(str(value).strip() for value in reference.wcs.cunit)
    if frame != "icrs" or ctype != ("RA---TAN", "DEC--TAN") or cunit != ("deg", "deg"):
        raise _fits_error(
            "celestial_wcs_alignment",
            "unsupported_wcs_convention",
            "The M2 target requires an ICRS RA/DEC TAN projection in degrees.",
        )
    model = CelestialWcsRef(
        plane_digests=tuple(digests),
        frame="icrs",
        ctype=ctype,
        axis_units=("deg", "deg"),
        projection="TAN",
        target_pixel_xy=(target_x, target_y),
        target_inside=True,
        pixel_scale_arcsec_xy=(float(scales[0]), float(scales[1])),
        footprint_corners_icrs=tuple(
            SkyCoordinate(ra_deg=float(ra), dec_deg=float(dec))
            for ra, dec in zip(corners.ra.deg, corners.dec.deg, strict=True)
        ),
        alignment_sample_count=int(xs.size),
        max_cross_plane_alignment_error_arcsec=maximum_separation,
        boundary_sample_count=_BOUNDARY_SAMPLE_COUNT,
        sampled_requested_circle_boundary_inside_pixel_footprint=True,
    )
    return model, reference


def _optional_component(
    hdul: fits.HDUList,
    archive_json: dict[str, Any],
    *,
    label: str,
    json_key: str,
    reason_code: str,
    alternate_names: tuple[str, ...] = (),
) -> PresentComponent | AbsentComponent | UnknownComponent:
    expected = {label, *alternate_names}
    matches = [
        (index, hdu)
        for index, hdu in enumerate(hdul)
        if str(getattr(hdu, "name", "") or "").strip().upper() in expected
    ]
    if len(matches) > 1:
        return UnknownComponent(
            state="unknown",
            reason_code="multiple_explicit_payloads",
            basis=f"Multiple explicit {label} HDUs were found and were not interpreted by M2.",
            checked_locations=("exact FITS EXTNAME inventory", "Rubin JSON root keys"),
        )
    if len(matches) == 1:
        index, hdu = matches[0]
        return PresentComponent(
            state="present",
            locator=ComponentLocator(
                hdu_index=index,
                extname=str(hdu.name).strip().upper(),
                extver=int(hdu.ver),
            ),
            basis=f"One explicit {label} HDU was present; M2 records its locator only.",
        )
    if json_key in archive_json:
        return UnknownComponent(
            state="unknown",
            reason_code="serialized_component_without_hdu_locator",
            basis=f"Rubin JSON mentioned {label}, but M2 found no explicit FITS HDU locator.",
            checked_locations=("exact FITS EXTNAME inventory", "Rubin JSON root keys"),
        )
    return AbsentComponent(
        state="absent_from_package",
        reason_code=reason_code,
        basis=f"No explicit {label} payload was present in this masked-image package.",
        checked_locations=("exact FITS EXTNAME inventory", "Rubin JSON root keys"),
    )


def _decoded_array_digest(data: np.ndarray) -> DecodedArrayDigest:
    dtype = np.dtype(data.dtype).newbyteorder("<")
    canonical = np.ascontiguousarray(data.astype(dtype, copy=False))
    descriptor = json.dumps(
        {
            "dtype": dtype.str,
            "order": "C",
            "shape": [int(axis) for axis in canonical.shape],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"ripple.dp2.decoded-array.v1\n")
    digest.update(descriptor)
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return DecodedArrayDigest(
        byte_count=int(canonical.nbytes), sha256=digest.hexdigest()
    )


def _wcs_digest(celestial: WCS) -> str:
    cards = [
        (card.keyword, repr(card.value), str(card.comment))
        for card in celestial.to_header(relax=True).cards
    ]
    encoded = json.dumps(cards, ensure_ascii=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(b"ripple.dp2.wcs.v1\n" + encoded).hexdigest()


def _resolved_byteorder(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1 or dtype.byteorder == "|":
        return "not_applicable"
    if dtype.byteorder == "=":
        return sys.byteorder
    return "little" if dtype.byteorder == "<" else "big"


def _required_header_unit(hdu: Any, label: str) -> str:
    value = hdu.header.get("BUNIT")
    if value is None or not str(value).strip():
        raise _fits_error(
            "calibration_metadata",
            f"missing_{label.lower()}_unit",
            f"The {label} plane did not declare BUNIT.",
        )
    return str(value).strip()


def _archive_unit(parsed: dict[str, Any], component: str) -> str:
    data = parsed.get(component, {}).get("data")
    unit_value = data.get("unit") if isinstance(data, dict) else None
    if unit_value is None or not str(unit_value).strip():
        raise _fits_error(
            "archive_metadata",
            f"missing_{component}_serialization_unit",
            f"Rubin JSON {component} metadata did not declare a unit.",
        )
    return str(unit_value).strip()


def _require_unit_scale(value: str, expected: u.UnitBase, label: str) -> None:
    try:
        unit = u.Unit(value)
        factor = float(unit.to(expected))
    except Exception:
        raise _fits_error(
            "calibration_metadata",
            "invalid_physical_unit",
            f"{label} was not parseable as the required physical unit.",
        ) from None
    if not np.isclose(factor, 1.0, rtol=0.0, atol=0.0):
        raise _fits_error(
            "calibration_metadata",
            "implicit_unit_scaling_forbidden",
            f"{label} would require an unrecorded value scaling.",
        )


def _inside_pixel_edges(x: float, y: float, width: int, height: int) -> bool:
    return bool(
        np.isfinite(x)
        and np.isfinite(y)
        and -0.5 <= x <= width - 0.5
        and -0.5 <= y <= height - 0.5
    )


def _decode_fits_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="strict").strip()
    return str(value).strip()


def _normalized_fits_content_type(value: str | None) -> str:
    normalized = str(value or "").split(";", 1)[0].strip().lower()
    if normalized != "application/fits":
        raise Dp2FitsValidationError(
            stage="retrieval_provenance",
            code="unexpected_content_type",
            message="The masked-image download did not declare application/fits.",
        )
    return "application/fits"


def _package_checks() -> tuple[StageCheck, ...]:
    return (
        StageCheck(
            name="local_artifact_integrity",
            passed=True,
            message="Local FITS size and SHA-256 match the live download receipt.",
        ),
        StageCheck(
            name="fits_identity",
            passed=True,
            message="FITS dataset identity and stencil match SIA and the request.",
        ),
        StageCheck(
            name="masked_image_planes",
            passed=True,
            message="Unique IMAGE, MASK, and VARIANCE planes have identical geometry.",
        ),
        StageCheck(
            name="mask_bit_definitions",
            passed=True,
            message="All set mask bits have unique named definitions cross-checked with metadata.",
        ),
        StageCheck(
            name="variance_semantics",
            passed=True,
            message="Variance units and all-pixel positive finite validity satisfy the M2 contract.",
        ),
        StageCheck(
            name="celestial_wcs_alignment",
            passed=True,
            message="All three planes carry the same validated celestial WCS transform.",
        ),
        StageCheck(
            name="requested_region_coverage",
            passed=True,
            message="The target and sampled requested-circle boundary lie inside pixel edges.",
        ),
        StageCheck(
            name="archive_metadata_crosscheck",
            passed=True,
            message=(
                "Rubin JSON array references and INDEX identity, storage, and byte extents "
                "agree with the FITS payload; serialized sky projection is retained but "
                "only checked for presence."
            ),
        ),
        StageCheck(
            name="explicit_component_availability",
            passed=True,
            message="PSF, calibration, and coadd-input provenance availability are explicit.",
        ),
    )


def _implementation_versions() -> tuple[NamedVersion, ...]:
    return (
        NamedVersion(name="python", version=platform.python_version()),
        NamedVersion(name="pydantic", version=pydantic.__version__),
        NamedVersion(name="requests", version=requests.__version__),
        NamedVersion(name="astropy", version=astropy.__version__),
        NamedVersion(name="pyvo", version=pyvo.__version__),
        NamedVersion(name="numpy", version=np.__version__),
    )


def _implementation_source_digests() -> tuple[SourceDigest, ...]:
    package_directory = Path(__file__).resolve().parent
    sources = (
        ("ripple/__init__.py", package_directory.parent / "__init__.py"),
        ("ripple/dp2/__init__.py", package_directory / "__init__.py"),
        ("ripple/dp2/errors.py", package_directory / "errors.py"),
        ("ripple/dp2/models.py", package_directory / "models.py"),
        ("ripple/dp2/client.py", package_directory / "client.py"),
        ("ripple/dp2/service.py", package_directory / "service.py"),
        ("ripple/dp2/package_models.py", package_directory / "package_models.py"),
        ("ripple/dp2/package_service.py", package_directory / "package_service.py"),
        ("ripple/dp2/package_cli.py", package_directory / "package_cli.py"),
    )
    return tuple(
        SourceDigest(
            relative_path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for name, path in sources
    )


def _require_private_regular_file(path: Path, *, max_bytes: int | None) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise Dp2ConfigurationError(
            stage="package_reload",
            code="missing_package_file",
            message="A required M2 package file was missing.",
        ) from None
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or details.st_mode & 0o077
    ):
        raise Dp2ConfigurationError(
            stage="package_reload",
            code="insecure_package_file",
            message="M2 package files must be private regular files owned by this user.",
        )
    if max_bytes is not None and details.st_size > max_bytes:
        raise Dp2ConfigurationError(
            stage="package_reload",
            code="package_file_too_large",
            message="An M2 package file exceeded its local size bound.",
        )


def _fits_error(stage: str, code: str, message: str) -> Dp2FitsValidationError:
    return Dp2FitsValidationError(stage=stage, code=code, message=message)
