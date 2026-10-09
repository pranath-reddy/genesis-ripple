"""Deterministic Stage-1 orchestration and FITS evidence construction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

import astropy
import numpy as np
import pydantic
import pyvo
import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from .errors import Dp2ConfigurationError, Dp2FitsValidationError
from .models import (
    ComponentEvidence,
    DatasetIdentity,
    DownloadEvidence,
    Dp2CutoutRequest,
    Dp2SmokeEvidence,
    FailureEvidence,
    FitsEvidence,
    FitsIdentityEvidence,
    HduSummary,
    RetrievalReceipt,
    SecurityEvidence,
    StageCheck,
    WcsEvidence,
)

_MAX_HDU_COUNT = 32
_MAX_DECODED_BYTES = 256 * 1024 * 1024
_MAX_AXIS_LENGTH = 8192
_FITS_BLOCK_BYTES = 2880
_MAX_HEADER_BLOCKS_PER_HDU = 64
_SAFE_SERVICE_ORIGIN = "https://data.lsst.cloud"
_FORBIDDEN_SERIALIZED_FRAGMENTS = (
    "bearer ",
    "authorization:",
    "cookie:",
    "set-cookie:",
    "access_token",
    "token=",
    "x-amz-",
    "x-goog-signature",
)
_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
)


class Dp2Gateway(Protocol):
    """Injected boundary between orchestration and the Rubin adapter."""

    def retrieve_one_cutout(
        self,
        request: Dp2CutoutRequest,
        destination: Path,
    ) -> RetrievalReceipt: ...


class Dp2SmokeService:
    """Run one bounded retrieval and construct allowlisted success evidence."""

    def __init__(self, gateway: Dp2Gateway) -> None:
        self._gateway = gateway

    def run(
        self,
        request: Dp2CutoutRequest,
        output_directory: Path,
    ) -> Dp2SmokeEvidence:
        output_directory = Path(output_directory)
        _require_private_run_directory(output_directory)

        destination = output_directory / "cutout.fits"
        receipt = self._gateway.retrieve_one_cutout(request, destination)
        if (
            receipt.proof_mode != "live_rubin_rsp"
            or receipt.authenticated is not True
            or receipt.service_origin != _SAFE_SERVICE_ORIGIN
            or receipt.sia_path != "/api/sia/dp2/query"
            or receipt.artifact_path != destination
        ):
            raise Dp2FitsValidationError(
                stage="evidence_provenance",
                code="invalid_live_receipt",
                message="The Stage-1 retrieval receipt did not match the requested local artifact.",
            )

        actual_bytes, actual_digest = _hash_file(destination)
        if actual_bytes != receipt.byte_count or actual_digest != receipt.sha256:
            raise Dp2FitsValidationError(
                stage="artifact_integrity",
                code="download_receipt_mismatch",
                message="The local FITS artifact did not match its download receipt.",
            )

        fits_evidence = inspect_fits(destination, request, receipt.dataset)
        checks = receipt.checks + (
            StageCheck(
                name="local_artifact_integrity",
                passed=True,
                message="Local FITS size and SHA-256 match the download receipt.",
            ),
            StageCheck(
                name="fits_science_image",
                passed=True,
                message="A non-empty numeric two-dimensional science image was found.",
            ),
            StageCheck(
                name="fits_celestial_wcs",
                passed=True,
                message="Celestial WCS maps the requested coordinate inside the cutout.",
            ),
            StageCheck(
                name="fits_identity",
                passed=True,
                message="FITS dataset identity and cutout stencil match SIA and the request.",
            ),
        )
        if request.soda_service_type in {
            "cutout-sync-maskedimage",
            "cutout-sync-exposure",
        }:
            if (
                fits_evidence.mask.state != "present"
                or fits_evidence.variance.state != "present"
            ):
                raise Dp2FitsValidationError(
                    stage="fits_components",
                    code="missing_mask_or_variance",
                    message="The requested masked-image product lacked an explicit mask or variance plane.",
                )
            checks += (
                StageCheck(
                    name="fits_mask_variance",
                    passed=True,
                    message="Explicit mask and variance planes were found.",
                ),
            )

        evidence = Dp2SmokeEvidence(
            completed_at_utc=datetime.now(timezone.utc),
            query=request,
            total_match_count=receipt.total_match_count,
            eligible_match_count=receipt.eligible_match_count,
            selection_rule=receipt.selection_rule,
            dataset=receipt.dataset,
            download=DownloadEvidence(
                filename=destination.name,
                byte_count=receipt.byte_count,
                sha256=receipt.sha256,
                content_type=receipt.content_type,
            ),
            fits=fits_evidence,
            checks=checks,
            security=SecurityEvidence(
                credential_source="environment",
                credential_name="RSP_TOKEN",
                credential_was_present=True,
                credential_value_recorded=False,
                authorization_headers_recorded=False,
                access_urls_recorded=False,
            ),
            implementation_versions={
                "python": platform.python_version(),
                "pydantic": pydantic.__version__,
                "requests": requests.__version__,
                "astropy": astropy.__version__,
                "pyvo": pyvo.__version__,
            },
            implementation_source_sha256=_implementation_source_hashes(),
        )
        write_model_json_atomic(output_directory / "evidence.json", evidence)
        return evidence


def create_private_run_directory(output_root: Path) -> Path:
    """Create one never-reused 0700 directory beneath a non-symlink root."""
    output_root = Path(output_root)
    if os.path.lexists(output_root):
        _reject_symlink_ancestors(output_root)
        if not output_root.is_dir():
            raise Dp2ConfigurationError(
                stage="local_output",
                code="invalid_output_root",
                message="The output root exists but is not a directory.",
            )
    else:
        _reject_symlink_ancestors(output_root.parent)
        output_root.mkdir(parents=True, exist_ok=False)
        _reject_symlink_ancestors(output_root)

    root_details = os.lstat(output_root)
    if (
        not stat.S_ISDIR(root_details.st_mode)
        or root_details.st_uid != os.geteuid()
        or root_details.st_mode & 0o022
    ):
        raise Dp2ConfigurationError(
            stage="local_output",
            code="insecure_output_root",
            message="The output root must be owned by this user and not group/world writable.",
        )

    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    created = Path(tempfile.mkdtemp(prefix=prefix, dir=output_root))
    created.chmod(0o700)
    _require_private_run_directory(created)
    return created


def _reject_symlink_ancestors(path: Path) -> None:
    if ".." in path.parts:
        raise Dp2ConfigurationError(
            stage="local_output",
            code="parent_traversal_forbidden",
            message="Parent traversal is forbidden in the Stage-1 output path.",
        )
    absolute = path.absolute()
    chain = list(reversed(absolute.parents)) + [absolute]
    for component in chain:
        if component == Path(component.anchor):
            continue
        try:
            mode = os.lstat(component).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise Dp2ConfigurationError(
                stage="local_output",
                code="symlinked_output_path",
                message="Symlinks are forbidden in the Stage-1 output path.",
            )


def _require_private_run_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise Dp2ConfigurationError(
            stage="local_output",
            code="missing_run_directory",
            message="The private run directory did not exist.",
        ) from None
    if not stat.S_ISDIR(details.st_mode) or details.st_mode & 0o077:
        raise Dp2ConfigurationError(
            stage="local_output",
            code="insecure_run_directory",
            message="The run directory must be a private 0700 non-symlink directory.",
        )


def inspect_fits(
    path: Path,
    request: Dp2CutoutRequest,
    dataset: DatasetIdentity,
) -> FitsEvidence:
    """Inspect a bounded FITS cutout and cross-check its identity and stencil."""
    try:
        details = os.lstat(path)
        if not stat.S_ISREG(details.st_mode):
            raise Dp2FitsValidationError(
                stage="fits_inspection",
                code="non_regular_fits_artifact",
                message="The local FITS artifact was not a regular file.",
            )
        _preflight_fits(path)
        with fits.open(
            path, mode="readonly", memmap=False, lazy_load_hdus=False
        ) as hdul:
            hdul.verify("exception")
            _validate_declared_fits_size(hdul)
            identity = _fits_identity(hdul[0].header, request, dataset)

            hdu_summaries: list[HduSummary] = []
            science_candidates: list[int] = []
            structural_images: list[int] = []
            names: dict[int, str] = {}

            for index, hdu in enumerate(hdul):
                name = str(getattr(hdu, "name", "") or f"HDU{index}").strip().upper()
                names[index] = name
                data = getattr(hdu, "data", None)
                shape = (
                    tuple(int(value) for value in data.shape)
                    if data is not None
                    else None
                )
                dtype = str(data.dtype) if data is not None else None
                numeric = bool(
                    data is not None and np.issubdtype(data.dtype, np.number)
                )
                finite_fraction = None
                if numeric and data.size:
                    finite_fraction = float(np.isfinite(data).sum() / data.size)
                unit_value = hdu.header.get("BUNIT")
                unit = str(unit_value).strip() if unit_value is not None else None
                hdu_summaries.append(
                    HduSummary(
                        index=index,
                        name=name,
                        shape=shape,
                        dtype=dtype,
                        unit=unit or None,
                        numeric=numeric,
                        finite_fraction=finite_fraction,
                    )
                )
                if numeric and data.ndim == 2 and data.size > 0:
                    structural_images.append(index)
                    if finite_fraction is not None and finite_fraction > 0:
                        science_candidates.append(index)

            explicit_images = [
                index
                for index in science_candidates
                if names[index] in {"IMAGE", "SCI", "SCIENCE"}
            ]
            if not explicit_images:
                raise Dp2FitsValidationError(
                    stage="fits_science_image",
                    code="missing_explicit_science_image",
                    message="No finite numeric HDU explicitly named IMAGE, SCI, or SCIENCE was found.",
                )
            image_index = explicit_images[0]
            image_data = np.asarray(hdul[image_index].data)
            science_image_sha256 = hashlib.sha256(
                np.ascontiguousarray(image_data).tobytes(order="C")
            ).hexdigest()

            mask_indices = [
                index for index in structural_images if "MASK" in names[index]
            ]
            variance_indices = [
                index for index in structural_images if "VAR" in names[index]
            ]
            mask = _component_from_indices("mask", mask_indices)
            variance = _component_from_indices("variance", variance_indices)

            try:
                celestial = WCS(hdul[image_index].header).celestial
                if celestial.pixel_n_dim != 2 or celestial.world_n_dim != 2:
                    raise ValueError("not a two-axis celestial WCS")
                x_pixel, y_pixel = celestial.world_to_pixel_values(
                    request.ra_deg,
                    request.dec_deg,
                )
                x_pixel = float(x_pixel)
                y_pixel = float(y_pixel)
                height, width = image_data.shape
                inside = bool(
                    np.isfinite(x_pixel)
                    and np.isfinite(y_pixel)
                    and 0 <= x_pixel < width
                    and 0 <= y_pixel < height
                )
                if not inside:
                    raise Dp2FitsValidationError(
                        stage="fits_celestial_wcs",
                        code="target_outside_cutout",
                        message="The FITS WCS did not place the requested coordinate inside the image.",
                    )
                scales = (
                    np.asarray(proj_plane_pixel_scales(celestial), dtype=float) * 3600.0
                )
                if (
                    scales.size != 2
                    or not np.all(np.isfinite(scales))
                    or np.any(scales <= 0)
                ):
                    raise ValueError("invalid projected pixel scale")
                wcs = WcsEvidence(
                    state="present",
                    basis="Celestial WCS parsed from the selected science-image HDU.",
                    hdu_index=image_index,
                    target_pixel_x=x_pixel,
                    target_pixel_y=y_pixel,
                    target_inside_image=True,
                    pixel_scale_arcsec=float(scales.mean()),
                )
            except Dp2FitsValidationError:
                raise
            except Exception as exc:
                raise Dp2FitsValidationError(
                    stage="fits_celestial_wcs",
                    code="invalid_or_missing_wcs",
                    message=f"Could not validate a celestial WCS ({type(exc).__name__}).",
                ) from None

            psf_indices = [index for index, name in names.items() if "PSF" in name]
            input_indices = [
                index
                for index, name in names.items()
                if "PROVENANCE" in name or "INPUT" in name
            ]
            return FitsEvidence(
                opened=True,
                hdu_count=len(hdul),
                hdus=tuple(hdu_summaries),
                science_image_sha256=science_image_sha256,
                image=ComponentEvidence(
                    state="present",
                    basis="Selected finite numeric two-dimensional HDU explicitly named IMAGE.",
                    hdu_index=image_index,
                ),
                mask=mask,
                variance=variance,
                wcs=wcs,
                identity=identity,
                psf=_optional_named_component("PSF", psf_indices),
                retrieval_provenance=ComponentEvidence(
                    state="present",
                    basis="Primary FITS BTLR and cutout-stencil headers identify retrieval provenance.",
                    hdu_index=0,
                ),
                input_provenance=_optional_named_component(
                    "input provenance", input_indices
                ),
            )
    except Dp2FitsValidationError:
        raise
    except Exception as exc:
        raise Dp2FitsValidationError(
            stage="fits_inspection",
            code="fits_inspection_failed",
            message=f"FITS inspection failed ({type(exc).__name__}).",
        ) from None


def _validate_declared_fits_size(hdul: fits.HDUList) -> None:
    if len(hdul) < 1 or len(hdul) > _MAX_HDU_COUNT:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_hdu_count",
            message="The FITS HDU count was outside the Stage-1 safety bound.",
        )
    total_bytes = 0
    for hdu in hdul:
        _, decoded_bytes = _declared_hdu_sizes(hdu.header)
        total_bytes += decoded_bytes
        if total_bytes > _MAX_DECODED_BYTES:
            raise Dp2FitsValidationError(
                stage="fits_resource_limits",
                code="decoded_fits_too_large",
                message="The FITS artifact exceeded the Stage-1 decoded-size bound.",
            )


def _declared_hdu_sizes(header: fits.Header) -> tuple[int, int]:
    """Return unpadded on-disk bytes and a conservative decoded-memory estimate."""
    try:
        stored_axes = int(header.get("NAXIS", 0) or 0)
        compressed_axes = int(header.get("ZNAXIS", 0) or 0)
        pcount = int(header.get("PCOUNT", 0) or 0)
        gcount = int(header.get("GCOUNT", 1) or 1)
        stored_lengths = [
            int(header.get(f"NAXIS{axis}", 0) or 0)
            for axis in range(1, stored_axes + 1)
        ]
        decoded_lengths = [
            int(header.get(f"ZNAXIS{axis}", 0) or 0)
            for axis in range(1, compressed_axes + 1)
        ]
        bitpix = int(header.get("BITPIX", 8) or 8)
        zbitpix = int(header.get("ZBITPIX", 0) or 0) if compressed_axes else None
    except (TypeError, ValueError, OverflowError):
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_size_header",
            message="A FITS HDU contained invalid resource-size headers.",
        ) from None
    if bool(header.get("GROUPS", False)):
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="random_groups_forbidden",
            message="FITS random-groups data are outside the Stage-1 product contract.",
        )
    if pcount < 0 or gcount < 1 or gcount > 1024:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_group_or_heap_size",
            message="A FITS HDU declared an invalid group count or heap size.",
        )
    if stored_axes < 0 or stored_axes > 8 or compressed_axes < 0 or compressed_axes > 8:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_axis_count",
            message="A FITS HDU declared an unsupported axis count.",
        )
    if any(
        length < 0 or length > _MAX_AXIS_LENGTH
        for length in stored_lengths + decoded_lengths
    ):
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="axis_too_large",
            message="A FITS HDU exceeded the Stage-1 decoded-axis bound.",
        )
    if bitpix not in {8, 16, 32, 64, -32, -64}:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_bitpix",
            message="A FITS HDU declared an unsupported BITPIX value.",
        )

    stored_elements = math.prod(stored_lengths) if stored_lengths else 0
    physical_bytes = gcount * (stored_elements * (abs(bitpix) // 8) + pcount)
    decoded_bytes = physical_bytes
    if compressed_axes:
        if zbitpix not in {8, 16, 32, 64, -32, -64}:
            raise Dp2FitsValidationError(
                stage="fits_resource_limits",
                code="invalid_zbitpix",
                message="A compressed FITS HDU declared an unsupported ZBITPIX value.",
            )
        decoded_bytes += math.prod(decoded_lengths) * (abs(zbitpix) // 8)
    return physical_bytes, decoded_bytes


def _preflight_fits(path: Path) -> None:
    """Scan bounded FITS headers before Astropy is allowed to materialize HDUs."""
    file_size = path.stat().st_size
    offset = 0
    hdu_count = 0
    total_decoded_bytes = 0
    try:
        with path.open("rb") as handle:
            while offset < file_size:
                hdu_count += 1
                if hdu_count > _MAX_HDU_COUNT:
                    raise Dp2FitsValidationError(
                        stage="fits_resource_limits",
                        code="invalid_hdu_count",
                        message="The FITS HDU count was outside the Stage-1 safety bound.",
                    )
                header_blocks: list[bytes] = []
                found_end = False
                for _ in range(_MAX_HEADER_BLOCKS_PER_HDU):
                    block = handle.read(_FITS_BLOCK_BYTES)
                    if len(block) != _FITS_BLOCK_BYTES:
                        raise Dp2FitsValidationError(
                            stage="fits_resource_limits",
                            code="truncated_fits_header",
                            message="A FITS header was truncated before its END card.",
                        )
                    header_blocks.append(block)
                    for start in range(0, _FITS_BLOCK_BYTES, 80):
                        if block[start : start + 8] == b"END     ":
                            found_end = True
                            break
                    if found_end:
                        break
                if not found_end:
                    raise Dp2FitsValidationError(
                        stage="fits_resource_limits",
                        code="fits_header_too_large",
                        message="A FITS header exceeded the Stage-1 header-size bound.",
                    )
                try:
                    raw_header = b"".join(header_blocks).decode("ascii")
                    header = fits.Header.fromstring(raw_header, sep="")
                except Exception:
                    raise Dp2FitsValidationError(
                        stage="fits_resource_limits",
                        code="invalid_fits_header",
                        message="A FITS header could not be parsed during bounded preflight.",
                    ) from None
                physical_bytes, decoded_bytes = _declared_hdu_sizes(header)
                total_decoded_bytes += decoded_bytes
                if total_decoded_bytes > _MAX_DECODED_BYTES:
                    raise Dp2FitsValidationError(
                        stage="fits_resource_limits",
                        code="decoded_fits_too_large",
                        message="The FITS artifact exceeded the Stage-1 decoded-size bound.",
                    )
                padded_data_bytes = (
                    (physical_bytes + _FITS_BLOCK_BYTES - 1) // _FITS_BLOCK_BYTES
                ) * _FITS_BLOCK_BYTES
                offset = handle.tell() + padded_data_bytes
                if offset > file_size:
                    raise Dp2FitsValidationError(
                        stage="fits_resource_limits",
                        code="truncated_fits_data",
                        message="A FITS HDU declared more data than the artifact contains.",
                    )
                handle.seek(padded_data_bytes, os.SEEK_CUR)
    except Dp2FitsValidationError:
        raise
    except OSError as exc:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="fits_preflight_io_failed",
            message=f"FITS preflight could not read the local artifact ({type(exc).__name__}).",
        ) from None
    if hdu_count < 1 or offset != file_size:
        raise Dp2FitsValidationError(
            stage="fits_resource_limits",
            code="invalid_fits_extent",
            message="The FITS artifact extent did not match its declared HDUs.",
        )


def _fits_identity(
    header: fits.Header,
    request: Dp2CutoutRequest,
    dataset: DatasetIdentity,
) -> FitsIdentityEvidence:
    pairs: dict[str, Any] = {}
    for index in range(1000):
        key_name = f"BTLRK{index:03d}"
        value_name = f"BTLRV{index:03d}"
        if key_name not in header:
            break
        pairs[str(header[key_name]).strip()] = header.get(value_name)

    butler_uuid = str(header.get("BTLRUUID", "")).replace("-", "").lower()
    publisher_query = parse_qs(urlparse(dataset.obs_publisher_did).query)
    publisher_ids = publisher_query.get("id", [])
    publisher_uuid = (
        publisher_ids[0].replace("-", "").lower() if len(publisher_ids) == 1 else ""
    )
    dataset_type = str(header.get("BTLRNAME", "")).strip()
    band = str(pairs.get("band", "")).strip()
    skymap = str(pairs.get("skymap", "")).strip()
    try:
        tract = int(pairs.get("tract"))
        patch = int(pairs.get("patch"))
        stencil_ra = float(header["ST_RA"])
        stencil_dec = float(header["ST_DEC"])
        stencil_radius = float(header["ST_RAD"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise Dp2FitsValidationError(
            stage="fits_identity",
            code="missing_identity_header",
            message="Required FITS dataset or cutout-stencil identity headers were missing.",
        ) from None
    stencil_type = str(header.get("ST_TYPE", "")).strip().upper()

    uuid_matches = bool(
        re.fullmatch(r"[0-9a-f]{32}", butler_uuid) and butler_uuid == publisher_uuid
    )
    sia_matches = bool(
        uuid_matches
        and dataset_type == "deep_coadd"
        and band == dataset.band_name
        and skymap == request.expected_skymap
        and tract == dataset.tract
        and patch == dataset.patch
    )
    request_matches = bool(
        stencil_type == "CIRCLE"
        and np.isclose(stencil_ra, request.ra_deg, rtol=0, atol=1e-10)
        and np.isclose(stencil_dec, request.dec_deg, rtol=0, atol=1e-10)
        and np.isclose(stencil_radius, request.cutout_radius_deg, rtol=0, atol=1e-10)
    )
    if not sia_matches or not request_matches:
        raise Dp2FitsValidationError(
            stage="fits_identity",
            code="fits_identity_mismatch",
            message="FITS identity or cutout-stencil headers did not match SIA and the request.",
        )

    cutout_version = header.get("CUTVERS")
    created_at = header.get("DATE-CUT")
    return FitsIdentityEvidence(
        butler_uuid=butler_uuid,
        dataset_type="deep_coadd",
        band_name=band,
        skymap="lsst_cells_v2",
        tract=5063,
        patch=34,
        stencil_type="CIRCLE",
        stencil_ra_deg=stencil_ra,
        stencil_dec_deg=stencil_dec,
        stencil_radius_deg=stencil_radius,
        cutout_service_version=str(cutout_version).strip()
        if cutout_version is not None
        else None,
        cutout_created_at=str(created_at).strip() if created_at is not None else None,
        matches_sia_dataset=True,
        matches_request=True,
    )


def _component_from_indices(label: str, indices: list[int]) -> ComponentEvidence:
    if indices:
        return ComponentEvidence(
            state="present",
            basis=f"An explicitly named {label} HDU was found.",
            hdu_index=indices[0],
        )
    return ComponentEvidence(
        state="absent",
        basis=f"No explicitly named {label} HDU was present in the opened FITS.",
    )


def _optional_named_component(label: str, indices: list[int]) -> ComponentEvidence:
    if indices:
        return ComponentEvidence(
            state="present",
            basis=f"An explicitly named {label} HDU was found.",
            hdu_index=indices[0],
        )
    return ComponentEvidence(
        state="unknown",
        basis=f"No explicitly named {label} HDU was found; no availability is inferred.",
    )


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _implementation_source_hashes() -> dict[str, str]:
    package_directory = Path(__file__).resolve().parent
    sources = {
        "ripple/__init__.py": package_directory.parent / "__init__.py",
        "ripple/dp2/__init__.py": package_directory / "__init__.py",
        "ripple/dp2/errors.py": package_directory / "errors.py",
        "ripple/dp2/models.py": package_directory / "models.py",
        "ripple/dp2/client.py": package_directory / "client.py",
        "ripple/dp2/service.py": package_directory / "service.py",
        "ripple/dp2/cli.py": package_directory / "cli.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources.items()
    }


def _assert_sanitized_payload(payload: Any) -> None:
    secret = os.environ.get("RSP_TOKEN", "").strip()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
            return
        if not isinstance(value, str):
            return
        lowered = value.lower()
        if any(ord(character) < 32 and character not in "\t" for character in value):
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="unsafe_control_character",
                message="Evidence serialization rejected unsafe control characters.",
            )
        if secret and secret in value:
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="credential_value_detected",
                message="Evidence serialization rejected a credential value.",
            )
        if any(fragment in lowered for fragment in _FORBIDDEN_SERIALIZED_FRAGMENTS):
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="credential_fragment_detected",
                message="Evidence serialization rejected credential-shaped text.",
            )
        if _JWT_PATTERN.search(value):
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="jwt_shaped_value_detected",
                message="Evidence serialization rejected token-shaped text.",
            )
        if _URL_PATTERN.search(value) and value != _SAFE_SERVICE_ORIGIN:
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="remote_url_detected",
                message="Evidence serialization rejected a remote access URL.",
            )

    try:
        visit(payload)
    finally:
        secret = ""


def write_model_json_atomic(
    path: Path,
    model: Dp2SmokeEvidence | FailureEvidence,
) -> None:
    """Write one approved Pydantic evidence model privately without replacement."""
    if not isinstance(model, (Dp2SmokeEvidence, FailureEvidence)):
        raise TypeError("only Stage-1 success or failure evidence can be serialized")
    path = Path(path)
    _require_private_run_directory(path.parent)
    if path.name not in {"evidence.json", "failure.json"}:
        raise Dp2ConfigurationError(
            stage="evidence_serialization",
            code="invalid_evidence_filename",
            message="The evidence filename was outside the Stage-1 allowlist.",
        )
    if os.path.lexists(path):
        raise Dp2ConfigurationError(
            stage="evidence_serialization",
            code="evidence_exists",
            message="Refused to overwrite an existing evidence artifact.",
        )

    payload = model.model_dump(mode="json")
    _assert_sanitized_payload(payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            raise Dp2ConfigurationError(
                stage="evidence_serialization",
                code="evidence_race_detected",
                message="The evidence destination appeared during serialization.",
            ) from None
        published = True
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)
    if not published:
        raise Dp2ConfigurationError(
            stage="evidence_serialization",
            code="evidence_publish_failed",
            message="The evidence artifact could not be published.",
        )
