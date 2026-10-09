"""Immutable runtime views of one or more verified M2 observations.

The models in this module do not read FITS files directly.  Construction from
disk always delegates to :func:`ripple.dp2.package_service.load_cutout_package`,
so the existing M2 manifest, file-digest, plane-digest, and WCS checks remain
the trust boundary for pixel data.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.wcs import WCS

from ripple.dp2.package_models import Dp2CutoutPackage
from ripple.dp2.package_service import LoadedDp2Cutout, load_cutout_package


PathInput = str | os.PathLike[str]


class ObservationBundleError(ValueError):
    """A set of individually valid M2 products cannot form one observation."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class ObservationPlane:
    """One verified, single-band M2 cutout and its read-only runtime arrays."""

    source_manifest_path: Path
    loaded: LoadedDp2Cutout

    def __post_init__(self) -> None:
        source_path = Path(self.source_manifest_path)
        if source_path.name != "package.json":
            raise ObservationBundleError(
                code="invalid_source_manifest_name",
                message="An observation plane must originate from an M2 package.json.",
            )
        object.__setattr__(self, "source_manifest_path", source_path)

        arrays = (self.loaded.image, self.loaded.mask, self.loaded.variance)
        if any(not isinstance(array, np.ndarray) for array in arrays):
            raise ObservationBundleError(
                code="invalid_runtime_plane",
                message="An observation plane contains a non-array pixel component.",
            )
        if not (
            self.loaded.image.shape
            == self.loaded.mask.shape
            == self.loaded.variance.shape
        ):
            raise ObservationBundleError(
                code="unaligned_runtime_components",
                message="An observation plane has inconsistent image, mask, and variance shapes.",
            )
        # ``setflags(write=False)`` on an owning ndarray is advisory: a caller
        # can turn the flag back on.  Rebuild every component over immutable
        # ``bytes`` storage so even ``setflags(write=True)`` is rejected.
        protected = LoadedDp2Cutout(
            package=self.loaded.package,
            image=_hard_readonly_copy(self.loaded.image),
            mask=_hard_readonly_copy(self.loaded.mask),
            variance=_hard_readonly_copy(self.loaded.variance),
            celestial_wcs=self.loaded.celestial_wcs,
        )
        object.__setattr__(self, "loaded", protected)

    @property
    def package(self) -> Dp2CutoutPackage:
        return self.loaded.package

    @property
    def band(self) -> str:
        return self.package.dataset.band_name

    @property
    def image(self) -> np.ndarray:
        return self.loaded.image

    @property
    def mask(self) -> np.ndarray:
        return self.loaded.mask

    @property
    def variance(self) -> np.ndarray:
        return self.loaded.variance

    @property
    def celestial_wcs(self) -> WCS:
        return self.loaded.celestial_wcs

    @property
    def shape_yx(self) -> tuple[int, int]:
        height, width = self.image.shape
        return int(height), int(width)


@dataclass(frozen=True)
class ObservationBundle:
    """Ordered, model-independent observation planes for one sky target.

    Plane order is retained because a later preprocessing adapter may use it as
    channel order.  Cross-band shapes are allowed to differ unless
    ``require_aligned_shapes`` is true; an adapter that accepts heterogeneous
    grids must then perform and record its own reviewed alignment operation.
    """

    planes: tuple[ObservationPlane, ...]
    require_aligned_shapes: bool = False

    def __post_init__(self) -> None:
        planes = tuple(self.planes)
        object.__setattr__(self, "planes", planes)
        if not planes:
            raise ObservationBundleError(
                code="empty_observation_bundle",
                message="An observation bundle requires at least one verified M2 plane.",
            )

        bands = tuple(plane.band for plane in planes)
        if len(bands) != len(set(bands)):
            raise ObservationBundleError(
                code="duplicate_observation_band",
                message="An observation bundle cannot contain the same band more than once.",
            )

        reference_target = _target_identity(planes[0])
        if any(
            not _same_target(reference_target, _target_identity(plane))
            for plane in planes[1:]
        ):
            raise ObservationBundleError(
                code="inconsistent_observation_target",
                message="Every observation plane in a bundle must describe the same sky target.",
            )

        reference_product = _product_identity(planes[0])
        if any(_product_identity(plane) != reference_product for plane in planes[1:]):
            raise ObservationBundleError(
                code="inconsistent_observation_product",
                message=(
                    "Every observation plane in a bundle must share the same release, "
                    "collection, product, sky-map cell, calibration level, and physical units."
                ),
            )

        if self.require_aligned_shapes:
            reference_shape = planes[0].shape_yx
            if any(plane.shape_yx != reference_shape for plane in planes[1:]):
                raise ObservationBundleError(
                    code="cross_band_shape_mismatch",
                    message="This observation bundle requires identical cross-band array shapes.",
                )

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(plane.band for plane in self.planes)

    @property
    def target_ra_deg(self) -> float:
        return float(self.planes[0].package.request.ra_deg)

    @property
    def target_dec_deg(self) -> float:
        return float(self.planes[0].package.request.dec_deg)

    @property
    def product_kind(self) -> str:
        return self.planes[0].package.product_kind

    @property
    def common_shape_yx(self) -> tuple[int, int] | None:
        reference = self.planes[0].shape_yx
        if all(plane.shape_yx == reference for plane in self.planes[1:]):
            return reference
        return None

    def plane_for_band(self, band: str) -> ObservationPlane:
        """Return the unique plane for ``band`` or raise a structured error."""

        for plane in self.planes:
            if plane.band == band:
                return plane
        raise ObservationBundleError(
            code="observation_band_unavailable",
            message=f"The observation bundle does not contain requested band {band!r}.",
        )


def load_observation_bundle(
    package_paths: PathInput | Iterable[PathInput],
    *,
    require_aligned_shapes: bool = False,
) -> ObservationBundle:
    """Verify one or more M2 packages and combine them into one runtime bundle.

    A single path is accepted directly.  For multiple bands, iterable order is
    preserved and therefore remains available to a model-specific adapter.
    """

    if isinstance(package_paths, (str, os.PathLike)):
        normalized_paths = (Path(package_paths),)
    else:
        normalized_paths = tuple(Path(path) for path in package_paths)
    if not normalized_paths:
        raise ObservationBundleError(
            code="empty_package_path_set",
            message="At least one M2 package path is required.",
        )

    planes = tuple(
        ObservationPlane(
            source_manifest_path=package_path,
            loaded=load_cutout_package(package_path),
        )
        for package_path in normalized_paths
    )
    return ObservationBundle(
        planes=planes,
        require_aligned_shapes=require_aligned_shapes,
    )


def _target_identity(plane: ObservationPlane) -> tuple[float, float]:
    request = plane.package.request
    return float(request.ra_deg), float(request.dec_deg)


def _same_target(
    reference: tuple[float, float],
    candidate: tuple[float, float],
) -> bool:
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)
        for left, right in zip(reference, candidate)
    )


def _product_identity(plane: ObservationPlane) -> tuple[object, ...]:
    package = plane.package
    return (
        package.schema_version,
        package.product_kind,
        package.retrieval.release,
        package.dataset.observation_collection,
        package.dataset.product_subtype,
        package.dataset.calibration_level,
        package.fits_identity.dataset_type,
        package.fits_identity.skymap,
        package.fits_identity.tract,
        package.fits_identity.patch,
        package.image.canonical_unit,
        package.variance.canonical_unit,
    )


def _hard_readonly_copy(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes storage."""

    contiguous = np.ascontiguousarray(array)
    immutable_storage = contiguous.tobytes(order="C")
    protected = np.frombuffer(immutable_storage, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    protected.setflags(write=False)
    return protected


__all__ = [
    "ObservationBundle",
    "ObservationBundleError",
    "ObservationPlane",
    "load_observation_bundle",
]
