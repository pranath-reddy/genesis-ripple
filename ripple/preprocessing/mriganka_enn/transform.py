"""Pure deterministic transformation for three-band Rubin DP2 ENN inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ripple.modeling.observation import ObservationBundle, ObservationPlane
from ripple.preprocessing.contracts import (
    CropGeometry,
    MaskBitSummary,
    NumericSummary,
    QualitySummary,
)
from ripple.preprocessing.errors import PreprocessingInputError

from .contracts import (
    BandName,
    CrossBandWcsEvidence,
    MrigankaEnnThreeBandRecipe,
    PairwiseWcsAlignment,
)


@dataclass(frozen=True)
class BandTransformResult:
    band: BandName
    normalized_channel: np.ndarray
    native_crop: np.ndarray
    mask_crop: np.ndarray
    variance_crop: np.ndarray
    fatal_mask: np.ndarray
    caution_mask: np.ndarray
    retained_mask: np.ndarray
    detected_mask: np.ndarray
    crop: CropGeometry
    quality: QualitySummary
    normalization_nonfinite_replacement_count: int

    def __post_init__(self) -> None:
        for array in (
            self.normalized_channel,
            self.native_crop,
            self.mask_crop,
            self.variance_crop,
            self.fatal_mask,
            self.caution_mask,
            self.retained_mask,
            self.detected_mask,
        ):
            array.setflags(write=False)


@dataclass(frozen=True)
class MrigankaEnnThreeBandTransformResult:
    model_input: np.ndarray
    channels: tuple[BandTransformResult, BandTransformResult, BandTransformResult]
    cross_band_wcs: CrossBandWcsEvidence

    def __post_init__(self) -> None:
        self.model_input.setflags(write=False)


@dataclass(frozen=True)
class _PreparedBand:
    band: BandName
    plane: ObservationPlane
    native_crop: np.ndarray
    mask_crop: np.ndarray
    variance_crop: np.ndarray
    fatal_mask: np.ndarray
    caution_mask: np.ndarray
    retained_mask: np.ndarray
    detected_mask: np.ndarray
    crop: CropGeometry
    mask_bits: tuple[MaskBitSummary, ...]
    fatal_count: int
    caution_count: int
    detected_count: int


def build_mriganka_enn_three_band_input(
    observation: ObservationBundle,
    recipe: MrigankaEnnThreeBandRecipe,
) -> MrigankaEnnThreeBandTransformResult:
    """Build one native-grid BCHW tensor without reprojection or scientific claims."""

    if len(observation.planes) != 3 or set(observation.bands) != {"g", "r", "i"}:
        raise PreprocessingInputError(
            stage="input_contract",
            code="exact_gri_band_set_required",
            message="The three-band ENN recipe requires exactly one verified g, r, and i package.",
        )

    prepared_by_band: dict[BandName, _PreparedBand] = {}
    for band in recipe.channel_bands:
        prepared_by_band[band] = _prepare_band_crop(
            observation.plane_for_band(band),
            recipe,
        )

    cross_band_wcs = _validate_cross_band_wcs(prepared_by_band, recipe)
    channels = tuple(
        _normalize_prepared_band(prepared_by_band[band], recipe)
        for band in recipe.channel_bands
    )
    if len(channels) != 3:
        raise AssertionError("recipe validation guarantees three channels")
    typed_channels = (channels[0], channels[1], channels[2])
    model_input = np.ascontiguousarray(
        np.stack(
            tuple(channel.normalized_channel for channel in typed_channels),
            axis=0,
        )[np.newaxis, :, :, :],
        dtype=np.float32,
    )
    if (
        tuple(model_input.shape) != recipe.output_shape_bchw
        or str(model_input.dtype) != recipe.output_dtype
        or not bool(np.isfinite(model_input).all())
        or float(model_input.min()) < 0.0
        or float(model_input.max()) > 1.0
    ):
        raise PreprocessingInputError(
            stage="normalization",
            code="invalid_model_tensor_contract",
            message="The derived tensor did not satisfy the fixed BCHW float32 [0,1] contract.",
        )
    return MrigankaEnnThreeBandTransformResult(
        model_input=model_input,
        channels=typed_channels,
        cross_band_wcs=cross_band_wcs,
    )


def _prepare_band_crop(
    plane: ObservationPlane,
    recipe: MrigankaEnnThreeBandRecipe,
) -> _PreparedBand:
    package = plane.package
    band = package.dataset.band_name
    if band not in {"g", "r", "i"}:
        raise PreprocessingInputError(
            stage="input_contract",
            code="unsupported_band",
            message="The three-band ENN recipe accepts only Rubin g, r, and i packages.",
        )
    typed_band: BandName = band
    if (
        package.image.canonical_unit != "nJy"
        or package.variance.canonical_unit != "nJy2"
    ):
        raise PreprocessingInputError(
            stage="input_contract",
            code="unsupported_physical_units",
            message="The three-band ENN adapter requires image nJy and variance nJy2 units.",
        )
    if plane.image.ndim != 2 or plane.mask.ndim != 2 or plane.variance.ndim != 2:
        raise PreprocessingInputError(
            stage="input_contract",
            code="non_2d_plane",
            message="Every band requires one two-dimensional image, mask, and variance plane.",
        )
    if not (plane.image.shape == plane.mask.shape == plane.variance.shape):
        raise PreprocessingInputError(
            stage="input_contract",
            code="unaligned_plane_shapes",
            message="Image, mask, and variance shapes disagree within one band.",
        )

    target = SkyCoord(
        package.request.ra_deg * u.deg,
        package.request.dec_deg * u.deg,
        frame="icrs",
    )
    runtime_x, runtime_y = plane.celestial_wcs.world_to_pixel(target)
    declared_x, declared_y = package.celestial_wcs.target_pixel_xy
    if not (
        math.isfinite(float(runtime_x))
        and math.isfinite(float(runtime_y))
        and math.isclose(float(runtime_x), declared_x, rel_tol=0.0, abs_tol=1e-8)
        and math.isclose(float(runtime_y), declared_y, rel_tol=0.0, abs_tol=1e-8)
    ):
        raise PreprocessingInputError(
            stage="wcs_centering",
            code="target_pixel_mismatch",
            message="A reloaded band WCS target position did not match its M2 package.",
        )

    crop_y, crop_x = recipe.crop_shape_yx
    x0 = math.floor(declared_x - (crop_x - 1) / 2.0 + 0.5)
    y0 = math.floor(declared_y - (crop_y - 1) / 2.0 + 0.5)
    x1 = x0 + crop_x
    y1 = y0 + crop_y
    source_y, source_x = plane.image.shape
    if x0 < 0 or y0 < 0 or x1 > source_x or y1 > source_y:
        raise PreprocessingInputError(
            stage="wcs_centering",
            code="crop_crosses_source_edge",
            message="A required native 64-pixel target crop crosses its source boundary.",
        )

    native_crop = np.ascontiguousarray(plane.image[y0:y1, x0:x1], dtype=np.float32)
    mask_crop = np.ascontiguousarray(plane.mask[y0:y1, x0:x1], dtype=np.int32)
    variance_crop = np.ascontiguousarray(
        plane.variance[y0:y1, x0:x1],
        dtype=np.float32,
    )
    if native_crop.shape != recipe.crop_shape_yx:
        raise PreprocessingInputError(
            stage="crop",
            code="unexpected_crop_shape",
            message="A native target crop did not have the required 64 by 64 shape.",
        )
    if not bool(np.isfinite(native_crop).all()):
        raise PreprocessingInputError(
            stage="quality",
            code="nonfinite_image_pixel",
            message="A target crop contains a nonfinite image value.",
        )
    if not bool(np.isfinite(variance_crop).all()) or not bool(
        (variance_crop > 0).all()
    ):
        raise PreprocessingInputError(
            stage="quality",
            code="invalid_variance_pixel",
            message="A target crop contains a nonpositive or nonfinite variance value.",
        )

    declared_bits = {item.name: item.value for item in package.mask.bits}
    required_bits = (
        set(recipe.mask_policy.fatal_bits)
        | set(recipe.mask_policy.caution_bits)
        | set(recipe.mask_policy.retained_bits)
    )
    if set(declared_bits) != required_bits:
        raise PreprocessingInputError(
            stage="mask_policy",
            code="unexpected_mask_schema",
            message="A DP2 mask schema did not match the frozen three-band recipe.",
        )

    unsigned_mask = mask_crop.astype(np.uint32, copy=False)

    def union(names: tuple[str, ...]) -> np.ndarray:
        combined = 0
        for name in names:
            combined |= declared_bits[name]
        return np.ascontiguousarray((unsigned_mask & np.uint32(combined)) != 0)

    fatal_mask = union(recipe.mask_policy.fatal_bits)
    caution_mask = union(recipe.mask_policy.caution_bits)
    retained_mask = union(recipe.mask_policy.retained_bits)
    detected_mask = np.ascontiguousarray(
        (unsigned_mask & np.uint32(declared_bits["DETECTED"])) != 0
    )
    pixel_count = int(native_crop.size)
    fatal_count = int(np.count_nonzero(fatal_mask))
    caution_count = int(np.count_nonzero(caution_mask))
    detected_count = int(np.count_nonzero(detected_mask))
    if fatal_count / pixel_count > recipe.mask_policy.maximum_fatal_fraction:
        raise PreprocessingInputError(
            stage="mask_policy",
            code="fatal_mask_pixel_present",
            message="A target crop contains a NO_DATA or SATURATED pixel forbidden by the recipe.",
        )

    bit_summaries: list[MaskBitSummary] = []
    for item in package.mask.bits:
        active = (unsigned_mask & np.uint32(item.value)) != 0
        count = int(np.count_nonzero(active))
        if item.name in recipe.mask_policy.fatal_bits:
            category = "fatal"
        elif item.name in recipe.mask_policy.caution_bits:
            category = "caution"
        else:
            category = "retained"
        bit_summaries.append(
            MaskBitSummary(
                name=item.name,
                value=item.value,
                category=category,
                set_pixel_count=count,
                set_pixel_fraction=float(count / pixel_count),
            )
        )

    pixel_scale_x, pixel_scale_y = package.celestial_wcs.pixel_scale_arcsec_xy
    target_crop_x = declared_x - x0
    target_crop_y = declared_y - y0
    center_x = (crop_x - 1) / 2.0
    center_y = (crop_y - 1) / 2.0
    geometry = CropGeometry(
        source_shape_yx=(int(source_y), int(source_x)),
        crop_bounds_xyxy=(int(x0), int(y0), int(x1), int(y1)),
        crop_shape_yx=recipe.crop_shape_yx,
        target_source_xy=(float(declared_x), float(declared_y)),
        target_crop_xy=(float(target_crop_x), float(target_crop_y)),
        crop_geometric_center_xy=(float(center_x), float(center_y)),
        target_offset_from_center_xy=(
            float(target_crop_x - center_x),
            float(target_crop_y - center_y),
        ),
        pixel_scale_arcsec_xy=(float(pixel_scale_x), float(pixel_scale_y)),
        field_of_view_arcsec_xy=(
            float(crop_x * pixel_scale_x),
            float(crop_y * pixel_scale_y),
        ),
    )
    return _PreparedBand(
        band=typed_band,
        plane=plane,
        native_crop=native_crop,
        mask_crop=mask_crop,
        variance_crop=variance_crop,
        fatal_mask=fatal_mask,
        caution_mask=caution_mask,
        retained_mask=retained_mask,
        detected_mask=detected_mask,
        crop=geometry,
        mask_bits=tuple(bit_summaries),
        fatal_count=fatal_count,
        caution_count=caution_count,
        detected_count=detected_count,
    )


def _validate_cross_band_wcs(
    prepared_by_band: dict[BandName, _PreparedBand],
    recipe: MrigankaEnnThreeBandRecipe,
) -> CrossBandWcsEvidence:
    grid = tuple(
        (float(x), float(y)) for y, x in product(recipe.wcs_sample_grid_xy, repeat=2)
    )
    footprint = ((-0.5, -0.5), (63.5, -0.5), (63.5, 63.5), (-0.5, 63.5))
    sample_points = footprint + grid
    crop_x = np.asarray([point[0] for point in sample_points], dtype=np.float64)
    crop_y = np.asarray([point[1] for point in sample_points], dtype=np.float64)
    sky_by_band: dict[BandName, SkyCoord] = {}
    for band in ("g", "r", "i"):
        prepared = prepared_by_band[band]
        x0, y0, _, _ = prepared.crop.crop_bounds_xyxy
        sky = prepared.plane.celestial_wcs.pixel_to_world(
            crop_x + float(x0),
            crop_y + float(y0),
        ).icrs
        if not bool(np.isfinite(sky.ra.deg).all() and np.isfinite(sky.dec.deg).all()):
            raise PreprocessingInputError(
                stage="cross_band_wcs",
                code="nonfinite_cross_band_wcs_sample",
                message="A cross-band native-grid WCS sample was nonfinite.",
            )
        sky_by_band[band] = sky

    pairwise: list[PairwiseWcsAlignment] = []
    for left, right in combinations(("g", "r", "i"), 2):
        separations = np.asarray(
            sky_by_band[left].separation(sky_by_band[right]).arcsec,
            dtype=np.float64,
        )
        if not bool(np.isfinite(separations).all()):
            raise PreprocessingInputError(
                stage="cross_band_wcs",
                code="nonfinite_cross_band_wcs_residual",
                message="A cross-band native-grid WCS residual was nonfinite.",
            )
        maximum = float(np.max(separations))
        if maximum > recipe.wcs_alignment_tolerance_arcsec:
            raise PreprocessingInputError(
                stage="cross_band_wcs",
                code="cross_band_native_grid_mismatch",
                message="The g/r/i crops do not share the required native celestial pixel grid.",
            )
        pairwise.append(
            PairwiseWcsAlignment(
                bands=(left, right),
                maximum_separation_arcsec=maximum,
            )
        )
    return CrossBandWcsEvidence(
        pairwise=(pairwise[0], pairwise[1], pairwise[2]),
        maximum_separation_arcsec=max(
            item.maximum_separation_arcsec for item in pairwise
        ),
    )


def _normalize_prepared_band(
    prepared: _PreparedBand,
    recipe: MrigankaEnnThreeBandRecipe,
) -> BandTransformResult:
    native_crop = prepared.native_crop
    image_min = np.float32(np.min(native_crop))
    shifted = np.ascontiguousarray(native_crop - image_min, dtype=np.float32)
    denominator_value = np.float32(np.max(shifted))
    denominator = float(denominator_value)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PreprocessingInputError(
            stage="normalization",
            code="zero_dynamic_range",
            message="Each channel requires a positive finite range for min-max normalization.",
        )
    normalized = np.ascontiguousarray(shifted, dtype=np.float32)
    np.divide(normalized, denominator_value, out=normalized)
    replacement_count = int(np.count_nonzero(~np.isfinite(normalized)))
    np.nan_to_num(normalized, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if replacement_count != 0 or not bool(np.isfinite(normalized).all()):
        raise PreprocessingInputError(
            stage="normalization",
            code="unexpected_nonfinite_normalized_value",
            message="Finite positive-range DP2 input unexpectedly required a nonfinite replacement.",
        )

    image_min_float = float(image_min)
    reconstructed = normalized.astype(np.float64) * denominator + image_min_float
    inverse_error = float(
        np.max(np.abs(reconstructed - native_crop.astype(np.float64)))
    )
    pixel_count = int(native_crop.size)
    quality = QualitySummary(
        fatal_pixel_count=prepared.fatal_count,
        fatal_pixel_fraction=float(prepared.fatal_count / pixel_count),
        caution_pixel_count=prepared.caution_count,
        caution_pixel_fraction=float(prepared.caution_count / pixel_count),
        detected_pixel_count=prepared.detected_count,
        detected_pixel_fraction=float(prepared.detected_count / pixel_count),
        mask_bits=prepared.mask_bits,
        psf_state=prepared.plane.package.psf.state,
        numeric=NumericSummary(
            image_min_njy=image_min_float,
            image_median_njy=float(np.median(native_crop)),
            image_max_njy=float(np.max(native_crop)),
            normalization_denominator_njy=denominator,
            normalized_min=float(np.min(normalized)),
            normalized_median=float(np.median(normalized)),
            normalized_max=float(np.max(normalized)),
            inverse_normalization_max_abs_error_njy=inverse_error,
            variance_min_njy2=float(np.min(prepared.variance_crop)),
            variance_median_njy2=float(np.median(prepared.variance_crop)),
            variance_max_njy2=float(np.max(prepared.variance_crop)),
            noise_sigma_median_njy=float(np.sqrt(np.median(prepared.variance_crop))),
        ),
    )
    return BandTransformResult(
        band=prepared.band,
        normalized_channel=normalized,
        native_crop=prepared.native_crop,
        mask_crop=prepared.mask_crop,
        variance_crop=prepared.variance_crop,
        fatal_mask=prepared.fatal_mask,
        caution_mask=prepared.caution_mask,
        retained_mask=prepared.retained_mask,
        detected_mask=prepared.detected_mask,
        crop=prepared.crop,
        quality=quality,
        normalization_nonfinite_replacement_count=replacement_count,
    )


__all__ = [
    "BandTransformResult",
    "MrigankaEnnThreeBandTransformResult",
    "build_mriganka_enn_three_band_input",
]
