"""Pure deterministic transformation for the provisional Mriganka 64px adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from ripple.dp2.package_service import LoadedDp2Cutout

from .contracts import (
    CropGeometry,
    MaskBitSummary,
    Mriganka64Recipe,
    NumericSummary,
    QualitySummary,
)
from .errors import PreprocessingInputError


@dataclass(frozen=True)
class MrigankaTransformResult:
    """In-memory result; arrays are made read-only immediately after construction."""

    model_input: np.ndarray
    native_crop: np.ndarray
    mask_crop: np.ndarray
    variance_crop: np.ndarray
    fatal_mask: np.ndarray
    caution_mask: np.ndarray
    retained_mask: np.ndarray
    detected_mask: np.ndarray
    crop: CropGeometry
    quality: QualitySummary

    def __post_init__(self) -> None:
        for array in (
            self.model_input,
            self.native_crop,
            self.mask_crop,
            self.variance_crop,
            self.fatal_mask,
            self.caution_mask,
            self.retained_mask,
            self.detected_mask,
        ):
            array.setflags(write=False)


def build_mriganka64_input(
    loaded: LoadedDp2Cutout,
    recipe: Mriganka64Recipe,
) -> MrigankaTransformResult:
    """Create a native-pixel, WCS-centered 64px preview input without resampling."""

    package = loaded.package
    if package.dataset.band_name != recipe.required_band:
        raise PreprocessingInputError(
            stage="input_contract",
            code="unsupported_band",
            message="The M3 recipe requires the configured single r-band observation.",
        )
    if loaded.image.ndim != 2 or loaded.mask.ndim != 2 or loaded.variance.ndim != 2:
        raise PreprocessingInputError(
            stage="input_contract",
            code="non_2d_plane",
            message="M3 requires exactly one aligned two-dimensional image, mask, and variance plane.",
        )
    if not (loaded.image.shape == loaded.mask.shape == loaded.variance.shape):
        raise PreprocessingInputError(
            stage="input_contract",
            code="unaligned_plane_shapes",
            message="M3 requires image, mask, and variance planes with identical shapes.",
        )

    sky = SkyCoord(
        package.request.ra_deg * u.deg, package.request.dec_deg * u.deg, frame="icrs"
    )
    runtime_x, runtime_y = loaded.celestial_wcs.world_to_pixel(sky)
    declared_x, declared_y = package.celestial_wcs.target_pixel_xy
    if not (
        math.isfinite(float(runtime_x))
        and math.isfinite(float(runtime_y))
        and math.isclose(float(runtime_x), declared_x, abs_tol=1e-8)
        and math.isclose(float(runtime_y), declared_y, abs_tol=1e-8)
    ):
        raise PreprocessingInputError(
            stage="wcs_centering",
            code="target_pixel_mismatch",
            message="The reloaded WCS target position did not match the M2 package.",
        )

    crop_y, crop_x = recipe.crop_shape_yx
    x0 = math.floor(declared_x - (crop_x - 1) / 2.0 + 0.5)
    y0 = math.floor(declared_y - (crop_y - 1) / 2.0 + 0.5)
    x1 = x0 + crop_x
    y1 = y0 + crop_y
    source_y, source_x = loaded.image.shape
    if x0 < 0 or y0 < 0 or x1 > source_x or y1 > source_y:
        raise PreprocessingInputError(
            stage="wcs_centering",
            code="crop_crosses_source_edge",
            message="The required native 64-pixel target crop crosses the source image boundary.",
        )

    native_crop = np.ascontiguousarray(loaded.image[y0:y1, x0:x1], dtype=np.float32)
    mask_crop = np.ascontiguousarray(loaded.mask[y0:y1, x0:x1], dtype=np.int32)
    variance_crop = np.ascontiguousarray(
        loaded.variance[y0:y1, x0:x1], dtype=np.float32
    )
    if native_crop.shape != recipe.crop_shape_yx:
        raise PreprocessingInputError(
            stage="crop",
            code="unexpected_crop_shape",
            message="The native target crop did not have the required 64 by 64 shape.",
        )
    if not bool(np.isfinite(native_crop).all()):
        raise PreprocessingInputError(
            stage="quality",
            code="nonfinite_image_pixel",
            message="The target crop contains a nonfinite image value.",
        )
    if not bool(np.isfinite(variance_crop).all()) or not bool(
        (variance_crop > 0).all()
    ):
        raise PreprocessingInputError(
            stage="quality",
            code="invalid_variance_pixel",
            message="The target crop contains a nonpositive or nonfinite variance value.",
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
            message="The DP2 mask schema did not match the frozen M3 recipe.",
        )

    unsigned_mask = mask_crop.astype(np.uint32, copy=False)

    def union(names: tuple[str, ...]) -> np.ndarray:
        value = 0
        for name in names:
            value |= declared_bits[name]
        return np.ascontiguousarray((unsigned_mask & np.uint32(value)) != 0)

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
    fatal_fraction = fatal_count / pixel_count
    if fatal_fraction > recipe.mask_policy.maximum_fatal_fraction:
        raise PreprocessingInputError(
            stage="mask_policy",
            code="fatal_mask_pixel_present",
            message="The target crop contains a NO_DATA or SATURATED pixel forbidden by this recipe.",
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

    image_min = float(np.min(native_crop))
    image_max = float(np.max(native_crop))
    denominator = image_max - image_min
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PreprocessingInputError(
            stage="normalization",
            code="zero_dynamic_range",
            message="Per-crop min-max normalization requires a positive finite dynamic range.",
        )
    normalized = np.ascontiguousarray(
        (native_crop - image_min) / denominator, dtype=np.float32
    )
    if not bool(np.isfinite(normalized).all()):
        raise PreprocessingInputError(
            stage="normalization",
            code="nonfinite_normalized_value",
            message="Min-max normalization produced a nonfinite value.",
        )
    model_input = np.ascontiguousarray(normalized[np.newaxis, np.newaxis, :, :])
    if (
        model_input.shape != recipe.output_shape_bchw
        or str(model_input.dtype) != recipe.output_dtype
    ):
        raise PreprocessingInputError(
            stage="normalization",
            code="invalid_model_tensor_contract",
            message="The derived tensor did not match the frozen B-C-H-W float32 contract.",
        )

    reconstructed = normalized.astype(np.float64) * denominator + image_min
    inverse_error = float(
        np.max(np.abs(reconstructed - native_crop.astype(np.float64)))
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
    quality = QualitySummary(
        fatal_pixel_count=fatal_count,
        fatal_pixel_fraction=float(fatal_fraction),
        caution_pixel_count=caution_count,
        caution_pixel_fraction=float(caution_count / pixel_count),
        detected_pixel_count=detected_count,
        detected_pixel_fraction=float(detected_count / pixel_count),
        mask_bits=tuple(bit_summaries),
        psf_state=package.psf.state,
        numeric=NumericSummary(
            image_min_njy=image_min,
            image_median_njy=float(np.median(native_crop)),
            image_max_njy=image_max,
            normalization_denominator_njy=float(denominator),
            normalized_min=float(np.min(normalized)),
            normalized_median=float(np.median(normalized)),
            normalized_max=float(np.max(normalized)),
            inverse_normalization_max_abs_error_njy=inverse_error,
            variance_min_njy2=float(np.min(variance_crop)),
            variance_median_njy2=float(np.median(variance_crop)),
            variance_max_njy2=float(np.max(variance_crop)),
            noise_sigma_median_njy=float(np.sqrt(np.median(variance_crop))),
        ),
    )
    return MrigankaTransformResult(
        model_input=model_input,
        native_crop=native_crop,
        mask_crop=mask_crop,
        variance_crop=variance_crop,
        fatal_mask=fatal_mask,
        caution_mask=caution_mask,
        retained_mask=retained_mask,
        detected_mask=detected_mask,
        crop=geometry,
        quality=quality,
    )
