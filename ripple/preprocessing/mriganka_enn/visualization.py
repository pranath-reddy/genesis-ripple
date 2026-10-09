"""Human-only grayscale QA for the three-band Mriganka ENN adapter."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ripple.preprocessing.contracts import PreviewArtifactRef
from ripple.preprocessing.errors import PreprocessingArtifactError

from .artifact_io import file_digest
from .transform import MrigankaEnnThreeBandTransformResult


def render_three_band_preprocessing_preview(
    *,
    result: MrigankaEnnThreeBandTransformResult,
    model_input_digest: str,
    destination: Path,
) -> PreviewArtifactRef:
    """Render per-band evidence without using a potentially misleading RGB mapping."""

    destination = Path(destination)
    if destination.name != "preprocessing_preview.png":
        raise _error(
            "preview_write",
            "preview_name_not_allowed",
            "The three-band preview filename is outside its fixed allowlist.",
        )
    if os.path.lexists(destination):
        raise _error(
            "preview_write",
            "preview_exists",
            "The three-band adapter refused to replace an existing preview.",
        )

    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch

        with matplotlib.rc_context():
            figure = Figure(figsize=(16, 13), constrained_layout=False)
            FigureCanvasAgg(figure)
            axes = figure.subplots(3, 4)
            figure.subplots_adjust(
                left=0.055,
                right=0.97,
                top=0.91,
                bottom=0.14,
                hspace=0.36,
                wspace=0.30,
            )
            channel_order = ", ".join(
                f"{index}→{channel.band}"
                for index, channel in enumerate(result.channels)
            )
            figure.suptitle(
                "RIPPLe M3 — Mriganka ENN g/r/i preprocessing QA "
                "(TECHNICAL / UNQUALIFIED)",
                fontsize=15,
                fontweight="bold",
            )

            colors = ["#222222", "#38a169", "#ed8936", "#e53e3e"]
            labels = ["clear", "DETECTED", "caution", "fatal"]
            mask_map = ListedColormap(colors)
            mask_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), mask_map.N)
            for index, channel in enumerate(result.channels):
                target_x, target_y = channel.crop.target_crop_xy
                axes[index, 0].imshow(
                    _display_asinh(channel.native_crop),
                    origin="lower",
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[index, 0].plot(
                    target_x,
                    target_y,
                    marker="+",
                    color="#00e5ff",
                    markersize=9,
                    markeredgewidth=1.4,
                )
                axes[index, 0].set_title(
                    f"ch {index} / {channel.band}: native crop nJy (display stretch)"
                )

                exact_channel = result.model_input[0, index]
                tensor_image = axes[index, 1].imshow(
                    exact_channel,
                    origin="lower",
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                axes[index, 1].plot(
                    target_x,
                    target_y,
                    marker="+",
                    color="#00e5ff",
                    markersize=9,
                    markeredgewidth=1.4,
                )
                axes[index, 1].set_title(
                    f"ch {index} / {channel.band}: exact tensor [0,{index}]"
                )
                figure.colorbar(
                    tensor_image,
                    ax=axes[index, 1],
                    fraction=0.046,
                    pad=0.04,
                )

                category = np.zeros(channel.mask_crop.shape, dtype=np.uint8)
                category[channel.detected_mask] = 1
                category[channel.caution_mask] = 2
                category[channel.fatal_mask] = 3
                axes[index, 2].imshow(
                    category,
                    origin="lower",
                    cmap=mask_map,
                    norm=mask_norm,
                )
                axes[index, 2].set_title(
                    f"ch {index} / {channel.band}: named DP2 mask policy"
                )
                if index == 0:
                    axes[index, 2].legend(
                        handles=[
                            Patch(facecolor=color, label=label)
                            for color, label in zip(colors, labels, strict=True)
                        ],
                        loc="lower right",
                        fontsize=7,
                        framealpha=0.85,
                    )

                sigma_image = axes[index, 3].imshow(
                    np.sqrt(channel.variance_crop),
                    origin="lower",
                    cmap="magma",
                )
                axes[index, 3].set_title(
                    f"ch {index} / {channel.band}: noise σ = √variance (nJy)"
                )
                figure.colorbar(
                    sigma_image,
                    ax=axes[index, 3],
                    fraction=0.046,
                    pad=0.04,
                )
                for axis in axes[index]:
                    axis.set_xlabel("x pixel in native crop")
                    axis.set_ylabel("y pixel in native crop")

            band_lines = []
            for channel in result.channels:
                fov_x, fov_y = channel.crop.field_of_view_arcsec_xy
                offset_x, offset_y = channel.crop.target_offset_from_center_xy
                band_lines.append(
                    f"{channel.band}: FOV {fov_x:.3f}×{fov_y:.3f} arcsec; "
                    f"target offset ({offset_x:+.3f},{offset_y:+.3f}) px; "
                    f"fatal/caution={channel.quality.fatal_pixel_count}/"
                    f"{channel.quality.caution_pixel_count}; PSF={channel.quality.psf_state}"
                )
            footer = (
                f"Configured channel mapping (UNVERIFIED against serialized HSC arrays): "
                f"{channel_order}\n"
                f"Common native-grid WCS max residual: "
                f"{result.cross_band_wcs.maximum_separation_arcsec:.3e} arcsec "
                f"(limit {result.cross_band_wcs.tolerance_arcsec:.1e}); no reprojection\n"
                + " | ".join(band_lines)
                + "\nPer-channel min–max + nan_to_num(0); BCHW float32; "
                f"tensor SHA-256 {model_input_digest[:20]}…; human QA only. "
                "Scientific use, probability language, thresholds, and candidate decisions BLOCKED."
            )
            figure.text(
                0.055,
                0.025,
                footer,
                ha="left",
                va="bottom",
                family="monospace",
                fontsize=8.2,
                linespacing=1.35,
                wrap=True,
            )

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".preprocessing_preview.png.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    figure.savefig(handle, format="png", dpi=140, facecolor="white")
                    handle.flush()
                    os.fsync(handle.fileno())
                figure.clear()
                temporary.chmod(0o600)
                os.link(temporary, destination, follow_symlinks=False)
                destination.chmod(0o600)
            except FileExistsError:
                raise _error(
                    "preview_write",
                    "preview_exists",
                    "The three-band adapter refused to replace an existing preview.",
                ) from None
            finally:
                temporary.unlink(missing_ok=True)
    except PreprocessingArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001 - isolate plotting backend failures
        raise _error(
            "preview_write",
            "preview_write_failed",
            f"The three-band QA preview could not be rendered ({type(exc).__name__}).",
        ) from None

    from PIL import Image

    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            pixel_size = tuple(int(value) for value in image.size)
    except Exception as exc:  # noqa: BLE001 - isolate third-party image decoder errors
        raise _error(
            "preview_write",
            "preview_verification_failed",
            f"The three-band QA preview could not be verified ({type(exc).__name__}).",
        ) from None
    byte_count, sha256 = file_digest(destination)
    return PreviewArtifactRef(
        pixel_size_xy=pixel_size,
        byte_count=byte_count,
        file_sha256=sha256,
    )


def _display_asinh(array: np.ndarray) -> np.ndarray:
    finite = np.asarray(array, dtype=np.float64)
    finite_values = finite[np.isfinite(finite)]
    if finite_values.size == 0:
        return np.zeros(finite.shape, dtype=np.float64)
    low, high = np.percentile(finite_values, (1.0, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
    if high <= low:
        return np.zeros(finite.shape, dtype=np.float64)
    scaled = np.clip((finite - low) / (high - low), 0.0, 1.0)
    return np.arcsinh(8.0 * scaled) / np.arcsinh(8.0)


def _error(stage: str, code: str, message: str) -> PreprocessingArtifactError:
    return PreprocessingArtifactError(stage=stage, code=code, message=message)


__all__ = ["render_three_band_preprocessing_preview"]
