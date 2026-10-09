"""Human-only visual QA for the provisional Mriganka M3 adapter."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .artifact_io import file_digest
from .contracts import PreviewArtifactRef
from .errors import PreprocessingArtifactError
from .mriganka import MrigankaTransformResult


def render_preprocessing_preview(
    *,
    source_image: np.ndarray,
    result: MrigankaTransformResult,
    model_input_digest: str,
    destination: Path,
) -> PreviewArtifactRef:
    """Render a six-panel QA figure that is never consumed by the classifier."""

    destination = Path(destination)
    if destination.name != "preprocessing_preview.png":
        raise PreprocessingArtifactError(
            stage="preview_write",
            code="preview_name_not_allowed",
            message="The M3 preview filename was outside the artifact allowlist.",
        )
    if os.path.lexists(destination):
        raise PreprocessingArtifactError(
            stage="preview_write",
            code="preview_exists",
            message="M3 refused to replace an existing preview.",
        )

    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.figure import Figure
        from matplotlib.patches import Patch, Rectangle

        with matplotlib.rc_context():
            figure = Figure(figsize=(16, 10), constrained_layout=True)
            FigureCanvasAgg(figure)
            axes = figure.subplots(2, 3)
            figure.suptitle(
                "RIPPLe M3 — Mriganka 64×64 preprocessing QA (PROVISIONAL / UNQUALIFIED)",
                fontsize=15,
                fontweight="bold",
            )

            x0, y0, x1, y1 = result.crop.crop_bounds_xyxy
            target_source_x, target_source_y = result.crop.target_source_xy
            target_crop_x, target_crop_y = result.crop.target_crop_xy

            full_display = _display_asinh(source_image)
            axes[0, 0].imshow(
                full_display, origin="lower", cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[0, 0].add_patch(
                Rectangle(
                    (x0 - 0.5, y0 - 0.5),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="#ff4d4d",
                    linewidth=1.8,
                )
            )
            axes[0, 0].plot(
                target_source_x,
                target_source_y,
                marker="+",
                color="#00e5ff",
                markersize=10,
                markeredgewidth=1.5,
            )
            axes[0, 0].set_title("1. Verified DP2 image — display stretch only")
            axes[0, 0].set_xlabel("x pixel (native)")
            axes[0, 0].set_ylabel("y pixel (native)")

            crop_display = _display_asinh(result.native_crop)
            axes[0, 1].imshow(
                crop_display, origin="lower", cmap="gray", vmin=0.0, vmax=1.0
            )
            axes[0, 1].plot(
                target_crop_x,
                target_crop_y,
                marker="+",
                color="#00e5ff",
                markersize=10,
                markeredgewidth=1.5,
            )
            axes[0, 1].set_title("2. Native 64×64 crop (nJy; display stretch only)")
            axes[0, 1].set_xlabel("x pixel in crop")
            axes[0, 1].set_ylabel("y pixel in crop")

            category = np.zeros(result.mask_crop.shape, dtype=np.uint8)
            category[result.detected_mask] = 1
            category[result.caution_mask] = 2
            category[result.fatal_mask] = 3
            colors = ["#222222", "#38a169", "#ed8936", "#e53e3e"]
            labels = ["clear", "DETECTED", "caution", "fatal"]
            mask_map = ListedColormap(colors)
            mask_norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), mask_map.N)
            axes[0, 2].imshow(category, origin="lower", cmap=mask_map, norm=mask_norm)
            axes[0, 2].set_title("3. Named DP2 mask policy (not mask != 0)")
            axes[0, 2].set_xlabel("x pixel in crop")
            axes[0, 2].set_ylabel("y pixel in crop")
            axes[0, 2].legend(
                handles=[
                    Patch(facecolor=color, label=label)
                    for color, label in zip(colors, labels)
                ],
                loc="lower right",
                fontsize=8,
                framealpha=0.85,
            )

            exact_tensor = result.model_input[0, 0]
            tensor_image = axes[1, 0].imshow(
                exact_tensor,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[1, 0].plot(
                target_crop_x,
                target_crop_y,
                marker="+",
                color="#00e5ff",
                markersize=10,
                markeredgewidth=1.5,
            )
            axes[1, 0].set_title("4. Exact saved provisional tensor [0,0], linear 0–1")
            axes[1, 0].set_xlabel("x pixel in tensor")
            axes[1, 0].set_ylabel("y pixel in tensor")
            figure.colorbar(tensor_image, ax=axes[1, 0], fraction=0.046, pad=0.04)

            sigma_image = axes[1, 1].imshow(
                np.sqrt(result.variance_crop),
                origin="lower",
                cmap="magma",
            )
            axes[1, 1].set_title("5. Per-pixel noise σ = √variance (nJy)")
            axes[1, 1].set_xlabel("x pixel in crop")
            axes[1, 1].set_ylabel("y pixel in crop")
            figure.colorbar(sigma_image, ax=axes[1, 1], fraction=0.046, pad=0.04)

            numeric = result.quality.numeric
            fov_x, fov_y = result.crop.field_of_view_arcsec_xy
            offset_x, offset_y = result.crop.target_offset_from_center_xy
            status_lines = (
                "STATUS: PROVISIONAL / UNQUALIFIED\n"
                "Classifier execution: BLOCKED\n\n"
                "Transform actually performed\n"
                "• r-band only\n"
                f"• native crop x={x0}:{x1}, y={y0}:{y1}\n"
                f"• field of view {fov_x:.3f} × {fov_y:.3f} arcsec\n"
                f"• target offset ({offset_x:+.3f}, {offset_y:+.3f}) px\n"
                "• no resize, rotation, flip, PSF operation, or augmentation\n"
                "• per-crop min–max normalization; BCHW float32\n\n"
                "Checks\n"
                f"• native range {numeric.image_min_njy:.3f} to {numeric.image_max_njy:.3f} nJy\n"
                f"• fatal pixels {result.quality.fatal_pixel_count}/4096\n"
                f"• caution pixels {result.quality.caution_pixel_count}/4096\n"
                f"• DETECTED pixels {result.quality.detected_pixel_count}/4096\n"
                f"• PSF {result.quality.psf_state}\n"
                f"• tensor SHA-256 {model_input_digest[:16]}…\n\n"
                "Why blocked\n"
                "Weights + training band/FOV/PSF/preparation contract are required."
            )
            axes[1, 2].axis("off")
            axes[1, 2].text(
                0.0,
                1.0,
                status_lines,
                transform=axes[1, 2].transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=9.2,
                linespacing=1.35,
            )
            axes[1, 2].set_title(
                "6. Machine-recorded recipe and qualification boundary"
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
                raise PreprocessingArtifactError(
                    stage="preview_write",
                    code="preview_exists",
                    message="M3 refused to replace an existing preview.",
                ) from None
            finally:
                if temporary.exists():
                    temporary.unlink()
    except PreprocessingArtifactError:
        raise
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="preview_write",
            code="preview_write_failed",
            message=f"The M3 QA preview could not be rendered ({type(exc).__name__}).",
        ) from None

    from PIL import Image

    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            pixel_size = tuple(int(value) for value in image.size)
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="preview_write",
            code="preview_verification_failed",
            message=f"The rendered M3 QA preview could not be verified ({type(exc).__name__}).",
        ) from None
    byte_count, sha256 = file_digest(destination)
    return PreviewArtifactRef(
        pixel_size_xy=pixel_size,
        byte_count=byte_count,
        file_sha256=sha256,
    )


def _display_asinh(array: np.ndarray) -> np.ndarray:
    """Return a robust display-only stretch without changing any saved science array."""

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
