"""Orchestration for deterministic M2-to-M3 model-input construction."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import astropy
import numpy as np
import pydantic

from ripple.dp2.errors import Dp2Error
from ripple.dp2.package_service import load_cutout_package

from .artifact_io import (
    file_digest,
    load_model_input_package,
    verify_model_input_artifacts,
    write_manifest_json_atomic,
    write_npy_new,
)
from .contracts import (
    Mriganka64Recipe,
    MrigankaModelInputPackage,
    NamedVersion,
    SourceDigest,
    SourcePackageRef,
)
from .errors import PreprocessingArtifactError, PreprocessingInputError
from .mriganka import build_mriganka64_input
from .visualization import render_preprocessing_preview


class Mriganka64Preprocessor:
    """Build the provisional 64px classifier input and its auditable QA package."""

    def __init__(self, recipe: Mriganka64Recipe | None = None) -> None:
        self.recipe = recipe or Mriganka64Recipe()

    def run(
        self,
        package_path: Path,
        output_directory: Path,
    ) -> MrigankaModelInputPackage:
        """Reverify M2, transform once, publish artifacts, then reload everything."""

        package_path = Path(package_path)
        output_directory = Path(output_directory)
        try:
            loaded = load_cutout_package(package_path)
        except Dp2Error as exc:
            raise PreprocessingInputError(
                stage="source_reload",
                code="m2_verification_failed",
                message=f"The M2 source package could not be reverified ({exc.code}).",
            ) from None

        output_root = output_directory.resolve(strict=True)
        try:
            source_relative_path = package_path.resolve(strict=True).relative_to(
                output_root
            )
        except (FileNotFoundError, ValueError):
            raise PreprocessingInputError(
                stage="source_provenance",
                code="source_outside_run",
                message="The M2 package must be bundled beneath the M3 run directory for replay.",
            ) from None
        source_relative = source_relative_path.as_posix()

        result = build_mriganka64_input(loaded, self.recipe)
        model_input_ref = write_npy_new(
            output_directory / "model_input.npy",
            result.model_input,
        )
        native_crop_ref = write_npy_new(
            output_directory / "native_crop_njy.npy",
            result.native_crop,
        )
        mask_crop_ref = write_npy_new(
            output_directory / "mask_crop.npy",
            result.mask_crop,
        )
        variance_crop_ref = write_npy_new(
            output_directory / "variance_crop_njy2.npy",
            result.variance_crop,
        )
        preview_ref = render_preprocessing_preview(
            source_image=loaded.image,
            result=result,
            model_input_digest=model_input_ref.decoded_array_sha256,
            destination=output_directory / "preprocessing_preview.png",
        )

        _, source_manifest_sha256 = file_digest(package_path)
        model_package = MrigankaModelInputPackage(
            created_at_utc=datetime.now(timezone.utc),
            source=SourcePackageRef(
                run_relative_manifest_path=source_relative,
                manifest_sha256=source_manifest_sha256,
                fits_sha256=loaded.package.artifact.sha256,
                m2_schema_version=loaded.package.schema_version,
                dataset_id=loaded.package.dataset.dataset_id,
                obs_id=loaded.package.dataset.obs_id,
                band=loaded.package.dataset.band_name,
                ra_deg=loaded.package.request.ra_deg,
                dec_deg=loaded.package.request.dec_deg,
            ),
            recipe=self.recipe,
            crop=result.crop,
            quality=result.quality,
            model_input=model_input_ref,
            native_crop=native_crop_ref,
            mask_crop=mask_crop_ref,
            variance_crop=variance_crop_ref,
            preview=preview_ref,
            implementation_versions=_implementation_versions(),
            implementation_sources=_implementation_source_digests(),
        )
        prepublication = verify_model_input_artifacts(output_directory, model_package)
        if prepublication.package != model_package:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="prepublication_package_mismatch",
                message="Prepublication M3 verification did not preserve the package contract.",
            )
        manifest_path = output_directory / "manifest.json"
        write_manifest_json_atomic(manifest_path, model_package)
        reloaded = load_model_input_package(manifest_path)
        if reloaded.package != model_package:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="manifest_round_trip_mismatch",
                message="The published M3 manifest did not round-trip exactly.",
            )
        expected_arrays = {
            "model_input": result.model_input,
            "native_crop": result.native_crop,
            "mask_crop": result.mask_crop,
            "variance_crop": result.variance_crop,
        }
        for name, expected in expected_arrays.items():
            if not np.array_equal(getattr(reloaded, name), expected):
                raise PreprocessingArtifactError(
                    stage="artifact_reload",
                    code="array_round_trip_mismatch",
                    message="A published M3 array did not round-trip exactly.",
                )
        return model_package


def _implementation_versions() -> tuple[NamedVersion, ...]:
    return (
        NamedVersion(name="python", version=platform.python_version()),
        NamedVersion(name="numpy", version=np.__version__),
        NamedVersion(name="astropy", version=astropy.__version__),
        NamedVersion(name="pydantic", version=pydantic.__version__),
        NamedVersion(name="matplotlib", version=version("matplotlib")),
        NamedVersion(name="pillow", version=version("pillow")),
    )


def _implementation_source_digests() -> tuple[SourceDigest, ...]:
    repository = Path(__file__).resolve().parents[2]
    relative_paths = (
        "ripple/preprocessing/__init__.py",
        "ripple/preprocessing/errors.py",
        "ripple/preprocessing/contracts.py",
        "ripple/preprocessing/mriganka.py",
        "ripple/preprocessing/artifact_io.py",
        "ripple/preprocessing/visualization.py",
        "ripple/preprocessing/service.py",
        "ripple/preprocessing/cli.py",
    )
    records: list[SourceDigest] = []
    for relative_path in relative_paths:
        _, sha256 = file_digest(repository / relative_path)
        records.append(SourceDigest(relative_path=relative_path, sha256=sha256))
    return tuple(records)
