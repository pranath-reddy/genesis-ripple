"""Publication service for the isolated three-band Mriganka ENN M3 adapter."""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import astropy
import numpy as np
import pydantic

from ripple.modeling.observation import ObservationBundle
from ripple.preprocessing.contracts import NamedVersion
from ripple.preprocessing.errors import (
    PreprocessingArtifactError,
    PreprocessingInputError,
)

from .artifact_io import (
    file_digest,
    implementation_source_ref,
    load_three_band_model_input_package,
    source_ref_from_plane,
    verify_three_band_model_input_artifacts,
    write_band_array_new,
    write_manifest_json_atomic,
    write_model_input_new,
)
from .contracts import (
    BandChannelEvidence,
    ImplementationSourceRef,
    MrigankaEnnThreeBandModelInputPackage,
    MrigankaEnnThreeBandRecipe,
    SourcePackageRef,
)
from .transform import build_mriganka_enn_three_band_input
from .visualization import render_three_band_preprocessing_preview


class MrigankaEnnThreeBandPreprocessor:
    """Build and independently replay one private g/r/i BCHW evidence package."""

    def __init__(self, recipe: MrigankaEnnThreeBandRecipe | None = None) -> None:
        self.recipe = recipe or MrigankaEnnThreeBandRecipe()

    def run(
        self,
        observation: ObservationBundle,
        output_directory: Path,
    ) -> MrigankaEnnThreeBandModelInputPackage:
        output_directory = Path(output_directory)
        output_root = output_directory.resolve(strict=True)
        sources: list[SourcePackageRef] = []
        for plane in observation.planes:
            try:
                source_path = plane.source_manifest_path.resolve(strict=True)
                source_relative = source_path.relative_to(output_root).as_posix()
            except (FileNotFoundError, ValueError):
                raise PreprocessingInputError(
                    stage="source_provenance",
                    code="source_outside_run",
                    message="Every M2 package must be bundled beneath the three-band run directory.",
                ) from None
            _, manifest_sha256 = file_digest(source_path)
            sources.append(
                source_ref_from_plane(
                    run_relative_manifest_path=source_relative,
                    manifest_sha256=manifest_sha256,
                    plane=plane,
                )
            )
        if len(sources) != 3:
            raise PreprocessingInputError(
                stage="input_contract",
                code="exact_three_sources_required",
                message="The three-band adapter requires exactly three bundled M2 packages.",
            )
        source_by_band = {source.band: source for source in sources}
        sources = [source_by_band[band] for band in self.recipe.channel_bands]

        result = build_mriganka_enn_three_band_input(observation, self.recipe)
        model_input_ref = write_model_input_new(
            output_directory / "model_input_bchw.npy",
            result.model_input,
        )
        source_by_band = {source.band: source for source in sources}
        channel_evidence: list[BandChannelEvidence] = []
        for index, channel in enumerate(result.channels):
            native_ref = write_band_array_new(
                output_directory / f"native_crop_{channel.band}_njy.npy",
                channel.native_crop,
                band=channel.band,
                role="native_image",
            )
            mask_ref = write_band_array_new(
                output_directory / f"mask_crop_{channel.band}.npy",
                channel.mask_crop,
                band=channel.band,
                role="mask",
            )
            variance_ref = write_band_array_new(
                output_directory / f"variance_crop_{channel.band}_njy2.npy",
                channel.variance_crop,
                band=channel.band,
                role="variance",
            )
            channel_evidence.append(
                BandChannelEvidence(
                    channel_index=index,
                    band=channel.band,
                    source_manifest_path=source_by_band[
                        channel.band
                    ].run_relative_manifest_path,
                    crop=channel.crop,
                    quality=channel.quality,
                    normalization_nonfinite_replacement_count=(
                        channel.normalization_nonfinite_replacement_count
                    ),
                    native_crop=native_ref,
                    mask_crop=mask_ref,
                    variance_crop=variance_ref,
                )
            )
        preview_ref = render_three_band_preprocessing_preview(
            result=result,
            model_input_digest=model_input_ref.decoded_array_sha256,
            destination=output_directory / "preprocessing_preview.png",
        )

        package = MrigankaEnnThreeBandModelInputPackage(
            created_at_utc=datetime.now(timezone.utc),
            sources=(sources[0], sources[1], sources[2]),
            recipe=self.recipe,
            channels=(
                channel_evidence[0],
                channel_evidence[1],
                channel_evidence[2],
            ),
            cross_band_wcs=result.cross_band_wcs,
            model_input=model_input_ref,
            preview=preview_ref,
            implementation_versions=_implementation_versions(),
            implementation_sources=_implementation_source_digests(),
        )
        prepublication = verify_three_band_model_input_artifacts(
            output_directory,
            package,
        )
        if prepublication.package != package:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="prepublication_package_mismatch",
                message="Prepublication verification changed the three-band package contract.",
            )

        manifest_path = output_directory / "manifest.json"
        write_manifest_json_atomic(manifest_path, package)
        reloaded = load_three_band_model_input_package(manifest_path)
        if reloaded.package != package:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="manifest_round_trip_mismatch",
                message="The three-band package manifest did not round-trip exactly.",
            )
        if not np.array_equal(reloaded.model_input, result.model_input):
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="model_input_round_trip_mismatch",
                message="The published three-band tensor did not round-trip exactly.",
            )
        expected_by_band = {channel.band: channel for channel in result.channels}
        for loaded_channel in reloaded.channels:
            expected = expected_by_band[loaded_channel.band]
            if not (
                np.array_equal(loaded_channel.native_crop, expected.native_crop)
                and np.array_equal(loaded_channel.mask_crop, expected.mask_crop)
                and np.array_equal(loaded_channel.variance_crop, expected.variance_crop)
            ):
                raise PreprocessingArtifactError(
                    stage="artifact_reload",
                    code="channel_array_round_trip_mismatch",
                    message="A published three-band channel artifact did not round-trip exactly.",
                )
        return package


def _implementation_versions() -> tuple[NamedVersion, ...]:
    return (
        NamedVersion(name="python", version=platform.python_version()),
        NamedVersion(name="numpy", version=np.__version__),
        NamedVersion(name="astropy", version=astropy.__version__),
        NamedVersion(name="pydantic", version=pydantic.__version__),
        NamedVersion(name="matplotlib", version=version("matplotlib")),
        NamedVersion(name="pillow", version=version("pillow")),
    )


def _implementation_source_digests() -> tuple[ImplementationSourceRef, ...]:
    repository = Path(__file__).resolve().parents[3]
    relative_paths = (
        "ripple/preprocessing/mriganka_enn/__init__.py",
        "ripple/preprocessing/mriganka_enn/contracts.py",
        "ripple/preprocessing/mriganka_enn/transform.py",
        "ripple/preprocessing/mriganka_enn/artifact_io.py",
        "ripple/preprocessing/mriganka_enn/service.py",
        "ripple/preprocessing/mriganka_enn/visualization.py",
        "ripple/modeling/mriganka_enn_adapter.py",
    )
    return tuple(
        implementation_source_ref(repository / relative_path, repository=repository)
        for relative_path in relative_paths
    )


__all__ = ["MrigankaEnnThreeBandPreprocessor"]
