"""Deterministic SLSim executor for the ten-image integration smoke.

This module deliberately imports astronomy dependencies only when ``generate``
is called.  Importing the wider orchestrator therefore does not require SLSim,
Astropy, Lenstronomy, or NumPy to be installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import random
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from ..schemas.simulation import (
    ArrayArtifactRef,
    BandMagnitudeRange,
    BandMagnitudeValue,
    ClosedFloatRange,
    EffectiveBandRendering,
    EplDeflectorParameters,
    LineOfSightParameters,
    RuntimeComponent,
    SersicSourceParameters,
    SimulationDatasetRecord,
    SimulationParameters,
    SimulationSampleRecord,
    SlsimSmokeSpec,
    canonical_model_bytes,
    canonical_spec_sha256,
    validate_finite_mapping,
)


class SlsimDependencyError(RuntimeError):
    """Raised when the optional simulation stack is not importable."""


class SlsimGenerationError(RuntimeError):
    """Raised when SLSim violates the declared output contract."""


@dataclass(frozen=True)
class _Dependencies:
    np: Any
    flat_lambda_cdm: Any
    source: Any
    deflector_from_table: Any
    lens: Any
    false_positive: Any
    los_individual: Any
    simulate_image: Any
    kwargs_single_band: Any
    slsim_version: str
    astropy_version: str
    lenstronomy_version: str


def _load_dependencies() -> _Dependencies:
    """Load the optional numerical stack at the execution boundary."""

    try:
        import numpy as np
        import astropy
        import lenstronomy
        import slsim
        from astropy.cosmology import FlatLambdaCDM
        from slsim.Deflectors.deflector_util import deflector_from_table
        from slsim.FalsePositives.false_positive import FalsePositive
        from slsim.ImageSimulation.image_quality_lenstronomy import kwargs_single_band
        from slsim.ImageSimulation.image_simulation import simulate_image
        from slsim.LOS.los_individual import LOSIndividual
        from slsim.Lenses.lens import Lens
        from slsim.Sources.source import Source
    except (ImportError, ModuleNotFoundError) as exc:
        raise SlsimDependencyError(
            "SLSim smoke generation requires numpy, astropy, lenstronomy, and "
            "the pinned SLSim source package"
        ) from exc

    return _Dependencies(
        np=np,
        flat_lambda_cdm=FlatLambdaCDM,
        source=Source,
        deflector_from_table=deflector_from_table,
        lens=Lens,
        false_positive=FalsePositive,
        los_individual=LOSIndividual,
        simulate_image=simulate_image,
        kwargs_single_band=kwargs_single_band,
        slsim_version=str(getattr(slsim, "__version__", "unknown")),
        astropy_version=str(getattr(astropy, "__version__", "unknown")),
        lenstronomy_version=str(getattr(lenstronomy, "__version__", "unknown")),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_ref(
    *,
    root: Path,
    path: Path,
    media_type: Literal["application/x-npy", "application/x-npz"],
    array_shape: tuple[int, ...] | None,
    dtype: str | None,
) -> ArrayArtifactRef:
    return ArrayArtifactRef(
        relative_path=path.relative_to(root).as_posix(),
        media_type=media_type,
        sha256=_sha256_file(path),
        byte_count=path.stat().st_size,
        array_shape=array_shape,
        dtype=dtype,
        contains_object_arrays=False,
    )


def _draw(rng: Any, interval: ClosedFloatRange) -> float:
    if interval.minimum == interval.maximum:
        return float(interval.minimum)
    return float(rng.uniform(interval.minimum, interval.maximum))


def _draw_annulus_offset(
    rng: Any, interval: ClosedFloatRange
) -> tuple[float, float, float]:
    """Draw uniformly in area within the declared annulus."""

    radius = math.sqrt(float(rng.uniform(interval.minimum**2, interval.maximum**2)))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    return radius * math.cos(angle), radius * math.sin(angle), radius


def _magnitude_values(
    rng: Any, ranges: tuple[BandMagnitudeRange, ...]
) -> tuple[BandMagnitudeValue, ...]:
    return tuple(
        BandMagnitudeValue(band=item.band, magnitude_ab=_draw(rng, item.magnitude_ab))
        for item in ranges
    )


def _magnitude_kwargs(values: tuple[BandMagnitudeValue, ...]) -> dict[str, float]:
    return {f"mag_{item.band}": item.magnitude_ab for item in values}


def _derive_unique_sample_seed(master_seed: int, sample_id: str, used: set[int]) -> int:
    """Derive a stable uint32 seed without relying on process-randomized hash()."""

    attempt = 0
    while True:
        payload = f"ripple-slsim-v1:{master_seed}:{sample_id}:{attempt}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        if seed not in used:
            return seed
        attempt += 1


@contextmanager
def _legacy_random_seed(np: Any, seed: int) -> Iterator[None]:
    """Scope the legacy global RNGs used internally by the current SLSim API."""

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SlsimGenerationError("LSST rendering configuration is non-finite")
        return value
    raise TypeError


def _effective_band_configuration(
    deps: _Dependencies, spec: SlsimSmokeSpec
) -> tuple[dict[str, Any], tuple[EffectiveBandRendering, ...]]:
    configs: dict[str, dict[str, Any]] = {}
    records: list[EffectiveBandRendering] = []
    render = spec.rendering

    for band in render.bands:
        config = dict(
            deps.kwargs_single_band(
                band=band,
                observatory=render.observatory,
                coadd_years=render.coadd_years,
            )
        )
        # These values are the declared smoke contract, even if an upstream
        # ObservationConfig changes its defaults in a later dependency release.
        config["pixel_scale"] = render.pixel_scale_arcsec
        config["psf_type"] = render.psf_type
        config["seeing"] = render.psf_fwhm_arcsec
        if render.exposure_time_seconds is not None:
            config["exposure_time"] = render.exposure_time_seconds
        if render.num_exposures is not None:
            config["num_exposures"] = render.num_exposures

        scalar_parameters: dict[str, Any] = {}
        omitted: list[str] = []
        for name, value in sorted(config.items()):
            try:
                scalar_parameters[str(name)] = _json_scalar(value)
            except TypeError:
                omitted.append(str(name))
        validate_finite_mapping(scalar_parameters)
        records.append(
            EffectiveBandRendering(
                band=band,
                parameters=scalar_parameters,
                omitted_non_scalar_parameter_names=tuple(omitted),
            )
        )
        configs[band] = config
    return configs, tuple(records)


def _build_system(
    *,
    deps: _Dependencies,
    spec: SlsimSmokeSpec,
    class_name: Literal["lens", "non_lens"],
    rng: Any,
    cosmo: Any,
) -> tuple[Any, SimulationParameters]:
    geometry = spec.lensing_geometry
    source_spec = spec.source_population
    deflector_spec = spec.deflector_population

    center_x = _draw(rng, geometry.deflector_center_component_arcsec)
    center_y = _draw(rng, geometry.deflector_center_component_arcsec)
    radius_range = (
        geometry.lens_source_radius_arcsec
        if class_name == "lens"
        else geometry.non_lens_intruder_radius_arcsec
    )
    offset_x, offset_y, source_radius = _draw_annulus_offset(rng, radius_range)

    source_magnitudes = _magnitude_values(rng, source_spec.magnitudes)
    source_parameters = SersicSourceParameters(
        redshift=_draw(rng, source_spec.redshift),
        angular_size_arcsec=_draw(rng, source_spec.angular_size_arcsec),
        magnitudes=source_magnitudes,
        ellipticity_e1=_draw(rng, source_spec.ellipticity_component),
        ellipticity_e2=_draw(rng, source_spec.ellipticity_component),
        sersic_index=_draw(rng, source_spec.sersic_index),
        center_x_arcsec=center_x + offset_x,
        center_y_arcsec=center_y + offset_y,
    )

    deflector_magnitudes = _magnitude_values(rng, deflector_spec.magnitudes)
    deflector_parameters = EplDeflectorParameters(
        redshift=_draw(rng, deflector_spec.redshift),
        angular_size_arcsec=_draw(rng, deflector_spec.angular_size_arcsec),
        magnitudes=deflector_magnitudes,
        light_ellipticity_e1=_draw(rng, deflector_spec.light_ellipticity_component),
        light_ellipticity_e2=_draw(rng, deflector_spec.light_ellipticity_component),
        mass_ellipticity_e1=_draw(rng, deflector_spec.mass_ellipticity_component),
        mass_ellipticity_e2=_draw(rng, deflector_spec.mass_ellipticity_component),
        einstein_radius_arcsec=_draw(rng, deflector_spec.einstein_radius_arcsec),
        power_law_slope=_draw(rng, deflector_spec.power_law_slope),
        sersic_index=_draw(rng, deflector_spec.sersic_index),
        center_x_arcsec=center_x,
        center_y_arcsec=center_y,
    )

    los_parameters = LineOfSightParameters(
        convergence=_draw(rng, geometry.line_of_sight_convergence),
        shear_gamma1=_draw(rng, geometry.line_of_sight_shear_component),
        shear_gamma2=_draw(rng, geometry.line_of_sight_shear_component),
    )

    source = deps.source(
        cosmo=cosmo,
        extended_source_type=source_spec.profile,
        z=source_parameters.redshift,
        angular_size=source_parameters.angular_size_arcsec,
        e1=source_parameters.ellipticity_e1,
        e2=source_parameters.ellipticity_e2,
        n_sersic=source_parameters.sersic_index,
        center_x=source_parameters.center_x_arcsec,
        center_y=source_parameters.center_y_arcsec,
        **_magnitude_kwargs(source_parameters.magnitudes),
    )
    deflector_table: dict[str, float] = {
        "z": deflector_parameters.redshift,
        "angular_size": deflector_parameters.angular_size_arcsec,
        "theta_E": deflector_parameters.einstein_radius_arcsec,
        "e1_light": deflector_parameters.light_ellipticity_e1,
        "e2_light": deflector_parameters.light_ellipticity_e2,
        "e1_mass": deflector_parameters.mass_ellipticity_e1,
        "e2_mass": deflector_parameters.mass_ellipticity_e2,
        "gamma_pl": deflector_parameters.power_law_slope,
        "n_sersic": deflector_parameters.sersic_index,
        "center_x": deflector_parameters.center_x_arcsec,
        "center_y": deflector_parameters.center_y_arcsec,
        **_magnitude_kwargs(deflector_parameters.magnitudes),
    }
    deflector = deps.deflector_from_table(
        table=deflector_table,
        mass_type=deflector_spec.mass_profile,
        extended_source_type=deflector_spec.light_profile,
        cosmo=cosmo,
    )
    los = deps.los_individual(
        kappa=los_parameters.convergence,
        gamma=[los_parameters.shear_gamma1, los_parameters.shear_gamma2],
    )

    if class_name == "lens":
        system = deps.lens(
            source_class=source,
            deflector_class=deflector,
            cosmo=cosmo,
            los_class=los,
            use_jax=geometry.use_jax_lens_models,
        )
        position_regime = "near_axis_lensed"
    else:
        system = deps.false_positive(
            source_class=source,
            deflector_class=deflector,
            cosmo=cosmo,
            los_class=los,
            include_deflector_light=geometry.include_deflector_light,
        )
        # FalsePositive currently omits Lens.use_jax from its public
        # constructor.  Fail loudly if that upstream compatibility point moves.
        if not hasattr(system, "_use_jax"):
            raise SlsimGenerationError(
                "SLSim FalsePositive no longer exposes the Lens JAX selection state"
            )
        system._use_jax = geometry.use_jax_lens_models
        position_regime = "projected_unlensed_intruder"

    parameters = SimulationParameters(
        source=source_parameters,
        deflector=deflector_parameters,
        line_of_sight=los_parameters,
        source_radius_from_deflector_arcsec=source_radius,
        source_position_regime=position_regime,
    )
    return system, parameters


def _render_system(
    *,
    deps: _Dependencies,
    spec: SlsimSmokeSpec,
    system: Any,
    band_configs: dict[str, dict[str, Any]],
) -> Any:
    channels: list[Any] = []
    render = spec.rendering
    for band in render.bands:
        image = deps.simulate_image(
            lens_class=system,
            band=band,
            num_pix=render.num_pix,
            add_noise=render.add_noise,
            add_background_counts=render.add_background_counts,
            observatory=render.observatory,
            kwargs_numerics={
                "supersampling_factor": render.supersampling_factor,
                "point_source_supersampling_factor": (
                    render.point_source_supersampling_factor
                ),
            },
            kwargs_single_band=dict(band_configs[band]),
            with_source=render.with_source,
            with_deflector=render.with_deflector,
            with_point_source=render.with_point_source,
            image_units_counts=False,
        )
        image = deps.np.asarray(image, dtype=deps.np.float32)
        expected_shape = (render.num_pix, render.num_pix)
        if image.shape != expected_shape:
            raise SlsimGenerationError(
                f"SLSim returned shape {image.shape!r}; expected {expected_shape!r}"
            )
        if not bool(deps.np.isfinite(image).all()):
            raise SlsimGenerationError("SLSim returned non-finite pixels")
        channels.append(image)
    return deps.np.stack(channels, axis=0).astype(deps.np.float32, copy=False)


def _runtime_components(deps: _Dependencies) -> tuple[RuntimeComponent, ...]:
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = str(deps.np.__version__)
    return (
        RuntimeComponent(component="python", version=platform.python_version()),
        RuntimeComponent(component="numpy", version=str(numpy_version)),
        RuntimeComponent(component="astropy", version=deps.astropy_version),
        RuntimeComponent(component="lenstronomy", version=deps.lenstronomy_version),
        RuntimeComponent(component="slsim", version=deps.slsim_version),
    )


class SlsimSmokeBackend:
    """Generate an exact ten-object lens/non-lens dataset with SLSim."""

    def generate(
        self,
        *,
        spec: SlsimSmokeSpec,
        output_dir: str | os.PathLike[str],
    ) -> SimulationDatasetRecord:
        deps = _load_dependencies()
        destination = Path(output_dir)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing simulation output: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)

        spec_sha256 = canonical_spec_sha256(spec)
        dataset_id = f"slsim-smoke-{spec_sha256[:16]}"
        band_configs, effective_rendering = _effective_band_configuration(deps, spec)
        cosmo = deps.flat_lambda_cdm(
            H0=spec.cosmology.hubble_constant_km_s_mpc,
            Om0=spec.cosmology.matter_density_omega_m,
        )

        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.stage-", dir=destination.parent
            )
        )
        moved = False
        try:
            samples_dir = stage / "samples"
            samples_dir.mkdir()
            image_arrays: list[Any] = []
            labels: list[int] = []
            sample_seeds: list[int] = []
            sample_ids: list[str] = []
            records: list[SimulationSampleRecord] = []
            used_seeds: set[int] = set()

            class_plan: list[tuple[Literal["lens", "non_lens"], int]] = [
                ("lens", index) for index in range(spec.lens_count)
            ] + [("non_lens", index) for index in range(spec.non_lens_count)]

            for ordinal, (class_name, class_index) in enumerate(class_plan):
                sample_prefix = "lens" if class_name == "lens" else "non-lens"
                sample_id = f"{sample_prefix}-{class_index:04d}"
                sample_seed = _derive_unique_sample_seed(
                    spec.master_seed, sample_id, used_seeds
                )
                used_seeds.add(sample_seed)
                rng = deps.np.random.default_rng(sample_seed)

                with _legacy_random_seed(deps.np, sample_seed):
                    system = None
                    parameters = None
                    draw_attempts = 0
                    for draw_attempts in range(
                        1, spec.lens_selection.maximum_draw_attempts_per_lens + 1
                    ):
                        system, parameters = _build_system(
                            deps=deps,
                            spec=spec,
                            class_name=class_name,
                            rng=rng,
                            cosmo=cosmo,
                        )
                        if class_name == "non_lens" or bool(
                            system.validity_test(
                                min_image_separation=(
                                    spec.lens_selection.minimum_image_separation_arcsec
                                ),
                                max_image_separation=(
                                    spec.lens_selection.maximum_image_separation_arcsec
                                ),
                            )
                        ):
                            break
                    else:
                        raise SlsimGenerationError(
                            f"could not draw a lens passing declared cuts for {sample_id}"
                        )
                    if system is None or parameters is None:
                        raise SlsimGenerationError("system draw produced no result")
                    image = _render_system(
                        deps=deps,
                        spec=spec,
                        system=system,
                        band_configs=band_configs,
                    )

                image_path = samples_dir / f"{sample_id}.npy"
                with image_path.open("xb") as stream:
                    deps.np.save(stream, image, allow_pickle=False)
                image_ref = _artifact_ref(
                    root=stage,
                    path=image_path,
                    media_type="application/x-npy",
                    array_shape=tuple(int(size) for size in image.shape),
                    dtype=str(image.dtype),
                )
                numeric_label = (
                    spec.label_mapping.lens
                    if class_name == "lens"
                    else spec.label_mapping.non_lens
                )
                records.append(
                    SimulationSampleRecord(
                        sample_id=sample_id,
                        ordinal=ordinal,
                        class_name=class_name,
                        numeric_label=numeric_label,
                        sample_seed=sample_seed,
                        draw_attempts=draw_attempts,
                        lens_selection_status=(
                            "passed" if class_name == "lens" else "not_applicable"
                        ),
                        bands=spec.rendering.bands,
                        image_shape_chw=tuple(int(size) for size in image.shape),
                        image_artifact=image_ref,
                        parameters=parameters,
                    )
                )
                image_arrays.append(image)
                labels.append(numeric_label)
                sample_seeds.append(sample_seed)
                sample_ids.append(sample_id)

            stacked_images = deps.np.stack(image_arrays, axis=0).astype(
                deps.np.float32, copy=False
            )
            dataset_path = stage / "dataset.npz"
            with dataset_path.open("xb") as stream:
                deps.np.savez_compressed(
                    stream,
                    images=stacked_images,
                    labels=deps.np.asarray(labels, dtype=deps.np.int64),
                    sample_seeds=deps.np.asarray(sample_seeds, dtype=deps.np.uint32),
                    sample_ids=deps.np.asarray(sample_ids, dtype="U16"),
                    bands=deps.np.asarray(spec.rendering.bands, dtype="U1"),
                    spec_sha256=deps.np.asarray(spec_sha256, dtype="U64"),
                )
            dataset_ref = _artifact_ref(
                root=stage,
                path=dataset_path,
                media_type="application/x-npz",
                array_shape=tuple(int(size) for size in stacked_images.shape),
                dtype=str(stacked_images.dtype),
            )

            dataset_record = SimulationDatasetRecord(
                dataset_id=dataset_id,
                generated_at_utc=datetime.now(timezone.utc),
                spec_sha256=spec_sha256,
                spec=spec,
                lens_count=spec.lens_count,
                non_lens_count=spec.non_lens_count,
                total_count=10,
                effective_rendering=effective_rendering,
                runtime_components=_runtime_components(deps),
                samples=tuple(records),
                dataset_artifact=dataset_ref,
                manifest_relative_path="simulation_manifest.json",
                supports_scientific_claims=False,
            )
            manifest_path = stage / dataset_record.manifest_relative_path
            with manifest_path.open("xb") as stream:
                stream.write(canonical_model_bytes(dataset_record))

            for artifact_path in stage.rglob("*"):
                if artifact_path.is_file():
                    artifact_path.chmod(0o444)

            os.replace(stage, destination)
            moved = True
            return dataset_record
        finally:
            if not moved and stage.exists():
                shutil.rmtree(stage)


def load_smoke_spec(path: str | os.PathLike[str]) -> SlsimSmokeSpec:
    """Load a strict JSON specification without executing simulation code."""

    spec_path = Path(path)
    return SlsimSmokeSpec.model_validate_json(spec_path.read_bytes(), strict=True)
