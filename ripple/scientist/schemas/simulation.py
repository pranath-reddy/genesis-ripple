"""Typed contracts for bounded SLSim lens/non-lens simulations.

The smoke and larger study datasets are synthetic artifacts, not scientifically
representative populations.  Every simulator choice is carried by a versioned
specification so the executor does not silently choose astronomy or instrument
parameters.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_SAFE_BAND_PATTERN = re.compile(r"^[ugrizy]$")


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class ClosedFloatRange(_ImmutableModel):
    """Inclusive finite interval used by one declared parameter draw."""

    minimum: float
    maximum: float

    @model_validator(mode="after")
    def _ordered(self) -> "ClosedFloatRange":
        if self.minimum > self.maximum:
            raise ValueError("range minimum exceeds maximum")
        return self


class BandMagnitudeRange(_ImmutableModel):
    band: Literal["u", "g", "r", "i", "z", "y"]
    magnitude_ab: ClosedFloatRange


class BandMagnitudeValue(_ImmutableModel):
    band: Literal["u", "g", "r", "i", "z", "y"]
    magnitude_ab: float


def _require_positive_range(interval: ClosedFloatRange, name: str) -> None:
    if interval.minimum <= 0.0:
        raise ValueError(f"{name} must be strictly positive")


def _require_component_range(interval: ClosedFloatRange, name: str) -> None:
    # Both independently drawn components remain inside unit ellipticity.
    maximum_absolute = max(abs(interval.minimum), abs(interval.maximum))
    if maximum_absolute * math.sqrt(2.0) >= 1.0:
        raise ValueError(f"{name} permits a non-physical ellipticity magnitude")


class CosmologySpec(_ImmutableModel):
    implementation: Literal["astropy.cosmology.FlatLambdaCDM"] = (
        "astropy.cosmology.FlatLambdaCDM"
    )
    hubble_constant_km_s_mpc: float = Field(default=70.0, gt=0.0)
    matter_density_omega_m: float = Field(default=0.3, gt=0.0, lt=1.0)


class SourcePopulationSpec(_ImmutableModel):
    profile: Literal["single_sersic"] = "single_sersic"
    redshift: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=1.2, maximum=2.0)
    )
    angular_size_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.08, maximum=0.18)
    )
    sersic_index: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.8, maximum=2.5)
    )
    ellipticity_component: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.18, maximum=0.18)
    )
    magnitudes: tuple[BandMagnitudeRange, ...] = Field(
        default_factory=lambda: (
            BandMagnitudeRange(
                band="r",
                magnitude_ab=ClosedFloatRange(minimum=21.5, maximum=24.0),
            ),
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def _unique_bands(self) -> "SourcePopulationSpec":
        bands = tuple(item.band for item in self.magnitudes)
        if len(bands) != len(set(bands)):
            raise ValueError("source magnitude bands must be unique")
        _require_positive_range(self.redshift, "source redshift")
        _require_positive_range(self.angular_size_arcsec, "source angular size")
        _require_positive_range(self.sersic_index, "source Sersic index")
        _require_component_range(
            self.ellipticity_component, "source ellipticity components"
        )
        return self


class DeflectorPopulationSpec(_ImmutableModel):
    light_profile: Literal["single_sersic"] = "single_sersic"
    mass_profile: Literal["EPL"] = "EPL"
    redshift: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.25, maximum=0.65)
    )
    angular_size_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.15, maximum=0.65)
    )
    sersic_index: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=2.0, maximum=5.0)
    )
    light_ellipticity_component: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.2, maximum=0.2)
    )
    mass_ellipticity_component: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.2, maximum=0.2)
    )
    einstein_radius_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.8, maximum=2.0)
    )
    power_law_slope: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=1.9, maximum=2.1)
    )
    magnitudes: tuple[BandMagnitudeRange, ...] = Field(
        default_factory=lambda: (
            BandMagnitudeRange(
                band="r",
                magnitude_ab=ClosedFloatRange(minimum=18.0, maximum=21.0),
            ),
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def _unique_bands(self) -> "DeflectorPopulationSpec":
        bands = tuple(item.band for item in self.magnitudes)
        if len(bands) != len(set(bands)):
            raise ValueError("deflector magnitude bands must be unique")
        _require_positive_range(self.redshift, "deflector redshift")
        _require_positive_range(self.angular_size_arcsec, "deflector angular size")
        _require_positive_range(self.sersic_index, "deflector Sersic index")
        _require_positive_range(self.einstein_radius_arcsec, "Einstein radius")
        _require_positive_range(self.power_law_slope, "power-law slope")
        _require_component_range(
            self.light_ellipticity_component, "deflector light ellipticity components"
        )
        _require_component_range(
            self.mass_ellipticity_component, "deflector mass ellipticity components"
        )
        return self


class LensingGeometrySpec(_ImmutableModel):
    """Declared placement and LOS ranges for the integration-only sample."""

    deflector_center_component_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.1, maximum=0.1)
    )
    lens_source_radius_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.02, maximum=0.25)
    )
    non_lens_intruder_radius_arcsec: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=0.7, maximum=2.3)
    )
    line_of_sight_convergence: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.02, maximum=0.02)
    )
    line_of_sight_shear_component: ClosedFloatRange = Field(
        default_factory=lambda: ClosedFloatRange(minimum=-0.04, maximum=0.04)
    )
    include_deflector_light: Literal[True] = True
    use_jax_lens_models: bool = False

    @model_validator(mode="after")
    def _placement_regimes_are_separate(self) -> "LensingGeometrySpec":
        if self.lens_source_radius_arcsec.minimum < 0.0:
            raise ValueError("lens source radii cannot be negative")
        if self.non_lens_intruder_radius_arcsec.minimum < 0.0:
            raise ValueError("non-lens intruder radii cannot be negative")
        if (
            self.lens_source_radius_arcsec.maximum
            >= self.non_lens_intruder_radius_arcsec.minimum
        ):
            raise ValueError(
                "lens and non-lens placement-radius regimes must not overlap"
            )
        if (
            max(
                abs(self.line_of_sight_convergence.minimum),
                abs(self.line_of_sight_convergence.maximum),
                abs(self.line_of_sight_shear_component.minimum),
                abs(self.line_of_sight_shear_component.maximum),
            )
            >= 1.0
        ):
            raise ValueError(
                "line-of-sight convergence and shear must have magnitude below one"
            )
        return self


class LensSelectionSpec(_ImmutableModel):
    """Executable strong-lens cuts used by the reviewed tutorial code."""

    minimum_image_separation_arcsec: float = Field(default=0.8, ge=0.0)
    maximum_image_separation_arcsec: float = Field(default=10.0, gt=0.0)
    maximum_draw_attempts_per_lens: int = Field(default=64, ge=1, le=10_000)

    @model_validator(mode="after")
    def _ordered(self) -> "LensSelectionSpec":
        if self.minimum_image_separation_arcsec >= self.maximum_image_separation_arcsec:
            raise ValueError("minimum lens-image separation must be below the maximum")
        return self


class LsstRenderingSpec(_ImmutableModel):
    """Explicit configuration passed to SLSim's LSST observation renderer."""

    renderer: Literal["slsim.ImageSimulation.simulate_image"] = (
        "slsim.ImageSimulation.simulate_image"
    )
    observatory: Literal["LSST"] = "LSST"
    bands: tuple[Literal["u", "g", "r", "i", "z", "y"], ...] = ("r",)
    num_pix: Literal[64] = 64
    pixel_scale_arcsec: Literal[0.2] = 0.2
    coadd_years: int = Field(default=10, ge=1, le=10)
    psf_type: Literal["GAUSSIAN"] = "GAUSSIAN"
    psf_fwhm_arcsec: float = Field(default=0.9, gt=0.0, le=5.0)
    noise_model: Literal["slsim.SimAPI.noise_for_model"] = (
        "slsim.SimAPI.noise_for_model"
    )
    add_noise: bool = True
    add_background_counts: bool = False
    exposure_time_seconds: float | None = Field(default=None, gt=0.0)
    num_exposures: int | None = Field(default=None, ge=1)
    supersampling_factor: int = Field(default=3, ge=1, le=10)
    point_source_supersampling_factor: int = Field(default=1, ge=1, le=10)
    with_source: Literal[True] = True
    with_deflector: Literal[True] = True
    with_point_source: Literal[False] = False
    image_units: Literal["counts_per_second"] = "counts_per_second"
    output_dtype: Literal["float32"] = "float32"
    output_axes: tuple[Literal["channel"], Literal["y"], Literal["x"]] = (
        "channel",
        "y",
        "x",
    )

    @field_validator("bands")
    @classmethod
    def _valid_bands(
        cls, value: tuple[Literal["u", "g", "r", "i", "z", "y"], ...]
    ) -> tuple[Literal["u", "g", "r", "i", "z", "y"], ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("rendering bands must be non-empty and unique")
        if any(_SAFE_BAND_PATTERN.fullmatch(band) is None for band in value):
            raise ValueError("rendering includes an unsupported LSST band")
        return value


class SourceRevision(_ImmutableModel):
    source_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_kind: Literal["executable_python", "notebook_code_cells"]
    repository_relative_path: str = Field(min_length=1, max_length=512)
    git_commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    evidence_scope: str = Field(min_length=1, max_length=256)

    @field_validator("repository_relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or value != parsed.as_posix()
            or value in {"", "."}
        ):
            raise ValueError(
                "source path must be a normalized repository-relative path"
            )
        return value

    @field_validator("evidence_scope")
    @classmethod
    def _trimmed_scope(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("evidence scope must be trimmed printable text")
        return value


class SimulationProvenance(_ImmutableModel):
    slsim_source: SourceRevision = Field(
        default_factory=lambda: SourceRevision(
            source_id="slsim-source",
            source_kind="executable_python",
            repository_relative_path="ripple/scientist/vendor/slsim/slsim",
            git_commit="ad62eefb74944f7ee76abf826d8c6329a5894d4b",
            evidence_scope="SLSim source, lens, false-positive, and image-rendering APIs",
        )
    )
    tutorial_code: SourceRevision = Field(
        default_factory=lambda: SourceRevision(
            source_id="slsim-tutorial-code",
            source_kind="notebook_code_cells",
            repository_relative_path="slsim-tutorials",
            git_commit="cbf2165f35c29278b255e4b679d51c84cf86e5d1",
            evidence_scope="Executable code cells for single-lens and false-positive examples",
        )
    )
    jaxtronomy_source: SourceRevision = Field(
        default_factory=lambda: SourceRevision(
            source_id="jaxtronomy-source",
            source_kind="executable_python",
            repository_relative_path="ripple/scientist/vendor/JAXtronomy/jaxtronomy",
            git_commit="a268deaf08dcabfb2919c480acbf39cc6596bf87",
            evidence_scope=(
                "SLSim import-time profile registry; JAX execution remains disabled"
            ),
        )
    )
    revisions_verified_by_executor: Literal[False] = False

    @model_validator(mode="after")
    def _canonical_role_identities(self) -> "SimulationProvenance":
        expected = {
            "slsim_source": (
                "slsim-source",
                "executable_python",
                "ripple/scientist/vendor/slsim/slsim",
                "ad62eefb74944f7ee76abf826d8c6329a5894d4b",
            ),
            "jaxtronomy_source": (
                "jaxtronomy-source",
                "executable_python",
                "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
                "a268deaf08dcabfb2919c480acbf39cc6596bf87",
            ),
            "tutorial_code": (
                "slsim-tutorial-code",
                "notebook_code_cells",
                "slsim-tutorials",
                "cbf2165f35c29278b255e4b679d51c84cf86e5d1",
            ),
        }
        for field_name, identity in expected.items():
            source = getattr(self, field_name)
            observed = (
                source.source_id,
                source.source_kind,
                source.repository_relative_path,
                source.git_commit,
            )
            if observed != identity:
                raise ValueError(
                    f"{field_name} must use its canonical role-specific identity"
                )
        return self


class LabelMapping(_ImmutableModel):
    non_lens: Literal[0] = 0
    lens: Literal[1] = 1


class SlsimSmokeSpec(_ImmutableModel):
    """Complete input to the exactly-ten-object integration smoke."""

    schema_version: Literal["ripple.scientist.slsim-smoke-spec.v1"] = (
        "ripple.scientist.slsim-smoke-spec.v1"
    )
    purpose: Literal["integration_smoke"] = "integration_smoke"
    scientific_status: Literal["synthetic_wiring_only"] = "synthetic_wiring_only"
    supports_scientific_claims: Literal[False] = False
    dataset_name: str = Field(
        default="slsim-lens-nonlens-smoke", pattern=_IDENTIFIER_PATTERN
    )
    lens_count: int = Field(default=5, ge=1, le=9)
    non_lens_count: int = Field(default=5, ge=1, le=9)
    master_seed: int = Field(default=20260917, ge=0, le=4_294_967_295)
    label_mapping: LabelMapping = Field(default_factory=LabelMapping)
    cosmology: CosmologySpec = Field(default_factory=CosmologySpec)
    source_population: SourcePopulationSpec = Field(
        default_factory=SourcePopulationSpec
    )
    deflector_population: DeflectorPopulationSpec = Field(
        default_factory=DeflectorPopulationSpec
    )
    lensing_geometry: LensingGeometrySpec = Field(default_factory=LensingGeometrySpec)
    lens_selection: LensSelectionSpec = Field(default_factory=LensSelectionSpec)
    rendering: LsstRenderingSpec = Field(default_factory=LsstRenderingSpec)
    provenance: SimulationProvenance = Field(default_factory=SimulationProvenance)

    @model_validator(mode="after")
    def _cross_validate(self) -> "SlsimSmokeSpec":
        if self.lens_count + self.non_lens_count != 10:
            raise ValueError("integration smoke must contain exactly ten images")
        expected_bands = set(self.rendering.bands)
        source_bands = {item.band for item in self.source_population.magnitudes}
        deflector_bands = {item.band for item in self.deflector_population.magnitudes}
        if source_bands != expected_bands:
            raise ValueError(
                "source magnitude bands must exactly match rendering bands"
            )
        if deflector_bands != expected_bands:
            raise ValueError(
                "deflector magnitude bands must exactly match rendering bands"
            )
        if (
            self.deflector_population.redshift.maximum
            >= self.source_population.redshift.minimum
        ):
            raise ValueError("source redshift range must lie wholly behind deflectors")
        if (
            self.lensing_geometry.lens_source_radius_arcsec.maximum
            >= self.deflector_population.einstein_radius_arcsec.minimum
        ):
            raise ValueError(
                "smoke lens source offsets must remain below every configured Einstein radius"
            )
        half_field_arcsec = (
            self.rendering.num_pix * self.rendering.pixel_scale_arcsec / 2.0
        )
        maximum_center_offset = max(
            abs(self.lensing_geometry.deflector_center_component_arcsec.minimum),
            abs(self.lensing_geometry.deflector_center_component_arcsec.maximum),
        )
        if (
            maximum_center_offset
            + self.lensing_geometry.non_lens_intruder_radius_arcsec.maximum
            + self.source_population.angular_size_arcsec.maximum
            >= half_field_arcsec
        ):
            raise ValueError(
                "configured non-lens intruders do not fit inside the image field"
            )
        return self


def _study_source_population() -> SourcePopulationSpec:
    return SourcePopulationSpec(
        magnitudes=tuple(
            BandMagnitudeRange(
                band=band,
                magnitude_ab=ClosedFloatRange(minimum=21.5, maximum=24.0),
            )
            for band in ("g", "r", "i")
        )
    )


def _study_deflector_population() -> DeflectorPopulationSpec:
    return DeflectorPopulationSpec(
        magnitudes=tuple(
            BandMagnitudeRange(
                band=band,
                magnitude_ab=ClosedFloatRange(minimum=18.0, maximum=21.0),
            )
            for band in ("g", "r", "i")
        )
    )


def _study_rendering() -> LsstRenderingSpec:
    return LsstRenderingSpec(bands=("g", "r", "i"))


class SlsimStudySpec(_ImmutableModel):
    """Synthetic multiband study input with bounded dynamic class counts."""

    schema_version: Literal["ripple.scientist.slsim-study-spec.v1"] = (
        "ripple.scientist.slsim-study-spec.v1"
    )
    purpose: Literal["scientific_training"] = "scientific_training"
    scientific_status: Literal["synthetic_benchmark_only"] = (
        "synthetic_benchmark_only"
    )
    supports_scientific_claims: Literal[False] = False
    dataset_name: str = Field(
        default="slsim-lens-nonlens-study", pattern=_IDENTIFIER_PATTERN
    )
    lens_count: int = Field(default=100, ge=3, le=10_000)
    non_lens_count: int = Field(default=100, ge=3, le=10_000)
    master_seed: int = Field(default=20260917, ge=0, le=4_294_967_295)
    label_mapping: LabelMapping = Field(default_factory=LabelMapping)
    cosmology: CosmologySpec = Field(default_factory=CosmologySpec)
    source_population: SourcePopulationSpec = Field(
        default_factory=_study_source_population
    )
    deflector_population: DeflectorPopulationSpec = Field(
        default_factory=_study_deflector_population
    )
    lensing_geometry: LensingGeometrySpec = Field(default_factory=LensingGeometrySpec)
    lens_selection: LensSelectionSpec = Field(default_factory=LensSelectionSpec)
    rendering: LsstRenderingSpec = Field(default_factory=_study_rendering)
    provenance: SimulationProvenance = Field(default_factory=SimulationProvenance)

    @model_validator(mode="after")
    def _cross_validate(self) -> "SlsimStudySpec":
        if self.rendering.bands != ("g", "r", "i"):
            raise ValueError("study rendering must use exactly ordered g, r, i bands")
        expected_bands = set(self.rendering.bands)
        source_bands = {item.band for item in self.source_population.magnitudes}
        deflector_bands = {item.band for item in self.deflector_population.magnitudes}
        if source_bands != expected_bands:
            raise ValueError(
                "source magnitude bands must exactly match rendering bands"
            )
        if deflector_bands != expected_bands:
            raise ValueError(
                "deflector magnitude bands must exactly match rendering bands"
            )
        if (
            self.deflector_population.redshift.maximum
            >= self.source_population.redshift.minimum
        ):
            raise ValueError("source redshift range must lie wholly behind deflectors")
        if (
            self.lensing_geometry.lens_source_radius_arcsec.maximum
            >= self.deflector_population.einstein_radius_arcsec.minimum
        ):
            raise ValueError(
                "study lens source offsets must remain below every configured Einstein radius"
            )
        half_field_arcsec = (
            self.rendering.num_pix * self.rendering.pixel_scale_arcsec / 2.0
        )
        maximum_center_offset = max(
            abs(self.lensing_geometry.deflector_center_component_arcsec.minimum),
            abs(self.lensing_geometry.deflector_center_component_arcsec.maximum),
        )
        if (
            maximum_center_offset
            + self.lensing_geometry.non_lens_intruder_radius_arcsec.maximum
            + self.source_population.angular_size_arcsec.maximum
            >= half_field_arcsec
        ):
            raise ValueError(
                "configured non-lens intruders do not fit inside the image field"
            )
        return self


SlsimSpec = Annotated[
    SlsimSmokeSpec | SlsimStudySpec,
    Field(discriminator="schema_version"),
]
SLSIM_SPEC_ADAPTER = TypeAdapter(SlsimSpec)


class ArrayArtifactRef(_ImmutableModel):
    """Digest plus array-safety metadata local to a simulation artifact."""

    relative_path: str = Field(min_length=1, max_length=512)
    media_type: Literal["application/x-npy", "application/x-npz"]
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_count: int = Field(gt=0)
    array_shape: tuple[int, ...] | None = None
    dtype: str | None = Field(default=None, min_length=1, max_length=32)
    contains_object_arrays: Literal[False] = False

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or value != parsed.as_posix()
            or value in {"", "."}
        ):
            raise ValueError("artifact path must be a normalized relative path")
        return value


class SersicSourceParameters(_ImmutableModel):
    redshift: float = Field(gt=0.0)
    angular_size_arcsec: float = Field(gt=0.0)
    magnitudes: tuple[BandMagnitudeValue, ...] = Field(min_length=1)
    ellipticity_e1: float
    ellipticity_e2: float
    sersic_index: float = Field(gt=0.0)
    center_x_arcsec: float
    center_y_arcsec: float


class EplDeflectorParameters(_ImmutableModel):
    redshift: float = Field(gt=0.0)
    angular_size_arcsec: float = Field(gt=0.0)
    magnitudes: tuple[BandMagnitudeValue, ...] = Field(min_length=1)
    light_ellipticity_e1: float
    light_ellipticity_e2: float
    mass_ellipticity_e1: float
    mass_ellipticity_e2: float
    einstein_radius_arcsec: float = Field(gt=0.0)
    power_law_slope: float = Field(gt=0.0)
    sersic_index: float = Field(gt=0.0)
    center_x_arcsec: float
    center_y_arcsec: float


class LineOfSightParameters(_ImmutableModel):
    convergence: float
    shear_gamma1: float
    shear_gamma2: float


class SimulationParameters(_ImmutableModel):
    source: SersicSourceParameters
    deflector: EplDeflectorParameters
    line_of_sight: LineOfSightParameters
    source_radius_from_deflector_arcsec: float = Field(ge=0.0)
    source_position_regime: Literal["near_axis_lensed", "projected_unlensed_intruder"]


class EffectiveBandRendering(_ImmutableModel):
    band: Literal["u", "g", "r", "i", "z", "y"]
    parameters: dict[str, JsonValue]
    omitted_non_scalar_parameter_names: tuple[str, ...] = ()


class RuntimeComponent(_ImmutableModel):
    component: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(min_length=1, max_length=128)


class SimulationSampleRecord(_ImmutableModel):
    sample_id: str = Field(pattern=r"^(?:lens|non-lens)-[0-9]{4}$")
    ordinal: int = Field(ge=0, le=9)
    class_name: Literal["lens", "non_lens"]
    numeric_label: Literal[0, 1]
    sample_seed: int = Field(ge=0, le=4_294_967_295)
    draw_attempts: int = Field(ge=1)
    lens_selection_status: Literal["passed", "not_applicable"]
    bands: tuple[Literal["u", "g", "r", "i", "z", "y"], ...] = Field(min_length=1)
    image_shape_chw: tuple[int, int, int]
    image_artifact: ArrayArtifactRef
    parameters: SimulationParameters

    @model_validator(mode="after")
    def _label_and_shape_match(self) -> "SimulationSampleRecord":
        expected_label = 1 if self.class_name == "lens" else 0
        if self.numeric_label != expected_label:
            raise ValueError("sample class and numeric label disagree")
        expected_selection = "passed" if self.class_name == "lens" else "not_applicable"
        if self.lens_selection_status != expected_selection:
            raise ValueError("sample class and lens-selection status disagree")
        if self.image_shape_chw[0] != len(self.bands):
            raise ValueError("sample channel count does not match band count")
        if self.image_artifact.array_shape != self.image_shape_chw:
            raise ValueError("sample artifact shape does not match sample metadata")
        return self


class SlsimStudySampleRecord(SimulationSampleRecord):
    """Sample metadata whose ordinal spans a dynamic study dataset."""

    ordinal: int = Field(ge=0, le=19_999)


class SimulationDatasetRecord(_ImmutableModel):
    schema_version: Literal["ripple.scientist.slsim-smoke-dataset.v1"] = (
        "ripple.scientist.slsim-smoke-dataset.v1"
    )
    dataset_id: str = Field(pattern=r"^slsim-smoke-[0-9a-f]{16}$")
    generated_at_utc: datetime
    spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    spec: SlsimSmokeSpec
    lens_count: int = Field(ge=1)
    non_lens_count: int = Field(ge=1)
    total_count: Literal[10] = 10
    effective_rendering: tuple[EffectiveBandRendering, ...] = Field(min_length=1)
    runtime_components: tuple[RuntimeComponent, ...] = Field(min_length=1)
    samples: tuple[SimulationSampleRecord, ...] = Field(min_length=10, max_length=10)
    dataset_artifact: ArrayArtifactRef
    manifest_relative_path: Literal["simulation_manifest.json"] = (
        "simulation_manifest.json"
    )
    supports_scientific_claims: Literal[False] = False
    qualification_boundary: Literal[
        "integration wiring only; not a population, performance, or science validation"
    ] = "integration wiring only; not a population, performance, or science validation"

    @field_validator("generated_at_utc")
    @classmethod
    def _aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _dataset_is_self_consistent(self) -> "SimulationDatasetRecord":
        if self.lens_count != self.spec.lens_count:
            raise ValueError("recorded lens count disagrees with specification")
        if self.non_lens_count != self.spec.non_lens_count:
            raise ValueError("recorded non-lens count disagrees with specification")
        if len(self.samples) != self.total_count:
            raise ValueError("sample records do not match total count")
        if tuple(sample.ordinal for sample in self.samples) != tuple(range(10)):
            raise ValueError("sample ordinals must be contiguous and ordered")
        if len({sample.sample_id for sample in self.samples}) != 10:
            raise ValueError("sample IDs must be unique")
        if len({sample.sample_seed for sample in self.samples}) != 10:
            raise ValueError("per-sample seeds must be unique")
        lens_count = sum(sample.class_name == "lens" for sample in self.samples)
        non_lens_count = sum(sample.class_name == "non_lens" for sample in self.samples)
        if (lens_count, non_lens_count) != (self.lens_count, self.non_lens_count):
            raise ValueError("sample class counts disagree with dataset counts")
        expected_shape = (
            10,
            len(self.spec.rendering.bands),
            self.spec.rendering.num_pix,
            self.spec.rendering.num_pix,
        )
        if self.dataset_artifact.array_shape != expected_shape:
            raise ValueError(
                "dataset artifact shape is inconsistent with specification"
            )
        return self


class SlsimStudyDatasetRecord(_ImmutableModel):
    """Content-addressed record for one dynamic synthetic SLSim study."""

    schema_version: Literal["ripple.scientist.slsim-study-dataset.v1"] = (
        "ripple.scientist.slsim-study-dataset.v1"
    )
    dataset_id: str = Field(pattern=r"^slsim-study-[0-9a-f]{16}$")
    generated_at_utc: datetime
    spec_sha256: str = Field(pattern=_SHA256_PATTERN)
    spec: SlsimStudySpec
    lens_count: int = Field(ge=3, le=10_000)
    non_lens_count: int = Field(ge=3, le=10_000)
    total_count: int = Field(ge=6, le=20_000)
    effective_rendering: tuple[EffectiveBandRendering, ...] = Field(min_length=3)
    runtime_components: tuple[RuntimeComponent, ...] = Field(min_length=1)
    samples: tuple[SlsimStudySampleRecord, ...] = Field(
        min_length=6, max_length=20_000
    )
    dataset_artifact: ArrayArtifactRef
    manifest_relative_path: Literal["simulation_manifest.json"] = (
        "simulation_manifest.json"
    )
    supports_scientific_claims: Literal[False] = False
    scientific_use_allowed: Literal[False] = False
    qualification_boundary: Literal[
        "synthetic benchmark only; not a population, model-performance, or science validation"
    ] = (
        "synthetic benchmark only; not a population, model-performance, or science validation"
    )

    @field_validator("generated_at_utc")
    @classmethod
    def _aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated timestamp must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _dataset_is_self_consistent(self) -> "SlsimStudyDatasetRecord":
        if self.lens_count != self.spec.lens_count:
            raise ValueError("recorded lens count disagrees with specification")
        if self.non_lens_count != self.spec.non_lens_count:
            raise ValueError("recorded non-lens count disagrees with specification")
        if self.total_count != self.lens_count + self.non_lens_count:
            raise ValueError("study total count disagrees with its class counts")
        if len(self.samples) != self.total_count:
            raise ValueError("sample records do not match total count")
        if tuple(sample.ordinal for sample in self.samples) != tuple(
            range(self.total_count)
        ):
            raise ValueError("sample ordinals must be contiguous and ordered")
        if len({sample.sample_id for sample in self.samples}) != self.total_count:
            raise ValueError("sample IDs must be unique")
        if len({sample.sample_seed for sample in self.samples}) != self.total_count:
            raise ValueError("per-sample seeds must be unique")
        lens_count = sum(sample.class_name == "lens" for sample in self.samples)
        non_lens_count = sum(sample.class_name == "non_lens" for sample in self.samples)
        if (lens_count, non_lens_count) != (self.lens_count, self.non_lens_count):
            raise ValueError("sample class counts disagree with dataset counts")
        expected_sample_shape = (
            len(self.spec.rendering.bands),
            self.spec.rendering.num_pix,
            self.spec.rendering.num_pix,
        )
        if any(
            sample.bands != self.spec.rendering.bands
            or sample.image_shape_chw != expected_sample_shape
            for sample in self.samples
        ):
            raise ValueError("study sample bands or shapes disagree with specification")
        if (
            tuple(item.band for item in self.effective_rendering)
            != self.spec.rendering.bands
        ):
            raise ValueError("effective rendering bands disagree with specification")
        expected_dataset_shape = (self.total_count, *expected_sample_shape)
        if self.dataset_artifact.array_shape != expected_dataset_shape:
            raise ValueError(
                "dataset artifact shape is inconsistent with specification"
            )
        return self


SlsimDatasetRecord = Annotated[
    SimulationDatasetRecord | SlsimStudyDatasetRecord,
    Field(discriminator="schema_version"),
]
SLSIM_DATASET_RECORD_ADAPTER = TypeAdapter(SlsimDatasetRecord)


def _canonical_model_payload_bytes(model: BaseModel) -> bytes:
    """Return canonical JSON payload bytes without presentation whitespace."""

    import json

    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_spec_sha256(spec: SlsimSpec) -> str:
    """Return the sole semantic identity used for a simulation specification."""

    return hashlib.sha256(_canonical_model_payload_bytes(spec)).hexdigest()


def canonical_model_bytes(model: BaseModel) -> bytes:
    """Return stable newline-terminated bytes for JSON artifact serialization."""

    return _canonical_model_payload_bytes(model) + b"\n"


def validate_finite_mapping(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Defensive helper for executor-created rendering metadata."""

    for key, value in values.items():
        if not key or key != key.strip():
            raise ValueError("rendering metadata keys must be non-empty and trimmed")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("rendering metadata contains a non-finite float")
    return values
