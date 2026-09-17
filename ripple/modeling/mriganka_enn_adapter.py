"""Allowlisted adapter for the provisional three-band Mriganka ENN input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ripple.preprocessing.errors import PreprocessingInputError
from ripple.preprocessing.mriganka_enn.artifact_io import (
    load_three_band_model_input_package,
)
from ripple.preprocessing.mriganka_enn.contracts import (
    MrigankaEnnThreeBandModelInputPackage,
    MrigankaEnnThreeBandRecipe,
)
from ripple.preprocessing.mriganka_enn.service import (
    MrigankaEnnThreeBandPreprocessor,
)
from ripple.preprocessing.mriganka_enn.transform import (
    build_mriganka_enn_three_band_input,
)

from .adapters import AdapterIdentity, CompatibilityIssue, CompatibilityResult
from .contracts import ModelManifest
from .observation import ObservationBundle

MODEL_ID = "deeplense.mriganka.enn-sda"
MANIFEST_ID = "mriganka-enn-three-band-dp2-provisional-v1"
ADAPTER_ID = "mriganka-enn-native64-three-band"
ADAPTER_VERSION = "v1"
RECIPE_ID = "mriganka-enn-dp2-native64-three-band-minmax-v1"
MODEL_VERSION = "sda-epoch-20-iteration-0-dp2-provisional-v1"
CHANNEL_BANDS = ("g", "r", "i")

_IMPLEMENTATION_IDS = (
    "ripple-verify-three-m2-v1",
    "ripple-select-gri-by-band-v1",
    "ripple-native-wcs-center-crop-v1",
    "ripple-validate-gri-crops-v1",
    "ripple-validate-common-native-grid-v1",
    "mriganka-upstream-channel-minmax-nan-to-num-v1",
    "ripple-bchw-three-channel-float32-v1",
)
_EXPECTED_STEPS: tuple[dict[str, Any], ...] = (
    {
        "order": 0,
        "operation": "verify-observation",
        "implementation_id": "ripple-verify-three-m2-v1",
        "parameters": {
            "package_count": 3,
            "product_kind": "rubin_dp2_deep_coadd_masked_image",
            "image_unit": "nJy",
            "variance_unit": "nJy2",
            "components": [
                "image",
                "mask",
                "variance",
                "celestial_wcs",
                "photometric_calibration",
            ],
        },
    },
    {
        "order": 1,
        "operation": "select-channels-by-band",
        "implementation_id": "ripple-select-gri-by-band-v1",
        "parameters": {
            "channel_bands": ["g", "r", "i"],
            "physical_band_mapping_status": "configured_unverified",
            "input_path_order": "ignored",
            "additional_bands": "reject",
        },
    },
    {
        "order": 2,
        "operation": "native-wcs-center-crop",
        "implementation_id": "ripple-native-wcs-center-crop-v1",
        "parameters": {
            "shape_yx": [64, 64],
            "centering": "nearest-native-pixel-window-around-requested-wcs-coordinate",
            "resampling": "none",
            "orientation_change": "none",
            "background_operation": "none",
            "field_of_view_policy": "native-64px-no-resampling-technical-only",
        },
    },
    {
        "order": 3,
        "operation": "validate-band-crops",
        "implementation_id": "ripple-validate-gri-crops-v1",
        "parameters": {
            "image_finite_required": True,
            "variance_positive_finite_required": True,
            "mask_schema_exact": True,
            "fatal_bits": ["NO_DATA", "SATURATED"],
            "caution_bits": [
                "INTERPOLATED",
                "COSMIC_RAY",
                "DETECTION_EDGE",
                "CLIPPED",
                "REJECTED",
                "INEXACT_PSF",
            ],
            "retained_bits": ["DETECTED"],
            "maximum_fatal_fraction": 0.0,
        },
    },
    {
        "order": 4,
        "operation": "validate-common-native-grid",
        "implementation_id": "ripple-validate-common-native-grid-v1",
        "parameters": {
            "comparison": "pairwise-sky-separation-at-common-crop-relative-pixels",
            "pixel_center_grid_xy": [0.0, 31.5, 63.0],
            "include_half_pixel_footprint_corners": True,
            "sample_count_per_pair": 13,
            "maximum_separation_arcsec": 1e-7,
            "mismatch_policy": "fail",
            "reprojection": "forbidden",
        },
    },
    {
        "order": 5,
        "operation": "channel-minmax-normalize",
        "implementation_id": "mriganka-upstream-channel-minmax-nan-to-num-v1",
        "parameters": {
            "scope": "each-channel-y-x",
            "operation_order": [
                "subtract-minimum",
                "divide-shifted-maximum",
                "nan-to-num-zero",
            ],
            "output_min": 0.0,
            "output_max": 1.0,
            "positive_dynamic_range_required": True,
        },
    },
    {
        "order": 6,
        "operation": "build-tensor",
        "implementation_id": "ripple-bchw-three-channel-float32-v1",
        "parameters": {
            "axes": ["batch", "channel", "y", "x"],
            "shape": [1, 3, 64, 64],
            "dtype": "float32",
        },
    },
)

_EXPECTED_OBSERVATION = {
    "accepted_product_kinds": ["rubin_dp2_deep_coadd_masked_image"],
    "required_bands": ["g", "r", "i"],
    "optional_bands": [],
    "required_components": {
        "image": True,
        "mask": "required",
        "variance": "required",
        "celestial_wcs": "required",
        "psf": "optional",
        "photometric_calibration": "required",
    },
    "accepted_image_units": ["nJy"],
    "pixel_scale_arcsec": None,
    "field_of_view_arcsec": None,
    "band_alignment": "common_pixel_grid_required",
    "missing_band_policy": "reject",
}

_EXPECTED_TENSOR = {
    "axes": ["batch", "channel", "y", "x"],
    "shape": [1, 3, 64, 64],
    "dtype": "float32",
    "unit": "dimensionless",
    "channel_semantics": [
        "configured g band at channel 0; HSC serialized mapping unresolved",
        "configured r band at channel 1; HSC serialized mapping unresolved",
        "configured i band at channel 2; HSC serialized mapping unresolved",
    ],
    "value_range": [0.0, 1.0],
}


class MrigankaEnnAdapterError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class MrigankaEnnThreeBandAdapter:
    """Bind one explicit g/r/i manifest to deterministic code-owned operations."""

    _identity = AdapterIdentity(
        adapter_id=ADAPTER_ID,
        adapter_version=ADAPTER_VERSION,
        implementation_ids=_IMPLEMENTATION_IDS,
        supported_model_ids=(MODEL_ID,),
    )

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def validate_manifest_contract(
        self,
        *,
        manifest: ModelManifest,
    ) -> tuple[CompatibilityIssue, ...]:
        issues: list[CompatibilityIssue] = []
        scalar_expectations = (
            (manifest.manifest_id, MANIFEST_ID, "manifest-id-mismatch", "manifest_id"),
            (manifest.model_id, MODEL_ID, "model-id-mismatch", "model_id"),
            (
                manifest.model_version,
                MODEL_VERSION,
                "model-version-mismatch",
                "model_version",
            ),
            (
                manifest.preprocessing.adapter_id,
                ADAPTER_ID,
                "adapter-id-mismatch",
                "preprocessing.adapter_id",
            ),
            (
                manifest.preprocessing.adapter_version,
                ADAPTER_VERSION,
                "adapter-version-mismatch",
                "preprocessing.adapter_version",
            ),
            (
                manifest.preprocessing.recipe_id,
                RECIPE_ID,
                "recipe-id-mismatch",
                "preprocessing.recipe_id",
            ),
        )
        for observed, expected, code, field in scalar_expectations:
            if observed != expected:
                issues.append(
                    _error(
                        code,
                        "The manifest identity does not match the reviewed three-band adapter.",
                        field,
                    )
                )

        observed_steps = tuple(
            {
                "order": step.order,
                "operation": step.operation,
                "implementation_id": step.implementation_id,
                "parameters": step.parameters,
            }
            for step in manifest.preprocessing.steps
        )
        if not _json_identical(observed_steps, _EXPECTED_STEPS):
            issues.append(
                _error(
                    "recipe-steps-mismatch",
                    "The manifest operations differ from the reviewed three-band recipe.",
                    "preprocessing.steps",
                )
            )
        if manifest.preprocessing.inference_augmentation != "none":
            issues.append(
                _error(
                    "inference-augmentation-mismatch",
                    "The three-band adapter applies no inference-time augmentation.",
                    "preprocessing.inference_augmentation",
                )
            )
        observation_contract = manifest.observation.model_dump(
            mode="json",
            exclude={"domain_notes"},
        )
        if not _json_identical(observation_contract, _EXPECTED_OBSERVATION):
            issues.append(
                _error(
                    "observation-contract-mismatch",
                    "The observation contract differs from the reviewed g/r/i adapter.",
                    "observation",
                )
            )
        if not _json_identical(
            manifest.tensor.model_dump(mode="json"), _EXPECTED_TENSOR
        ):
            issues.append(
                _error(
                    "tensor-contract-mismatch",
                    "The tensor contract differs from the reviewed BCHW output.",
                    "tensor",
                )
            )
        qualification = manifest.qualification
        if not (
            qualification.state == "preprocessing_preview_only"
            and qualification.preprocessing_execution_allowed
            and not qualification.model_execution_allowed
            and not qualification.scientific_use_allowed
        ):
            issues.append(
                _error(
                    "qualification-gate-mismatch",
                    "The provisional three-band manifest must remain preprocessing-preview-only.",
                    "qualification",
                )
            )
        return tuple(issues)

    def check_compatibility(
        self,
        *,
        manifest: ModelManifest,
        observation: ObservationBundle,
    ) -> CompatibilityResult:
        issues = list(self.validate_manifest_contract(manifest=manifest))
        if len(observation.planes) != 3 or set(observation.bands) != {"g", "r", "i"}:
            issues.append(
                _error(
                    "observation-band-set-incompatible",
                    "The reviewed adapter requires exactly one g, r, and i observation plane.",
                    "observation.required_bands",
                )
            )
        if observation.product_kind not in manifest.observation.accepted_product_kinds:
            issues.append(
                _error(
                    "product-kind-incompatible",
                    "The observation product kind is not accepted by this manifest.",
                    "observation.accepted_product_kinds",
                )
            )
        for plane in observation.planes:
            if (
                plane.package.image.canonical_unit
                not in manifest.observation.accepted_image_units
            ):
                issues.append(
                    _error(
                        f"{plane.band}-image-unit-incompatible",
                        "One observation plane uses an unsupported image unit.",
                        "observation.accepted_image_units",
                    )
                )
            if plane.package.variance.canonical_unit != "nJy2":
                issues.append(
                    _error(
                        f"{plane.band}-variance-unit-incompatible",
                        "One observation plane uses an unsupported variance unit.",
                        "observation.required_components",
                    )
                )
        if not any(issue.severity == "error" for issue in issues):
            try:
                build_mriganka_enn_three_band_input(
                    observation,
                    MrigankaEnnThreeBandRecipe(channel_bands=CHANNEL_BANDS),
                )
            except PreprocessingInputError as exc:
                issues.append(
                    _error(
                        f"preprocessing-{exc.code.replace('_', '-')}",
                        exc.safe_message,
                        "observation.band_alignment",
                    )
                )
        if any(issue.severity == "error" for issue in issues):
            return CompatibilityResult.rejected(
                manifest=manifest,
                adapter=self.identity,
                issues=tuple(issues),
            )
        return CompatibilityResult.accepted(
            manifest=manifest,
            adapter=self.identity,
            warnings=tuple(issues),
        )

    def preprocess(
        self,
        *,
        manifest: ModelManifest,
        observation: ObservationBundle,
        output_directory: Path,
    ) -> MrigankaEnnThreeBandModelInputPackage:
        compatibility = self.check_compatibility(
            manifest=manifest,
            observation=observation,
        )
        if not compatibility.compatible:
            raise MrigankaEnnAdapterError(
                code="observation_incompatible",
                message="The observation is incompatible with the reviewed three-band adapter.",
            )
        recipe = MrigankaEnnThreeBandRecipe(channel_bands=CHANNEL_BANDS)
        package = MrigankaEnnThreeBandPreprocessor(recipe).run(
            observation,
            Path(output_directory),
        )
        output_contract = (
            package.recipe.recipe_id,
            package.model_input.axes,
            package.model_input.shape,
            package.model_input.dtype,
            package.model_input.unit,
        )
        manifest_contract = (
            manifest.preprocessing.recipe_id,
            manifest.tensor.axes,
            manifest.tensor.shape,
            manifest.tensor.dtype,
            manifest.tensor.unit,
        )
        if output_contract != manifest_contract:
            raise MrigankaEnnAdapterError(
                code="adapter-output-contract-mismatch",
                message="The three-band package does not match its selected model manifest.",
            )
        return package

    def load_package(
        self,
        *,
        output_directory: Path,
    ) -> MrigankaEnnThreeBandModelInputPackage:
        return load_three_band_model_input_package(
            Path(output_directory) / "manifest.json"
        ).package


def _error(
    code: str,
    message: str,
    manifest_field: str | None,
) -> CompatibilityIssue:
    return CompatibilityIssue(
        code=code,
        severity="error",
        message=message,
        manifest_field=manifest_field,
    )


def _json_identical(left: object, right: object) -> bool:
    def encode(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    return encode(left) == encode(right)


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "CHANNEL_BANDS",
    "MANIFEST_ID",
    "MODEL_ID",
    "MODEL_VERSION",
    "RECIPE_ID",
    "MrigankaEnnAdapterError",
    "MrigankaEnnThreeBandAdapter",
]
