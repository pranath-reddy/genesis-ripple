"""Allowlisted wrapper around the existing provisional Mriganka M3 path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ripple.preprocessing.contracts import MrigankaModelInputPackage
from ripple.preprocessing.artifact_io import load_model_input_package
from ripple.preprocessing.service import Mriganka64Preprocessor

from .adapters import AdapterIdentity, CompatibilityIssue, CompatibilityResult
from .contracts import ModelManifest
from .observation import ObservationBundle, ObservationBundleError


_MODEL_ID = "deeplense.mriganka.domain-adaptation"
_ADAPTER_ID = "mriganka-native64-minmax"
_ADAPTER_VERSION = "v1"
_RECIPE_ID = "mriganka-domain-adaptation-native64-minmax-v1"
_IMPLEMENTATION_IDS = (
    "ripple-verify-m2-v1",
    "ripple-select-band-v1",
    "ripple-native-center-crop-v1",
    "ripple-validate-mriganka-crop-v1",
    "ripple-per-image-minmax-v1",
    "ripple-bchw-float32-v1",
)
_EXPECTED_STEPS: tuple[dict[str, Any], ...] = (
    {
        "order": 0,
        "operation": "verify-observation",
        "implementation_id": "ripple-verify-m2-v1",
        "parameters": {
            "product_kind": "rubin_dp2_deep_coadd_masked_image",
            "image_unit": "nJy",
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
        "operation": "select-band",
        "implementation_id": "ripple-select-band-v1",
        "parameters": {"band": "r", "additional_bands": "reject"},
    },
    {
        "order": 2,
        "operation": "native-center-crop",
        "implementation_id": "ripple-native-center-crop-v1",
        "parameters": {
            "shape_yx": [64, 64],
            "centering": "nearest-native-pixel-window-around-requested-wcs-coordinate",
            "resampling": "none",
            "orientation_change": "none",
            "background_operation": "none",
        },
    },
    {
        "order": 3,
        "operation": "validate-crop",
        "implementation_id": "ripple-validate-mriganka-crop-v1",
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
        "operation": "minmax-normalize",
        "implementation_id": "ripple-per-image-minmax-v1",
        "parameters": {
            "scope": "crop",
            "output_min": 0.0,
            "output_max": 1.0,
            "positive_dynamic_range_required": True,
        },
    },
    {
        "order": 5,
        "operation": "build-tensor",
        "implementation_id": "ripple-bchw-float32-v1",
        "parameters": {
            "axes": ["batch", "channel", "y", "x"],
            "shape": [1, 1, 64, 64],
            "dtype": "float32",
        },
    },
)

_EXPECTED_OBSERVATION = {
    "accepted_product_kinds": ["rubin_dp2_deep_coadd_masked_image"],
    "required_bands": ["r"],
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
    "band_alignment": "not_applicable",
    "missing_band_policy": "reject",
}

_EXPECTED_TENSOR = {
    "axes": ["batch", "channel", "y", "x"],
    "shape": [1, 1, 64, 64],
    "dtype": "float32",
    "unit": "dimensionless",
    "channel_semantics": ["single selected r-band image"],
    "value_range": [0.0, 1.0],
}


class MrigankaAdapterError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class Mriganka64Adapter:
    """Expose the existing v1 transform through the generic adapter protocol.

    The manifest describes the transform and its evidence, but this class owns
    the executable implementation.  Every declared step and parameter must
    match the reviewed v1 recipe exactly before any artifacts are written.
    """

    _identity = AdapterIdentity(
        adapter_id=_ADAPTER_ID,
        adapter_version=_ADAPTER_VERSION,
        implementation_ids=_IMPLEMENTATION_IDS,
        supported_model_ids=(_MODEL_ID,),
    )

    @property
    def identity(self) -> AdapterIdentity:
        return self._identity

    def validate_manifest_contract(
        self,
        *,
        manifest: ModelManifest,
    ) -> tuple[CompatibilityIssue, ...]:
        """Validate every manifest field that controls this adapter's behavior."""

        issues: list[CompatibilityIssue] = []
        if manifest.model_id != _MODEL_ID:
            issues.append(
                _error(
                    "model-id-mismatch",
                    "The Mriganka adapter is not registered for this model ID.",
                    "model_id",
                )
            )
        if manifest.preprocessing.adapter_id != _ADAPTER_ID:
            issues.append(
                _error(
                    "adapter-id-mismatch",
                    "The manifest does not select the Mriganka adapter.",
                    "preprocessing.adapter_id",
                )
            )
        if manifest.preprocessing.adapter_version != _ADAPTER_VERSION:
            issues.append(
                _error(
                    "adapter-version-mismatch",
                    "The manifest requests a different Mriganka adapter version.",
                    "preprocessing.adapter_version",
                )
            )
        if manifest.preprocessing.recipe_id != _RECIPE_ID:
            issues.append(
                _error(
                    "recipe-id-mismatch",
                    "The manifest recipe ID does not match the reviewed v1 recipe.",
                    "preprocessing.recipe_id",
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
                    "The manifest transform steps or parameters differ from the reviewed v1 recipe.",
                    "preprocessing.steps",
                )
            )
        if manifest.preprocessing.inference_augmentation != "none":
            issues.append(
                _error(
                    "inference-augmentation-contract-mismatch",
                    "The reviewed v1 adapter applies no inference-time augmentation.",
                    "preprocessing.inference_augmentation",
                )
            )

        observation_contract = manifest.observation.model_dump(
            mode="json", exclude={"domain_notes"}
        )
        if not _json_identical(observation_contract, _EXPECTED_OBSERVATION):
            issues.append(
                _error(
                    "observation-contract-mismatch",
                    "The manifest observation contract differs from the reviewed v1 adapter.",
                    "observation",
                )
            )
        if not _json_identical(
            manifest.tensor.model_dump(mode="json"),
            _EXPECTED_TENSOR,
        ):
            issues.append(
                _error(
                    "tensor-contract-mismatch",
                    "The manifest tensor contract differs from the reviewed v1 output.",
                    "tensor",
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
        missing_bands = tuple(
            band
            for band in manifest.observation.required_bands
            if band not in observation.bands
        )
        if missing_bands:
            issues.append(
                _error(
                    "required-band-missing",
                    "The observation bundle is missing a required model band.",
                    "observation.required_bands",
                )
            )
        if observation.bands != ("r",):
            issues.append(
                _error(
                    "observation-band-set-incompatible",
                    "The reviewed v1 adapter accepts exactly one r-band observation plane.",
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

        try:
            plane = observation.plane_for_band("r")
        except ObservationBundleError:
            plane = None
        if plane is not None:
            image_unit = plane.package.image.canonical_unit
            if image_unit not in manifest.observation.accepted_image_units:
                issues.append(
                    _error(
                        "image-unit-incompatible",
                        "The observation image unit is not accepted by this manifest.",
                        "observation.accepted_image_units",
                    )
                )
            if manifest.observation.required_components.psf == "required":
                if plane.package.psf.state != "present":
                    issues.append(
                        _error(
                            "required-psf-missing",
                            "The manifest requires a PSF component that is unavailable.",
                            "observation.required_components.psf",
                        )
                    )
            pixel_scale = manifest.observation.pixel_scale_arcsec
            if pixel_scale is not None and any(
                not pixel_scale.minimum <= scale <= pixel_scale.maximum
                for scale in plane.package.celestial_wcs.pixel_scale_arcsec_xy
            ):
                issues.append(
                    _error(
                        "pixel-scale-incompatible",
                        "The observation pixel scale lies outside the manifest range.",
                        "observation.pixel_scale_arcsec",
                    )
                )

        errors = tuple(issue for issue in issues if issue.severity == "error")
        if errors:
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
    ) -> MrigankaModelInputPackage:
        compatibility = self.check_compatibility(
            manifest=manifest,
            observation=observation,
        )
        if not compatibility.compatible:
            raise MrigankaAdapterError(
                code="observation_incompatible",
                message="The observation is incompatible with the reviewed Mriganka adapter.",
            )

        source = observation.plane_for_band("r").source_manifest_path
        package = Mriganka64Preprocessor().run(source, Path(output_directory))
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
            raise MrigankaAdapterError(
                code="adapter-output-contract-mismatch",
                message="The produced M3 package does not match the selected model manifest.",
            )
        return package

    def load_package(self, *, output_directory: Path) -> MrigankaModelInputPackage:
        """Reload and reverify every adapter-owned artifact in a completed run."""

        return load_model_input_package(
            Path(output_directory) / "manifest.json"
        ).package


def _error(code: str, message: str, manifest_field: str) -> CompatibilityIssue:
    return CompatibilityIssue(
        code=code,
        severity="error",
        message=message,
        manifest_field=manifest_field,
    )


def _json_identical(left: object, right: object) -> bool:
    """Compare JSON values without Python's ``False == 0`` coercion."""

    def encode(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    return encode(left) == encode(right)


__all__ = ["Mriganka64Adapter", "MrigankaAdapterError"]
