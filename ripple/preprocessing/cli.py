"""CLI for provisional Mriganka-specific deterministic M3 preprocessing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from ripple.dp2.errors import Dp2Error
from ripple.modeling.observation import ObservationBundleError
from ripple.modeling.registry import RegistryError
from ripple.modeling.service import (
    ModelingServiceError,
    build_default_registry,
    run_registered_preprocessing,
)

from .errors import PreprocessingError


_MRIGANKA_MANIFEST_ID = "mriganka-domain-adaptation-provisional-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one verified Rubin DP2 M2 package into the fixed provisional "
            "64x64 single-channel Mriganka classifier-input contract. This command "
            "does not load weights or run a classifier."
        )
    )
    parser.add_argument(
        "--package",
        type=Path,
        required=True,
        help="Path to a verified M2 package.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/m3_mriganka"),
        help="Local root beneath which a new private M3 run directory is created.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = build_default_registry()
        completed = run_registered_preprocessing(
            registry=registry,
            manifest_id=_MRIGANKA_MANIFEST_ID,
            package_paths=args.package,
            output_root=args.output_root,
            invocation_interface="mriganka_cli",
        )
    except (
        Dp2Error,
        ModelingServiceError,
        ObservationBundleError,
        PreprocessingError,
        RegistryError,
        ValidationError,
    ) as exc:
        code = getattr(exc, "code", "invalid_contract")
        print(f"RIPPLe M3: FAILED ({code})", file=sys.stderr)
        return 3
    except Exception:
        print("RIPPLe M3: FAILED unexpectedly", file=sys.stderr)
        return 10

    package = completed.envelope.package
    run_directory = completed.run_directory
    print("RIPPLe M3: SUCCESS (PREPROCESSING ONLY; CLASSIFIER BLOCKED)")
    print(f"manifest: {run_directory / 'manifest.json'}")
    print(f"preview: {run_directory / package.preview.filename}")
    print(f"model_input: {run_directory / package.model_input.filename}")
    print(f"tensor_shape_bchw: {package.model_input.shape}")
    print(f"tensor_dtype: {package.model_input.dtype}")
    print(f"tensor_sha256: {package.model_input.decoded_array_sha256}")
    print(f"crop_bounds_xyxy: {package.crop.crop_bounds_xyxy}")
    print(f"field_of_view_arcsec_xy: {package.crop.field_of_view_arcsec_xy}")
    print(f"qualification: {package.compatibility.state}")
    print(
        f"classifier_execution_allowed: {package.compatibility.classifier_execution_allowed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
