"""Offline CLI for the model-aware preprocessing registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from ripple.dp2.errors import Dp2Error
from ripple.preprocessing.errors import PreprocessingError

from .manifest_io import ModelManifestIOError
from .observation import ObservationBundleError
from .registry import RegistryError
from .service import (
    ModelingServiceError,
    build_default_registry,
    run_registered_preprocessing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect registered model contracts or run one approved deterministic "
            "preprocessing adapter. No classifier or LLM is invoked."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "list", help="List registered model manifests and qualification gates."
    )

    preprocess = commands.add_parser(
        "preprocess",
        help="Build a model input with the adapter selected by a registered manifest.",
    )
    preprocess.add_argument("--manifest-id", required=True)
    preprocess.add_argument(
        "--package",
        type=Path,
        action="append",
        required=True,
        help="Verified M2 package.json path; repeat for a future multiband model.",
    )
    preprocess.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/m3_model_aware"),
    )
    preprocess.add_argument(
        "--require-aligned-shapes",
        action="store_true",
        help="Reject differing cross-band shapes before adapter compatibility checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = build_default_registry()
        if args.command == "list":
            for reference in registry.manifest_references():
                resolved = registry.inspect(reference)
                qualification = resolved.manifest.qualification
                print(f"manifest_id: {reference.manifest_id}")
                print(f"model_id: {reference.model_id}")
                print(f"model_version: {reference.model_version}")
                print(f"manifest_sha256: {reference.sha256}")
                print(
                    f"adapter: {resolved.adapter_identity.adapter_id}@{resolved.adapter_identity.adapter_version}"
                )
                print(f"qualification: {qualification.state}")
                print(
                    f"preprocessing_allowed: {qualification.preprocessing_execution_allowed}"
                )
                print(
                    f"model_execution_allowed: {qualification.model_execution_allowed}"
                )
                print(f"scientific_use_allowed: {qualification.scientific_use_allowed}")
                print()
            return 0

        completed = run_registered_preprocessing(
            registry=registry,
            manifest_id=args.manifest_id,
            package_paths=tuple(args.package),
            output_root=args.output_root,
            require_aligned_shapes=args.require_aligned_shapes,
            invocation_interface="modeling_cli",
        )
    except (
        Dp2Error,
        ModelManifestIOError,
        ModelingServiceError,
        ObservationBundleError,
        PreprocessingError,
        RegistryError,
        ValidationError,
    ) as exc:
        code = getattr(exc, "code", "invalid_contract")
        print(f"RIPPLe model-aware preprocessing: FAILED ({code})", file=sys.stderr)
        return 3
    except Exception:
        print("RIPPLe model-aware preprocessing: FAILED unexpectedly", file=sys.stderr)
        return 10

    package = completed.envelope.package
    print("RIPPLe model-aware preprocessing: SUCCESS")
    print(f"run_directory: {completed.run_directory}")
    print(f"selected_manifest: {completed.selected_manifest_path}")
    print(f"run_envelope: {completed.envelope_path}")
    print(f"completion: {completed.completion_path}")
    print(f"model_id: {completed.envelope.manifest.model_id}")
    print(
        "adapter: "
        f"{completed.envelope.adapter.adapter_id}@{completed.envelope.adapter.adapter_version}"
    )
    print(f"recipe_id: {completed.envelope.recipe_id}")
    if hasattr(package, "model_input"):
        print(f"model_input: {completed.run_directory / package.model_input.filename}")
        print(f"tensor_shape: {package.model_input.shape}")
        print(f"tensor_dtype: {package.model_input.dtype}")
    if hasattr(package, "preview"):
        print(f"preview: {completed.run_directory / package.preview.filename}")
    selected_qualification = registry.inspect(
        completed.envelope.manifest
    ).manifest.qualification
    classifier_state = (
        "ALLOWED BY SELECTED MODEL MANIFEST"
        if selected_qualification.model_execution_allowed
        else "BLOCKED BY SELECTED MODEL MANIFEST"
    )
    print(f"classifier_execution: {classifier_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
