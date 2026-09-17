"""Command-line entry point for the bounded M4 integration runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .checkpoint_io import CheckpointLoadError
from .service import (
    M4InferenceError,
    builtin_bundle_manifest_path,
    run_m4_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Mriganka ENN checkpoint pair on one compatible "
            "3x64x64 float32 NPY artifact. Results are integration evidence only."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=builtin_bundle_manifest_path(),
        help="Pinned checkpoint-bundle manifest.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Local directory containing the encoder and classifier checkpoint files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One float32 NPY array with shape 3x64x64.",
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        help=(
            "Explicit audited bridge.json for a technical Rubin DP2 integration. "
            "Without it, --input must be an exactly pinned HSC reference array."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/m4_mriganka"),
        help="Parent directory for a new immutable M4 run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        completed = run_m4_inference(
            manifest_path=args.manifest,
            checkpoint_root=args.checkpoint_root,
            input_path=args.input,
            output_root=args.output_root,
            bridge_manifest_path=args.bridge_manifest,
        )
    except (CheckpointLoadError, M4InferenceError, ValidationError) as exc:
        code = getattr(exc, "code", "invalid_contract")
        print(f"RIPPLe M4 inference: FAILED ({code})", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - keep CLI failures sanitized
        print("RIPPLe M4 inference: FAILED unexpectedly", file=sys.stderr)
        return 10

    result = completed.result
    print("RIPPLe M4 inference: SUCCESS (INTEGRATION ONLY)")
    print(f"run_directory: {completed.run_directory}")
    print(f"completion: {completed.completion_path}")
    print(f"bundle_id: {result.bundle_id}")
    print(f"execution_scope: {result.execution_scope_used}")
    if result.bridge_provenance is not None:
        print(
            f"bridge_manifest_sha256: {result.bridge_provenance.bridge_manifest.sha256}"
        )
        print(
            "bridge_completion_sha256: "
            f"{result.bridge_provenance.bridge_completion.sha256}"
        )
        print(
            "channel_mapping_status: "
            f"{result.bridge_provenance.record.channel_mapping.status}"
        )
    print(f"input_sha256: {result.input.file.sha256}")
    print(f"encoder_sha256: {result.encoder_checkpoint.sha256}")
    print(f"classifier_sha256: {result.classifier_checkpoint.sha256}")
    print(f"logits: {result.logits}")
    print(f"scores: {result.scores}")
    print(f"class_index_1_softmax_score: {result.lens_score:.10f}")
    print("score_is_calibrated_probability: false")
    print(f"reference_check: {result.reference_verification.status}")
    print("candidate_decision: NOT MADE")
    print("rubin_dp2_scientific_use: BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
