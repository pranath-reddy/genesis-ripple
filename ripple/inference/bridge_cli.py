"""CLI for the audited three-band M3-to-M4 tensor bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from .m3_bridge import M3ToM4BridgeError, run_m3_to_m4_bridge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reverify one completed three-band M3 run and mechanically select "
            "batch index zero as a 3x64x64 M4 input. Scientific use remains blocked."
        )
    )
    parser.add_argument(
        "--m3-run",
        type=Path,
        required=True,
        help="Completed registered three-band M3 run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/m3_m4_bridge"),
        help="Parent directory for a new immutable bridge run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        completed = run_m3_to_m4_bridge(
            m3_run_directory=args.m3_run,
            output_root=args.output_root,
        )
    except (M3ToM4BridgeError, ValidationError) as exc:
        code = getattr(exc, "code", "invalid_bridge_contract")
        print(f"RIPPLe M3-to-M4 bridge: FAILED ({code})", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - keep CLI failures sanitized
        print("RIPPLe M3-to-M4 bridge: FAILED unexpectedly", file=sys.stderr)
        return 10

    record = completed.record
    print("RIPPLe M3-to-M4 bridge: SUCCESS (TECHNICAL INTEGRATION ONLY)")
    print(f"run_directory: {completed.run_directory}")
    print(f"completion: {completed.completion_path}")
    print(f"bridge_manifest: {completed.bridge_path}")
    print(f"model_input_chw: {completed.model_input_path}")
    print(f"source_m3_completion_sha256: {record.source_m3.completion_record.sha256}")
    print(f"source_m3_envelope_sha256: {record.source_m3.run_envelope.sha256}")
    print(
        f"source_m3_model_input_sha256: {record.source_m3.model_input_bchw.file.sha256}"
    )
    print(f"output_model_input_sha256: {record.model_input_chw.file.sha256}")
    print(f"channel_mapping_status: {record.channel_mapping.status}")
    print(f"configured_bands: {record.channel_mapping.configured_bands}")
    print(f"operation: {record.payload_integrity.operation}")
    print("payload_identical: true")
    print("candidate_decision: NOT MADE")
    print("rubin_dp2_scientific_use: BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
