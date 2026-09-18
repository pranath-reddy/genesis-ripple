"""Command-line entry point for the controlled architecture/model study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .paths import UnsafePathError, checked_real_directory, checked_real_file
from .schemas.simulation import SlsimStudySpec
from .schemas.study import BedrockPricingSnapshot
from .schemas.study_run import ArchitectureModelStudyConfiguration
from .tools.slsim_backend import load_study_spec
from .workflows.study import run_architecture_model_study


class StudyCliError(RuntimeError):
    """Safe local validation failure suitable for sanitized CLI handling."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ripple-scientist-study")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--configuration", required=True)

    run = commands.add_parser("run")
    run.add_argument("--configuration", required=True)
    run.add_argument("--repository-root")
    run.add_argument("--run-id")
    return parser


def _configuration(
    path_value: str,
) -> tuple[Path, ArchitectureModelStudyConfiguration]:
    try:
        path = checked_real_file(path_value)
    except UnsafePathError:
        raise StudyCliError("study configuration must be a real file") from None
    configuration = ArchitectureModelStudyConfiguration.model_validate_json(
        path.read_bytes(), strict=True
    )
    return path, configuration


def _related_file(
    configuration_path: Path,
    configured_path: str,
    *,
    description: str,
) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = configuration_path.parent / path
    try:
        return checked_real_file(path)
    except UnsafePathError:
        raise StudyCliError(f"{description} must be a real file") from None


def _validated_inputs(
    configuration_path: Path,
    configuration: ArchitectureModelStudyConfiguration,
) -> tuple[SlsimStudySpec, BedrockPricingSnapshot]:
    specification_path = _related_file(
        configuration_path,
        configuration.simulation_spec,
        description="SLSim study specification",
    )
    pricing_path = _related_file(
        configuration_path,
        configuration.pricing_snapshot,
        description="pricing snapshot",
    )
    specification = load_study_spec(specification_path)
    pricing = BedrockPricingSnapshot.model_validate_json(
        pricing_path.read_bytes(), strict=True
    )

    rates = {(rate.model_id, rate.region): rate for rate in pricing.rates}
    for model in configuration.bedrock_models:
        rate = rates.get((model.pricing_model_id, configuration.bedrock_region))
        if rate is None:
            raise StudyCliError(
                "pricing snapshot does not cover every configured model and region"
            )
        if rate.provider != model.provider or rate.category != model.category:
            raise StudyCliError(
                "pricing snapshot metadata disagrees with a configured model"
            )
    return specification, pricing


def _repository_root(path_value: str | None) -> Path:
    candidate = (
        path_value
        if path_value is not None
        else Path(__file__).resolve().parents[2]
    )
    try:
        return checked_real_directory(candidate)
    except UnsafePathError:
        raise StudyCliError("repository root must be a real directory") from None


def _print_json(payload: dict[str, object], *, error: bool = False) -> None:
    print(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        file=sys.stderr if error else sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        configuration_path, configuration = _configuration(
            arguments.configuration
        )
        specification, pricing = _validated_inputs(
            configuration_path, configuration
        )

        if arguments.command == "validate":
            pricing_basis_counts: dict[str, int] = {}
            for rate in pricing.rates:
                pricing_basis_counts[rate.pricing_basis] = (
                    pricing_basis_counts.get(rate.pricing_basis, 0) + 1
                )
            _print_json(
                {
                    "status": "valid",
                    "study_id": configuration.study_id,
                    "architecture_arm_count": len(configuration.arms),
                    "bedrock_model_count": len(configuration.bedrock_models),
                    "sample_count": (
                        specification.lens_count
                        + specification.non_lens_count
                    ),
                    "bands": list(specification.rendering.bands),
                    "image_shape_chw": [
                        len(specification.rendering.bands),
                        specification.rendering.num_pix,
                        specification.rendering.num_pix,
                    ],
                    "pricing_snapshot_id": pricing.snapshot_id,
                    "pricing_rate_count": len(pricing.rates),
                    "pricing_basis_counts": pricing_basis_counts,
                    "rates_are_estimates_not_invoices": True,
                    "supports_scientific_claims": (
                        specification.supports_scientific_claims
                    ),
                    "scientific_performance_claim_allowed": (
                        configuration.scientific_performance_claim_allowed
                    ),
                    "network_or_remote_action_performed": False,
                }
            )
            return 0

        completed = run_architecture_model_study(
            configuration,
            configuration_base=configuration_path.parent,
            repository_root=_repository_root(arguments.repository_root),
            run_id=arguments.run_id,
        )
        qualification = completed.study.qualification
        mriganka = completed.study.mriganka_zero_shot
        _print_json(
            {
                "status": "complete",
                "run_id": completed.run_directory.name,
                "study_id": completed.study.study_id,
                "dataset_id": completed.study.dataset.dataset_id,
                "selected_searched_arm_id": (
                    completed.study.architecture_plan.selected_arm_id
                ),
                "baseline_arm_id": configuration.baseline_arm_id,
                "run_directory": str(completed.run_directory),
                "completion": str(completed.completion_path),
                "study_aggregate": str(completed.aggregate_path),
                "paper_table_manifest": str(
                    completed.paper_tables.manifest_path
                ),
                "paper_table_manifest_sha256": (
                    completed.paper_tables.manifest_sha256
                ),
                "paper_table_artifacts": [
                    str(path) for path in completed.paper_tables.artifact_paths
                ],
                "scientific_performance_claim_allowed": (
                    qualification.scientific_performance_claim_allowed
                ),
                "fair_architecture_comparison_allowed": (
                    qualification.fair_architecture_comparison_allowed
                ),
                "held_out_test_used_once_after_selection": (
                    qualification.held_out_test_used_once_after_selection
                ),
                "mriganka_fair_training_comparison_allowed": (
                    False
                    if mriganka is None
                    else mriganka.fair_training_comparison_allowed
                ),
                "rates_are_estimates_not_invoices": True,
                "network_or_remote_action_performed": True,
            }
        )
        return 0
    except Exception as exc:
        # Never echo configuration contents, SDK text, remote stderr, or credentials.
        _print_json(
            {
                "status": "failed",
                "error_code": (
                    "study_validation_failed"
                    if arguments.command == "validate"
                    else "study_execution_failed"
                ),
                "error_type": type(exc).__name__,
                "safe_message": (
                    "study configuration validation failed"
                    if arguments.command == "validate"
                    else (
                        "study execution failed; inspect persisted local run "
                        "evidence if a run directory was created"
                    )
                ),
            },
            error=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
