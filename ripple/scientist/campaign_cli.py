"""Local CLI for the self-contained RIPPLe simulation smoke campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .paths import UnsafePathError, checked_real_directory, checked_real_file
from .providers import load_bedrock_agent_settings
from .schemas.campaign import SimulationCampaignConfiguration
from .tools.slsim_backend import load_smoke_spec
from .workflows.campaign import CampaignExecutionError, run_simulation_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ripple-scientist-campaign")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--configuration", required=True)

    run = commands.add_parser("run")
    run.add_argument("--configuration", required=True)
    run.add_argument("--repository-root")
    run.add_argument("--run-id")
    return parser


def _configuration(path_value: str) -> tuple[Path, SimulationCampaignConfiguration]:
    try:
        path = checked_real_file(path_value)
    except UnsafePathError:
        raise CampaignExecutionError("campaign configuration must be a real file")
    return path, SimulationCampaignConfiguration.model_validate_json(
        path.read_bytes(), strict=True
    )


def _spec_path(
    configuration_path: Path, configuration: SimulationCampaignConfiguration
) -> Path:
    path = Path(configuration.request.simulation_spec).expanduser()
    if not path.is_absolute():
        path = configuration_path.parent / path
    try:
        return checked_real_file(path)
    except UnsafePathError:
        raise CampaignExecutionError(
            "simulation specification must be a real file"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        configuration_path, configuration = _configuration(arguments.configuration)
        spec = load_smoke_spec(_spec_path(configuration_path, configuration))
        if arguments.command == "validate":
            settings = load_bedrock_agent_settings()
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "request_id": configuration.request.request_id,
                        "sample_count": spec.lens_count + spec.non_lens_count,
                        "smoke_only": configuration.smoke_only,
                        "provider": settings.runtime_identity().model_dump(mode="json"),
                        "network_or_remote_action_performed": False,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0

        try:
            repository_root = checked_real_directory(
                arguments.repository_root
                if arguments.repository_root
                else Path(__file__).resolve().parents[2]
            )
        except UnsafePathError:
            raise CampaignExecutionError(
                "repository root must be a real directory"
            ) from None
        completed = run_simulation_campaign(
            configuration,
            configuration_base=configuration_path.parent,
            repository_root=repository_root,
            provider_settings=load_bedrock_agent_settings(),
            run_id=arguments.run_id,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": completed.completion.run_id,
                    "run_directory": str(completed.run_directory),
                    "completion": str(completed.completion_path),
                    "technical_report": str(completed.report_path),
                    "final_state": str(completed.final_state_path),
                    "selected_candidate_id": (
                        completed.completion.selected_candidate_id
                    ),
                    "smoke_only": completed.completion.smoke_only,
                    "scientific_performance_claim_allowed": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except Exception as exc:
        # Do not echo SDK errors, remote stderr, Pydantic inputs, or credentials.
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "safe_message": (
                        "campaign validation or execution failed; inspect the local "
                        "run evidence when a run directory was created"
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
