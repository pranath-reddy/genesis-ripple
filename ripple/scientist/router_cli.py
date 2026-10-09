"""Command-line entry point for the bounded three-route RIPPLe dispatcher."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .router import (
    RouteDispatchError,
    load_pipeline_request,
    plan_pipeline_route,
    run_pipeline_route,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ripple-scientist")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--simulation-configuration", type=Path)
        command.add_argument(
            "--preprocessing-output-root",
            type=Path,
            help=(
                "Known-model route output root for the complete DP2 g/r/i, M3, "
                "bridge, M4, technical-report, and completion artifact tree."
            ),
        )
        command.add_argument("--researcher-output-root", type=Path)
        command.add_argument("--repository-root", type=Path)
        if name == "run":
            command.add_argument("--campaign-run-id")
            command.add_argument(
                "--researcher-agent-provider",
                choices=("bedrock", "openai"),
                default=None,
                help="Live provider for researcher_model only; defaults to bedrock.",
            )
    return parser


def _render(value: object) -> str:
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", exclude_none=False)  # type: ignore[attr-defined]
    else:
        payload = value
    return json.dumps(payload, sort_keys=True, allow_nan=False)


async def _main_async(arguments: argparse.Namespace) -> int:
    request_path = arguments.request.expanduser()
    request = load_pipeline_request(request_path)
    options = {
        "simulation_configuration": arguments.simulation_configuration,
        "preprocessing_output_root": arguments.preprocessing_output_root,
        "researcher_output_root": arguments.researcher_output_root,
        "repository_root": arguments.repository_root,
    }
    if arguments.command == "plan":
        print(_render(plan_pipeline_route(request, **options)))
        return 0

    result = await run_pipeline_route(
        request,
        request_base_directory=request_path.absolute().parent,
        **options,
        campaign_run_id=arguments.campaign_run_id,
        researcher_agent_provider=arguments.researcher_agent_provider,
    )
    print(_render(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        return asyncio.run(_main_async(arguments))
    except Exception as exc:  # noqa: BLE001 - keep CLI failures sanitized
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": getattr(exc, "code", "route_execution_failed"),
                    "error_type": type(exc).__name__,
                    "safe_message": (
                        exc.safe_message
                        if isinstance(exc, RouteDispatchError)
                        else "The selected route failed; inspect its persisted local evidence."
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
