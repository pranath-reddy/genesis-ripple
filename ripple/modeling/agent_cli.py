"""Prepare source evidence locally or run the optional live onboarding agent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .agent import PydanticAIUnavailableError, run_model_onboarding_agent
from .agent_io import (
    AgentArtifactIOError,
    load_onboarding_request,
    load_source_inventory,
    write_agent_outcome,
    write_onboarding_request,
    write_source_inventory,
)
from .agent_provider import (
    LiveAgentConfigurationError,
    build_live_agent_model_from_environment,
)
from .onboarding import ModelOnboardingRequest
from .service import build_default_registry
from .source_inventory import SourceInventoryError, inventory_local_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare code-only model evidence without an LLM, or run a bounded "
            "proposal-only PydanticAI onboarding agent."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Create a typed onboarding request and hash-only source inventory; no LLM/key.",
    )
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--checkpoint", action="append", default=[])
    prepare.add_argument("--primary-literature", action="append", default=[])
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-version", required=True)
    prepare.add_argument("--display-name", required=True)
    prepare.add_argument(
        "--scientific-task",
        required=True,
        choices=(
            "binary_lens_classification",
            "multiclass_lens_classification",
            "lens_reconstruction",
            "parameter_regression",
        ),
    )
    prepare.add_argument("--target-observation-domain", required=True)
    prepare.add_argument("--research-question", required=True)
    prepare.add_argument("--requested-by", required=True)
    prepare.add_argument("--request-output", type=Path, required=True)
    prepare.add_argument("--inventory-output", type=Path, required=True)

    run = commands.add_parser(
        "run",
        help="Run the live two-phase proposal agent; requires local environment variables.",
    )
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--inventory", type=Path, required=True)
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--allow-literature-network", action="store_true")
    run.add_argument(
        "--provider",
        choices=("openai", "bedrock"),
        default="openai",
        help=(
            "Live reasoning provider. The default preserves the existing OpenAI "
            "Responses behavior; bedrock uses the configured local AWS profile."
        ),
    )
    run.add_argument("--retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            return _prepare(args)
        return asyncio.run(_run_live(args))
    except (
        AgentArtifactIOError,
        LiveAgentConfigurationError,
        PydanticAIUnavailableError,
        SourceInventoryError,
        ValidationError,
        ValueError,
    ) as exc:
        code = getattr(exc, "code", "invalid_agent_request")
        print(f"RIPPLe model onboarding: FAILED ({code})", file=sys.stderr)
        return 3
    except Exception:
        # Provider/network exceptions can contain request details. Keep CLI output bounded.
        print("RIPPLe model onboarding: FAILED unexpectedly", file=sys.stderr)
        return 10


def _prepare(args: argparse.Namespace) -> int:
    if args.request_output == args.inventory_output:
        raise ValueError("request and inventory outputs must be different paths")
    requested_at = datetime.now(timezone.utc)
    request_payload = {
        "requested_at_utc": requested_at.isoformat(),
        "requested_by": args.requested_by,
        "model_id": args.model_id,
        "candidate_model_version": args.model_version,
        "display_name": args.display_name,
        "scientific_task": args.scientific_task,
        "target_observation_domain": args.target_observation_domain,
        "source_locators": tuple(args.source),
        "checkpoint_locators": tuple(args.checkpoint),
        "primary_literature_locators": tuple(args.primary_literature),
        "research_question": args.research_question,
    }
    identity = hashlib.sha256(
        json.dumps(
            request_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    request = ModelOnboardingRequest(
        request_id=f"request-{identity}",
        requested_at_utc=requested_at,
        requested_by=args.requested_by,
        model_id=args.model_id,
        candidate_model_version=args.model_version,
        display_name=args.display_name,
        scientific_task=args.scientific_task,
        target_observation_domain=args.target_observation_domain,
        source_locators=tuple(args.source),
        checkpoint_locators=tuple(args.checkpoint),
        primary_literature_locators=tuple(args.primary_literature),
        research_question=args.research_question,
    )
    inventory = inventory_local_sources(
        request,
        repository_root=args.repository_root,
        relative_sources=tuple(args.source),
        relative_checkpoints=tuple(args.checkpoint),
    )
    write_onboarding_request(args.request_output, request)
    try:
        write_source_inventory(args.inventory_output, inventory)
    except Exception:
        # Request publication is immutable; report the partial preparation explicitly.
        print(
            f"RIPPLe model onboarding: request written, inventory failed: {args.request_output}",
            file=sys.stderr,
        )
        raise
    print("RIPPLe model onboarding preparation: SUCCESS")
    print(f"request: {args.request_output}")
    print(f"inventory: {args.inventory_output}")
    print(f"inventory_id: {inventory.inventory_id}")
    print(f"artifacts: {len(inventory.artifacts)}")
    print("llm_invoked: false")
    print("api_key_required: false")
    return 0


async def _run_live(args: argparse.Namespace) -> int:
    request = load_onboarding_request(args.request)
    inventory = load_source_inventory(args.inventory)
    model, configuration = build_live_agent_model_from_environment(args.provider)
    outcome = await run_model_onboarding_agent(
        model=model,
        request=request,
        inventory=inventory,
        repository_root=args.repository_root,
        registry=build_default_registry(),
        allow_literature_network=args.allow_literature_network,
        retries=args.retries,
    )
    write_agent_outcome(args.output, outcome)
    print("RIPPLe live model onboarding: SUCCESS")
    print(f"outcome: {args.output}")
    print(f"provider: {configuration.provider}")
    print(f"model_id: {configuration.model_id}")
    print(f"qualification: {outcome.deterministic_qualification.outcome}")
    print("preprocessing_execution_performed: false")
    print("model_execution_performed: false")
    print("credential_recorded: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
