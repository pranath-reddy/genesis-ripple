"""Tool-calling architecture planner for the simulation-trained route."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.usage import UsageLimits

from ..providers import (
    BedrockAgentSettings,
    build_bedrock_converse_model,
    expand_bedrock_agent_settings,
)
from ..schemas.architecture import (
    ArchitectureFamily,
    ArchitectureSearchPlan,
    DatasetSummary,
)


_SYSTEM_PROMPT = """\
You propose candidate neural-network structures for RIPPLe binary strong-lens
classification. Deterministic code builds, trains, measures, and selects them.

PROCEDURE — complete every step in this order:
1. Call `inspect_dataset_contract` for image dimensions, channels, classes,
   frozen split counts, dataset purpose, and manifest identity.
2. Call `inspect_buildable_architecture_space` for the exact executable search
   space and structural limits.
3. Call `inspect_prior_measurements` for results already measured on this exact
   dataset version.
4. Return one ArchitectureSearchPlan containing 2–4 new candidates.

BUILDABLE FAMILIES:
- `cnn`: staged 3x3 convolutions; depths are convolutions per stage and widths
  are stage channels.
- `resnet`: BasicBlock stages; depths are blocks and widths are stage channels.
- `vit`: patch embedding plus transformer blocks; sum(depths) is block count and
  widths[-1] is embedding size.
- `mlpmixer`: patch embedding plus mixer blocks; sum(depths) is block count and
  widths[-1] is hidden size.
- `hybrid`: convolutional stages followed by attention; the final depth/width
  describe the attention section.
- `equivariant`: C4 group convolutions over four 90-degree orientations; mark
  physics_informed=true.

CANDIDATE RULES:
- Stay inside the tool-reported bounds. Anything else is rejected by code.
- Make candidates structurally distinct and vary family or capacity; do not
  designate a family as known best.
- Reason only from supplied dataset metadata and measured history.
- Never choose input shape, channel count, class count, labels, sample splits,
  or qualification. Code binds those fields from the manifest.
- For an `integration_smoke` dataset set smoke_only=true. Its metrics can verify
  plumbing but cannot support a scientific architecture claim.

OUTPUT: give each candidate a short identifier, executable depths and widths,
and one evidence-bounded rationale. Summarize why the set is a useful comparison;
do not predict accuracy.\
"""


@dataclass
class ArchitecturePlannerDependencies:
    dataset: DatasetSummary
    prior_measurements: tuple[dict[str, object], ...] = ()
    called_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchitecturePlannerRun:
    plan: ArchitectureSearchPlan
    called_tools: tuple[str, ...]
    request_count: int
    tool_call_count: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    elapsed_seconds: float


def build_architecture_planner(model: Any, *, retries: int = 1) -> Any:
    agent = Agent(
        model=model,
        deps_type=ArchitecturePlannerDependencies,
        output_type=PromptedOutput(
            ArchitectureSearchPlan,
            name="ripple_architecture_search_plan",
            description="Bounded candidate structures for code-side short training.",
        ),
        system_prompt=_SYSTEM_PROMPT,
        retries=retries,
        name="ripple_architecture_planner",
    )

    @agent.tool
    def inspect_dataset_contract(
        ctx: RunContext[ArchitecturePlannerDependencies],
    ) -> dict[str, object]:
        """Return aggregate manifest metadata; never returns pixel arrays."""

        ctx.deps.called_tools.append("inspect_dataset_contract")
        return ctx.deps.dataset.model_dump(mode="json")

    @agent.tool
    def inspect_buildable_architecture_space(
        ctx: RunContext[ArchitecturePlannerDependencies],
    ) -> dict[str, object]:
        """Return the exact families and structural limits implemented by code."""

        if "inspect_dataset_contract" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the dataset contract before the model space.")
        ctx.deps.called_tools.append("inspect_buildable_architecture_space")
        return {
            "families": [family.value for family in ArchitectureFamily],
            "stage_count": {"minimum": 2, "maximum": 4},
            "depth_per_stage": {"minimum": 1, "maximum": 6},
            "width_per_stage": {"minimum": 8, "maximum": 1024},
            "dimensions_bound_by_code": [
                "input_shape",
                "channels",
                "num_classes",
            ],
        }

    @agent.tool
    def inspect_prior_measurements(
        ctx: RunContext[ArchitecturePlannerDependencies],
    ) -> dict[str, object]:
        """Return compact prior metrics, never checkpoints or predictions."""

        if "inspect_buildable_architecture_space" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the buildable space before prior measurements.")
        ctx.deps.called_tools.append("inspect_prior_measurements")
        return {
            "measurements": list(ctx.deps.prior_measurements),
            "count": len(ctx.deps.prior_measurements),
        }

    return agent


def run_architecture_planner(
    dataset: DatasetSummary,
    *,
    prior_measurements: tuple[dict[str, object], ...] = (),
    settings: BedrockAgentSettings | None = None,
) -> ArchitecturePlannerRun:
    """Perform one live, bounded architecture-planning turn with tool evidence."""

    resolved = expand_bedrock_agent_settings(
        settings or BedrockAgentSettings(),
        max_output_tokens=1200,
        output_token_limit=4000,
        input_token_limit=20000,
        request_limit=7,
    )
    dependencies = ArchitecturePlannerDependencies(
        dataset=dataset,
        prior_measurements=prior_measurements,
    )
    agent = build_architecture_planner(
        build_bedrock_converse_model(resolved), retries=resolved.retries
    )
    started = time.monotonic()
    result = agent.run_sync(
        (
            "Use every required tool in order. Propose a small, diverse search plan "
            "for binary strong-lens classification within the supplied dataset purpose."
        ),
        deps=dependencies,
        usage_limits=UsageLimits(
            request_limit=resolved.request_limit,
            tool_calls_limit=3,
            input_tokens_limit=resolved.input_token_limit,
            output_tokens_limit=resolved.output_token_limit,
            total_tokens_limit=resolved.total_token_limit,
            count_tokens_before_request=False,
        ),
    )
    elapsed_seconds = time.monotonic() - started
    required = {
        "inspect_dataset_contract",
        "inspect_buildable_architecture_space",
        "inspect_prior_measurements",
    }
    if not required <= set(dependencies.called_tools):
        raise RuntimeError("architecture planner skipped a required evidence tool")
    expected_smoke = dataset.purpose == "integration_smoke"
    if result.output.smoke_only is not expected_smoke:
        raise RuntimeError(
            "architecture plan changed the dataset qualification boundary"
        )
    usage = result.usage
    return ArchitecturePlannerRun(
        plan=result.output,
        called_tools=tuple(dependencies.called_tools),
        request_count=usage.requests,
        tool_call_count=usage.tool_calls,
        input_tokens=usage.input_tokens or 0,
        cache_write_tokens=usage.cache_write_tokens or 0,
        cache_read_tokens=usage.cache_read_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        elapsed_seconds=elapsed_seconds,
    )
