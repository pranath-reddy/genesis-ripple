"""Validation-only, allowlisted tuning agent for closed-loop studies."""

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
from ..schemas.tuning import TuningAction, TuningDecision, TuningMeasurement
from ..tools.tuning import eligible_tuning_actions


_SYSTEM_PROMPT = """\
You are the bounded tuning planner in a binary strong-lens architecture study.
Code owns the dataset, split isolation, action allowlist, parameter changes,
training, metrics, stopping budget, and final selection.

PROCEDURE:
1. Call inspect_tuning_protocol.
2. Call inspect_current_measurement.
3. Call inspect_prior_trajectory.
4. Select exactly one action returned by inspect_tuning_protocol.

Use only validation and training measurements. Never request, infer, or discuss
test performance. Prefer an action that directly addresses the measured
generalization gap or optimization state. A stop action is valid only when code
includes it in the eligible list. Do not predict accuracy or make a scientific
claim. Return the exact candidate identifier, iteration, and eligible action
list supplied by the tools.\
"""


@dataclass
class TuningDependencies:
    measurement: TuningMeasurement
    prior_measurements: tuple[TuningMeasurement, ...]
    eligible_actions: tuple[TuningAction, ...]
    minimum_iterations: int
    maximum_iterations: int
    minimum_validation_balanced_accuracy: float
    maximum_generalization_gap: float
    called_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TuningAgentRun:
    decision: TuningDecision
    called_tools: tuple[str, ...]
    request_count: int
    tool_call_count: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    elapsed_seconds: float


def build_tuning_agent(model: Any, *, retries: int = 1) -> Any:
    agent = Agent(
        model=model,
        deps_type=TuningDependencies,
        output_type=PromptedOutput(
            TuningDecision,
            name="ripple_tuning_decision",
            description="One validation-only action from a code-owned allowlist.",
        ),
        system_prompt=_SYSTEM_PROMPT,
        retries=retries,
        name="ripple_tuning_planner",
    )

    @agent.tool
    def inspect_tuning_protocol(
        ctx: RunContext[TuningDependencies],
    ) -> dict[str, object]:
        ctx.deps.called_tools.append("inspect_tuning_protocol")
        return {
            "candidate_id": ctx.deps.measurement.candidate_id,
            "iteration": ctx.deps.measurement.iteration,
            "eligible_actions": [item.value for item in ctx.deps.eligible_actions],
            "minimum_iterations": ctx.deps.minimum_iterations,
            "maximum_iterations": ctx.deps.maximum_iterations,
            "minimum_validation_balanced_accuracy": (
                ctx.deps.minimum_validation_balanced_accuracy
            ),
            "maximum_generalization_gap": ctx.deps.maximum_generalization_gap,
            "selection_data": "train_and_validation_only",
            "test_split_available_to_agent": False,
        }

    @agent.tool
    def inspect_current_measurement(
        ctx: RunContext[TuningDependencies],
    ) -> dict[str, object]:
        if "inspect_tuning_protocol" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the tuning protocol before measurements.")
        ctx.deps.called_tools.append("inspect_current_measurement")
        return ctx.deps.measurement.model_dump(mode="json")

    @agent.tool
    def inspect_prior_trajectory(
        ctx: RunContext[TuningDependencies],
    ) -> dict[str, object]:
        if "inspect_current_measurement" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the current measurement before its history.")
        ctx.deps.called_tools.append("inspect_prior_trajectory")
        return {
            "measurements": [
                item.model_dump(mode="json") for item in ctx.deps.prior_measurements
            ],
            "count": len(ctx.deps.prior_measurements),
        }

    return agent


def run_tuning_agent(
    measurement: TuningMeasurement,
    *,
    prior_measurements: tuple[TuningMeasurement, ...] = (),
    minimum_iterations: int,
    maximum_iterations: int,
    minimum_validation_balanced_accuracy: float,
    maximum_generalization_gap: float,
    settings: BedrockAgentSettings | None = None,
) -> TuningAgentRun:
    eligible = eligible_tuning_actions(
        measurement,
        minimum_iterations=minimum_iterations,
        maximum_iterations=maximum_iterations,
        minimum_validation_balanced_accuracy=minimum_validation_balanced_accuracy,
        maximum_generalization_gap=maximum_generalization_gap,
    )
    resolved = expand_bedrock_agent_settings(
        settings or BedrockAgentSettings(),
        max_output_tokens=700,
        output_token_limit=2400,
        input_token_limit=14000,
        request_limit=7,
    )
    dependencies = TuningDependencies(
        measurement=measurement,
        prior_measurements=prior_measurements,
        eligible_actions=eligible,
        minimum_iterations=minimum_iterations,
        maximum_iterations=maximum_iterations,
        minimum_validation_balanced_accuracy=minimum_validation_balanced_accuracy,
        maximum_generalization_gap=maximum_generalization_gap,
    )
    started = time.monotonic()
    result = build_tuning_agent(
        build_bedrock_converse_model(resolved), retries=resolved.retries
    ).run_sync(
        "Inspect all required evidence tools in order and select one eligible action.",
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
        "inspect_tuning_protocol",
        "inspect_current_measurement",
        "inspect_prior_trajectory",
    }
    if not required <= set(dependencies.called_tools):
        raise RuntimeError("tuning planner skipped a required evidence tool")
    decision = result.output
    if (
        decision.candidate_id != measurement.candidate_id
        or decision.iteration != measurement.iteration
        or decision.eligible_actions != eligible
    ):
        raise RuntimeError("tuning planner changed code-owned decision inputs")
    usage = result.usage
    return TuningAgentRun(
        decision=decision,
        called_tools=tuple(dependencies.called_tools),
        request_count=usage.requests,
        tool_call_count=usage.tool_calls,
        input_tokens=usage.input_tokens or 0,
        cache_write_tokens=usage.cache_write_tokens or 0,
        cache_read_tokens=usage.cache_read_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = ["TuningAgentRun", "build_tuning_agent", "run_tuning_agent"]
