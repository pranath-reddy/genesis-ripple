"""Tool-calling coordinator constrained by a persisted run state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.usage import UsageLimits

from ..providers import BedrockAgentSettings, build_bedrock_converse_model
from ..schemas.orchestration import CoordinatorDecision, RunState


_SYSTEM_PROMPT = """\
You choose the next transition in one RIPPLe gravitational-lens workflow. The
surrounding program, not you, executes scientific and infrastructure operations.

PROCEDURE — complete these steps in order on every turn:
1. Call `inspect_run_state` to obtain the branch, current phase, evidence,
   scientific gates, blockers, and resource use.
2. Call `list_next_allowed_actions` to obtain the code-owned action set.
3. Return one CoordinatorDecision whose `selected_action` exactly matches one
   action returned by that tool.

DECISION RULES:
- Treat tool results as authoritative; do not fill missing facts from general
  knowledge or the user prompt.
- A closed gate cannot be bypassed. Select an evidence-gathering action when one
  is offered; otherwise select the explicit blocked/stop action.
- Prefer the least costly action that advances the current phase without
  weakening a scientific contract.
- Do not repeat an action when the state records it as exhausted or uninformative.
- Use `no_action` only when it is explicitly returned by the action tool.

BOUNDARY: you never manipulate pixel arrays, simulate images, load checkpoints,
calculate metrics, call SSH/cloud services, or invent preprocessing parameters.
Those operations belong to approved deterministic tools.

OUTPUT: briefly identify the evidence used, the selected action, expected cost,
and unresolved risks. Do not propose a second action.\
"""


@dataclass
class CoordinatorDependencies:
    state: RunState
    called_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CoordinatorRun:
    decision: CoordinatorDecision
    called_tools: tuple[str, ...]
    request_count: int
    input_tokens: int
    output_tokens: int


def build_coordinator_agent(model: Any, *, retries: int = 1) -> Any:
    agent = Agent(
        model=model,
        deps_type=CoordinatorDependencies,
        output_type=PromptedOutput(
            CoordinatorDecision,
            name="ripple_coordinator_decision",
            description="One allowlisted next action with evidence and bounded rationale.",
        ),
        system_prompt=_SYSTEM_PROMPT,
        retries=retries,
        name="ripple_scientist_coordinator",
    )

    @agent.tool
    def inspect_run_state(
        ctx: RunContext[CoordinatorDependencies],
    ) -> dict[str, object]:
        """Return scientific gates, blockers, artifacts, and remaining budget."""

        ctx.deps.called_tools.append("inspect_run_state")
        state = ctx.deps.state
        return {
            "run_id": state.run_id,
            "branch": state.branch,
            "phase": state.phase,
            "status": state.status,
            "artifact_ids": [artifact.artifact_id for artifact in state.artifacts],
            "gates": [gate.model_dump(mode="json") for gate in state.gates],
            "blockers": list(state.blockers),
            "budget": state.budget.model_dump(mode="json"),
            "usage": state.usage.model_dump(mode="json"),
        }

    @agent.tool
    def list_next_allowed_actions(
        ctx: RunContext[CoordinatorDependencies],
    ) -> dict[str, object]:
        """Return the exact code-owned action allowlist for this state."""

        if "inspect_run_state" not in ctx.deps.called_tools:
            raise ModelRetry("Call inspect_run_state before requesting actions.")
        ctx.deps.called_tools.append("list_next_allowed_actions")
        actions = list(ctx.deps.state.next_allowed_actions)
        if not actions:
            actions = ["no_action"]
        return {
            "allowed_actions": actions,
            "selection_rule": "selected_action must exactly match one returned value",
        }

    return agent


def run_coordinator(
    state: RunState,
    *,
    settings: BedrockAgentSettings | None = None,
) -> CoordinatorRun:
    """Run one bounded live decision and verify that required tools were called."""

    resolved = settings or BedrockAgentSettings()
    dependencies = CoordinatorDependencies(state=state)
    agent = build_coordinator_agent(
        build_bedrock_converse_model(resolved), retries=resolved.retries
    )
    result = agent.run_sync(
        (
            "Inspect the persisted state using both required tools, then choose the "
            "single next action. Do not use knowledge outside tool results."
        ),
        deps=dependencies,
        usage_limits=UsageLimits(
            request_limit=resolved.request_limit,
            tool_calls_limit=min(4, resolved.request_limit * 2),
            input_tokens_limit=resolved.input_token_limit,
            output_tokens_limit=resolved.output_token_limit,
            total_tokens_limit=resolved.total_token_limit,
            count_tokens_before_request=False,
        ),
    )
    required = {"inspect_run_state", "list_next_allowed_actions"}
    if not required <= set(dependencies.called_tools):
        raise RuntimeError("coordinator returned without calling both required tools")
    allowed = set(state.next_allowed_actions) or {"no_action"}
    if result.output.selected_action not in allowed:
        raise RuntimeError("coordinator selected an action outside persisted state")
    usage = result.usage
    return CoordinatorRun(
        decision=result.output,
        called_tools=tuple(dependencies.called_tools),
        request_count=usage.requests,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )
