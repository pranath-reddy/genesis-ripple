"""Pre-training architecture judge, mirroring the AI Scientist search phase."""

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
    ArchitectureJudgeVerdict,
    ArchitectureSearchPlan,
    DatasetSummary,
)


_SYSTEM_PROMPT = """\
You rank architecture candidates before short training in the RIPPLe search
workflow. This is a proposal-stage comparison; measured validation metrics later
determine the winner in ordinary code.

PROCEDURE — complete these steps in order:
1. Call `inspect_dataset_contract` for dataset size, tensor shape, classes,
   split sizes, and qualification purpose.
2. Call `inspect_candidate_specs` for every proposed family, depth, width, and
   rationale.
3. Call `inspect_short_training_budget` for the exact shortlist size and smoke
   limits.
4. Rank every candidate and return the leading candidates as the shortlist.

JUDGING RULES:
- Use only the supplied dataset numbers and candidate specifications.
- Consider capacity relative to sample count, structural diversity, and whether
  the candidate can produce a useful plumbing comparison.
- Do not assume one family is inherently superior and do not cite external
  benchmark knowledge.
- Include every candidate exactly once in `ranking`.
- `shortlist` must be the first N ranked identifiers, where N is returned by the
  budget tool.
- For an integration smoke, set smoke_only=true. Ranking is operational and
  cannot be described as a scientific model-selection result.

OUTPUT: provide the full identifier ranking, the exact prefix shortlist, and a
short evidence-bounded justification.\
"""


@dataclass
class ArchitectureJudgeDependencies:
    dataset: DatasetSummary
    plan: ArchitectureSearchPlan
    shortlist_size: int
    called_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ArchitectureJudgeRun:
    verdict: ArchitectureJudgeVerdict
    called_tools: tuple[str, ...]
    request_count: int
    tool_call_count: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    elapsed_seconds: float


def build_architecture_judge(model: Any, *, retries: int = 1) -> Any:
    agent = Agent(
        model=model,
        deps_type=ArchitectureJudgeDependencies,
        output_type=PromptedOutput(
            ArchitectureJudgeVerdict,
            name="ripple_architecture_judge_verdict",
            description="Full pre-training ranking and budget-sized shortlist.",
        ),
        system_prompt=_SYSTEM_PROMPT,
        retries=retries,
        name="ripple_architecture_judge",
    )

    @agent.tool
    def inspect_dataset_contract(
        ctx: RunContext[ArchitectureJudgeDependencies],
    ) -> dict[str, object]:
        """Return aggregate dataset facts and no pixel content."""

        ctx.deps.called_tools.append("inspect_dataset_contract")
        return ctx.deps.dataset.model_dump(mode="json")

    @agent.tool
    def inspect_candidate_specs(
        ctx: RunContext[ArchitectureJudgeDependencies],
    ) -> dict[str, object]:
        """Return every validated candidate proposed by the generator."""

        if "inspect_dataset_contract" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the dataset before candidate specifications.")
        ctx.deps.called_tools.append("inspect_candidate_specs")
        return {
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in ctx.deps.plan.candidates
            ]
        }

    @agent.tool
    def inspect_short_training_budget(
        ctx: RunContext[ArchitectureJudgeDependencies],
    ) -> dict[str, object]:
        """Return the code-owned number of candidates that may be trained."""

        if "inspect_candidate_specs" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect candidates before the training budget.")
        ctx.deps.called_tools.append("inspect_short_training_budget")
        return {
            "shortlist_size": ctx.deps.shortlist_size,
            "candidate_count": len(ctx.deps.plan.candidates),
            "purpose": ctx.deps.dataset.purpose,
        }

    return agent


def run_architecture_judge(
    dataset: DatasetSummary,
    plan: ArchitectureSearchPlan,
    *,
    shortlist_size: int,
    settings: BedrockAgentSettings | None = None,
) -> ArchitectureJudgeRun:
    if shortlist_size < 1 or shortlist_size > len(plan.candidates):
        raise ValueError("shortlist size must fit the proposed candidate count")
    resolved = expand_bedrock_agent_settings(
        settings or BedrockAgentSettings(),
        max_output_tokens=900,
        output_token_limit=3000,
        input_token_limit=18000,
        request_limit=7,
    )
    dependencies = ArchitectureJudgeDependencies(
        dataset=dataset,
        plan=plan,
        shortlist_size=shortlist_size,
    )
    started = time.monotonic()
    result = build_architecture_judge(
        build_bedrock_converse_model(resolved), retries=resolved.retries
    ).run_sync(
        "Inspect all three required tools, rank all candidates, and shortlist the exact budgeted prefix.",
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
        "inspect_candidate_specs",
        "inspect_short_training_budget",
    }
    if not required <= set(dependencies.called_tools):
        raise RuntimeError("architecture judge skipped a required evidence tool")
    proposed_ids = tuple(candidate.candidate_id for candidate in plan.candidates)
    if set(result.output.ranking) != set(proposed_ids) or len(
        result.output.ranking
    ) != len(proposed_ids):
        raise RuntimeError(
            "architecture judge ranking does not cover the candidate set"
        )
    if len(result.output.shortlist) != shortlist_size:
        raise RuntimeError("architecture judge changed the code-owned shortlist size")
    expected_smoke = dataset.purpose == "integration_smoke"
    if result.output.smoke_only is not expected_smoke:
        raise RuntimeError(
            "architecture judge changed the dataset qualification boundary"
        )
    usage = result.usage
    return ArchitectureJudgeRun(
        verdict=result.output,
        called_tools=tuple(dependencies.called_tools),
        request_count=usage.requests,
        tool_call_count=usage.tool_calls,
        input_tokens=usage.input_tokens or 0,
        cache_write_tokens=usage.cache_write_tokens or 0,
        cache_read_tokens=usage.cache_read_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        elapsed_seconds=elapsed_seconds,
    )
