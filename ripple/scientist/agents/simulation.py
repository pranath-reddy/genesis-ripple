"""Tool-calling plan gate for the SLSim simulation route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field
from pydantic_ai import Agent, ModelRetry, PromptedOutput, RunContext
from pydantic_ai.usage import UsageLimits

from ..providers import (
    BedrockAgentSettings,
    build_bedrock_converse_model,
    expand_bedrock_agent_settings,
)
from ..schemas.common import ComputeBudget, FrozenModel, SHA256_PATTERN
from ..schemas.simulation import SlsimSmokeSpec, canonical_spec_sha256


class SimulationPlanDecision(FrozenModel):
    action: Literal["generate_smoke", "block"]
    spec_sha256: str = Field(pattern=SHA256_PATTERN)
    backend: Literal["slsim_smoke_backend"] = "slsim_smoke_backend"
    expected_image_count: int = Field(ge=1)
    expected_classes: tuple[Literal["non_lens", "lens"], Literal["non_lens", "lens"]]
    rationale: str = Field(min_length=1, max_length=1600)
    caveats: tuple[str, ...]
    supports_scientific_claims: Literal[False] = False


_SYSTEM_PROMPT = """\
You decide whether one already-typed SLSim request may enter the deterministic
simulation tool. You do not alter the request and you do not generate pixels.

PROCEDURE — complete these steps in order:
1. Call `inspect_simulation_spec` for class counts, bands, instrument settings,
   seed policy, purpose, and immutable specification digest.
2. Call `inspect_simulator_contract` for the pinned executable APIs and source
   revisions implemented by the backend.
3. Call `check_simulation_budget` to compare the exact requested count with the
   run budget.
4. Return one SimulationPlanDecision.

ACTIONS:
- `generate_smoke`: only when the typed specification, backend contract, and
  budget all agree.
- `block`: when any tool reports a mismatch, missing contract, or exhausted
  budget.

RULES:
- Preserve the digest returned by the specification tool exactly.
- Do not invent bands, PSF, noise, redshifts, geometry, counts, or seeds.
- The supported smoke contains both lenses and projected non-lens false
  positives. It is not an observationally representative population.
- Set supports_scientific_claims=false and state the smoke limitation in
  caveats, even when generation is allowed.
- Do not ask for a second confirmation in prose; workflow authorization is
  handled outside this agent.

OUTPUT: choose one action and explain it only from tool evidence.\
"""


@dataclass
class SimulationPlannerDependencies:
    spec: SlsimSmokeSpec
    budget: ComputeBudget
    called_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationPlannerRun:
    decision: SimulationPlanDecision
    called_tools: tuple[str, ...]
    request_count: int
    input_tokens: int
    output_tokens: int


def build_simulation_planner(model: Any, *, retries: int = 1) -> Any:
    agent = Agent(
        model=model,
        deps_type=SimulationPlannerDependencies,
        output_type=PromptedOutput(
            SimulationPlanDecision,
            name="ripple_simulation_plan_decision",
            description="One bounded decision for the typed SLSim request.",
        ),
        system_prompt=_SYSTEM_PROMPT,
        retries=retries,
        name="ripple_simulation_planner",
    )

    @agent.tool
    def inspect_simulation_spec(
        ctx: RunContext[SimulationPlannerDependencies],
    ) -> dict[str, object]:
        """Return the immutable scientific request summary and digest."""

        ctx.deps.called_tools.append("inspect_simulation_spec")
        spec = ctx.deps.spec
        return {
            "spec_sha256": canonical_spec_sha256(spec),
            "purpose": spec.purpose,
            "scientific_status": spec.scientific_status,
            "supports_scientific_claims": spec.supports_scientific_claims,
            "lens_count": spec.lens_count,
            "non_lens_count": spec.non_lens_count,
            "bands": list(spec.rendering.bands),
            "num_pix": spec.rendering.num_pix,
            "pixel_scale_arcsec": spec.rendering.pixel_scale_arcsec,
            "psf_fwhm_arcsec": spec.rendering.psf_fwhm_arcsec,
            "noise_model": spec.rendering.noise_model,
            "master_seed": spec.master_seed,
        }

    @agent.tool
    def inspect_simulator_contract(
        ctx: RunContext[SimulationPlannerDependencies],
    ) -> dict[str, object]:
        """Return the executable API and pinned source identities."""

        if "inspect_simulation_spec" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the simulation specification first.")
        ctx.deps.called_tools.append("inspect_simulator_contract")
        provenance = ctx.deps.spec.provenance
        return {
            "backend": "slsim_smoke_backend",
            "lens_api": "slsim.Lenses.lens.Lens",
            "non_lens_api": "slsim.FalsePositives.false_positive.FalsePositive",
            "renderer_api": ctx.deps.spec.rendering.renderer,
            "slsim_revision": provenance.slsim_source.git_commit,
            "jaxtronomy_revision": provenance.jaxtronomy_source.git_commit,
            "jax_execution_enabled": ctx.deps.spec.lensing_geometry.use_jax_lens_models,
            "tutorial_revision": provenance.tutorial_code.git_commit,
            "output_contract": "numeric float32 CHW arrays with content hashes",
        }

    @agent.tool
    def check_simulation_budget(
        ctx: RunContext[SimulationPlannerDependencies],
    ) -> dict[str, object]:
        """Compare the exact request with the code-owned simulation limit."""

        if "inspect_simulator_contract" not in ctx.deps.called_tools:
            raise ModelRetry("Inspect the simulator contract before checking budget.")
        ctx.deps.called_tools.append("check_simulation_budget")
        requested = ctx.deps.spec.lens_count + ctx.deps.spec.non_lens_count
        return {
            "requested_simulations": requested,
            "maximum_simulations": ctx.deps.budget.max_simulations,
            "within_budget": requested <= ctx.deps.budget.max_simulations,
        }

    return agent


def run_simulation_planner(
    spec: SlsimSmokeSpec,
    budget: ComputeBudget,
    *,
    settings: BedrockAgentSettings | None = None,
) -> SimulationPlannerRun:
    resolved = expand_bedrock_agent_settings(
        settings or BedrockAgentSettings(),
        max_output_tokens=900,
        output_token_limit=3000,
        input_token_limit=18000,
        request_limit=7,
    )
    dependencies = SimulationPlannerDependencies(spec=spec, budget=budget)
    result = build_simulation_planner(
        build_bedrock_converse_model(resolved), retries=resolved.retries
    ).run_sync(
        "Inspect all three required tools in order and decide whether the exact smoke request can run.",
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
    required = {
        "inspect_simulation_spec",
        "inspect_simulator_contract",
        "check_simulation_budget",
    }
    if not required <= set(dependencies.called_tools):
        raise RuntimeError("simulation planner skipped a required evidence tool")
    expected_digest = canonical_spec_sha256(spec)
    if result.output.spec_sha256 != expected_digest:
        raise RuntimeError("simulation planner changed the specification identity")
    expected_count = spec.lens_count + spec.non_lens_count
    if result.output.expected_image_count != expected_count:
        raise RuntimeError("simulation planner changed the requested image count")
    if result.output.expected_classes != ("non_lens", "lens"):
        raise RuntimeError("simulation planner changed the binary label contract")
    within_budget = expected_count <= budget.max_simulations
    if (result.output.action == "generate_smoke") is not within_budget:
        raise RuntimeError(
            "simulation planner action disagrees with deterministic budget check"
        )
    usage = result.usage
    return SimulationPlannerRun(
        decision=result.output,
        called_tools=tuple(dependencies.called_tools),
        request_count=usage.requests,
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )
