"""Append-only state transitions and allowlisted agent/tool dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from ..agents.coordinator import CoordinatorRun, run_coordinator
from ..artifacts import ArtifactStore, sha256_file
from ..providers import BedrockAgentSettings
from ..schemas.common import BudgetUsage, canonical_json_sha256
from ..schemas.orchestration import (
    DecisionRecord,
    PipelineRunRequest,
    RunState,
    ToolOutcome,
)


class WorkflowStateError(RuntimeError):
    pass


ToolCallable = Callable[[RunState], ToolOutcome]


def _budget_add(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    return BudgetUsage(
        llm_requests=left.llm_requests + right.llm_requests,
        tool_calls=left.tool_calls + right.tool_calls,
        simulations=left.simulations + right.simulations,
        training_runs=left.training_runs + right.training_runs,
        gpu_seconds=left.gpu_seconds + right.gpu_seconds,
        storage_bytes=left.storage_bytes + right.storage_bytes,
    )


def account_state_journal_storage(state: RunState) -> RunState:
    """Charge this state's exact serialized bytes once to its storage usage.

    The serialized size depends on the decimal representation of
    ``usage.storage_bytes`` itself. Iterate to the small fixed point before the
    immutable state is written. Callers must pass a not-yet-written state whose
    usage already includes all earlier state files and newly created artifacts.
    """

    base_storage_bytes = state.usage.storage_bytes
    state_file_bytes = 0
    payload = state.model_dump(mode="python")
    for _ in range(16):
        usage = BudgetUsage(
            llm_requests=state.usage.llm_requests,
            tool_calls=state.usage.tool_calls,
            simulations=state.usage.simulations,
            training_runs=state.usage.training_runs,
            gpu_seconds=state.usage.gpu_seconds,
            storage_bytes=base_storage_bytes + state_file_bytes,
        )
        payload["usage"] = usage
        candidate = RunState.model_validate(payload, strict=True)
        measured_bytes = len(ArtifactStore.json_bytes(candidate))
        if measured_bytes == state_file_bytes:
            return candidate
        state_file_bytes = measured_bytes
    raise WorkflowStateError("state journal byte accounting did not converge")


def create_initial_state(request: PipelineRunRequest, *, run_id: str) -> RunState:
    first_action = {
        "mriganka_dp2": "verify_observation",
        "researcher_model": "inventory_researcher_model",
        "simulation_training": "plan_simulation",
    }[request.branch]
    return RunState(
        run_id=run_id,
        request_id=request.request_id,
        branch=request.branch,
        phase="initialized",
        status="created",
        sequence=0,
        request_sha256=canonical_json_sha256(request),
        budget=request.budget,
        next_allowed_actions=(first_action,),
    )


class StateJournal:
    def __init__(self, root: str | Path) -> None:
        self.store = ArtifactStore(root, create=False)

    def append(self, state: RunState) -> Path:
        return self.store.write_json(f"states/state-{state.sequence:04d}.json", state)

    def state_sha256(self, state: RunState) -> str:
        path = self.store.resolve(f"states/state-{state.sequence:04d}.json")
        return sha256_file(path)


@dataclass(frozen=True)
class WorkflowStep:
    coordinator: CoordinatorRun
    outcome: ToolOutcome
    state: RunState


class AgenticWorkflow:
    """LLM chooses an allowlisted action; code executes and validates it."""

    def __init__(
        self,
        *,
        tools: Mapping[str, ToolCallable],
        journal: StateJournal,
        provider_settings: BedrockAgentSettings | None = None,
    ) -> None:
        if not tools:
            raise ValueError("workflow requires at least one approved tool")
        self._tools = dict(tools)
        self._journal = journal
        self._provider_settings = provider_settings

    def start(self, state: RunState) -> RunState:
        if state.sequence != 0 or state.status != "created":
            raise WorkflowStateError("only a new state can start a journal")
        unknown = set(state.next_allowed_actions) - set(self._tools)
        if unknown:
            raise WorkflowStateError("initial state exposes an unregistered tool")
        charged_state = account_state_journal_storage(state)
        self._journal.append(charged_state)
        return charged_state

    def step(self, state: RunState) -> WorkflowStep:
        if state.status in {"blocked", "failed", "complete"}:
            raise WorkflowStateError("terminal workflow state cannot advance")
        coordinator = run_coordinator(state, settings=self._provider_settings)
        action = coordinator.decision.selected_action
        if action not in state.next_allowed_actions or action not in self._tools:
            raise WorkflowStateError(
                "coordinator action is not executable in this state"
            )
        outcome = self._tools[action](state)
        unknown = set(outcome.next_allowed_actions) - set(self._tools)
        if unknown:
            raise WorkflowStateError("tool outcome exposed an unregistered action")
        decision = DecisionRecord(
            decision_id=f"decision-{state.sequence + 1:04d}",
            agent_name="ripple-scientist-coordinator",
            phase=state.phase,
            observed_evidence_ids=coordinator.decision.evidence_ids,
            allowed_actions=state.next_allowed_actions,
            selected_action=action,
            rationale=coordinator.decision.rationale,
            expected_cost=coordinator.decision.expected_cost,
            created_at_utc=datetime.now(timezone.utc),
        )
        llm_usage = BudgetUsage(
            llm_requests=coordinator.request_count,
            tool_calls=len(coordinator.called_tools),
        )
        usage = _budget_add(_budget_add(state.usage, llm_usage), outcome.usage_delta)
        next_state = RunState(
            run_id=state.run_id,
            request_id=state.request_id,
            branch=state.branch,
            phase=outcome.phase,
            status=outcome.status,
            sequence=state.sequence + 1,
            request_sha256=state.request_sha256,
            previous_state_sha256=self._journal.state_sha256(state),
            artifacts=state.artifacts + outcome.artifacts,
            decisions=state.decisions + (decision,),
            budget=state.budget,
            usage=usage,
            gates=outcome.gates,
            blockers=outcome.blockers,
            next_allowed_actions=outcome.next_allowed_actions,
        )
        next_state = account_state_journal_storage(next_state)
        self._journal.append(next_state)
        return WorkflowStep(coordinator=coordinator, outcome=outcome, state=next_state)
