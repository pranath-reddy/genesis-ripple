"""End-to-end controller for the ten-image simulation/training smoke.

Language-model calls remain in the local control plane.  The remote host only
receives validated JSON inputs and executes the deterministic worker CLI.  The
controller persists each typed decision, remote invocation, copied scientific
artifact, state transition, and final technical report under one immutable run
directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, JsonValue, TypeAdapter

from ..agents import (
    run_architecture_judge,
    run_architecture_planner,
    run_simulation_planner,
)
from ..artifacts import ArtifactStore, sha256_file
from ..paths import checked_absolute_path, checked_real_directory, checked_real_file
from ..providers import BedrockAgentSettings, load_bedrock_agent_settings
from ..schemas.architecture import (
    ArchitectureEvaluation,
    ArchitectureSearchResult,
    BoundArchitecture,
    DatasetSummary,
)
from ..schemas.campaign import (
    AgentStageEvidence,
    RemoteStageEvidence,
    ScientistSourceRevisions,
    SimulationCampaignCompletion,
    SimulationCampaignConfiguration,
    SourceSyncEvidence,
    SyncedSourceComponent,
)
from ..schemas.common import (
    ArtifactRef,
    BudgetUsage,
    ScientificGate,
    canonical_json_sha256,
    utc_now,
)
from ..schemas.dataset import DatasetManifest
from ..schemas.orchestration import DecisionRecord, RunState
from ..schemas.remote import RemoteWorkerSettings
from ..schemas.report import SyntheticTrainingSmokeEvidence
from ..schemas.simulation import SlsimSmokeSpec, canonical_spec_sha256
from ..schemas.training import FinalEvaluationRecord, TrainingRunRecord
from ..tools import (
    copy_remote_output,
    copy_to_remote_run,
    load_dataset_manifest,
    load_smoke_spec,
    run_remote_worker,
    sync_worker_source,
    SourceTreeDigest,
)
from ..tools.model_builders import RIPPLE_MODEL_BUILDERS_REVISION
from ..tools.report import assemble_synthetic_training_technical_report
from .state_machine import (
    StateJournal,
    account_state_journal_storage,
    create_initial_state,
)


_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
_AGENT_STAGE_COUNT = 3
_AGENT_STAGE_REQUEST_FLOOR = 7
_AGENT_STAGE_TOOL_CALLS = 3
_CAMPAIGN_FIXED_TOOL_CALLS = 12
_CAMPAIGN_MINIMUM_STORAGE_BYTES = 64 * 1024 * 1024
_ACTION_STORAGE_RESERVATION_BYTES = 8 * 1024 * 1024
_EXPECTED_RUNTIME_SOURCE_LOCKS = {
    "slsim": (
        "https://github.com/LSST-strong-lensing/slsim.git",
        "ad62eefb74944f7ee76abf826d8c6329a5894d4b",
        "ripple/scientist/vendor/slsim/slsim",
        "lens, false-positive, and LSST image simulation",
    ),
    "JAXtronomy": (
        "https://github.com/lenstronomy/JAXtronomy.git",
        "a268deaf08dcabfb2919c480acbf39cc6596bf87",
        "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
        "SLSim import compatibility; JAX model execution remains disabled",
    ),
}
_EXPECTED_PRECEDENT_REPOSITORIES = {
    "DeepLense-AI-Scientist": (
        "https://github.com/ML4SCI/DeepLense-AI-Scientist.git",
        "dbea8485f8bf64250e2aec82dd8999d354085996",
        "ripple/scientist/tools/model_builders.py",
        "ripple-model-builders-v1",
    ),
    "slsim-tutorials": (
        "https://github.com/LSST-strong-lensing/slsim-tutorials.git",
        "cbf2165f35c29278b255e4b679d51c84cf86e5d1",
        "design reference for the bounded single-lens and false-positive smoke specification",
    ),
}


class CampaignExecutionError(RuntimeError):
    """Credential-free campaign failure with a local evidence location."""


@dataclass(frozen=True)
class CompletedSimulationCampaign:
    run_directory: Path
    completion_path: Path
    report_path: Path
    final_state_path: Path
    completion: SimulationCampaignCompletion
    final_state: RunState


def _add_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    return BudgetUsage(
        llm_requests=left.llm_requests + right.llm_requests,
        tool_calls=left.tool_calls + right.tool_calls,
        simulations=left.simulations + right.simulations,
        training_runs=left.training_runs + right.training_runs,
        gpu_seconds=left.gpu_seconds + right.gpu_seconds,
        storage_bytes=left.storage_bytes + right.storage_bytes,
    )


def _safe_run_id(request_id: str) -> str:
    suffix = uuid.uuid4().hex[:12]
    prefix = request_id[:100].rstrip("._-")
    value = f"{prefix}-{suffix}"
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("could not derive a safe campaign run ID")
    return value


def _tree_digest(root: Path) -> tuple[str, int]:
    """Hash the exact regular-file set rsync transfers from one source tree."""

    try:
        root_state = os.lstat(root)
    except FileNotFoundError:
        raise CampaignExecutionError("a required vendored source tree is unavailable")
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        raise CampaignExecutionError("a required vendored source tree is unavailable")
    files: list[Path] = []
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        directory_path = Path(directory)
        retained_names: list[str] = []
        for name in names:
            path = directory_path / name
            details = os.lstat(path)
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise CampaignExecutionError(
                    "source synchronization only permits regular files and directories"
                )
            if name != "__pycache__":
                retained_names.append(name)
        names[:] = retained_names
        for name in filenames:
            path = directory_path / name
            details = os.lstat(path)
            if not stat.S_ISREG(details.st_mode):
                raise CampaignExecutionError(
                    "source synchronization only permits regular files and directories"
                )
            if path.suffix != ".pyc":
                files.append(path)
    files.sort(key=lambda value: value.relative_to(root).as_posix())
    if not files:
        raise CampaignExecutionError("a required source tree has no regular files")
    digest = hashlib.sha256()
    count = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise CampaignExecutionError(
                    "source file changed type while it was being hashed"
                )
            content_length = details.st_size
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(content_length.to_bytes(8, "big"))
            consumed = 0
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                while chunk := handle.read(1024 * 1024):
                    consumed += len(chunk)
                    digest.update(chunk)
            if consumed != content_length:
                raise CampaignExecutionError(
                    "source file changed while it was being hashed"
                )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        count += 1
    return digest.hexdigest(), count


def _directory_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def _last_json_object(stdout: str) -> dict[str, JsonValue]:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return _JSON_OBJECT.validate_python(decoded, strict=True)
    raise CampaignExecutionError("remote worker returned no final JSON object")


class SimulationCampaignRunner:
    """Run the bounded smoke campaign and persist an append-only evidence chain."""

    def __init__(
        self,
        configuration: SimulationCampaignConfiguration,
        *,
        configuration_base: str | os.PathLike[str] | None = None,
        repository_root: str | os.PathLike[str] | None = None,
        provider_settings: BedrockAgentSettings | None = None,
        run_id: str | None = None,
    ) -> None:
        self.configuration = configuration
        self.configuration_base = checked_real_directory(
            configuration_base or Path.cwd()
        )
        self.repository_root = checked_real_directory(
            Path(repository_root or Path(__file__).resolve().parents[3])
        )
        self.provider_settings = provider_settings or load_bedrock_agent_settings()
        self._agent_request_ceiling = max(
            _AGENT_STAGE_REQUEST_FLOOR,
            self.provider_settings.request_limit,
        )
        preflight_usage = BudgetUsage(
            llm_requests=_AGENT_STAGE_COUNT * self._agent_request_ceiling,
            tool_calls=(
                _CAMPAIGN_FIXED_TOOL_CALLS
                + _AGENT_STAGE_COUNT * _AGENT_STAGE_TOOL_CALLS
                + 3 * configuration.architecture_shortlist_size
            ),
            simulations=10,
            training_runs=configuration.architecture_shortlist_size,
            gpu_seconds=float(configuration.architecture_shortlist_size + 1),
            storage_bytes=_CAMPAIGN_MINIMUM_STORAGE_BYTES,
        )
        if not preflight_usage.fits(configuration.request.budget):
            raise CampaignExecutionError(
                "campaign budget cannot cover the code-owned worst-case action plan"
            )
        self.run_id = run_id or _safe_run_id(configuration.request.request_id)
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a normalized campaign identifier")

        output_root = Path(configuration.request.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = self.configuration_base / output_root
        output_root = checked_real_directory(
            checked_absolute_path(output_root),
            create=True,
        )
        self.run_directory = output_root / self.run_id
        try:
            self.run_directory.mkdir(mode=0o700)
        except FileExistsError:
            raise CampaignExecutionError("campaign run directory already exists")
        except OSError as exc:
            raise CampaignExecutionError(
                f"campaign run directory creation failed ({type(exc).__name__})"
            ) from None
        run_state = os.lstat(self.run_directory)
        if (
            not stat.S_ISDIR(run_state.st_mode)
            or run_state.st_uid != os.getuid()
            or stat.S_IMODE(run_state.st_mode) & 0o077
        ):
            raise CampaignExecutionError(
                "campaign run directory is not private and caller-owned"
            )

        self.store = ArtifactStore(self.run_directory, create=False)
        self.journal = StateJournal(self.run_directory)
        self.state = account_state_journal_storage(
            create_initial_state(configuration.request, run_id=self.run_id)
        )
        self.journal.append(self.state)
        self._pending_artifacts: list[ArtifactRef] = []
        self._accounted_storage_bytes = _directory_bytes(self.run_directory)
        self._failure_usage_charge = BudgetUsage()
        self._action_storage_reservation_bytes = 0
        self._source_sync: SourceSyncEvidence | None = None
        self._source_sync_reference: ArtifactRef | None = None

        request = configuration.request
        self.remote = RemoteWorkerSettings(
            host=request.gpu_host,
            python=request.gpu_python,
            remote_root=request.gpu_remote_root,
        )
        self.remote_run_root = f"{self.remote.remote_root}/runs/{self.run_id}"

    def _require_budget_capacity(
        self,
        action: str,
        usage_delta: BudgetUsage,
    ) -> None:
        current_bytes = _directory_bytes(self.run_directory)
        unaccounted_storage = max(0, current_bytes - self._accounted_storage_bytes)
        projected_delta = BudgetUsage(
            llm_requests=usage_delta.llm_requests,
            tool_calls=usage_delta.tool_calls,
            simulations=usage_delta.simulations,
            training_runs=usage_delta.training_runs,
            gpu_seconds=usage_delta.gpu_seconds,
            storage_bytes=usage_delta.storage_bytes + unaccounted_storage,
        )
        if not _add_usage(self.state.usage, projected_delta).fits(self.state.budget):
            raise CampaignExecutionError(
                f"budget exhausted before approved action: {action}"
            )

    def _remaining_storage_bytes(self) -> int:
        return max(
            0,
            self.state.budget.max_storage_bytes
            - _directory_bytes(self.run_directory)
            - self._action_storage_reservation_bytes,
        )

    def _prepare_action(self, action: str, usage_charge: BudgetUsage) -> None:
        """Reserve evidence capacity and retain a conservative failure charge."""

        capacity = BudgetUsage(
            llm_requests=usage_charge.llm_requests,
            tool_calls=usage_charge.tool_calls,
            simulations=usage_charge.simulations,
            training_runs=usage_charge.training_runs,
            gpu_seconds=usage_charge.gpu_seconds,
            storage_bytes=_ACTION_STORAGE_RESERVATION_BYTES,
        )
        self._require_budget_capacity(action, capacity)
        self._failure_usage_charge = usage_charge
        self._action_storage_reservation_bytes = _ACTION_STORAGE_RESERVATION_BYTES

    def _set_failure_usage_charge(self, usage_charge: BudgetUsage) -> None:
        """Update the conservative debit while a multi-action stage is running."""

        capacity = BudgetUsage(
            llm_requests=usage_charge.llm_requests,
            tool_calls=usage_charge.tool_calls,
            simulations=usage_charge.simulations,
            training_runs=usage_charge.training_runs,
            gpu_seconds=usage_charge.gpu_seconds,
            storage_bytes=self._action_storage_reservation_bytes,
        )
        self._require_budget_capacity("update_inflight_usage", capacity)
        self._failure_usage_charge = usage_charge

    def _remaining_gpu_seconds(self, *, pending_seconds: float = 0.0) -> float:
        return (
            self.state.budget.max_gpu_seconds
            - self.state.usage.gpu_seconds
            - pending_seconds
        )

    def _resolve_input(self, configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = self.configuration_base / path
        try:
            return checked_real_file(path)
        except ValueError:
            raise CampaignExecutionError(
                "campaign input must be a real local file"
            ) from None

    def _write_artifact(
        self,
        *,
        artifact_id: str,
        role: str,
        relative_path: str,
        value: BaseModel | dict[str, Any],
        producer: str,
        input_artifact_ids: tuple[str, ...] = (),
    ) -> tuple[Path, ArtifactRef]:
        self._require_budget_capacity(
            f"write_artifact:{artifact_id}",
            BudgetUsage(storage_bytes=len(ArtifactStore.json_bytes(value))),
        )
        path = self.store.write_json(relative_path, value)
        reference = self.store.reference(
            artifact_id=artifact_id,
            role=role,
            relative_path=relative_path,
            media_type="application/json",
            producer=producer,
            configuration=self.configuration,
            input_artifact_ids=input_artifact_ids,
        )
        self._pending_artifacts.append(reference)
        return path, reference

    def _reference_file(
        self,
        *,
        artifact_id: str,
        role: str,
        path: Path,
        media_type: str,
        producer: str,
        input_artifact_ids: tuple[str, ...] = (),
    ) -> ArtifactRef:
        resolved = checked_real_file(path)
        relative = resolved.relative_to(self.run_directory).as_posix()
        reference = self.store.reference(
            artifact_id=artifact_id,
            role=role,
            relative_path=relative,
            media_type=media_type,
            producer=producer,
            configuration=self.configuration,
            input_artifact_ids=input_artifact_ids,
        )
        self._pending_artifacts.append(reference)
        return reference

    def _append_state(
        self,
        *,
        phase: str,
        status: str = "running",
        next_allowed_actions: tuple[str, ...] = (),
        usage_delta: BudgetUsage | None = None,
        decisions: tuple[DecisionRecord, ...] = (),
        gates: tuple[ScientificGate, ...] | None = None,
        blockers: tuple[str, ...] = (),
    ) -> RunState:
        self._require_budget_capacity(
            f"append_state:{phase}",
            usage_delta or BudgetUsage(),
        )
        current_bytes = _directory_bytes(self.run_directory)
        storage_delta = max(0, current_bytes - self._accounted_storage_bytes)
        delta = usage_delta or BudgetUsage()
        delta = BudgetUsage(
            llm_requests=delta.llm_requests,
            tool_calls=delta.tool_calls,
            simulations=delta.simulations,
            training_runs=delta.training_runs,
            gpu_seconds=delta.gpu_seconds,
            storage_bytes=delta.storage_bytes + storage_delta,
        )
        artifacts = tuple(self._pending_artifacts)
        next_state = RunState(
            run_id=self.state.run_id,
            request_id=self.state.request_id,
            branch=self.state.branch,
            phase=phase,
            status=status,
            sequence=self.state.sequence + 1,
            request_sha256=self.state.request_sha256,
            previous_state_sha256=self.journal.state_sha256(self.state),
            artifacts=self.state.artifacts + artifacts,
            decisions=self.state.decisions + decisions,
            budget=self.state.budget,
            usage=_add_usage(self.state.usage, delta),
            gates=self.state.gates if gates is None else gates,
            blockers=blockers,
            next_allowed_actions=next_allowed_actions,
        )
        next_state = account_state_journal_storage(next_state)
        self.journal.append(next_state)
        self.state = next_state
        self._pending_artifacts.clear()
        self._accounted_storage_bytes = _directory_bytes(self.run_directory)
        self._failure_usage_charge = BudgetUsage()
        self._action_storage_reservation_bytes = 0
        return next_state

    def _decision(
        self,
        *,
        agent_name: str,
        phase: str,
        allowed_actions: tuple[str, ...],
        selected_action: str,
        rationale: str,
        expected_cost: str,
        evidence_ids: tuple[str, ...],
    ) -> DecisionRecord:
        return DecisionRecord(
            decision_id=f"decision-{self.state.sequence + 1:04d}",
            agent_name=agent_name,
            phase=phase,
            observed_evidence_ids=evidence_ids,
            allowed_actions=allowed_actions,
            selected_action=selected_action,
            rationale=rationale,
            expected_cost=expected_cost,
            created_at_utc=utc_now(),
        )

    def _agent_evidence(
        self,
        *,
        stage: str,
        output_reference: ArtifactRef,
        called_tools: tuple[str, ...],
        request_count: int,
        input_tokens: int,
        output_tokens: int,
    ) -> ArtifactRef:
        evidence = AgentStageEvidence(
            stage=stage,
            output_artifact_id=output_reference.artifact_id,
            output_sha256=output_reference.sha256,
            called_tools=called_tools,
            request_count=request_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        _, reference = self._write_artifact(
            artifact_id=f"{stage}-agent-evidence",
            role="agent-stage-evidence",
            relative_path=f"evidence/agents/{stage}.json",
            value=evidence,
            producer="ripple-campaign-runner",
            input_artifact_ids=(output_reference.artifact_id,),
        )
        return reference

    def _invoke_remote(
        self,
        *,
        operation: str,
        arguments: Sequence[str],
        slug: str,
        execution_timeout_seconds: int | None = None,
    ) -> tuple[dict[str, JsonValue], ArtifactRef, float]:
        if self._source_sync is None or self._source_sync_reference is None:
            raise CampaignExecutionError(
                "remote execution requires verified source-sync evidence"
            )
        result = run_remote_worker(
            settings=self.remote,
            operation=operation,  # type: ignore[arg-type]
            arguments=arguments,
            execution_timeout_seconds=execution_timeout_seconds,
            source_sync=self._source_sync,
        )
        _, command_reference = self._write_artifact(
            artifact_id=f"remote-{slug}-command",
            role="remote-command-result",
            relative_path=f"evidence/remote/{slug}-command.json",
            value=result,
            producer="ripple-remote-executor",
            input_artifact_ids=(self._source_sync_reference.artifact_id,),
        )
        if not result.succeeded or result.exit_code != 0:
            raise CampaignExecutionError(
                f"remote {operation} failed; inspect {command_reference.relative_path}"
            )
        payload = _last_json_object(result.stdout)
        if (
            payload.get("verified_source_manifest_sha256")
            != self._source_sync.source_manifest_sha256
        ):
            raise CampaignExecutionError(
                "remote worker did not echo the verified source manifest"
            )
        evidence = RemoteStageEvidence(
            result=result,
            parsed_payload=payload,
            completed_at_utc=utc_now(),
        )
        _, evidence_reference = self._write_artifact(
            artifact_id=f"remote-{slug}-evidence",
            role="remote-stage-evidence",
            relative_path=f"evidence/remote/{slug}-evidence.json",
            value=evidence,
            producer="ripple-campaign-runner",
            input_artifact_ids=(command_reference.artifact_id,),
        )
        return payload, evidence_reference, result.elapsed_seconds

    @staticmethod
    def _require_payload(payload: dict[str, JsonValue], **expected: object) -> None:
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CampaignExecutionError(
                    f"remote worker payload violated the {key} contract"
                )

    @staticmethod
    def _payload_nonnegative_float(payload: dict[str, JsonValue], key: str) -> float:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CampaignExecutionError(
                f"remote worker payload violated the {key} numeric contract"
            )
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise CampaignExecutionError(
                f"remote worker payload violated the {key} numeric contract"
            )
        return parsed

    def _sync_sources(
        self,
        source_revisions: ScientistSourceRevisions,
        source_revisions_ref: ArtifactRef,
    ) -> ArtifactRef:
        local_components: list[SourceTreeDigest] = []
        definitions = (
            (
                "ripple_scientist",
                self.repository_root / "ripple" / "scientist",
                "ripple/scientist",
                "ripple/scientist",
            ),
            (
                "slsim",
                self.repository_root
                / "ripple"
                / "scientist"
                / "vendor"
                / "slsim"
                / "slsim",
                "ripple/scientist/vendor/slsim/slsim",
                "ripple/scientist/vendor/slsim/slsim",
            ),
            (
                "jaxtronomy",
                self.repository_root
                / "ripple"
                / "scientist"
                / "vendor"
                / "JAXtronomy"
                / "jaxtronomy",
                "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
                "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
            ),
        )
        runtime_locks = {item.name: item for item in source_revisions.runtime_sources}
        for name, path, local_relative, remote_relative in definitions:
            digest, count = _tree_digest(path)
            if name in {"slsim", "jaxtronomy"}:
                lock_name = "slsim" if name == "slsim" else "JAXtronomy"
                locked = runtime_locks[lock_name]
                if (
                    local_relative != locked.local_path
                    or digest != locked.transfer_tree_sha256
                ):
                    raise CampaignExecutionError(
                        f"{lock_name} transfer tree does not match its source lock"
                    )
            local_components.append(
                SourceTreeDigest(
                    component=name,
                    remote_relative_path=remote_relative,
                    sha256=digest,
                    regular_file_count=count,
                )
            )
        package_initializer = self.repository_root / "ripple" / "__init__.py"
        if not package_initializer.is_file() or package_initializer.is_symlink():
            raise CampaignExecutionError("ripple package initializer is unavailable")
        package_initializer_sha256 = sha256_file(package_initializer)
        verified = sync_worker_source(
            local_repository=self.repository_root,
            settings=self.remote,
            expected_components=tuple(local_components),
            expected_package_initializer_sha256=package_initializer_sha256,
        )
        remote_components = {item.component: item for item in verified.components}
        components = tuple(
            SyncedSourceComponent(
                component=local.component,
                local_relative_path=next(
                    definition[2]
                    for definition in definitions
                    if definition[0] == local.component
                ),
                remote_relative_path=local.remote_relative_path,
                source_tree_sha256=local.sha256,
                regular_file_count=local.regular_file_count,
                remote_tree_sha256=remote_components[local.component].sha256,
                remote_regular_file_count=(
                    remote_components[local.component].regular_file_count
                ),
            )
            for local in local_components
        )
        verified_at = utc_now()
        evidence = SourceSyncEvidence(
            host=self.remote.host,
            remote_root=self.remote.remote_root,
            components=components,
            package_initializer_sha256=package_initializer_sha256,
            remote_package_initializer_sha256=verified.package_initializer_sha256,
            source_manifest_sha256=verified.source_manifest_sha256,
            remote_source_root=verified.source_root,
            synchronized_at_utc=verified_at,
            verified_at_utc=verified_at,
        )
        _, reference = self._write_artifact(
            artifact_id="worker-source-sync",
            role="source-sync-evidence",
            relative_path="evidence/source-sync.json",
            value=evidence,
            producer="ripple-remote-executor",
            input_artifact_ids=(source_revisions_ref.artifact_id,),
        )
        self._source_sync = evidence
        self._source_sync_reference = reference
        return reference

    def _load_source_revisions(self) -> tuple[ScientistSourceRevisions, ArtifactRef]:
        lock_path = (
            self.repository_root
            / "ripple"
            / "scientist"
            / "locks"
            / "source-revisions.json"
        )
        if not lock_path.is_file() or lock_path.is_symlink():
            raise CampaignExecutionError(
                "scientist source-revision lock is unavailable"
            )
        revisions = ScientistSourceRevisions.model_validate_json(
            lock_path.read_bytes(), strict=True
        )
        runtime_locks = {item.name: item for item in revisions.runtime_sources}
        for name, expected in _EXPECTED_RUNTIME_SOURCE_LOCKS.items():
            observed = runtime_locks[name]
            if (
                observed.repository,
                observed.commit,
                observed.local_path,
                observed.runtime_role,
            ) != expected:
                raise CampaignExecutionError(
                    f"{name} source lock has a non-canonical runtime identity"
                )
        precedents = {item.name: item for item in revisions.source_precedent_only}
        precedent = precedents["DeepLense-AI-Scientist"]
        if (
            precedent.repository,
            precedent.commit,
            precedent.port_path,
            precedent.local_revision,
        ) != _EXPECTED_PRECEDENT_REPOSITORIES["DeepLense-AI-Scientist"]:
            raise CampaignExecutionError(
                "AI Scientist precedent lock has a non-canonical identity"
            )
        tutorial_precedent = precedents["slsim-tutorials"]
        if (
            tutorial_precedent.repository,
            tutorial_precedent.commit,
            tutorial_precedent.evidence_role,
        ) != _EXPECTED_PRECEDENT_REPOSITORIES["slsim-tutorials"]:
            raise CampaignExecutionError(
                "SLSim tutorial precedent lock has a non-canonical identity"
            )
        port_path = checked_real_file(self.repository_root / precedent.port_path)
        if (
            not port_path.is_file()
            or port_path.is_symlink()
            or self.repository_root not in port_path.parents
        ):
            raise CampaignExecutionError("local architecture builder is unavailable")
        if sha256_file(port_path) != precedent.port_sha256:
            raise CampaignExecutionError(
                "local architecture builder does not match its source lock"
            )
        _, reference = self._write_artifact(
            artifact_id="scientist-source-revisions",
            role="source-revision-lock",
            relative_path="inputs/source-revisions.json",
            value=revisions,
            producer="ripple-campaign-runner",
        )
        return revisions, reference

    def _load_and_persist_inputs(
        self,
    ) -> tuple[
        SlsimSmokeSpec,
        Path,
        ArtifactRef,
        ArtifactRef,
        ArtifactRef,
        ArtifactRef,
        ScientistSourceRevisions,
    ]:
        spec = load_smoke_spec(
            self._resolve_input(self.configuration.request.simulation_spec)
        )
        configuration_path, configuration_reference = self._write_artifact(
            artifact_id="campaign-configuration",
            role="campaign-configuration",
            relative_path="inputs/campaign.json",
            value=self.configuration,
            producer="ripple-campaign-runner",
        )
        del configuration_path
        spec_path, spec_reference = self._write_artifact(
            artifact_id="simulation-specification",
            role="simulation-specification",
            relative_path="inputs/simulation-spec.json",
            value=spec,
            producer="ripple-campaign-runner",
            input_artifact_ids=(configuration_reference.artifact_id,),
        )
        _, provider_reference = self._write_artifact(
            artifact_id="bedrock-runtime-identity",
            role="agent-runtime-identity",
            relative_path="inputs/bedrock-runtime-identity.json",
            value=self.provider_settings.runtime_identity(),
            producer="ripple-campaign-runner",
            input_artifact_ids=(configuration_reference.artifact_id,),
        )
        source_revisions, source_revisions_reference = self._load_source_revisions()
        runtime_revisions = {
            item.name: item.commit for item in source_revisions.runtime_sources
        }
        precedent_revisions = {
            item.name: item.commit for item in source_revisions.source_precedent_only
        }
        if (
            spec.provenance.slsim_source.git_commit != runtime_revisions["slsim"]
            or spec.provenance.jaxtronomy_source.git_commit
            != runtime_revisions["JAXtronomy"]
            or spec.provenance.tutorial_code.git_commit
            != precedent_revisions["slsim-tutorials"]
        ):
            raise CampaignExecutionError(
                "simulation specification disagrees with the source-revision lock"
            )
        return (
            spec,
            spec_path,
            configuration_reference,
            spec_reference,
            provider_reference,
            source_revisions_reference,
            source_revisions,
        )

    def _dataset_summary(
        self, manifest: DatasetManifest, manifest_sha256: str
    ) -> DatasetSummary:
        shape = manifest.samples[0].shape
        return DatasetSummary(
            dataset_id=manifest.dataset_id,
            manifest_sha256=manifest_sha256,
            image_shape=(shape[1], shape[2]),
            channels=shape[0],
            class_names=manifest.class_names,
            split_counts={
                "train": len(manifest.splits.train_sample_ids),
                "validation": len(manifest.splits.validation_sample_ids),
                "test": len(manifest.splits.test_sample_ids),
            },
            purpose=manifest.purpose,
            notes=manifest.qualification_notes,
        )

    def _fail(self, phase: str, exception: Exception) -> None:
        if self.state.status in {"blocked", "failed", "complete"}:
            return
        failure = {
            "schema_version": "ripple.simulation-campaign-failure.v1",
            "run_id": self.run_id,
            "phase": phase,
            "exception_type": type(exception).__name__,
            "safe_message": "campaign stage failed; inspect persisted stage evidence",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self._write_artifact(
                artifact_id=f"failure-{self.state.sequence + 1:04d}",
                role="campaign-failure",
                relative_path="failure.json",
                value=failure,
                producer="ripple-campaign-runner",
            )
            self._append_state(
                phase=phase,
                status="failed",
                usage_delta=self._failure_usage_charge,
                blockers=(
                    "campaign stage failed; inspect failure.json and stage evidence",
                ),
            )
        except Exception:
            # Preserve the original failure if even the final evidence write cannot run.
            return

    def run(self) -> CompletedSimulationCampaign:
        phase = "load_inputs"
        try:
            (
                spec,
                spec_path,
                config_ref,
                spec_ref,
                provider_ref,
                source_revisions_ref,
                source_revisions,
            ) = self._load_and_persist_inputs()

            phase = "plan_simulation"
            self._prepare_action(
                "plan_simulation",
                BudgetUsage(
                    llm_requests=self._agent_request_ceiling,
                    tool_calls=_AGENT_STAGE_TOOL_CALLS,
                ),
            )
            simulation_run = run_simulation_planner(
                spec,
                self.configuration.request.budget,
                settings=self.provider_settings,
            )
            _, simulation_plan_ref = self._write_artifact(
                artifact_id="simulation-plan",
                role="simulation-plan",
                relative_path="evidence/simulation-plan.json",
                value=simulation_run.decision,
                producer="ripple-simulation-planner",
                input_artifact_ids=(spec_ref.artifact_id, provider_ref.artifact_id),
            )
            simulation_agent_ref = self._agent_evidence(
                stage="simulation_planner",
                output_reference=simulation_plan_ref,
                called_tools=simulation_run.called_tools,
                request_count=simulation_run.request_count,
                input_tokens=simulation_run.input_tokens,
                output_tokens=simulation_run.output_tokens,
            )
            if simulation_run.decision.action != "generate_smoke":
                self._append_state(
                    phase="simulation_planning_blocked",
                    status="blocked",
                    usage_delta=BudgetUsage(
                        llm_requests=simulation_run.request_count,
                        tool_calls=len(simulation_run.called_tools),
                    ),
                    blockers=("typed simulation planner blocked execution",),
                )
                raise CampaignExecutionError(
                    "typed simulation planner blocked execution"
                )
            self._append_state(
                phase="simulation_planned",
                next_allowed_actions=("sync_worker_source",),
                usage_delta=BudgetUsage(
                    llm_requests=simulation_run.request_count,
                    tool_calls=len(simulation_run.called_tools),
                ),
                decisions=(
                    self._decision(
                        agent_name="ripple-simulation-planner",
                        phase="plan_simulation",
                        allowed_actions=("generate_smoke", "block"),
                        selected_action=simulation_run.decision.action,
                        rationale=simulation_run.decision.rationale,
                        expected_cost="ten deterministic SLSim renders",
                        evidence_ids=(
                            spec_ref.artifact_id,
                            simulation_agent_ref.artifact_id,
                        ),
                    ),
                ),
            )

            phase = "sync_worker_source"
            self._prepare_action(
                "sync_worker_source",
                BudgetUsage(tool_calls=1),
            )
            self._sync_sources(source_revisions, source_revisions_ref)
            self._append_state(
                phase="worker_source_synced",
                next_allowed_actions=("verify_worker_environment",),
                usage_delta=BudgetUsage(tool_calls=1),
            )

            phase = "verify_worker_environment"
            self._prepare_action(
                "verify_worker_environment",
                BudgetUsage(tool_calls=1),
            )
            environment_payload, environment_ref, _ = self._invoke_remote(
                operation="environment",
                arguments=("--require-cuda", "--require-scientific-stack"),
                slug="environment",
            )
            self._require_payload(
                environment_payload,
                cuda_available=True,
                timeout_available=True,
                worker_source=(
                    f"{self._source_sync.remote_source_root}/ripple/scientist/worker.py"
                ),
            )
            scientific_stack = environment_payload.get("scientific_stack")
            if not isinstance(scientific_stack, dict):
                raise CampaignExecutionError(
                    "remote environment omitted scientific source evidence"
                )
            expected_sources = {
                "slsim_source": (
                    f"{self._source_sync.remote_source_root}"
                    "/ripple/scientist/vendor/slsim/slsim/__init__.py"
                ),
                "jaxtronomy_source": (
                    f"{self._source_sync.remote_source_root}"
                    "/ripple/scientist/vendor/JAXtronomy/jaxtronomy/__init__.py"
                ),
            }
            if any(
                scientific_stack.get(key) != value
                for key, value in expected_sources.items()
            ):
                raise CampaignExecutionError(
                    "remote scientific imports did not use the verified source tree"
                )
            self._append_state(
                phase="worker_environment_verified",
                next_allowed_actions=("simulate_smoke",),
                usage_delta=BudgetUsage(tool_calls=1),
            )

            phase = "simulate_smoke"
            self._prepare_action(
                "simulate_smoke",
                BudgetUsage(tool_calls=2, simulations=10),
            )
            remote_spec = copy_to_remote_run(
                local_path=spec_path,
                remote_relative_path=f"{self.run_id}/inputs/simulation-spec.json",
                settings=self.remote,
            )
            remote_simulation_root = f"{self.remote_run_root}/simulation"
            simulation_payload, remote_simulation_ref, _ = self._invoke_remote(
                operation="simulate",
                arguments=(
                    "--spec",
                    remote_spec,
                    "--output-dir",
                    remote_simulation_root,
                ),
                slug="simulation",
            )
            self._require_payload(
                simulation_payload,
                sample_count=10,
                supports_scientific_claims=False,
            )
            self._append_state(
                phase="simulation_generated",
                next_allowed_actions=("build_dataset",),
                usage_delta=BudgetUsage(tool_calls=2, simulations=10),
            )

            phase = "build_dataset"
            self._prepare_action(
                "build_dataset",
                BudgetUsage(tool_calls=2),
            )
            dataset_payload, remote_dataset_ref, _ = self._invoke_remote(
                operation="build-dataset",
                arguments=("--simulation-root", remote_simulation_root),
                slug="dataset-build",
            )
            copied_simulation_root = copy_remote_output(
                remote_path=remote_simulation_root,
                local_parent=self.run_directory / "remote",
                settings=self.remote,
                maximum_bytes=self._remaining_storage_bytes(),
            )
            manifest_path = copied_simulation_root / "dataset_manifest.json"
            simulation_manifest_path = (
                copied_simulation_root / "simulation_manifest.json"
            )
            manifest = load_dataset_manifest(manifest_path)
            manifest_sha256 = sha256_file(manifest_path)
            self._require_payload(
                dataset_payload,
                dataset_id=manifest.dataset_id,
                manifest_sha256=manifest_sha256,
            )
            dataset_ref = self._reference_file(
                artifact_id="dataset-manifest",
                role="dataset-manifest",
                path=manifest_path,
                media_type="application/json",
                producer="ripple-dataset-builder",
                input_artifact_ids=(
                    spec_ref.artifact_id,
                    remote_simulation_ref.artifact_id,
                    remote_dataset_ref.artifact_id,
                ),
            )
            self._reference_file(
                artifact_id="simulation-manifest",
                role="simulation-manifest",
                path=simulation_manifest_path,
                media_type="application/json",
                producer="ripple-slsim-worker",
                input_artifact_ids=(spec_ref.artifact_id,),
            )
            dataset_summary = self._dataset_summary(manifest, manifest_sha256)
            _, dataset_summary_ref = self._write_artifact(
                artifact_id="dataset-summary",
                role="dataset-summary",
                relative_path="evidence/dataset-summary.json",
                value=dataset_summary,
                producer="ripple-campaign-runner",
                input_artifact_ids=(dataset_ref.artifact_id,),
            )
            self._append_state(
                phase="dataset_built",
                next_allowed_actions=("generate_architectures",),
                usage_delta=BudgetUsage(tool_calls=2),
            )

            phase = "generate_architectures"
            self._prepare_action(
                "generate_architectures",
                BudgetUsage(
                    llm_requests=self._agent_request_ceiling,
                    tool_calls=_AGENT_STAGE_TOOL_CALLS,
                ),
            )
            architecture_run = run_architecture_planner(
                dataset_summary,
                settings=self.provider_settings,
            )
            architecture_plan_path, architecture_plan_ref = self._write_artifact(
                artifact_id="architecture-search-plan",
                role="architecture-search-plan",
                relative_path="evidence/architecture-plan.json",
                value=architecture_run.plan,
                producer="ripple-architecture-generator",
                input_artifact_ids=(
                    dataset_summary_ref.artifact_id,
                    provider_ref.artifact_id,
                ),
            )
            architecture_agent_ref = self._agent_evidence(
                stage="architecture_generator",
                output_reference=architecture_plan_ref,
                called_tools=architecture_run.called_tools,
                request_count=architecture_run.request_count,
                input_tokens=architecture_run.input_tokens,
                output_tokens=architecture_run.output_tokens,
            )
            self._append_state(
                phase="architecture_plan_created",
                next_allowed_actions=("judge_architectures",),
                usage_delta=BudgetUsage(
                    llm_requests=architecture_run.request_count,
                    tool_calls=len(architecture_run.called_tools),
                ),
                decisions=(
                    self._decision(
                        agent_name="ripple-architecture-generator",
                        phase="generate_architectures",
                        allowed_actions=("propose_architectures",),
                        selected_action="propose_architectures",
                        rationale=architecture_run.plan.comparison_rationale,
                        expected_cost="one bounded architecture-planning turn",
                        evidence_ids=(
                            dataset_summary_ref.artifact_id,
                            architecture_agent_ref.artifact_id,
                        ),
                    ),
                ),
            )

            phase = "judge_architectures"
            self._prepare_action(
                "judge_architectures",
                BudgetUsage(
                    llm_requests=self._agent_request_ceiling,
                    tool_calls=_AGENT_STAGE_TOOL_CALLS,
                ),
            )
            judge_run = run_architecture_judge(
                dataset_summary,
                architecture_run.plan,
                shortlist_size=self.configuration.architecture_shortlist_size,
                settings=self.provider_settings,
            )
            _, judge_ref = self._write_artifact(
                artifact_id="architecture-judge-verdict",
                role="architecture-judge-verdict",
                relative_path="evidence/architecture-judge.json",
                value=judge_run.verdict,
                producer="ripple-architecture-judge",
                input_artifact_ids=(architecture_plan_ref.artifact_id,),
            )
            judge_agent_ref = self._agent_evidence(
                stage="architecture_judge",
                output_reference=judge_ref,
                called_tools=judge_run.called_tools,
                request_count=judge_run.request_count,
                input_tokens=judge_run.input_tokens,
                output_tokens=judge_run.output_tokens,
            )
            self._append_state(
                phase="architectures_shortlisted",
                next_allowed_actions=("train_shortlist",),
                usage_delta=BudgetUsage(
                    llm_requests=judge_run.request_count,
                    tool_calls=len(judge_run.called_tools),
                ),
                decisions=(
                    self._decision(
                        agent_name="ripple-architecture-judge",
                        phase="judge_architectures",
                        allowed_actions=("shortlist",),
                        selected_action="shortlist",
                        rationale=judge_run.verdict.reasoning,
                        expected_cost=(
                            f"{len(judge_run.verdict.shortlist)} bounded CUDA training runs"
                        ),
                        evidence_ids=(
                            architecture_plan_ref.artifact_id,
                            judge_agent_ref.artifact_id,
                        ),
                    ),
                ),
            )

            phase = "train_shortlist"
            self._prepare_action(
                "train_shortlist",
                BudgetUsage(
                    tool_calls=1 + 3 * len(judge_run.verdict.shortlist),
                    training_runs=len(judge_run.verdict.shortlist),
                    gpu_seconds=float(len(judge_run.verdict.shortlist) + 1),
                ),
            )
            training_configuration_path, training_configuration_ref = (
                self._write_artifact(
                    artifact_id="training-configuration",
                    role="training-configuration",
                    relative_path="inputs/training-configuration.json",
                    value=self.configuration.training,
                    producer="ripple-campaign-runner",
                    input_artifact_ids=(config_ref.artifact_id,),
                )
            )
            remote_training_configuration = copy_to_remote_run(
                local_path=training_configuration_path,
                remote_relative_path=f"{self.run_id}/inputs/training-configuration.json",
                settings=self.remote,
            )
            candidates = {
                candidate.candidate_id: candidate
                for candidate in architecture_run.plan.candidates
            }
            training_records: dict[str, TrainingRunRecord] = {}
            local_training_directories: dict[str, Path] = {}
            total_gpu_seconds = 0.0
            shortlisted_ids = judge_run.verdict.shortlist
            remote_action_overhead_seconds = self.remote.connect_timeout_seconds + 30
            for candidate_index, candidate_id in enumerate(shortlisted_ids):
                remaining_gpu_slots = len(shortlisted_ids) - candidate_index + 1
                available_gpu_seconds = self._remaining_gpu_seconds(
                    pending_seconds=total_gpu_seconds
                )
                training_timeout_seconds = (
                    math.floor(available_gpu_seconds / remaining_gpu_slots)
                    - remote_action_overhead_seconds
                )
                if training_timeout_seconds < 1:
                    raise CampaignExecutionError(
                        "GPU budget exhausted before shortlisted training"
                    )
                candidate = candidates[candidate_id]
                bound = BoundArchitecture(
                    candidate=candidate,
                    input_shape=dataset_summary.image_shape,
                    channels=dataset_summary.channels,
                    num_classes=len(dataset_summary.class_names),
                    source_backend=self.configuration.architecture_implementation,
                    source_revision=RIPPLE_MODEL_BUILDERS_REVISION,
                )
                bound_path, bound_ref = self._write_artifact(
                    artifact_id=f"bound-{candidate_id}",
                    role="bound-architecture",
                    relative_path=f"inputs/architectures/{candidate_id}.json",
                    value=bound,
                    producer="ripple-campaign-runner",
                    input_artifact_ids=(
                        architecture_plan_ref.artifact_id,
                        judge_ref.artifact_id,
                        dataset_ref.artifact_id,
                    ),
                )
                self._set_failure_usage_charge(
                    BudgetUsage(
                        tool_calls=1 + 3 * candidate_index + 1,
                        training_runs=candidate_index,
                        gpu_seconds=total_gpu_seconds,
                    )
                )
                remote_bound = copy_to_remote_run(
                    local_path=bound_path,
                    remote_relative_path=(
                        f"{self.run_id}/inputs/architectures/{candidate_id}.json"
                    ),
                    settings=self.remote,
                )
                remote_training_dir = f"{self.remote_run_root}/training/{candidate_id}"
                self._set_failure_usage_charge(
                    BudgetUsage(
                        tool_calls=1 + 3 * (candidate_index + 1),
                        training_runs=candidate_index + 1,
                        gpu_seconds=(
                            total_gpu_seconds
                            + training_timeout_seconds
                            + remote_action_overhead_seconds
                        ),
                    )
                )
                (
                    training_payload,
                    remote_training_ref,
                    training_command_seconds,
                ) = self._invoke_remote(
                    operation="train",
                    arguments=(
                        "--dataset-root",
                        remote_simulation_root,
                        "--manifest",
                        f"{remote_simulation_root}/dataset_manifest.json",
                        "--architecture",
                        remote_bound,
                        "--training-config",
                        remote_training_configuration,
                        "--output-dir",
                        remote_training_dir,
                    ),
                    slug=f"train-{candidate_id}",
                    execution_timeout_seconds=training_timeout_seconds,
                )
                worker_training_seconds = self._payload_nonnegative_float(
                    training_payload,
                    "worker_elapsed_seconds",
                )
                if worker_training_seconds > training_timeout_seconds:
                    raise CampaignExecutionError(
                        "remote training worker exceeded its enforced GPU timeout"
                    )
                if worker_training_seconds > training_command_seconds:
                    raise CampaignExecutionError(
                        "remote training timing evidence is inconsistent"
                    )
                local_training_dir = copy_remote_output(
                    remote_path=remote_training_dir,
                    local_parent=self.run_directory / "remote" / "training",
                    settings=self.remote,
                    maximum_bytes=self._remaining_storage_bytes(),
                )
                training_record_path = local_training_dir / "training_record.json"
                training_record = TrainingRunRecord.model_validate_json(
                    training_record_path.read_bytes(), strict=True
                )
                self._require_payload(
                    training_payload,
                    run_id=training_record.run_id,
                    weights_sha256=training_record.weights_sha256,
                    smoke_only=True,
                )
                if training_record.architecture.candidate.candidate_id != candidate_id:
                    raise CampaignExecutionError(
                        "remote training record changed the candidate identity"
                    )
                if training_record.dataset_manifest_sha256 != manifest_sha256:
                    raise CampaignExecutionError(
                        "remote training record used another dataset manifest"
                    )
                if training_record.elapsed_seconds > training_timeout_seconds:
                    raise CampaignExecutionError(
                        "remote training record exceeded its enforced GPU timeout"
                    )
                weights_path = (
                    local_training_dir / training_record.weights_relative_path
                )
                checkpoint_path = (
                    local_training_dir
                    / training_record.checkpoint_metadata_relative_path
                )
                if sha256_file(weights_path) != training_record.weights_sha256:
                    raise CampaignExecutionError(
                        "copied training weights failed hashing"
                    )
                if (
                    sha256_file(checkpoint_path)
                    != training_record.checkpoint_metadata_sha256
                ):
                    raise CampaignExecutionError(
                        "copied checkpoint metadata failed hashing"
                    )
                training_record_ref = self._reference_file(
                    artifact_id=f"training-record-{candidate_id}",
                    role="training-record",
                    path=training_record_path,
                    media_type="application/json",
                    producer="ripple-torch-worker",
                    input_artifact_ids=(
                        bound_ref.artifact_id,
                        training_configuration_ref.artifact_id,
                        dataset_ref.artifact_id,
                        remote_training_ref.artifact_id,
                    ),
                )
                self._reference_file(
                    artifact_id=f"checkpoint-{candidate_id}",
                    role="model-checkpoint",
                    path=weights_path,
                    media_type="application/octet-stream",
                    producer="ripple-torch-worker",
                    input_artifact_ids=(training_record_ref.artifact_id,),
                )
                self._reference_file(
                    artifact_id=f"checkpoint-metadata-{candidate_id}",
                    role="checkpoint-metadata",
                    path=checkpoint_path,
                    media_type="application/json",
                    producer="ripple-torch-worker",
                    input_artifact_ids=(training_record_ref.artifact_id,),
                )
                training_records[candidate_id] = training_record
                local_training_directories[candidate_id] = local_training_dir
                total_gpu_seconds += training_command_seconds
            self._append_state(
                phase="shortlisted_candidates_trained",
                next_allowed_actions=("select_by_validation",),
                usage_delta=BudgetUsage(
                    tool_calls=1 + 3 * len(training_records),
                    training_runs=len(training_records),
                    gpu_seconds=total_gpu_seconds,
                ),
            )

            phase = "select_by_validation"
            self._prepare_action(
                "select_by_validation",
                BudgetUsage(tool_calls=1),
            )
            ordered = sorted(
                training_records.items(),
                key=lambda item: (
                    -item[1].best_validation.balanced_accuracy,
                    item[1].best_validation.loss,
                    item[0],
                ),
            )
            selected_candidate_id, selected_training_record = ordered[0]
            evaluations = tuple(
                ArchitectureEvaluation(
                    candidate_id=candidate_id,
                    run_id=record.run_id,
                    checkpoint_sha256=record.weights_sha256,
                    validation_loss=record.best_validation.loss,
                    validation_accuracy=record.best_validation.accuracy,
                    validation_balanced_accuracy=(
                        record.best_validation.balanced_accuracy
                    ),
                    elapsed_gpu_seconds=record.elapsed_seconds,
                    epochs_completed=len(record.epochs),
                    smoke_only=True,
                )
                for candidate_id, record in training_records.items()
            )
            search_result = ArchitectureSearchResult(
                plan_sha256=sha256_file(architecture_plan_path),
                evaluations=evaluations,
                selected_candidate_id=selected_candidate_id,
                selection_metric="validation_balanced_accuracy",
                scientific_selection_claim_allowed=False,
            )
            _, search_result_ref = self._write_artifact(
                artifact_id="architecture-search-result",
                role="architecture-search-result",
                relative_path="evidence/architecture-search-result.json",
                value=search_result,
                producer="ripple-deterministic-selector",
                input_artifact_ids=tuple(
                    f"training-record-{candidate_id}"
                    for candidate_id in training_records
                ),
            )
            self._append_state(
                phase="candidate_selected",
                next_allowed_actions=("evaluate_heldout_once",),
                usage_delta=BudgetUsage(tool_calls=1),
                decisions=(
                    self._decision(
                        agent_name="ripple-deterministic-selector",
                        phase="select_by_validation",
                        allowed_actions=tuple(training_records),
                        selected_action=selected_candidate_id,
                        rationale=(
                            "Selected by descending measured validation balanced "
                            "accuracy, then lower validation loss, then candidate ID."
                        ),
                        expected_cost="one final held-out evaluation",
                        evidence_ids=(search_result_ref.artifact_id,),
                    ),
                ),
            )

            phase = "evaluate_heldout_once"
            self._prepare_action(
                "evaluate_heldout_once",
                BudgetUsage(tool_calls=3, gpu_seconds=1.0),
            )
            selected_remote_training_dir = (
                f"{self.remote_run_root}/training/{selected_candidate_id}"
            )
            remote_evaluation_path = (
                f"{self.remote_run_root}/evaluation/final-evaluation.json"
            )
            remote_action_overhead_seconds = self.remote.connect_timeout_seconds + 30
            evaluation_timeout_seconds = (
                math.floor(self._remaining_gpu_seconds())
                - remote_action_overhead_seconds
            )
            if evaluation_timeout_seconds < 1:
                raise CampaignExecutionError(
                    "GPU budget exhausted before held-out evaluation"
                )
            self._set_failure_usage_charge(
                BudgetUsage(
                    tool_calls=3,
                    gpu_seconds=float(
                        evaluation_timeout_seconds + remote_action_overhead_seconds
                    ),
                )
            )
            (
                evaluation_payload,
                remote_evaluation_ref,
                evaluation_command_seconds,
            ) = self._invoke_remote(
                operation="evaluate",
                arguments=(
                    "--dataset-root",
                    remote_simulation_root,
                    "--manifest",
                    f"{remote_simulation_root}/dataset_manifest.json",
                    "--training-dir",
                    selected_remote_training_dir,
                    "--training-record",
                    f"{selected_remote_training_dir}/training_record.json",
                    "--output",
                    remote_evaluation_path,
                ),
                slug="heldout-evaluation",
                execution_timeout_seconds=evaluation_timeout_seconds,
            )
            worker_evaluation_seconds = self._payload_nonnegative_float(
                evaluation_payload,
                "elapsed_seconds",
            )
            if worker_evaluation_seconds > evaluation_timeout_seconds:
                raise CampaignExecutionError(
                    "held-out evaluation exceeded its enforced GPU timeout"
                )
            evaluation_gpu_seconds = self._payload_nonnegative_float(
                evaluation_payload,
                "worker_elapsed_seconds",
            )
            if evaluation_gpu_seconds > evaluation_timeout_seconds:
                raise CampaignExecutionError(
                    "held-out worker exceeded its enforced GPU timeout"
                )
            if evaluation_gpu_seconds > evaluation_command_seconds:
                raise CampaignExecutionError(
                    "held-out evaluation timing evidence is inconsistent"
                )
            evaluation_gpu_seconds = evaluation_command_seconds
            local_evaluation_path = copy_remote_output(
                remote_path=remote_evaluation_path,
                local_parent=self.run_directory / "remote" / "evaluation",
                settings=self.remote,
                maximum_bytes=self._remaining_storage_bytes(),
            )
            local_consumed_path = copy_remote_output(
                remote_path=f"{remote_evaluation_path}.consumed",
                local_parent=self.run_directory / "remote" / "evaluation",
                settings=self.remote,
                maximum_bytes=self._remaining_storage_bytes(),
            )
            evaluation = FinalEvaluationRecord.model_validate_json(
                local_evaluation_path.read_bytes(), strict=True
            )
            self._require_payload(
                evaluation_payload,
                evaluation_id=evaluation.evaluation_id,
                smoke_only=True,
            )
            if evaluation.selected_training_run_id != selected_training_record.run_id:
                raise CampaignExecutionError(
                    "held-out evaluation used a non-selected training run"
                )
            if evaluation.checkpoint_sha256 != selected_training_record.weights_sha256:
                raise CampaignExecutionError(
                    "held-out evaluation used a different checkpoint"
                )
            evaluation_ref = self._reference_file(
                artifact_id="final-heldout-evaluation",
                role="final-heldout-evaluation",
                path=local_evaluation_path,
                media_type="application/json",
                producer="ripple-torch-worker",
                input_artifact_ids=(
                    f"training-record-{selected_candidate_id}",
                    search_result_ref.artifact_id,
                    remote_evaluation_ref.artifact_id,
                ),
            )
            self._reference_file(
                artifact_id="heldout-consumption-marker",
                role="heldout-consumption-marker",
                path=local_consumed_path,
                media_type="application/json",
                producer="ripple-torch-worker",
                input_artifact_ids=(evaluation_ref.artifact_id,),
            )
            self._append_state(
                phase="heldout_evaluated_once",
                next_allowed_actions=("assemble_technical_report",),
                usage_delta=BudgetUsage(
                    tool_calls=3,
                    gpu_seconds=evaluation_gpu_seconds,
                ),
            )

            phase = "assemble_technical_report"
            self._prepare_action(
                "assemble_technical_report",
                BudgetUsage(tool_calls=1),
            )
            lens_count = sum(item.label_name == "lens" for item in manifest.samples)
            non_lens_count = sum(
                item.label_name == "non_lens" for item in manifest.samples
            )
            report_evidence = SyntheticTrainingSmokeEvidence(
                dataset_id=manifest.dataset_id,
                selected_training_run_id=selected_training_record.run_id,
                checkpoint_sha256=selected_training_record.weights_sha256,
                evaluation_artifact_id=evaluation_ref.artifact_id,
                total_samples=len(manifest.samples),
                lens_samples=lens_count,
                non_lens_samples=non_lens_count,
                has_sky_coordinates=False,
            )
            report = assemble_synthetic_training_technical_report(
                report_id=f"report-{self.run_id}",
                evidence=report_evidence,
            )
            report_path, report_ref = self._write_artifact(
                artifact_id="technical-report",
                role="technical-report",
                relative_path="report/technical-report.json",
                value=report,
                producer="ripple-deterministic-report",
                input_artifact_ids=(
                    evaluation_ref.artifact_id,
                    search_result_ref.artifact_id,
                    dataset_ref.artifact_id,
                ),
            )
            final_sequence = self.state.sequence + 1
            provisional_completion = SimulationCampaignCompletion(
                run_id=self.run_id,
                request_id=self.configuration.request.request_id,
                campaign_configuration_sha256=canonical_json_sha256(self.configuration),
                simulation_spec_sha256=canonical_spec_sha256(spec),
                dataset_id=manifest.dataset_id,
                dataset_manifest_sha256=manifest_sha256,
                architecture_plan_sha256=sha256_file(architecture_plan_path),
                selected_candidate_id=selected_candidate_id,
                selected_training_run_id=selected_training_record.run_id,
                selected_checkpoint_sha256=selected_training_record.weights_sha256,
                validation_balanced_accuracy=(
                    selected_training_record.best_validation.balanced_accuracy
                ),
                final_evaluation_id=evaluation.evaluation_id,
                final_evaluation_sha256=sha256_file(local_evaluation_path),
                technical_report_id=report.report_id,
                technical_report_sha256=sha256_file(report_path),
                final_state_sequence=final_sequence,
                final_state_sha256="0" * 64,
                completed_at_utc=utc_now(),
                smoke_only=True,
                scientific_performance_claim_allowed=False,
            )
            completion_size = len(ArtifactStore.json_bytes(provisional_completion))
            final_state = self._append_state(
                phase="technical_report_assembled",
                status="complete",
                usage_delta=BudgetUsage(
                    tool_calls=1,
                    storage_bytes=completion_size,
                ),
                gates=(
                    ScientificGate(
                        gate_id="integration-smoke-complete",
                        status="open",
                        evidence_artifact_ids=(
                            report_ref.artifact_id,
                            evaluation_ref.artifact_id,
                        ),
                    ),
                    ScientificGate(
                        gate_id="scientific-performance-claim",
                        status="closed",
                        evidence_artifact_ids=(
                            dataset_ref.artifact_id,
                            evaluation_ref.artifact_id,
                        ),
                        reasons=(
                            "Ten synthetic samples validate wiring only.",
                            "Architecture selection and test metrics are not statistically meaningful.",
                        ),
                    ),
                    ScientificGate(
                        gate_id="lenscat-candidate-validation",
                        status="not_applicable",
                        evidence_artifact_ids=(report_ref.artifact_id,),
                        reasons=(
                            "Synthetic smoke samples have no celestial coordinates.",
                        ),
                    ),
                ),
            )
            final_state_path = self.run_directory / (
                f"states/state-{final_state.sequence:04d}.json"
            )
            completion = provisional_completion.model_copy(
                update={"final_state_sha256": sha256_file(final_state_path)}
            )
            if len(ArtifactStore.json_bytes(completion)) != completion_size:
                raise CampaignExecutionError(
                    "completion marker changed size after final-state binding"
                )
            if (
                _directory_bytes(self.run_directory) + completion_size
                > self.state.budget.max_storage_bytes
            ):
                raise CampaignExecutionError(
                    "completion marker no longer fits its precharged storage budget"
                )
            # This root marker is the publication commit and is intentionally the
            # final filesystem write of a successful campaign.
            completion_path = self.store.write_json("completion.json", completion)
            return CompletedSimulationCampaign(
                run_directory=self.run_directory,
                completion_path=completion_path,
                report_path=report_path,
                final_state_path=final_state_path,
                completion=completion,
                final_state=final_state,
            )
        except Exception as exc:
            if self.state.status == "complete":
                raise
            self._fail(phase, exc)
            if isinstance(exc, CampaignExecutionError):
                raise
            raise CampaignExecutionError(
                f"campaign failed during {phase}; inspect {self.run_directory}"
            ) from exc


def run_simulation_campaign(
    configuration: SimulationCampaignConfiguration,
    *,
    configuration_base: str | os.PathLike[str] | None = None,
    repository_root: str | os.PathLike[str] | None = None,
    provider_settings: BedrockAgentSettings | None = None,
    run_id: str | None = None,
) -> CompletedSimulationCampaign:
    """Convenience entry point used by the CLI and Python callers."""

    return SimulationCampaignRunner(
        configuration,
        configuration_base=configuration_base,
        repository_root=repository_root,
        provider_settings=provider_settings,
        run_id=run_id,
    ).run()


__all__ = [
    "CampaignExecutionError",
    "CompletedSimulationCampaign",
    "SimulationCampaignRunner",
    "run_simulation_campaign",
]
