"""Controlled synthetic architecture training and Bedrock comparison workflow.

Bedrock calls run only on the local control plane.  The SSH worker receives
validated JSON and verified source trees, never cloud credentials.  The final
held-out split remains unopened until validation-only tuning and selection are
complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import JsonValue, TypeAdapter

from ..agents.architecture import run_architecture_planner
from ..agents.architecture_judge import run_architecture_judge
from ..agents.tuning import run_tuning_agent
from ..artifacts import ArtifactStore, sha256_file
from ..paths import checked_real_directory, checked_real_file
from ..providers.bedrock import (
    BedrockAgentSettings,
    run_bedrock_typed_output_smoke_with_usage,
)
from ..schemas.architecture import BoundArchitecture, DatasetSummary
from ..schemas.campaign import (
    ScientistSourceRevisions,
    SourceSyncEvidence,
    SyncedSourceComponent,
)
from ..schemas.common import canonical_json_sha256
from ..schemas.dataset import DatasetManifest
from ..schemas.remote import RemoteCommandResult
from ..schemas.study import (
    AgentStageEvidenceV2,
    AgentStageFailure,
    ArchitecturePlan,
    ArchitecturePlanCandidate,
    ArmStudyResult,
    ArmTrajectoryPoint,
    BedrockModelAvailabilityRecord,
    BedrockModelMatrix,
    BedrockPricingSnapshot,
    ClassificationMetrics,
    ComparisonRegime,
    FinalEvaluationSummary,
    MrigankaZeroShotSummary,
    ScientificQualification,
    SearchProtocol,
    StudyAggregateInput,
    StudyDatasetIdentity,
    TokenUsage,
    TuningPolicy,
)
from ..schemas.study_run import (
    ArchitectureModelStudyConfiguration,
    StudyArchitectureArm,
    StudyBedrockModel,
)
from ..schemas.training import FinalEvaluationRecord, TrainingRunRecord
from ..schemas.tuning import TuningAction, TuningMeasurement
from ..tools.costing import apply_rate_based_costing, attach_rate_based_cost
from ..tools.dataset_builder import load_dataset_manifest
from ..tools.model_builders import RIPPLE_MODEL_BUILDERS_REVISION
from ..tools.paper_tables import RenderedPaperTables, render_paper_tables
from ..tools.remote_executor import (
    SourceTreeDigest,
    copy_remote_output,
    copy_to_remote_run,
    run_remote_worker,
    sync_worker_source,
)
from ..tools.slsim_backend import load_study_spec
from ..tools.tuning import apply_tuning_action, eligible_tuning_actions
from .campaign import _tree_digest


_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


class StudyExecutionError(RuntimeError):
    """Safe failure raised after the run has retained local evidence."""


@dataclass(frozen=True)
class CompletedArchitectureModelStudy:
    run_directory: Path
    aggregate_path: Path
    completion_path: Path
    paper_tables: RenderedPaperTables
    study: StudyAggregateInput


@dataclass
class _TrainedArm:
    configuration: StudyArchitectureArm
    trajectories: list[ArmTrajectoryPoint]
    records: list[TrainingRunRecord]
    local_directories: list[Path]
    remote_directories: list[str]
    total_worker_seconds: float = 0.0
    total_remote_wall_seconds: float = 0.0
    selected_iteration: int = 0
    test_evaluation: FinalEvaluationRecord | None = None
    test_evaluated_at_utc: datetime | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("could not derive a safe artifact slug")
    return slug[:96].rstrip("-")


def _safe_run_id(study_id: str) -> str:
    timestamp = _utc_now().strftime("%Y%m%dT%H%M%SZ").lower()
    prefix = study_id[:88].rstrip("._-")
    run_id = f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("could not derive a safe study run ID")
    return run_id


def _resolve_input(base: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = base / path
    return checked_real_file(path)


def _last_json_object(stdout: str) -> dict[str, JsonValue]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return _JSON_OBJECT.validate_python(value, strict=True)
    raise StudyExecutionError("remote worker returned no final JSON object")


def _usage(
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> TokenUsage:
    cached = cache_write_tokens + cache_read_tokens
    if cached > input_tokens:
        raise StudyExecutionError("provider token cache arithmetic is inconsistent")
    return TokenUsage(
        uncached_input_tokens=input_tokens - cached,
        cache_write_input_tokens=cache_write_tokens,
        cache_read_input_tokens=cache_read_tokens,
        measured_total_input_tokens=input_tokens,
        output_tokens=output_tokens,
        source="bedrock_usage",
    )


def _settings(
    configuration: ArchitectureModelStudyConfiguration,
    model_id: str,
) -> BedrockAgentSettings:
    return BedrockAgentSettings(
        profile_name=configuration.bedrock_profile_name,
        region_name=configuration.bedrock_region,
        model_id=model_id,
    )


def _safe_failure(exc: Exception) -> tuple[str, str]:
    code = getattr(exc, "code", None)
    safe_code = str(code) if isinstance(code, str) else type(exc).__name__
    return safe_code[:256], (
        "The bounded model invocation failed. Raw provider text is intentionally "
        "excluded from study artifacts."
    )


def _stage_success(
    *,
    evidence_id: str,
    request_id: str,
    arm_id: str | None,
    round_index: int,
    phase: Literal["phase_a", "phase_b", "other"],
    stage: str,
    model: StudyBedrockModel,
    region: str,
    started: datetime,
    completed: datetime,
    provider_request_count: int,
    tool_call_count: int,
    usage: TokenUsage,
    output_artifact_id: str,
    output_sha256: str,
    called_tools: Sequence[str],
    pricing: BedrockPricingSnapshot,
) -> AgentStageEvidenceV2:
    record = AgentStageEvidenceV2(
        evidence_id=evidence_id,
        request_id=request_id,
        arm_id=arm_id,
        round_index=round_index,
        phase=phase,
        stage=stage,
        provider_request_count=provider_request_count,
        tool_call_count=tool_call_count,
        model_id=model.model_id,
        pricing_model_id=model.pricing_model_id,
        inference_profile_id=model.inference_profile_id,
        provider=model.provider,
        region=region,
        status="succeeded",
        started_at_utc=started,
        completed_at_utc=completed,
        wall_time_seconds=(completed - started).total_seconds(),
        token_usage=usage,
        output_artifact_id=output_artifact_id,
        output_sha256=output_sha256,
        called_tools=tuple(called_tools),
    )
    return attach_rate_based_cost(record, pricing)


def _stage_failure(
    *,
    evidence_id: str,
    request_id: str,
    arm_id: str | None,
    round_index: int,
    phase: Literal["phase_a", "phase_b", "other"],
    stage: str,
    model: StudyBedrockModel,
    region: str,
    started: datetime,
    completed: datetime,
    exc: Exception,
) -> AgentStageEvidenceV2:
    code, message = _safe_failure(exc)
    return AgentStageEvidenceV2(
        evidence_id=evidence_id,
        request_id=request_id,
        arm_id=arm_id,
        round_index=round_index,
        phase=phase,
        stage=stage,
        model_id=model.model_id,
        pricing_model_id=model.pricing_model_id,
        inference_profile_id=model.inference_profile_id,
        provider=model.provider,
        region=region,
        status="failed",
        started_at_utc=started,
        completed_at_utc=completed,
        wall_time_seconds=(completed - started).total_seconds(),
        failure=AgentStageFailure(
            error_type=type(exc).__name__,
            error_code=code,
            message=message,
            retryable=False,
        ),
    )


def _sync_sources(
    *,
    repository_root: Path,
    configuration: ArchitectureModelStudyConfiguration,
) -> SourceSyncEvidence:
    revisions_path = checked_real_file(
        repository_root / "ripple" / "scientist" / "locks" / "source-revisions.json"
    )
    revisions = ScientistSourceRevisions.model_validate_json(
        revisions_path.read_bytes(), strict=True
    )
    runtime_locks = {item.name: item for item in revisions.runtime_sources}
    definitions = (
        (
            "ripple_scientist",
            repository_root / "ripple" / "scientist",
            "ripple/scientist",
        ),
        (
            "slsim",
            repository_root / "ripple" / "scientist" / "vendor" / "slsim" / "slsim",
            "ripple/scientist/vendor/slsim/slsim",
        ),
        (
            "jaxtronomy",
            repository_root
            / "ripple"
            / "scientist"
            / "vendor"
            / "JAXtronomy"
            / "jaxtronomy",
            "ripple/scientist/vendor/JAXtronomy/jaxtronomy",
        ),
    )
    local: list[SourceTreeDigest] = []
    for component, path, remote_relative in definitions:
        digest, count = _tree_digest(path)
        lock_name = {"slsim": "slsim", "jaxtronomy": "JAXtronomy"}.get(component)
        if lock_name is not None:
            locked = runtime_locks[lock_name]
            if (
                locked.local_path != remote_relative
                or locked.transfer_tree_sha256 != digest
            ):
                raise StudyExecutionError(
                    f"{lock_name} transfer tree differs from its pinned source lock"
                )
        local.append(
            SourceTreeDigest(
                component=component,
                remote_relative_path=remote_relative,
                sha256=digest,
                regular_file_count=count,
            )
        )
    package_initializer = checked_real_file(repository_root / "ripple" / "__init__.py")
    initializer_sha256 = sha256_file(package_initializer)
    verified = sync_worker_source(
        local_repository=repository_root,
        settings=configuration.remote,
        expected_components=tuple(local),
        expected_package_initializer_sha256=initializer_sha256,
    )
    remote = {item.component: item for item in verified.components}
    components = tuple(
        SyncedSourceComponent(
            component=item.component,
            local_relative_path=item.remote_relative_path,
            remote_relative_path=item.remote_relative_path,
            source_tree_sha256=item.sha256,
            regular_file_count=item.regular_file_count,
            remote_tree_sha256=remote[item.component].sha256,
            remote_regular_file_count=remote[item.component].regular_file_count,
        )
        for item in local
    )
    observed = _utc_now()
    return SourceSyncEvidence(
        host=configuration.remote.host,
        remote_root=configuration.remote.remote_root,
        components=components,
        package_initializer_sha256=initializer_sha256,
        remote_package_initializer_sha256=verified.package_initializer_sha256,
        source_manifest_sha256=verified.source_manifest_sha256,
        remote_source_root=verified.source_root,
        synchronized_at_utc=observed,
        verified_at_utc=observed,
    )


def _catalog_and_account(
    configuration: ArchitectureModelStudyConfiguration,
) -> tuple[str, set[str]]:
    try:
        import boto3
    except ImportError:
        raise StudyExecutionError("boto3 is unavailable in the control environment")
    session = boto3.Session(
        profile_name=configuration.bedrock_profile_name,
        region_name=configuration.bedrock_region,
    )
    identity = session.client("sts").get_caller_identity()
    account = identity.get("Account")
    if not isinstance(account, str) or not account:
        raise StudyExecutionError("AWS identity did not return an account identifier")
    fingerprint = hashlib.sha256(account.encode("utf-8")).hexdigest()
    client = session.client("bedrock")
    identifiers: set[str] = set()
    foundation = client.list_foundation_models()
    for item in foundation.get("modelSummaries", ()):
        model_id = item.get("modelId")
        if isinstance(model_id, str):
            identifiers.add(model_id)
    token: str | None = None
    while True:
        payload = {} if token is None else {"nextToken": token}
        page = client.list_inference_profiles(**payload)
        for item in page.get("inferenceProfileSummaries", ()):
            profile_id = item.get("inferenceProfileId")
            if isinstance(profile_id, str):
                identifiers.add(profile_id)
        next_token = page.get("nextToken")
        if not isinstance(next_token, str) or not next_token:
            break
        token = next_token
    return fingerprint, identifiers


def _classification(metrics: Any) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=metrics.accuracy,
        balanced_accuracy=metrics.balanced_accuracy,
        roc_auc=metrics.roc_auc,
        loss=metrics.loss,
        sample_count=metrics.sample_count,
    )


class ArchitectureModelStudyRunner:
    def __init__(
        self,
        configuration: ArchitectureModelStudyConfiguration,
        *,
        configuration_base: str | os.PathLike[str],
        repository_root: str | os.PathLike[str],
        run_id: str | None = None,
    ) -> None:
        self.configuration = configuration
        self.configuration_base = checked_real_directory(configuration_base)
        self.repository_root = checked_real_directory(repository_root)
        self.run_id = run_id or _safe_run_id(configuration.study_id)
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id is not a normalized study identifier")
        output_root = Path(configuration.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = self.repository_root / output_root
        output_root = checked_real_directory(output_root, create=True)
        self.run_directory = output_root / self.run_id
        self.run_directory.mkdir(mode=0o700)
        self.store = ArtifactStore(self.run_directory, create=False)
        self.remote_run_root = (
            f"{configuration.remote.remote_root}/runs/{self.run_id}"
        )
        self.stage_runs: list[AgentStageEvidenceV2] = []
        self.training_runs = 0
        self.remote_wall_seconds = 0.0
        self.phase = "initialization"
        self._write_progress(
            status="running",
            activity="controller_initialized",
        )

    def _write_progress(
        self,
        *,
        status: Literal["running", "complete", "failed"],
        activity: str,
        **details: JsonValue,
    ) -> None:
        provider_requests = sum(
            (
                stage.provider_request_count
                if stage.provider_request_count is not None
                else 7
            )
            for stage in self.stage_runs
        )
        measured_tokens = sum(
            stage.token_usage.measured_total_tokens
            for stage in self.stage_runs
            if stage.token_usage is not None
        )
        estimated_cost = sum(
            (
                stage.cost.total_estimated_cost_usd
                for stage in self.stage_runs
                if stage.cost is not None
                and stage.cost.status == "estimated"
                and stage.cost.total_estimated_cost_usd is not None
            ),
            Decimal(0),
        )
        payload: dict[str, JsonValue] = {
            "schema_version": "ripple.architecture-model-study-progress.v1",
            "run_id": self.run_id,
            "status": status,
            "phase": self.phase,
            "activity": activity,
            "training_runs_completed": self.training_runs,
            "training_runs_budget": self.configuration.maximum_training_runs,
            "provider_requests_observed_or_reserved": provider_requests,
            "provider_request_budget": self.configuration.maximum_provider_requests,
            "measured_tokens": measured_tokens,
            "measured_token_budget": self.configuration.maximum_measured_tokens,
            "estimated_rate_based_cost_usd": str(estimated_cost),
            "remote_wall_seconds": self.remote_wall_seconds,
            "updated_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            **details,
        }
        progress_path = self.run_directory / "progress.json"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".progress.json.",
            suffix=".tmp",
            dir=self.run_directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, progress_path)
            progress_path.chmod(0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _set_phase(self, phase: str, *, activity: str) -> None:
        self.phase = phase
        self._write_progress(status="running", activity=activity)

    def _write(self, relative_path: str, value: Any) -> Path:
        path = self.store.write_json(relative_path, value)
        self._check_budgets()
        return path

    def _local_storage_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.run_directory.rglob("*")
            if path.is_file() and not path.is_symlink()
        )

    def _check_budgets(self) -> None:
        provider_requests = sum(
            (
                stage.provider_request_count
                if stage.provider_request_count is not None
                else 7
            )
            for stage in self.stage_runs
        )
        measured_tokens = sum(
            stage.token_usage.measured_total_tokens
            for stage in self.stage_runs
            if stage.token_usage is not None
        )
        estimated_cost = sum(
            (
                stage.cost.total_estimated_cost_usd
                for stage in self.stage_runs
                if stage.cost is not None
                and stage.cost.status == "estimated"
                and stage.cost.total_estimated_cost_usd is not None
            ),
            Decimal(0),
        )
        limits = self.configuration
        if provider_requests > limits.maximum_provider_requests:
            raise StudyExecutionError("study provider-request budget was exceeded")
        if measured_tokens > limits.maximum_measured_tokens:
            raise StudyExecutionError("study measured-token budget was exceeded")
        if estimated_cost > Decimal(str(limits.maximum_rate_based_cost_usd)):
            raise StudyExecutionError("study rate-based API cost budget was exceeded")
        if self.training_runs > limits.maximum_training_runs:
            raise StudyExecutionError("study training-run budget was exceeded")
        if self.remote_wall_seconds > limits.maximum_remote_wall_seconds:
            raise StudyExecutionError("study remote-wall-time budget was exceeded")
        if self._local_storage_bytes() > limits.maximum_local_storage_bytes:
            raise StudyExecutionError("study local-storage budget was exceeded")

    def _append_stage(self, stage: AgentStageEvidenceV2) -> None:
        self.stage_runs.append(stage)
        self._check_budgets()

    def _remote(
        self,
        *,
        source_sync: SourceSyncEvidence,
        operation: Literal[
            "environment", "simulate", "build-dataset", "train", "evaluate"
        ],
        slug: str,
        arguments: Sequence[str] = (),
        timeout: int | None = None,
    ) -> tuple[RemoteCommandResult, dict[str, JsonValue]]:
        if operation == "train" and self.training_runs + 1 > (
            self.configuration.maximum_training_runs
        ):
            raise StudyExecutionError("training-run budget exhausted before action")
        if self.remote_wall_seconds >= self.configuration.maximum_remote_wall_seconds:
            raise StudyExecutionError("remote-wall-time budget exhausted before action")
        self._write_progress(
            status="running",
            activity="remote_operation_started",
            remote_operation=operation,
            remote_slug=slug,
        )
        result = run_remote_worker(
            settings=self.configuration.remote,
            operation=operation,
            arguments=arguments,
            execution_timeout_seconds=timeout,
            source_sync=source_sync,
        )
        self.remote_wall_seconds += result.elapsed_seconds
        if operation == "train" and result.succeeded:
            self.training_runs += 1
        self._write(f"evidence/remote/{slug}.json", result)
        self._write_progress(
            status="running",
            activity="remote_operation_completed",
            remote_operation=operation,
            remote_slug=slug,
            remote_operation_succeeded=result.succeeded,
        )
        if not result.succeeded:
            raise StudyExecutionError(
                f"remote worker failed during {operation}; inspect local evidence"
            )
        return result, _last_json_object(result.stdout)

    def _model_matrix(
        self,
        *,
        dataset_summary: DatasetSummary,
        pricing: BedrockPricingSnapshot,
    ) -> BedrockModelMatrix:
        self._set_phase(
            "bedrock_model_matrix",
            activity="bedrock_catalog_and_account_check",
        )
        account_fingerprint, catalog = _catalog_and_account(self.configuration)
        availability: list[BedrockModelAvailabilityRecord] = []
        for index, model in enumerate(self.configuration.bedrock_models):
            self._write_progress(
                status="running",
                activity="bedrock_model_benchmark",
                model_index=index + 1,
                model_count=len(self.configuration.bedrock_models),
                model_id=model.model_id,
            )
            slug = _safe_slug(model.model_id)
            started = _utc_now()
            evidence_id = f"bedrock-smoke-{index:02d}"
            request_id = f"request-bedrock-smoke-{index:02d}"
            try:
                result = run_bedrock_typed_output_smoke_with_usage(
                    _settings(self.configuration, model.model_id)
                )
                completed = _utc_now()
                path = self._write(
                    f"evidence/bedrock/{slug}/smoke-output.json",
                    result.output,
                )
                token_usage = _usage(
                    input_tokens=result.input_tokens,
                    cache_write_tokens=result.cache_write_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    output_tokens=result.output_tokens,
                )
                self._append_stage(
                    _stage_success(
                        evidence_id=evidence_id,
                        request_id=request_id,
                        arm_id=None,
                        round_index=0,
                        phase="other",
                        stage="bedrock-smoke",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=started,
                        completed=completed,
                        provider_request_count=result.requests,
                        tool_call_count=result.tool_calls,
                        usage=token_usage,
                        output_artifact_id=f"bedrock-smoke-output-{index:02d}",
                        output_sha256=sha256_file(path),
                        called_tools=(),
                        pricing=pricing,
                    )
                )
                availability.append(
                    BedrockModelAvailabilityRecord(
                        availability_id=f"availability-{index:02d}",
                        model_id=model.model_id,
                        inference_profile_id=model.inference_profile_id,
                        display_name=model.display_name,
                        provider=model.provider,
                        region=self.configuration.bedrock_region,
                        category=model.category,
                        catalog_status=(
                            "listed"
                            if model.model_id in catalog
                            or (
                                model.inference_profile_id is not None
                                and model.inference_profile_id in catalog
                            )
                            else "not_listed"
                        ),
                        access_status="enabled",
                        invocation_status="succeeded",
                        supports_converse=True,
                        checked_at_utc=completed,
                    )
                )
            except Exception as exc:
                completed = _utc_now()
                self._append_stage(
                    _stage_failure(
                        evidence_id=evidence_id,
                        request_id=request_id,
                        arm_id=None,
                        round_index=0,
                        phase="other",
                        stage="bedrock-smoke",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=started,
                        completed=completed,
                        exc=exc,
                    )
                )
                code, message = _safe_failure(exc)
                availability.append(
                    BedrockModelAvailabilityRecord(
                        availability_id=f"availability-{index:02d}",
                        model_id=model.model_id,
                        inference_profile_id=model.inference_profile_id,
                        display_name=model.display_name,
                        provider=model.provider,
                        region=self.configuration.bedrock_region,
                        category=model.category,
                        catalog_status=(
                            "listed"
                            if model.model_id in catalog
                            or (
                                model.inference_profile_id is not None
                                and model.inference_profile_id in catalog
                            )
                            else "not_listed"
                        ),
                        access_status=(
                            "access_denied"
                            if "access" in code.lower()
                            else "unavailable"
                        ),
                        invocation_status="failed",
                        supports_converse=None,
                        checked_at_utc=completed,
                        failure_code=code,
                        failure_message=message,
                    )
                )

        enabled = {
            item.model_id
            for item in availability
            if item.invocation_status == "succeeded"
        }
        for index, model in enumerate(self.configuration.bedrock_models):
            if model.model_id not in enabled:
                continue
            slug = _safe_slug(model.model_id)
            planner_started = _utc_now()
            planner_evidence_id = f"architecture-generator-{index:02d}"
            try:
                planner = run_architecture_planner(
                    dataset_summary,
                    settings=_settings(self.configuration, model.model_id),
                )
                planner_completed = _utc_now()
                planner_path = self._write(
                    f"evidence/bedrock/{slug}/architecture-plan.json",
                    planner.plan,
                )
                self._append_stage(
                    _stage_success(
                        evidence_id=planner_evidence_id,
                        request_id=f"request-architecture-generator-{index:02d}",
                        arm_id=None,
                        round_index=0,
                        phase="phase_a",
                        stage="architecture-generator",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=planner_started,
                        completed=planner_completed,
                        provider_request_count=planner.request_count,
                        tool_call_count=planner.tool_call_count,
                        usage=_usage(
                            input_tokens=planner.input_tokens,
                            cache_write_tokens=planner.cache_write_tokens,
                            cache_read_tokens=planner.cache_read_tokens,
                            output_tokens=planner.output_tokens,
                        ),
                        output_artifact_id=f"architecture-plan-{index:02d}",
                        output_sha256=sha256_file(planner_path),
                        called_tools=planner.called_tools,
                        pricing=pricing,
                    )
                )
            except Exception as exc:
                planner_completed = _utc_now()
                self._append_stage(
                    _stage_failure(
                        evidence_id=planner_evidence_id,
                        request_id=f"request-architecture-generator-{index:02d}",
                        arm_id=None,
                        round_index=0,
                        phase="phase_a",
                        stage="architecture-generator",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=planner_started,
                        completed=planner_completed,
                        exc=exc,
                    )
                )
                continue

            judge_started = _utc_now()
            judge_evidence_id = f"architecture-judge-{index:02d}"
            try:
                judge = run_architecture_judge(
                    dataset_summary,
                    planner.plan,
                    shortlist_size=min(
                        self.configuration.architecture_shortlist_size,
                        len(planner.plan.candidates),
                    ),
                    settings=_settings(self.configuration, model.model_id),
                )
                judge_completed = _utc_now()
                judge_path = self._write(
                    f"evidence/bedrock/{slug}/architecture-judge.json",
                    judge.verdict,
                )
                self._append_stage(
                    _stage_success(
                        evidence_id=judge_evidence_id,
                        request_id=f"request-architecture-judge-{index:02d}",
                        arm_id=None,
                        round_index=0,
                        phase="phase_a",
                        stage="architecture-judge",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=judge_started,
                        completed=judge_completed,
                        provider_request_count=judge.request_count,
                        tool_call_count=judge.tool_call_count,
                        usage=_usage(
                            input_tokens=judge.input_tokens,
                            cache_write_tokens=judge.cache_write_tokens,
                            cache_read_tokens=judge.cache_read_tokens,
                            output_tokens=judge.output_tokens,
                        ),
                        output_artifact_id=f"architecture-judge-output-{index:02d}",
                        output_sha256=sha256_file(judge_path),
                        called_tools=judge.called_tools,
                        pricing=pricing,
                    )
                )
            except Exception as exc:
                judge_completed = _utc_now()
                self._append_stage(
                    _stage_failure(
                        evidence_id=judge_evidence_id,
                        request_id=f"request-architecture-judge-{index:02d}",
                        arm_id=None,
                        round_index=0,
                        phase="phase_a",
                        stage="architecture-judge",
                        model=model,
                        region=self.configuration.bedrock_region,
                        started=judge_started,
                        completed=judge_completed,
                        exc=exc,
                    )
                )

        matrix = BedrockModelMatrix(
            matrix_id=f"model-matrix-{self.run_id[-8:]}",
            account_fingerprint_sha256=account_fingerprint,
            availability=tuple(availability),
            pricing=pricing,
        )
        self._write("evidence/bedrock/model-matrix.json", matrix)
        return matrix

    def _tuning_decision(
        self,
        *,
        arm: StudyArchitectureArm,
        iteration: int,
        measurement: TuningMeasurement,
        prior: tuple[TuningMeasurement, ...],
        pricing: BedrockPricingSnapshot,
        tuning_model: StudyBedrockModel,
        tuning_available: bool,
    ) -> tuple[TuningAction, str, str]:
        evidence_id = f"tuning-{arm.arm_id}-{iteration:02d}"
        request_id = f"request-{evidence_id}"
        eligible = eligible_tuning_actions(
            measurement,
            minimum_iterations=self.configuration.minimum_tuning_iterations,
            maximum_iterations=self.configuration.maximum_tuning_iterations,
            minimum_validation_balanced_accuracy=(
                self.configuration.minimum_validation_balanced_accuracy_for_early_stop
            ),
            maximum_generalization_gap=(
                self.configuration.maximum_generalization_gap_for_early_stop
            ),
        )
        started = _utc_now()
        try:
            if not tuning_available:
                raise StudyExecutionError("configured tuning model is unavailable")
            result = run_tuning_agent(
                measurement,
                prior_measurements=prior,
                minimum_iterations=self.configuration.minimum_tuning_iterations,
                maximum_iterations=self.configuration.maximum_tuning_iterations,
                minimum_validation_balanced_accuracy=(
                    self.configuration.minimum_validation_balanced_accuracy_for_early_stop
                ),
                maximum_generalization_gap=(
                    self.configuration.maximum_generalization_gap_for_early_stop
                ),
                settings=_settings(self.configuration, tuning_model.model_id),
            )
            completed = _utc_now()
            decision_path = self._write(
                f"evidence/tuning/{arm.arm_id}/iteration-{iteration:02d}.json",
                result.decision,
            )
            self._append_stage(
                _stage_success(
                    evidence_id=evidence_id,
                    request_id=request_id,
                    arm_id=arm.arm_id,
                    round_index=iteration,
                    phase="phase_b",
                    stage="tuning-planner",
                    model=tuning_model,
                    region=self.configuration.bedrock_region,
                    started=started,
                    completed=completed,
                    provider_request_count=result.request_count,
                    tool_call_count=result.tool_call_count,
                    usage=_usage(
                        input_tokens=result.input_tokens,
                        cache_write_tokens=result.cache_write_tokens,
                        cache_read_tokens=result.cache_read_tokens,
                        output_tokens=result.output_tokens,
                    ),
                    output_artifact_id=f"tuning-output-{arm.arm_id}-{iteration:02d}",
                    output_sha256=sha256_file(decision_path),
                    called_tools=result.called_tools,
                    pricing=pricing,
                )
            )
            return result.decision.action, result.decision.rationale, evidence_id
        except Exception as exc:
            completed = _utc_now()
            self._append_stage(
                _stage_failure(
                    evidence_id=evidence_id,
                    request_id=request_id,
                    arm_id=arm.arm_id,
                    round_index=iteration,
                    phase="phase_b",
                    stage="tuning-planner",
                    model=tuning_model,
                    region=self.configuration.bedrock_region,
                    started=started,
                    completed=completed,
                    exc=exc,
                )
            )
            action = (
                TuningAction.STOP
                if TuningAction.STOP in eligible
                else eligible[0]
            )
            fallback = {
                "schema_version": "ripple.code-owned-tuning-fallback.v1",
                "arm_id": arm.arm_id,
                "iteration": iteration,
                "action": action.value,
                "eligible_actions": [item.value for item in eligible],
                "reason": (
                    "The model stage failed; deterministic code selected the first "
                    "eligible action, or the mandatory budget stop."
                ),
                "scientific_claim_allowed": False,
            }
            self._write(
                f"evidence/tuning/{arm.arm_id}/iteration-{iteration:02d}-fallback.json",
                fallback,
            )
            return action, fallback["reason"], evidence_id

    def _train_arms(
        self,
        *,
        source_sync: SourceSyncEvidence,
        manifest: DatasetManifest,
        manifest_sha256: str,
        pricing: BedrockPricingSnapshot,
        model_matrix: BedrockModelMatrix,
    ) -> dict[str, _TrainedArm]:
        self._set_phase(
            "validation_only_training",
            activity="prepare_training_arms",
        )
        tuning_model = next(
            item
            for item in self.configuration.bedrock_models
            if item.model_id == self.configuration.tuning_model_id
        )
        tuning_available = any(
            item.model_id == tuning_model.model_id
            and item.invocation_status == "succeeded"
            for item in model_matrix.availability
        )
        dataset_shape = manifest.samples[0].shape
        trained: dict[str, _TrainedArm] = {}
        for arm_index, arm in enumerate(self.configuration.arms):
            state = _TrainedArm(
                configuration=arm,
                trajectories=[],
                records=[],
                local_directories=[],
                remote_directories=[],
            )
            bound = BoundArchitecture(
                candidate=arm.candidate,
                input_shape=dataset_shape[1:],
                channels=dataset_shape[0],
                num_classes=len(manifest.class_names),
                source_revision=RIPPLE_MODEL_BUILDERS_REVISION,
            )
            bound_path = self._write(
                f"inputs/architectures/{arm.arm_id}.json",
                bound,
            )
            remote_bound = copy_to_remote_run(
                local_path=bound_path,
                remote_relative_path=(
                    f"{self.run_id}/inputs/architectures/{arm.arm_id}.json"
                ),
                settings=self.configuration.remote,
            )
            current = self.configuration.base_training.model_copy(
                update={"seed": self.configuration.random_seeds[0]}
            )
            measurements: list[TuningMeasurement] = []
            for iteration in range(self.configuration.maximum_tuning_iterations):
                self._write_progress(
                    status="running",
                    activity="training_iteration",
                    arm_id=arm.arm_id,
                    arm_index=arm_index + 1,
                    arm_count=len(self.configuration.arms),
                    iteration=iteration + 1,
                    maximum_iterations=self.configuration.maximum_tuning_iterations,
                )
                training_path = self._write(
                    f"inputs/training/{arm.arm_id}/iteration-{iteration:02d}.json",
                    current,
                )
                remote_training_configuration = copy_to_remote_run(
                    local_path=training_path,
                    remote_relative_path=(
                        f"{self.run_id}/inputs/training/{arm.arm_id}/"
                        f"iteration-{iteration:02d}.json"
                    ),
                    settings=self.configuration.remote,
                )
                remote_training_dir = (
                    f"{self.remote_run_root}/training/{arm.arm_id}/"
                    f"iteration-{iteration:02d}"
                )
                result, payload = self._remote(
                    source_sync=source_sync,
                    operation="train",
                    slug=f"train-{arm.arm_id}-{iteration:02d}",
                    arguments=(
                        "--dataset-root",
                        f"{self.remote_run_root}/dataset",
                        "--manifest",
                        f"{self.remote_run_root}/dataset/dataset_manifest.json",
                        "--architecture",
                        remote_bound,
                        "--training-config",
                        remote_training_configuration,
                        "--output-dir",
                        remote_training_dir,
                    ),
                    timeout=self.configuration.remote_training_timeout_seconds,
                )
                local_training_dir = copy_remote_output(
                    remote_path=remote_training_dir,
                    local_parent=self.run_directory
                    / "remote"
                    / "training"
                    / arm.arm_id,
                    settings=self.configuration.remote,
                    maximum_bytes=self.configuration.maximum_transfer_bytes,
                )
                record = TrainingRunRecord.model_validate_json(
                    (local_training_dir / "training_record.json").read_bytes(),
                    strict=True,
                )
                if (
                    record.dataset_manifest_sha256 != manifest_sha256
                    or record.architecture != bound
                    or record.configuration != current
                    or record.architecture.candidate.candidate_id != arm.arm_id
                ):
                    raise StudyExecutionError(
                        "remote training record changed a bound study input"
                    )
                if payload.get("weights_sha256") != record.weights_sha256:
                    raise StudyExecutionError("remote checkpoint digest evidence differs")
                checkpoint_train = record.best_checkpoint_train
                if checkpoint_train is None:
                    raise StudyExecutionError(
                        "training record omitted selected-checkpoint train metrics"
                    )
                gap = checkpoint_train.accuracy - record.best_validation.accuracy
                measurement = TuningMeasurement(
                    candidate_id=arm.arm_id,
                    iteration=iteration,
                    train_accuracy=checkpoint_train.accuracy,
                    validation_loss=record.best_validation.loss,
                    validation_accuracy=record.best_validation.accuracy,
                    validation_balanced_accuracy=(
                        record.best_validation.balanced_accuracy
                    ),
                    validation_roc_auc=record.best_validation.roc_auc,
                    generalization_gap=gap,
                    configuration=current,
                )
                action, rationale, evidence_id = self._tuning_decision(
                    arm=arm,
                    iteration=iteration,
                    measurement=measurement,
                    prior=tuple(measurements),
                    pricing=pricing,
                    tuning_model=tuning_model,
                    tuning_available=tuning_available,
                )
                measurements.append(measurement)
                state.trajectories.append(
                    ArmTrajectoryPoint(
                        arm_id=arm.arm_id,
                        iteration=iteration,
                        training_run_id=record.run_id,
                        checkpoint_sha256=record.weights_sha256,
                        train_metrics=_classification(checkpoint_train),
                        validation_metrics=_classification(record.best_validation),
                        accuracy_generalization_gap=gap,
                        planner_decision=action.value,
                        planner_rationale=rationale,
                        stage_evidence_ids=(evidence_id,),
                        worker_seconds=record.elapsed_seconds,
                        remote_wall_seconds=result.elapsed_seconds,
                    )
                )
                state.records.append(record)
                state.local_directories.append(local_training_dir)
                state.remote_directories.append(remote_training_dir)
                state.total_worker_seconds += record.elapsed_seconds
                state.total_remote_wall_seconds += result.elapsed_seconds
                if action == TuningAction.STOP:
                    break
                current = apply_tuning_action(current, action)

            if len(state.records) < self.configuration.minimum_tuning_iterations:
                raise StudyExecutionError("an arm stopped before its minimum iterations")
            state.selected_iteration = min(
                range(len(state.records)),
                key=lambda item: (
                    -state.records[item].best_validation.balanced_accuracy,
                    state.records[item].best_validation.loss,
                    item,
                ),
            )
            trained[arm.arm_id] = state
        return trained

    def _evaluate_locked_arms(
        self,
        *,
        source_sync: SourceSyncEvidence,
        manifest_sha256: str,
        trained: dict[str, _TrainedArm],
        selected_arm_id: str,
    ) -> None:
        self._set_phase(
            "locked_heldout_evaluation",
            activity="evaluate_locked_arms",
        )
        evaluation_arms = (selected_arm_id, self.configuration.baseline_arm_id)
        if len(set(evaluation_arms)) != 2:
            raise StudyExecutionError("searched selection and baseline are not distinct")
        for arm_id in evaluation_arms:
            self._write_progress(
                status="running",
                activity="heldout_evaluation",
                arm_id=arm_id,
            )
            state = trained[arm_id]
            selected = state.selected_iteration
            record = state.records[selected]
            remote_output = f"{self.remote_run_root}/evaluation/{arm_id}/final.json"
            result, payload = self._remote(
                source_sync=source_sync,
                operation="evaluate",
                slug=f"evaluate-{arm_id}",
                arguments=(
                    "--dataset-root",
                    f"{self.remote_run_root}/dataset",
                    "--manifest",
                    f"{self.remote_run_root}/dataset/dataset_manifest.json",
                    "--training-dir",
                    state.remote_directories[selected],
                    "--training-record",
                    f"{state.remote_directories[selected]}/training_record.json",
                    "--output",
                    remote_output,
                ),
                timeout=self.configuration.remote_evaluation_timeout_seconds,
            )
            local_parent = self.run_directory / "remote" / "evaluation" / arm_id
            output_path = copy_remote_output(
                remote_path=remote_output,
                local_parent=local_parent,
                settings=self.configuration.remote,
                maximum_bytes=self.configuration.maximum_transfer_bytes,
            )
            copy_remote_output(
                remote_path=f"{remote_output}.consumed",
                local_parent=local_parent,
                settings=self.configuration.remote,
                maximum_bytes=self.configuration.maximum_transfer_bytes,
            )
            evaluation = FinalEvaluationRecord.model_validate_json(
                output_path.read_bytes(), strict=True
            )
            if (
                evaluation.selected_training_run_id != record.run_id
                or evaluation.checkpoint_sha256 != record.weights_sha256
                or evaluation.dataset_manifest_sha256 != manifest_sha256
            ):
                raise StudyExecutionError("held-out evaluation changed the locked input")
            if payload.get("evaluation_id") != evaluation.evaluation_id:
                raise StudyExecutionError("held-out evaluation identity differs")
            state.test_evaluation = evaluation
            state.test_evaluated_at_utc = _utc_now()
            worker_seconds = payload.get("worker_elapsed_seconds")
            if isinstance(worker_seconds, (int, float)) and not isinstance(
                worker_seconds, bool
            ):
                state.total_worker_seconds += float(worker_seconds)
            state.total_remote_wall_seconds += result.elapsed_seconds

    def run(self) -> CompletedArchitectureModelStudy:
        started_at = _utc_now()
        try:
            self._set_phase("load_inputs", activity="load_and_validate_inputs")
            specification_path = _resolve_input(
                self.configuration_base,
                self.configuration.simulation_spec,
            )
            pricing_path = _resolve_input(
                self.configuration_base,
                self.configuration.pricing_snapshot,
            )
            specification = load_study_spec(specification_path)
            pricing = BedrockPricingSnapshot.model_validate_json(
                pricing_path.read_bytes(), strict=True
            )
            self._write("inputs/study-configuration.json", self.configuration)
            source_revisions = ScientistSourceRevisions.model_validate_json(
                checked_real_file(
                    self.repository_root
                    / "ripple"
                    / "scientist"
                    / "locks"
                    / "source-revisions.json"
                ).read_bytes(),
                strict=True,
            )
            precedent = next(
                item
                for item in source_revisions.source_precedent_only
                if item.name == "DeepLense-AI-Scientist"
            )
            if sha256_file(
                checked_real_file(self.repository_root / precedent.port_path)
            ) != precedent.port_sha256:
                raise StudyExecutionError(
                    "architecture builder differs from its pinned source precedent lock"
                )
            self._write("inputs/source-revisions.json", source_revisions)
            local_specification_path = self._write(
                "inputs/simulation-specification.json", specification
            )
            self._write("inputs/bedrock-pricing-snapshot.json", pricing)

            self._set_phase("sync_worker_source", activity="sync_verified_source")
            source_sync = _sync_sources(
                repository_root=self.repository_root,
                configuration=self.configuration,
            )
            self._write("evidence/source-sync.json", source_sync)
            self._remote(
                source_sync=source_sync,
                operation="environment",
                slug="environment",
                arguments=("--require-cuda", "--require-scientific-stack"),
            )

            self._set_phase("simulate_dataset", activity="generate_synthetic_dataset")
            remote_specification = copy_to_remote_run(
                local_path=local_specification_path,
                remote_relative_path=f"{self.run_id}/inputs/simulation-specification.json",
                settings=self.configuration.remote,
            )
            self._remote(
                source_sync=source_sync,
                operation="simulate",
                slug="simulate-dataset",
                arguments=(
                    "--spec",
                    remote_specification,
                    "--output-dir",
                    f"{self.remote_run_root}/dataset",
                ),
                timeout=self.configuration.remote_simulation_timeout_seconds,
            )
            self._remote(
                source_sync=source_sync,
                operation="build-dataset",
                slug="build-dataset-manifest",
                arguments=(
                    "--simulation-root",
                    f"{self.remote_run_root}/dataset",
                ),
            )
            local_dataset = copy_remote_output(
                remote_path=f"{self.remote_run_root}/dataset",
                local_parent=self.run_directory / "remote",
                settings=self.configuration.remote,
                maximum_bytes=self.configuration.maximum_transfer_bytes,
            )
            manifest_path = local_dataset / "dataset_manifest.json"
            manifest = load_dataset_manifest(manifest_path)
            manifest_sha256 = sha256_file(manifest_path)
            if (
                manifest.purpose != "scientific_training"
                or manifest.scientific_use_allowed
                or manifest.bands != specification.rendering.bands
            ):
                raise StudyExecutionError("dataset qualification or band contract changed")
            split_counts = {
                "train": len(manifest.splits.train_sample_ids),
                "validation": len(manifest.splits.validation_sample_ids),
                "test": len(manifest.splits.test_sample_ids),
            }
            dataset_summary = DatasetSummary(
                dataset_id=manifest.dataset_id,
                manifest_sha256=manifest_sha256,
                image_shape=manifest.samples[0].shape[1:],
                channels=manifest.samples[0].shape[0],
                class_names=manifest.class_names,
                split_counts=split_counts,
                purpose=manifest.purpose,
                notes=manifest.qualification_notes,
            )
            self._write("evidence/dataset-summary.json", dataset_summary)

            model_matrix = self._model_matrix(
                dataset_summary=dataset_summary,
                pricing=pricing,
            )
            trained = self._train_arms(
                source_sync=source_sync,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                pricing=pricing,
                model_matrix=model_matrix,
            )
            searched = [
                state
                for state in trained.values()
                if state.configuration.role == "searched"
            ]
            selected_state = min(
                searched,
                key=lambda state: (
                    -state.records[
                        state.selected_iteration
                    ].best_validation.balanced_accuracy,
                    state.records[state.selected_iteration].best_validation.loss,
                    state.configuration.arm_id,
                ),
            )
            selected_arm_id = selected_state.configuration.arm_id
            selection_lock = {
                "schema_version": "ripple.study-selection-lock.v1",
                "selected_searched_arm_id": selected_arm_id,
                "baseline_arm_id": self.configuration.baseline_arm_id,
                "selection_metric": "validation_balanced_accuracy",
                "tie_breakers": ["validation_loss_ascending", "arm_id_ascending"],
                "test_split_opened_before_lock": False,
                "locked_checkpoint_sha256": {
                    arm_id: state.records[state.selected_iteration].weights_sha256
                    for arm_id, state in trained.items()
                },
                "locked_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            }
            self._write("evidence/selection-lock.json", selection_lock)
            self._evaluate_locked_arms(
                source_sync=source_sync,
                manifest_sha256=manifest_sha256,
                trained=trained,
                selected_arm_id=selected_arm_id,
            )

            arm_results: list[ArmStudyResult] = []
            for arm in self.configuration.arms:
                state = trained[arm.arm_id]
                selected = state.records[state.selected_iteration]
                evaluation = state.test_evaluation
                arm_results.append(
                    ArmStudyResult(
                        arm_id=arm.arm_id,
                        candidate_id=arm.candidate.candidate_id,
                        display_name=arm.display_name,
                        architecture_family=arm.candidate.family.value,
                        status="succeeded",
                        parameter_count=selected.parameter_count,
                        tuning_iterations=len(state.trajectories),
                        total_worker_seconds=state.total_worker_seconds,
                        total_remote_wall_seconds=state.total_remote_wall_seconds,
                        trajectory=tuple(state.trajectories),
                        final_evaluation=FinalEvaluationSummary(
                            evaluation_id=f"final-{arm.arm_id}",
                            arm_id=arm.arm_id,
                            selected_training_run_id=selected.run_id,
                            checkpoint_sha256=selected.weights_sha256,
                            validation_metrics=_classification(
                                selected.best_validation
                            ),
                            test_metrics=(
                                None
                                if evaluation is None
                                else _classification(evaluation.metrics)
                            ),
                            test_evaluated_at_utc=state.test_evaluated_at_utc,
                        ),
                        fair_training_comparison_allowed=True,
                    )
                )

            architecture_plan = ArchitecturePlan(
                plan_id=f"architecture-plan-{self.run_id[-8:]}",
                candidates=tuple(
                    ArchitecturePlanCandidate(
                        candidate_id=arm.candidate.candidate_id,
                        arm_id=arm.arm_id,
                        display_name=arm.display_name,
                        architecture_family=arm.candidate.family.value,
                        role=arm.role,
                        plan_order=index,
                        specification_sha256=canonical_json_sha256(arm.candidate),
                    )
                    for index, arm in enumerate(self.configuration.arms)
                ),
                selected_arm_id=selected_arm_id,
            )
            mriganka_manifest_path = checked_real_file(
                self.repository_root
                / "ripple"
                / "inference"
                / "manifests"
                / "mriganka-enn-sda-epoch20.hsc-eval.v1.json"
            )
            mriganka_manifest = json.loads(
                mriganka_manifest_path.read_text(encoding="utf-8")
            )
            mriganka_checkpoint_sha256 = canonical_json_sha256(
                {
                    "encoder_sha256": mriganka_manifest["encoder"]["sha256"],
                    "classifier_sha256": mriganka_manifest["classifier"]["sha256"],
                }
            )
            mriganka = MrigankaZeroShotSummary(
                status="blocked",
                checkpoint_sha256=mriganka_checkpoint_sha256,
                learned_parameter_count=3_391_746,
                materialized_state_element_count=41_656_409,
                reason=(
                    "The frozen HSC-trained checkpoint is retained as a separate "
                    "resource reference. A same-dataset score is blocked because its "
                    "physical channel order, angular field/resampling, PSF, calibration, "
                    "and HSC-to-synthetic/Rubin domain compatibility are unresolved."
                ),
            )
            dataset_identity = StudyDatasetIdentity(
                dataset_id=manifest.dataset_id,
                manifest_sha256=manifest_sha256,
                purpose=manifest.purpose,
                data_origin="synthetic_simulation",
                source=(
                    "SLSim synthetic g/r/i benchmark at pinned revision "
                    f"{manifest.simulator.revision}"
                ),
                bands=manifest.bands,
                sample_shape_chw=manifest.samples[0].shape,
                class_names=manifest.class_names,
                split_counts=split_counts,
                scientific_use_allowed=manifest.scientific_use_allowed,
                supports_scientific_claims=False,
                notes=manifest.qualification_notes,
            )
            qualification = ScientificQualification(
                evidence_level="synthetic_benchmark_unqualified",
                scientific_performance_claim_allowed=False,
                fair_architecture_comparison_allowed=True,
                held_out_test_used_once_after_selection=True,
                limitations=(
                    "The dataset is a controlled synthetic benchmark and is not a representative Rubin population.",
                    "This run uses one random seed; uncertainty across seeds is not measured.",
                    "Only the selected searched arm and the pre-registered ResNet-34 baseline open the held-out split.",
                    "Rate-based API costs are not AWS invoices and exclude GPU, storage, transfer, tax, support, and discounts.",
                    "Mriganka remains a separately reported external HSC checkpoint, not a fair trained-arm peer.",
                ),
            )
            search_protocol = SearchProtocol(
                protocol_id=f"search-protocol-{self.run_id[-8:]}",
                primary_selection_metric="validation_balanced_accuracy",
                comparison_regime=ComparisonRegime(
                    regime_id=f"shared-slsim-{self.run_id[-8:]}",
                    description=(
                        "All six native arms use one content-addressed dataset, frozen "
                        "splits, seed, optimizer family, epoch budget, and validation-only "
                        "selection. Mriganka is excluded from this regime."
                    ),
                ),
                phase_a_description=(
                    "Each successfully invoked Bedrock model receives the same typed "
                    "connectivity, architecture-generation, and judging contracts. "
                    "These calls measure orchestration behavior and cost; they do not "
                    "silently replace the pre-registered training arms."
                ),
                phase_b_description=(
                    "Each pre-registered arm is trained from scratch for bounded "
                    "validation-only tuning iterations. Deterministic code applies "
                    "allowlisted changes, locks selection, then opens held-out data."
                ),
                random_seeds=self.configuration.random_seeds,
            )
            tuning_policy = TuningPolicy(
                policy_id=f"tuning-policy-{self.run_id[-8:]}",
                maximum_iterations=self.configuration.maximum_tuning_iterations,
                allowed_decisions=tuple(item.value for item in TuningAction),
                stop_decision=TuningAction.STOP.value,
                generalization_gap_threshold=(
                    self.configuration.maximum_generalization_gap_for_early_stop
                ),
                minimum_validation_improvement=(
                    self.configuration.minimum_validation_improvement
                ),
                per_iteration_epoch_budget=self.configuration.base_training.epochs,
                augmentation_policy=(
                    "D4 augmentation may only be enabled by the allowlisted enable_d4 action."
                ),
                policy_description=(
                    "The agent can select only code-owned actions. Stop is unavailable "
                    "before the minimum iterations and mandatory at the hard budget."
                ),
            )
            aggregate = StudyAggregateInput(
                study_id=self.configuration.study_id,
                title=self.configuration.title,
                dataset=dataset_identity,
                qualification=qualification,
                search_protocol=search_protocol,
                tuning_policy=tuning_policy,
                model_matrix=model_matrix,
                architecture_plan=architecture_plan,
                stage_runs=tuple(self.stage_runs),
                arm_results=tuple(arm_results),
                mriganka_zero_shot=mriganka,
                started_at_utc=started_at,
                completed_at_utc=_utc_now(),
            )
            aggregate = apply_rate_based_costing(aggregate)
            aggregate_path = self._write("study/study-aggregate.json", aggregate)
            self._set_phase("paper_tables", activity="render_paper_tables")
            paper_tables = render_paper_tables(
                aggregate,
                self.run_directory / "paper_tables",
            )

            self._set_phase("completion", activity="build_completion_manifest")
            self._write_progress(
                status="complete",
                activity="study_complete",
                selected_searched_arm_id=selected_arm_id,
                baseline_arm_id=self.configuration.baseline_arm_id,
            )
            artifacts: list[dict[str, JsonValue]] = []
            for path in sorted(self.run_directory.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    relative = path.relative_to(self.run_directory).as_posix()
                    if relative in {"completion.json", "failure.json"}:
                        continue
                    artifacts.append(
                        {
                            "relative_path": relative,
                            "byte_count": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
            completion = {
                "schema_version": "ripple.architecture-model-study-completion.v1",
                "status": "complete",
                "run_id": self.run_id,
                "study_id": aggregate.study_id,
                "dataset_id": aggregate.dataset.dataset_id,
                "selected_searched_arm_id": selected_arm_id,
                "baseline_arm_id": self.configuration.baseline_arm_id,
                "study_aggregate": {
                    "relative_path": aggregate_path.relative_to(
                        self.run_directory
                    ).as_posix(),
                    "sha256": sha256_file(aggregate_path),
                },
                "paper_table_manifest": {
                    "relative_path": paper_tables.manifest_path.relative_to(
                        self.run_directory
                    ).as_posix(),
                    "sha256": paper_tables.manifest_sha256,
                },
                "artifact_count_before_completion": len(artifacts),
                "artifacts": artifacts,
                "scientific_performance_claim_allowed": False,
                "mriganka_fair_training_comparison_allowed": False,
                "completion_written_last": True,
                "completed_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
            }
            completion_path = self._write("completion.json", completion)
            return CompletedArchitectureModelStudy(
                run_directory=self.run_directory,
                aggregate_path=aggregate_path,
                completion_path=completion_path,
                paper_tables=paper_tables,
                study=aggregate,
            )
        except Exception as exc:
            if not (self.run_directory / "completion.json").exists():
                try:
                    self._write_progress(
                        status="failed",
                        activity="study_failed",
                        error_type=type(exc).__name__,
                    )
                except Exception:
                    pass
                failure = {
                    "schema_version": "ripple.architecture-model-study-failure.v1",
                    "status": "failed",
                    "run_id": self.run_id,
                    "phase": self.phase,
                    "error_type": type(exc).__name__,
                    "safe_message": (
                        "The study failed; inspect the already persisted local evidence. "
                        "Raw cloud and remote exception text is intentionally omitted."
                    ),
                    "failed_at_utc": _utc_now().isoformat().replace("+00:00", "Z"),
                }
                try:
                    self._write("failure.json", failure)
                except Exception:
                    pass
            if isinstance(exc, StudyExecutionError):
                raise
            raise StudyExecutionError(
                f"study failed during {self.phase}; inspect {self.run_directory}"
            ) from exc


def run_architecture_model_study(
    configuration: ArchitectureModelStudyConfiguration,
    *,
    configuration_base: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    run_id: str | None = None,
) -> CompletedArchitectureModelStudy:
    return ArchitectureModelStudyRunner(
        configuration,
        configuration_base=configuration_base,
        repository_root=repository_root,
        run_id=run_id,
    ).run()


__all__ = [
    "ArchitectureModelStudyRunner",
    "CompletedArchitectureModelStudy",
    "StudyExecutionError",
    "run_architecture_model_study",
]
