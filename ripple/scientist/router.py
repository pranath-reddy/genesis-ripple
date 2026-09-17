"""Bounded dispatcher for the three RIPPLe research routes.

The dispatcher owns routing only.  Scientific work remains in registered,
typed implementations: the DP2/Mriganka path may retrieve, preprocess, bridge,
and run its explicitly unqualified technical classifier integration; the
researcher path may inspect an immutable source snapshot and answer an explicit
analysis goal, and the synthetic path delegates to the existing campaign
runner.  This module has no generic Python execution facility and never invokes
LensCat before a real candidate report.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from ripple.dp2.client import Dp2Client
from ripple.dp2.errors import Dp2Error
from ripple.dp2.models import Dp2ClientConfig, Dp2CutoutRequest
from ripple.dp2.package_service import (
    Dp2PackageService,
    LoadedDp2Cutout,
    load_cutout_package,
)
from ripple.dp2.service import create_private_run_directory
from ripple.modeling.contracts import ModelManifestRef
from ripple.modeling.manifest_io import load_model_manifest
from ripple.modeling.service import (
    build_default_registry,
    run_registered_preprocessing,
)
from ripple.preprocessing.mriganka_enn.contracts import (
    MrigankaEnnThreeBandModelInputPackage,
)

from .artifacts import ArtifactStore, sha256_file
from .paths import checked_absolute_path, checked_real_directory, checked_real_file
from .schemas.campaign import SimulationCampaignConfiguration
from .schemas.common import BudgetUsage, canonical_json_sha256, utc_now
from .schemas.orchestration import (
    PIPELINE_REQUEST_ADAPTER,
    MrigankaDp2Request,
    PipelineRunRequest,
    ResearcherModelRequest,
    SimulationTrainingRequest,
)
from .schemas.report import MrigankaDp2TechnicalReport
from .schemas.repository import (
    RepositoryIntakeManifest,
    RepositoryIntakePolicy,
    RepositoryIntakeRequest,
)
from .schemas.routes import (
    MrigankaBridgeResult,
    MrigankaDp2BandResult,
    MrigankaDp2RouteCompletion,
    MrigankaDp2RouteResult,
    MrigankaM3Result,
    MrigankaM4Result,
    MrigankaRouteArtifactRef,
    PipelineRouteResult,
    ResearcherModelRouteResult,
    RoutePlan,
    SimulationTrainingRouteResult,
)

_MAX_REQUEST_BYTES = 2 * 1024 * 1024
_MAX_CAMPAIGN_CONFIGURATION_BYTES = 2 * 1024 * 1024
_MIB = 1024 * 1024
_MRIGANKA_TOOL_CALLS = 7
_MRIGANKA_M2_STAGE_RESERVATION_BYTES = 66 * _MIB
_MRIGANKA_M3_FIXED_RESERVATION_BYTES = 54 * _MIB
_MRIGANKA_BRIDGE_STAGE_RESERVATION_BYTES = 12 * _MIB
_MRIGANKA_M4_STAGE_RESERVATION_BYTES = 16 * _MIB
_MRIGANKA_MINIMUM_STORAGE_BYTES = _MRIGANKA_M2_STAGE_RESERVATION_BYTES
_MRIGANKA_MAX_ARTIFACT_BYTES = 2 * 1024**3
_RESEARCHER_FIXED_TOOL_CALLS = 1
_RESEARCHER_MIN_AGENT_TOOL_CALLS = 8
_RESEARCHER_ARTIFACT_RESERVATION_BYTES = 16 * _MIB
_RESEARCHER_MIN_STORAGE_BYTES = 40 * _MIB


def _research_run_id(request_id: str) -> str:
    """Create one immutable attempt ID while retaining the logical request ID."""

    prefix = request_id[:100].rstrip("._-")
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class RouteDispatchError(RuntimeError):
    """Credential-free route error suitable for a bounded CLI status."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _directory_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _write_budgeted_json(
    store: ArtifactStore,
    relative_path: str,
    value: BaseModel | dict[str, Any],
    *,
    maximum_storage_bytes: int,
) -> Path:
    encoded_size = len(store.json_bytes(value))
    if _directory_bytes(store.root) + encoded_size > maximum_storage_bytes:
        raise RouteDispatchError(
            code="route_storage_budget_exhausted",
            message="The next immutable artifact would exceed the route storage budget.",
        )
    return store.write_json(relative_path, value)


def _require_researcher_budget(request: ResearcherModelRequest) -> None:
    minimum_tools = _RESEARCHER_FIXED_TOOL_CALLS + _RESEARCHER_MIN_AGENT_TOOL_CALLS
    if (
        request.budget.max_llm_requests < 10
        or request.budget.max_tool_calls < minimum_tools
        or request.budget.max_storage_bytes < _RESEARCHER_MIN_STORAGE_BYTES
    ):
        raise RouteDispatchError(
            code="researcher_agent_budget_too_small",
            message=(
                "The researcher route needs at least ten model requests, nine total "
                "tool calls, and forty MiB of storage before repository intake starts."
            ),
        )


def _bounded_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        candidate = checked_real_file(path)
        state = os.lstat(candidate)
    except (FileNotFoundError, ValueError):
        raise RouteDispatchError(
            code=f"{label}_not_found",
            message=f"The required {label.replace('_', ' ')} file is unavailable.",
        ) from None
    if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
        raise RouteDispatchError(
            code=f"invalid_{label}",
            message=f"The {label.replace('_', ' ')} must be a regular non-symlink file.",
        )
    if state.st_size <= 0 or state.st_size > maximum_bytes:
        raise RouteDispatchError(
            code=f"{label}_size_out_of_bounds",
            message=f"The {label.replace('_', ' ')} is outside its fixed byte bound.",
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _stable_file_identity(state) != _stable_file_identity(opened):
            raise RouteDispatchError(
                code=f"{label}_changed_before_read",
                message=f"The {label.replace('_', ' ')} changed before it was read.",
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise RouteDispatchError(
                    code=f"{label}_size_changed",
                    message=f"The {label.replace('_', ' ')} exceeded its byte bound.",
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if (
            _stable_file_identity(opened) != _stable_file_identity(final)
            or byte_count != final.st_size
        ):
            raise RouteDispatchError(
                code=f"{label}_changed_during_read",
                message=f"The {label.replace('_', ' ')} changed while it was read.",
            )
        return b"".join(chunks)
    except RouteDispatchError:
        raise
    except OSError as exc:
        raise RouteDispatchError(
            code=f"{label}_read_failed",
            message=(
                f"The {label.replace('_', ' ')} could not be read "
                f"({type(exc).__name__})."
            ),
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _stable_file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
    )


def _declared_file(value: str, *, base_directory: Path, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    _bounded_regular_file(path, maximum_bytes=64 * 1024 * 1024, label=label)
    return checked_real_file(path)


def _mriganka_declared_path(value: str, *, base_directory: Path) -> Path:
    """Lexically resolve a request-relative known-route path without following links."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    # The shipped request intentionally walks from configs/scientist back to the
    # repository root. Collapse that traversal lexically, then let checked_real_*
    # reject every symlink component and require the final on-disk type.
    return Path(os.path.normpath(os.fspath(path.absolute())))


def _declared_mriganka_file(
    value: str,
    *,
    base_directory: Path,
    label: str,
) -> Path:
    path = _mriganka_declared_path(value, base_directory=base_directory)
    _bounded_regular_file(path, maximum_bytes=64 * 1024 * 1024, label=label)
    return checked_real_file(path)


def _declared_mriganka_directory(
    value: str,
    *,
    base_directory: Path,
    label: str,
) -> Path:
    path = _mriganka_declared_path(value, base_directory=base_directory)
    try:
        return checked_real_directory(path)
    except ValueError:
        raise RouteDispatchError(
            code=f"{label}_not_found",
            message=f"The required {label.replace('_', ' ')} directory is unavailable.",
        ) from None


def _mriganka_artifact_ref(path: Path) -> MrigankaRouteArtifactRef:
    descriptor: int | None = None
    try:
        candidate = checked_real_file(path)
        before = os.lstat(candidate)
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(opened):
            raise OSError("artifact identity changed before hashing")
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            byte_count += len(chunk)
            if byte_count > _MRIGANKA_MAX_ARTIFACT_BYTES:
                raise OSError("artifact exceeded its route reference bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(candidate)
        if (
            _stable_file_identity(opened) != _stable_file_identity(after)
            or _stable_file_identity(after) != _stable_file_identity(path_after)
            or byte_count != after.st_size
        ):
            raise OSError("artifact identity changed while hashing")
    except (OSError, ValueError):
        raise RouteDispatchError(
            code="mriganka_artifact_unavailable",
            message="A required known-route artifact is unavailable or unsafe.",
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return MrigankaRouteArtifactRef(
        path=str(candidate),
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _enforce_mriganka_storage_budget(
    run_directory: Path,
    *,
    maximum_storage_bytes: int,
) -> None:
    if _directory_bytes(run_directory) > maximum_storage_bytes:
        raise RouteDispatchError(
            code="mriganka_storage_budget_exhausted",
            message="The known-model route exceeded its caller-owned storage budget.",
        )


def _reserve_mriganka_stage_storage(
    run_directory: Path,
    *,
    maximum_storage_bytes: int,
    additional_bytes: int,
    stage: str,
) -> None:
    """Refuse a stage before writing unless its contract maximum fits."""

    if _directory_bytes(run_directory) + additional_bytes > maximum_storage_bytes:
        raise RouteDispatchError(
            code=f"mriganka_{stage}_storage_reservation_failed",
            message=(
                "The remaining route storage budget cannot cover the next stage's "
                "contract maximum."
            ),
        )


def load_pipeline_request(path: Path) -> PipelineRunRequest:
    """Load the strict discriminated request without executing a route."""

    encoded = _bounded_regular_file(
        Path(path), maximum_bytes=_MAX_REQUEST_BYTES, label="pipeline_request"
    )
    try:
        return PIPELINE_REQUEST_ADAPTER.validate_json(encoded, strict=True)
    except (ValueError, ValidationError):
        raise RouteDispatchError(
            code="invalid_pipeline_request",
            message="The pipeline request failed its strict discriminated contract.",
        ) from None


def plan_pipeline_route(
    request: PipelineRunRequest,
    *,
    simulation_configuration: Path | None = None,
    preprocessing_output_root: Path | None = None,
    researcher_output_root: Path | None = None,
    repository_root: Path | None = None,
) -> RoutePlan:
    """Describe the exact executable stages and any missing caller-owned inputs."""

    if isinstance(request, SimulationTrainingRequest):
        required = ("simulation_configuration",)
        missing = () if simulation_configuration is not None else required
        stages = (
            "validate_exact_campaign_configuration",
            "run_existing_simulation_campaign",
            "stop_with_synthetic_technical_report",
        )
        boundary = (
            "Synthetic integration smoke only; LensCat is not applicable because "
            "the simulated samples have no celestial coordinates."
        )
        lenscat_policy = "not_applicable_synthetic_without_coordinates"
    elif isinstance(request, MrigankaDp2Request):
        required = ("preprocessing_output_root", "RSP_TOKEN")
        missing_items: list[str] = []
        if preprocessing_output_root is None:
            missing_items.append("preprocessing_output_root")
        if not os.environ.get("RSP_TOKEN"):
            missing_items.append("RSP_TOKEN")
        if request.budget.max_tool_calls < _MRIGANKA_TOOL_CALLS:
            missing_items.append(
                f"request_budget.max_tool_calls>={_MRIGANKA_TOOL_CALLS}"
            )
        if request.budget.max_storage_bytes < _MRIGANKA_MINIMUM_STORAGE_BYTES:
            missing_items.append(
                f"request_budget.max_storage_bytes>={_MRIGANKA_MINIMUM_STORAGE_BYTES}"
            )
        missing = tuple(missing_items)
        stages = (
            "retrieve_and_verify_real_dp2_gri_packages",
            "resolve_exact_registered_model_manifest",
            "run_registered_three_band_preprocessing",
            "run_audited_m3_to_m4_bridge",
            "run_unqualified_dp2_technical_inference",
            "assemble_non_candidate_technical_report",
            "publish_route_completion_last",
        )
        boundary = (
            "The classifier output is an uncalibrated technical integration score, "
            "not a probability or candidate decision; scientific use and LensCat "
            "remain blocked."
        )
        lenscat_policy = "final_only_after_real_candidate_evidence"
    else:
        required = ("researcher_output_root",)
        missing_items = [] if researcher_output_root is not None else list(required)
        if request.budget.max_llm_requests < 10:
            missing_items.append("request_budget.max_llm_requests>=10")
        if request.budget.max_tool_calls < 9:
            missing_items.append("request_budget.max_tool_calls>=9")
        if request.budget.max_storage_bytes < _RESEARCHER_MIN_STORAGE_BYTES:
            missing_items.append("request_budget.max_storage_bytes>=41943040")
        missing = tuple(missing_items)
        stages = (
            "create_safe_immutable_repository_snapshot",
            "run_open_ended_source_research_planner",
            "synthesize_evidence_linked_terminal_result",
            "stop_before_any_researcher_code_execution",
        )
        boundary = (
            "The agent may choose its analysis strategy but can only list, search, "
            "and read the immutable snapshot; researcher code is never imported or run."
        )
        lenscat_policy = "final_only_after_real_candidate_evidence"
    return RoutePlan(
        branch=request.branch,
        request_id=request.request_id,
        can_execute_with_supplied_inputs=not missing,
        required_runtime_inputs=required,
        missing_runtime_inputs=missing,
        stages=stages,
        terminal_boundary=boundary,
        lenscat_policy=lenscat_policy,
    )


async def run_pipeline_route(
    request: PipelineRunRequest,
    *,
    request_base_directory: Path,
    simulation_configuration: Path | None = None,
    preprocessing_output_root: Path | None = None,
    researcher_output_root: Path | None = None,
    repository_root: Path | None = None,
    campaign_run_id: str | None = None,
    researcher_agent_provider: Literal["bedrock", "openai"] | None = None,
) -> PipelineRouteResult:
    """Execute exactly one typed route using only its approved implementation."""

    if researcher_agent_provider not in {None, "bedrock", "openai"}:
        raise RouteDispatchError(
            code="researcher_agent_provider_invalid",
            message="The researcher agent provider must be bedrock or openai.",
        )
    if researcher_agent_provider is not None and not isinstance(
        request, ResearcherModelRequest
    ):
        raise RouteDispatchError(
            code="researcher_agent_provider_not_applicable",
            message=(
                "A researcher agent provider may only be selected for the "
                "researcher_model route."
            ),
        )

    base = checked_real_directory(request_base_directory)
    plan = plan_pipeline_route(
        request,
        simulation_configuration=simulation_configuration,
        preprocessing_output_root=preprocessing_output_root,
        researcher_output_root=researcher_output_root,
        repository_root=repository_root,
    )
    if not plan.can_execute_with_supplied_inputs:
        raise RouteDispatchError(
            code="route_runtime_inputs_missing",
            message="The selected route is missing explicit caller-owned runtime inputs.",
        )
    if isinstance(request, SimulationTrainingRequest):
        assert simulation_configuration is not None
        return await asyncio.to_thread(
            _run_simulation_route,
            request,
            configuration_path=simulation_configuration,
            request_base_directory=base,
            repository_root=repository_root,
            campaign_run_id=campaign_run_id,
        )
    if isinstance(request, MrigankaDp2Request):
        assert preprocessing_output_root is not None
        return await asyncio.to_thread(
            _run_mriganka_route,
            request,
            request_base_directory=base,
            output_root=preprocessing_output_root,
        )
    assert isinstance(request, ResearcherModelRequest)
    assert researcher_output_root is not None
    return await _run_researcher_route(
        request,
        output_root=researcher_output_root,
        provider=researcher_agent_provider or "bedrock",
    )


def _run_simulation_route(
    request: SimulationTrainingRequest,
    *,
    configuration_path: Path,
    request_base_directory: Path,
    repository_root: Path | None,
    campaign_run_id: str | None,
) -> SimulationTrainingRouteResult:
    # Keep the live PydanticAI/Bedrock and SSH campaign stack outside imports of
    # the DP2-only route.
    from .providers import load_bedrock_agent_settings
    from .workflows.campaign import run_simulation_campaign

    path = checked_real_file(configuration_path)
    encoded = _bounded_regular_file(
        path,
        maximum_bytes=_MAX_CAMPAIGN_CONFIGURATION_BYTES,
        label="simulation_configuration",
    )
    try:
        configuration = SimulationCampaignConfiguration.model_validate_json(
            encoded, strict=True
        )
    except (ValueError, ValidationError):
        raise RouteDispatchError(
            code="invalid_simulation_configuration",
            message="The simulation campaign configuration failed strict validation.",
        ) from None
    if configuration.request != request:
        raise RouteDispatchError(
            code="simulation_request_configuration_mismatch",
            message="The campaign configuration does not contain the exact routed request.",
        )
    request_spec_path = _declared_file(
        request.simulation_spec,
        base_directory=request_base_directory,
        label="simulation_specification",
    )
    configuration_spec_path = _declared_file(
        configuration.request.simulation_spec,
        base_directory=path.parent,
        label="simulation_specification",
    )
    request_output_root = Path(request.output_root).expanduser()
    if not request_output_root.is_absolute():
        request_output_root = request_base_directory / request_output_root
    request_output_root = checked_absolute_path(request_output_root)
    configuration_output_root = Path(configuration.request.output_root).expanduser()
    if not configuration_output_root.is_absolute():
        configuration_output_root = path.parent / configuration_output_root
    configuration_output_root = checked_absolute_path(configuration_output_root)
    if (
        request_spec_path != configuration_spec_path
        or request_output_root != configuration_output_root
    ):
        raise RouteDispatchError(
            code="simulation_request_path_context_mismatch",
            message=(
                "The routed request and campaign configuration resolve their "
                "simulation input or output root differently."
            ),
        )
    root = (
        checked_real_directory(repository_root)
        if repository_root is not None
        else checked_real_directory(Path(__file__).absolute().parents[2])
    )
    completed = run_simulation_campaign(
        configuration,
        configuration_base=path.parent,
        repository_root=root,
        provider_settings=load_bedrock_agent_settings(),
        run_id=campaign_run_id,
    )
    return SimulationTrainingRouteResult(
        request_id=request.request_id,
        campaign_run_id=completed.completion.run_id,
        campaign_run_directory=str(completed.run_directory),
        completion_path=str(completed.completion_path),
        completion_sha256=sha256_file(completed.completion_path),
        technical_report_path=str(completed.report_path),
        final_state_path=str(completed.final_state_path),
    )


def _run_mriganka_route(
    request: MrigankaDp2Request,
    *,
    request_base_directory: Path,
    output_root: Path,
) -> MrigankaDp2RouteResult:
    if request.budget.max_tool_calls < _MRIGANKA_TOOL_CALLS:
        raise RouteDispatchError(
            code="mriganka_budget_too_small",
            message=(
                "The complete Mriganka technical route requires seven approved "
                "deterministic tool calls."
            ),
        )
    if request.budget.max_storage_bytes < _MRIGANKA_MINIMUM_STORAGE_BYTES:
        raise RouteDispatchError(
            code="mriganka_storage_budget_too_small",
            message=(
                "The storage budget is below the fixed per-stage reservation "
                "required by the known-model route."
            ),
        )

    manifest_path = _declared_mriganka_file(
        request.model_manifest,
        base_directory=request_base_directory,
        label="model_manifest",
    )
    inference_manifest_path = _declared_mriganka_file(
        request.inference_bundle_manifest,
        base_directory=request_base_directory,
        label="inference_bundle_manifest",
    )
    checkpoint_root = _declared_mriganka_directory(
        request.checkpoint_root,
        base_directory=request_base_directory,
        label="checkpoint_root",
    )

    manifest = load_model_manifest(manifest_path)
    reference = ModelManifestRef.from_manifest(manifest)
    registry = build_default_registry()
    resolved = registry.inspect(reference)
    expected_registration = (
        "mriganka-enn-three-band-dp2-provisional-v1",
        "deeplense.mriganka.enn-sda",
        "mriganka-enn-native64-three-band",
        "v1",
    )
    observed_registration = (
        resolved.manifest.manifest_id,
        resolved.manifest.model_id,
        resolved.adapter_identity.adapter_id,
        resolved.adapter_identity.adapter_version,
    )
    if observed_registration != expected_registration:
        raise RouteDispatchError(
            code="mriganka_three_band_manifest_required",
            message=(
                "The known-model route requires the exact registered three-band "
                "Mriganka preprocessing manifest."
            ),
        )
    if not resolved.manifest.qualification.preprocessing_execution_allowed:
        raise RouteDispatchError(
            code="mriganka_preprocessing_gate_closed",
            message="The registered manifest does not authorize preprocessing.",
        )

    # Keep the heavyweight checkpoint runtime out of import-only planning paths.
    from ripple.inference.m3_bridge import run_m3_to_m4_bridge
    from ripple.inference.service import (
        builtin_bundle_manifest_path,
        load_bundle_manifest,
        run_m4_inference,
    )

    bundle_manifest = load_bundle_manifest(inference_manifest_path)
    builtin_bundle = load_bundle_manifest(builtin_bundle_manifest_path())
    if bundle_manifest != builtin_bundle:
        raise RouteDispatchError(
            code="mriganka_exact_inference_bundle_required",
            message=(
                "The known-model route requires the exact pinned Mriganka ENN "
                "checkpoint-bundle manifest."
            ),
        )

    route_run_directory = checked_real_directory(
        create_private_run_directory(Path(output_root))
    )
    route_run_id = f"mriganka-{uuid.uuid4().hex[:12]}"
    route_store = ArtifactStore(route_run_directory, create=False)

    try:
        client = Dp2Client.from_environment(Dp2ClientConfig())
        m2_service = Dp2PackageService(client)
    except Dp2Error as exc:
        raise RouteDispatchError(
            code=f"mriganka_m2_{exc.code}",
            message="The authenticated Rubin DP2 client could not be initialized.",
        ) from None

    m2_runs: list[tuple[str, Path, LoadedDp2Cutout]] = []
    for band in request.bands:
        _reserve_mriganka_stage_storage(
            route_run_directory,
            maximum_storage_bytes=request.budget.max_storage_bytes,
            additional_bytes=_MRIGANKA_M2_STAGE_RESERVATION_BYTES,
            stage=f"m2_{band}",
        )
        try:
            m2_run_directory = checked_real_directory(
                create_private_run_directory(route_run_directory / "m2" / band)
            )
            package = m2_service.run(
                Dp2CutoutRequest(
                    ra_deg=request.target.ra_deg,
                    dec_deg=request.target.dec_deg,
                    band_name=band,
                    soda_service_type="cutout-sync-maskedimage",
                ),
                m2_run_directory,
            )
            loaded = load_cutout_package(m2_run_directory / "package.json")
        except Dp2Error as exc:
            raise RouteDispatchError(
                code=f"mriganka_m2_{band}_{exc.code}",
                message=(
                    f"The authenticated Rubin DP2 {band}-band package stage failed "
                    "without recording credential material."
                ),
            ) from None
        if loaded.package != package:
            raise RouteDispatchError(
                code="mriganka_m2_round_trip_mismatch",
                message="A published M2 package changed during strict reload.",
            )
        if not (
            math.isclose(
                float(loaded.package.request.ra_deg),
                request.target.ra_deg,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            and math.isclose(
                float(loaded.package.request.dec_deg),
                request.target.dec_deg,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            and loaded.package.dataset.band_name == band
        ):
            raise RouteDispatchError(
                code="mriganka_m2_identity_mismatch",
                message="A verified M2 package does not match its requested target or band.",
            )
        m2_runs.append((band, m2_run_directory, loaded))
        _enforce_mriganka_storage_budget(
            route_run_directory,
            maximum_storage_bytes=request.budget.max_storage_bytes,
        )

    m2_package_paths = tuple(
        run_directory / "package.json" for _, run_directory, _ in m2_runs
    )
    m2_copy_bytes = sum(
        _directory_bytes(run_directory) for _, run_directory, _ in m2_runs
    )
    _reserve_mriganka_stage_storage(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
        additional_bytes=m2_copy_bytes + _MRIGANKA_M3_FIXED_RESERVATION_BYTES,
        stage="m3",
    )
    completed_m3 = run_registered_preprocessing(
        registry=registry,
        manifest_id=manifest.manifest_id,
        package_paths=m2_package_paths,
        output_root=route_run_directory / "m3",
        require_aligned_shapes=False,
        invocation_interface="agent_tool",
    )
    _enforce_mriganka_storage_budget(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
    )
    invocation_inputs = completed_m3.envelope.invocation.inputs
    if len(invocation_inputs) != 3 or tuple(
        item.band for item in invocation_inputs
    ) != ("g", "r", "i"):
        raise RouteDispatchError(
            code="mriganka_preprocessing_provenance_invalid",
            message="The completed M3 run did not bind exactly one g/r/i M2 package.",
        )
    try:
        m3_package = MrigankaEnnThreeBandModelInputPackage.model_validate(
            completed_m3.envelope.package.model_dump(mode="python"),
            strict=True,
        )
    except ValidationError:
        raise RouteDispatchError(
            code="mriganka_three_band_package_invalid",
            message="The registered M3 stage did not publish its exact three-band contract.",
        ) from None
    if any(
        not (
            math.isclose(
                source.ra_deg,
                request.target.ra_deg,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            and math.isclose(
                source.dec_deg,
                request.target.dec_deg,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        )
        for source in m3_package.sources
    ):
        raise RouteDispatchError(
            code="mriganka_processed_target_mismatch",
            message="The completed M3 run describes a different sky coordinate.",
        )

    _reserve_mriganka_stage_storage(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
        additional_bytes=_MRIGANKA_BRIDGE_STAGE_RESERVATION_BYTES,
        stage="bridge",
    )
    completed_bridge = run_m3_to_m4_bridge(
        m3_run_directory=completed_m3.run_directory,
        output_root=route_run_directory / "bridge",
    )
    _enforce_mriganka_storage_budget(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
    )

    _reserve_mriganka_stage_storage(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
        additional_bytes=_MRIGANKA_M4_STAGE_RESERVATION_BYTES,
        stage="m4",
    )
    completed_m4 = run_m4_inference(
        manifest_path=inference_manifest_path,
        checkpoint_root=checkpoint_root,
        input_path=completed_bridge.model_input_path,
        output_root=route_run_directory / "m4",
        bridge_manifest_path=completed_bridge.bridge_path,
    )
    if (
        completed_m4.result.execution_scope_used
        != "unqualified_dp2_technical_integration"
        or completed_m4.result.bridge_provenance is None
        or completed_m4.result.score_is_calibrated_probability
        or completed_m4.result.decision_threshold_applied
        or completed_m4.result.candidate_decision_made
        or completed_m4.result.scientific_use_allowed
    ):
        raise RouteDispatchError(
            code="mriganka_m4_scope_violation",
            message="M4 did not preserve the required technical-integration-only boundary.",
        )
    _enforce_mriganka_storage_budget(
        route_run_directory,
        maximum_storage_bytes=request.budget.max_storage_bytes,
    )

    from .tools.report import assemble_mriganka_dp2_technical_report

    report_id = f"report-{route_run_id}"
    report = assemble_mriganka_dp2_technical_report(
        report_id=report_id,
        request_id=request.request_id,
        target=request.target,
        m2_packages=(
            m2_runs[0][2].package,
            m2_runs[1][2].package,
            m2_runs[2][2].package,
        ),
        m3_package=m3_package,
        m3_completion=completed_m3.completion,
        m3_completion_sha256=sha256_file(completed_m3.completion_path),
        bridge_record=completed_bridge.record,
        bridge_completion=completed_bridge.completion,
        bridge_completion_sha256=sha256_file(completed_bridge.completion_path),
        m4_result=completed_m4.result,
        m4_completion=completed_m4.completion,
        m4_completion_sha256=sha256_file(completed_m4.completion_path),
    )
    report_path = _write_budgeted_json(
        route_store,
        "report/technical-report.json",
        report,
        maximum_storage_bytes=request.budget.max_storage_bytes,
    )
    try:
        reloaded_report = MrigankaDp2TechnicalReport.model_validate_json(
            _bounded_regular_file(
                report_path,
                maximum_bytes=16 * _MIB,
                label="mriganka_technical_report",
            ),
            strict=True,
        )
    except (ValueError, ValidationError):
        raise RouteDispatchError(
            code="mriganka_technical_report_round_trip_invalid",
            message="The technical report failed strict reload after publication.",
        ) from None
    if reloaded_report != report:
        raise RouteDispatchError(
            code="mriganka_technical_report_round_trip_mismatch",
            message="The technical report changed during strict reload.",
        )

    m2_results = tuple(
        MrigankaDp2BandResult(
            band=band,
            dataset_id=loaded.package.dataset.dataset_id,
            run_directory=str(run_directory),
            package_manifest=_mriganka_artifact_ref(run_directory / "package.json"),
            fits_artifact=_mriganka_artifact_ref(
                run_directory / loaded.package.artifact.filename
            ),
            image_decoded_sha256=loaded.package.image.digest.sha256,
            mask_decoded_sha256=loaded.package.mask.digest.sha256,
            variance_decoded_sha256=loaded.package.variance.digest.sha256,
        )
        for band, run_directory, loaded in m2_runs
    )
    if len(m2_results) != 3:  # pragma: no cover - fixed request contract
        raise RouteDispatchError(
            code="mriganka_m2_result_count_invalid",
            message="The route did not retain exactly three M2 results.",
        )
    typed_m2_results = (m2_results[0], m2_results[1], m2_results[2])

    m3_result = MrigankaM3Result(
        run_directory=str(completed_m3.run_directory),
        model_manifest_id=manifest.manifest_id,
        model_manifest_sha256=reference.sha256,
        package_manifest=_mriganka_artifact_ref(
            completed_m3.run_directory / "manifest.json"
        ),
        completion=_mriganka_artifact_ref(completed_m3.completion_path),
        model_input_bchw=_mriganka_artifact_ref(
            completed_m3.run_directory / m3_package.model_input.filename
        ),
        qa_preview=_mriganka_artifact_ref(
            completed_m3.run_directory / m3_package.preview.filename
        ),
        cross_band_wcs_maximum_separation_arcsec=(
            m3_package.cross_band_wcs.maximum_separation_arcsec
        ),
    )
    bridge_result = MrigankaBridgeResult(
        run_directory=str(completed_bridge.run_directory),
        bridge_manifest=_mriganka_artifact_ref(completed_bridge.bridge_path),
        completion=_mriganka_artifact_ref(completed_bridge.completion_path),
        model_input_chw=_mriganka_artifact_ref(completed_bridge.model_input_path),
    )
    m4_result = MrigankaM4Result(
        run_directory=str(completed_m4.run_directory),
        bundle_id=completed_m4.result.bundle_id,
        bundle_manifest_sha256=completed_m4.result.bundle_manifest_sha256,
        inference_result=_mriganka_artifact_ref(completed_m4.result_path),
        completion=_mriganka_artifact_ref(completed_m4.completion_path),
        embedding=_mriganka_artifact_ref(completed_m4.embedding_path),
        raw_logits=completed_m4.result.logits,
        uncalibrated_softmax_components=completed_m4.result.scores,
    )
    report_ref = _mriganka_artifact_ref(report_path)
    completion_fields = {
        "route_run_id": route_run_id,
        "request_id": request.request_id,
        "request_sha256": canonical_json_sha256(request),
        "completed_at_utc": utc_now(),
        "m2_package_manifest_sha256": tuple(
            item.package_manifest.sha256 for item in typed_m2_results
        ),
        "m3_model_manifest_sha256": reference.sha256,
        "m3_completion_sha256": m3_result.completion.sha256,
        "bridge_completion_sha256": bridge_result.completion.sha256,
        "m4_bundle_manifest_sha256": m4_result.bundle_manifest_sha256,
        "m4_inference_result_sha256": m4_result.inference_result.sha256,
        "m4_completion_sha256": m4_result.completion.sha256,
        "technical_report_id": report.report_id,
        "technical_report_sha256": report_ref.sha256,
        "unresolved_scientific_blockers": report.unresolved_scientific_blockers,
    }
    bytes_before_completion = _directory_bytes(route_run_directory)
    predicted_storage_bytes = bytes_before_completion
    for _ in range(8):
        usage = BudgetUsage(
            tool_calls=_MRIGANKA_TOOL_CALLS,
            storage_bytes=predicted_storage_bytes,
        )
        completion = MrigankaDp2RouteCompletion(
            **completion_fields,
            usage=usage,
        )
        next_prediction = bytes_before_completion + len(
            route_store.json_bytes(completion)
        )
        if next_prediction == predicted_storage_bytes:
            break
        predicted_storage_bytes = next_prediction
    else:  # pragma: no cover - decimal byte length converges in at most two changes
        raise RouteDispatchError(
            code="mriganka_storage_accounting_did_not_converge",
            message="The durable route storage accounting did not converge.",
        )
    if not completion.usage.fits(request.budget):
        raise RouteDispatchError(
            code="mriganka_budget_accounting_mismatch",
            message="Final known-route usage would exceed its declared budget.",
        )
    # This is the route-level publication commit and intentionally the last write.
    completion_path = _write_budgeted_json(
        route_store,
        "completion.json",
        completion,
        maximum_storage_bytes=request.budget.max_storage_bytes,
    )
    try:
        reloaded_completion = MrigankaDp2RouteCompletion.model_validate_json(
            _bounded_regular_file(
                completion_path,
                maximum_bytes=2 * _MIB,
                label="mriganka_route_completion",
            ),
            strict=True,
        )
    except (ValueError, ValidationError):
        raise RouteDispatchError(
            code="mriganka_route_completion_round_trip_invalid",
            message="The route completion failed strict reload after publication.",
        ) from None
    if reloaded_completion != completion:
        raise RouteDispatchError(
            code="mriganka_route_completion_round_trip_mismatch",
            message="The route completion changed during strict reload.",
        )
    actual_storage_bytes = _directory_bytes(route_run_directory)
    if actual_storage_bytes != completion.usage.storage_bytes:
        raise RouteDispatchError(
            code="mriganka_durable_storage_accounting_mismatch",
            message="The durable route storage usage does not match the artifact tree.",
        )
    completion_ref = _mriganka_artifact_ref(completion_path)
    return MrigankaDp2RouteResult(
        request_id=request.request_id,
        route_run_id=route_run_id,
        route_run_directory=str(route_run_directory),
        target=request.target,
        m2_packages=typed_m2_results,
        m3=m3_result,
        bridge=bridge_result,
        m4=m4_result,
        technical_report_id=report.report_id,
        technical_report=report_ref,
        route_completion=completion_ref,
        usage=completion.usage,
        unresolved_scientific_blockers=report.unresolved_scientific_blockers,
    )


async def _run_researcher_route(
    request: ResearcherModelRequest,
    *,
    output_root: Path,
    provider: Literal["bedrock", "openai"],
) -> ResearcherModelRouteResult:
    from ripple.modeling import (
        OpenResearchRequest,
        ResearchAgentLimits,
        run_open_research_agent,
    )

    # Imported lazily so the deterministic DP2 and planning paths do not need a
    # live-provider SDK or inspect credential state.
    from ripple.modeling.agent_provider import (
        build_live_agent_model_from_environment,
    )

    from .tools import build_intake_repository_snapshot, create_repository_intake
    from .tools.repository_snapshot import IntakeRepositorySnapshot

    _require_researcher_budget(request)
    requests_per_phase = min(50, request.budget.max_llm_requests // 2)
    agent_tool_budget = request.budget.max_tool_calls - _RESEARCHER_FIXED_TOOL_CALLS
    tools_per_phase = min(160, agent_tool_budget // 2)
    limits = ResearchAgentLimits(
        maximum_read_calls=min(16, max(1, tools_per_phase - 3)),
        maximum_read_lines_total=min(
            1200,
            max(200, (tools_per_phase - 3) * 100),
        ),
        maximum_model_requests_per_phase=requests_per_phase,
        maximum_tool_calls_per_phase=tools_per_phase,
        # PydanticAI counts the complete accumulated conversation on every
        # request.  Iterative list/search/read turns therefore need a larger
        # cumulative input allowance than a single model context, while the
        # request/tool/read bounds above still cap the agent's work.
        maximum_input_tokens_per_phase=96_000,
        maximum_output_tokens_per_phase=4_096,
        maximum_total_tokens_per_phase=100_096,
    )

    root = checked_real_directory(output_root, create=True)
    run_id = _research_run_id(request.request_id)
    run_directory = root / run_id
    try:
        run_directory.mkdir(mode=0o700)
    except FileExistsError:
        raise RouteDispatchError(
            code="research_run_already_exists",
            message="The immutable research run directory already exists.",
        ) from None
    except OSError as exc:
        raise RouteDispatchError(
            code="research_run_creation_failed",
            message=f"The research run directory could not be created ({type(exc).__name__}).",
        ) from None
    run_state = os.lstat(run_directory)
    if (
        not stat.S_ISDIR(run_state.st_mode)
        or run_state.st_uid != os.getuid()
        or stat.S_IMODE(run_state.st_mode) & 0o077
    ):
        raise RouteDispatchError(
            code="research_run_not_private",
            message="The research run directory is not private and caller-owned.",
        )
    store = ArtifactStore(run_directory, create=False)
    maximum_storage_bytes = request.budget.max_storage_bytes
    stage = "persist_pipeline_request"
    failure_usage = {
        "accounting_method": "observed_before_agent",
        "llm_requests": 0,
        "tool_calls": 0,
        "is_conservative_upper_bound": False,
    }
    try:
        pipeline_request_path = _write_budgeted_json(
            store,
            "pipeline-request.json",
            request,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        research_request = OpenResearchRequest(
            request_id=request.request_id,
            end_goal=request.end_goal,
            success_criteria=request.success_criteria,
            target_observation_domain=request.target_observation_domain,
        )
        research_request_path = _write_budgeted_json(
            store,
            "open-research-request.json",
            research_request,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        limits_path = _write_budgeted_json(
            store,
            "effective-research-limits.json",
            limits,
            maximum_storage_bytes=maximum_storage_bytes,
        )

        stage = "create_repository_intake"
        failure_usage = {
            **failure_usage,
            "tool_calls": _RESEARCHER_FIXED_TOOL_CALLS,
        }
        intake_id = f"intake-{canonical_json_sha256(request)[:24]}"
        available_intake_bytes = (
            maximum_storage_bytes
            - _directory_bytes(run_directory)
            - _RESEARCHER_ARTIFACT_RESERVATION_BYTES
        )
        if available_intake_bytes < 16 * _MIB:
            raise RouteDispatchError(
                code="researcher_storage_budget_exhausted",
                message="The repository intake reservation does not fit the storage budget.",
            )
        default_policy = RepositoryIntakePolicy()
        intake_policy = RepositoryIntakePolicy.model_validate(
            {
                **default_policy.model_dump(mode="python"),
                "max_total_bytes": min(
                    default_policy.max_total_bytes,
                    available_intake_bytes,
                ),
                "max_git_transfer_bytes": min(
                    default_policy.max_git_transfer_bytes,
                    available_intake_bytes,
                ),
            },
            strict=True,
        )

        def create_snapshot() -> tuple[
            RepositoryIntakeManifest,
            IntakeRepositorySnapshot,
        ]:
            manifest = create_repository_intake(
                RepositoryIntakeRequest(
                    intake_id=intake_id,
                    requested_at_utc=utc_now(),
                    source=request.repository_source,
                ),
                run_directory=run_directory,
                policy=intake_policy,
            )
            return manifest, build_intake_repository_snapshot(
                manifest=manifest,
                intake_directory=run_directory / "repository-intake",
                request_id=request.request_id,
            )

        intake_manifest, snapshot = await asyncio.to_thread(create_snapshot)
        if _directory_bytes(run_directory) > maximum_storage_bytes:
            raise RouteDispatchError(
                code="researcher_storage_budget_exhausted",
                message="Repository intake exceeded the route storage budget.",
            )
        intake_directory = run_directory / "repository-intake"
        binding_path = _write_budgeted_json(
            store,
            "repository-snapshot-binding.json",
            snapshot.binding,
            maximum_storage_bytes=maximum_storage_bytes,
        )

        stage = "initialize_live_model"
        model, provider_identity = await asyncio.to_thread(
            build_live_agent_model_from_environment,
            provider=provider,
        )
        provider_path = _write_budgeted_json(
            store,
            "provider-runtime-identity.json",
            provider_identity,
            maximum_storage_bytes=maximum_storage_bytes,
        )

        stage = "run_open_research_agent"
        failure_usage = {
            "accounting_method": "agent_phase_ceiling",
            "llm_requests": 2 * requests_per_phase,
            "tool_calls": (_RESEARCHER_FIXED_TOOL_CALLS + 2 * tools_per_phase),
            "is_conservative_upper_bound": True,
        }
        outcome = await run_open_research_agent(
            model=model,
            request=research_request,
            snapshot=snapshot,
            limits=limits,
            retries=5 if provider == "bedrock" else 2,
        )
        actual_requests = outcome.usage.planner_requests + outcome.usage.result_requests
        actual_tools = (
            _RESEARCHER_FIXED_TOOL_CALLS
            + outcome.usage.planner_tool_calls
            + outcome.usage.result_tool_calls
        )
        failure_usage = {
            "accounting_method": "provider_reported_exact",
            "llm_requests": actual_requests,
            "tool_calls": actual_tools,
            "is_conservative_upper_bound": False,
        }
        if (
            actual_requests > request.budget.max_llm_requests
            or actual_tools > request.budget.max_tool_calls
        ):
            raise RouteDispatchError(
                code="researcher_agent_budget_accounting_mismatch",
                message="Research-agent usage exceeded the caller-owned route budget.",
            )

        stage = "persist_research_outcome"
        plan_path = _write_budgeted_json(
            store,
            "research-plan.json",
            outcome.plan,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        result_path = _write_budgeted_json(
            store,
            "final-research-result.json",
            outcome.result,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        destination = _write_budgeted_json(
            store,
            "open-research-outcome.json",
            outcome,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        manifest_path = intake_directory / "manifest.json"
        completed = outcome.result.status == "completed"
        route_result = ResearcherModelRouteResult(
            request_id=request.request_id,
            run_id=run_id,
            run_directory=str(run_directory),
            status="completed" if completed else "blocked",
            repository_acquisition=(
                "https_git_clone"
                if intake_manifest.source_kind == "git_https"
                else "local_snapshot"
            ),
            repository_intake_id=intake_manifest.intake_id,
            repository_tree_sha256=intake_manifest.tree_sha256,
            pipeline_request_path=str(pipeline_request_path),
            pipeline_request_sha256=sha256_file(pipeline_request_path),
            repository_intake_manifest_path=str(manifest_path),
            repository_intake_manifest_sha256=sha256_file(manifest_path),
            repository_snapshot_binding_path=str(binding_path),
            repository_snapshot_binding_sha256=sha256_file(binding_path),
            open_research_request_path=str(research_request_path),
            open_research_request_sha256=sha256_file(research_request_path),
            effective_limits_path=str(limits_path),
            effective_limits_sha256=sha256_file(limits_path),
            provider_runtime_identity_path=str(provider_path),
            provider_runtime_identity_sha256=sha256_file(provider_path),
            research_plan_path=str(plan_path),
            research_plan_sha256=sha256_file(plan_path),
            final_result_path=str(result_path),
            final_result_sha256=sha256_file(result_path),
            research_outcome_path=str(destination),
            research_outcome_sha256=sha256_file(destination),
            research_plan_id=outcome.plan.plan_id,
            final_result_id=outcome.result.result_id,
            cited_evidence_count=len(outcome.result.cited_evidence_ids),
            analysis_completed=outcome.result.analysis_completed,
            terminal_reason=(
                "research_goal_answered_from_cited_source_evidence"
                if completed
                else "research_goal_not_satisfied_from_available_source_evidence"
            ),
        )
        _write_budgeted_json(
            store,
            "route-result.json",
            route_result,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        return route_result
    except Exception as exc:
        failure_path = run_directory / "failure.json"
        if not failure_path.exists():
            try:
                _write_budgeted_json(
                    store,
                    "failure.json",
                    {
                        "schema_version": "ripple.researcher-route-failure.v1",
                        "request_id": request.request_id,
                        "run_id": run_id,
                        "failed_stage": stage,
                        "error_type": type(exc).__name__,
                        "error_code": getattr(exc, "code", "researcher_route_failed"),
                        "safe_message": "The researcher route stopped before publishing a terminal research outcome.",
                        "usage": failure_usage,
                        "storage_bytes_before_failure_artifact": _directory_bytes(
                            run_directory
                        ),
                        "source_execution_performed": False,
                        "preprocessing_execution_performed": False,
                        "model_execution_performed": False,
                    },
                    maximum_storage_bytes=maximum_storage_bytes,
                )
            except Exception:  # noqa: BLE001,S110 - failure evidence is best effort
                pass
        raise


__all__ = [
    "RouteDispatchError",
    "load_pipeline_request",
    "plan_pipeline_route",
    "run_pipeline_route",
]
