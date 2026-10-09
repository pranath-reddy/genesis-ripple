"""Render deterministic CSV and LaTeX paper tables from one study JSON record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from ..schemas.common import canonical_json_sha256
from ..schemas.study import (
    AgentStageEvidenceV2,
    ArmStudyResult,
    ClassificationMetrics,
    ModelStudyTotal,
    PaperTableArtifact,
    PaperTableManifest,
    StudyAggregateInput,
)
from .costing import apply_rate_based_costing, verify_reported_model_totals


NA = "N/A"


class PaperTableError(RuntimeError):
    """Base failure for loading, rendering, or verifying paper tables."""


class ImmutableArtifactError(PaperTableError):
    """Raised instead of overwriting a different existing artifact."""


@dataclass(frozen=True)
class RenderedPaperTables:
    output_directory: Path
    manifest_path: Path
    manifest_sha256_path: Path
    manifest_sha256: str
    artifact_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _Table:
    number: int
    slug: str
    caption: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    alignments: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.headers) != len(self.alignments):
            raise PaperTableError("table headers and alignments have different widths")
        if any(len(row) != len(self.headers) for row in self.rows):
            raise PaperTableError("paper-table row width differs from its header")
        if any(alignment not in {"l", "c", "r"} for alignment in self.alignments):
            raise PaperTableError("paper table contains an invalid alignment")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise PaperTableError(f"study JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise PaperTableError(f"study JSON contains non-finite number {value!r}")


def load_study_json(path: str | Path) -> tuple[StudyAggregateInput, bytes]:
    """Load strict study JSON while rejecting duplicate keys and non-finite values."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        text = raw.decode("utf-8")
        json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
        study = StudyAggregateInput.model_validate_json(raw)
    except PaperTableError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise PaperTableError(f"invalid study JSON {source}: {exc}") from exc
    return study, raw


def _clean(value: object) -> str:
    return " ".join(str(value).split())


def _metric(value: float | None) -> str:
    return NA if value is None else f"{value:.4f}"


def _count(value: int | None) -> str:
    return NA if value is None else str(value)


def _millions(value: int | None) -> str:
    return NA if value is None else f"{value / 1_000_000:.3f}"


def _seconds(value: float | None) -> str:
    return NA if value is None else f"{value:.1f}"


def _cost(value: Decimal | None) -> str:
    return NA if value is None else f"{value:.6f}"


def _category(value: str) -> str:
    return {
        "paid_api": "Paid API",
        "free_open_source": "Free Open-Source",
        "external_api": "External API",
    }[value]


def _plan_order(study: StudyAggregateInput) -> dict[str, int]:
    return {
        candidate.arm_id: candidate.plan_order
        for candidate in study.architecture_plan.candidates
    }


def _ordered_arms(study: StudyAggregateInput) -> tuple[ArmStudyResult, ...]:
    order = _plan_order(study)
    return tuple(
        sorted(
            study.arm_results,
            key=lambda arm: (order[arm.arm_id], arm.arm_id),
        )
    )


def _qualification_sentence(study: StudyAggregateInput) -> str:
    level = study.qualification.evidence_level.replace("_", " ")
    claim = (
        "scientific performance claims are allowed by the recorded qualification"
        if study.qualification.scientific_performance_claim_allowed
        else (
            "scientific performance claims are not allowed by the recorded "
            "qualification"
        )
    )
    comparison = (
        "same-regime architecture comparison is allowed"
        if study.qualification.fair_architecture_comparison_allowed
        else "architecture comparison is not qualified"
    )
    return f"Evidence level: {level}; {comparison}; {claim}."


def _table_1(study: StudyAggregateInput) -> _Table:
    rows: list[tuple[str, ...]] = []
    for arm in _ordered_arms(study):
        if not arm.trajectory:
            rows.append(
                (
                    _clean(arm.display_name),
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                    NA,
                    f"{arm.status}: {_clean(arm.failure_reason or '')}",
                )
            )
            continue
        for point in arm.trajectory:
            rows.append(
                (
                    _clean(arm.display_name),
                    str(point.iteration),
                    _metric(point.train_metrics.accuracy),
                    _metric(point.train_metrics.balanced_accuracy),
                    _metric(point.train_metrics.roc_auc),
                    _metric(point.validation_metrics.accuracy),
                    _metric(point.validation_metrics.balanced_accuracy),
                    _metric(point.validation_metrics.roc_auc),
                    _metric(point.accuracy_generalization_gap),
                    _clean(point.planner_decision),
                )
            )
    caption = (
        "Closed-loop tuning trajectories on the frozen "
        f"{study.dataset.dataset_id} split. Values are recorded measurements; "
        f"{_qualification_sentence(study)}"
    )
    return _Table(
        number=1,
        slug="tuning_trajectories",
        caption=caption,
        headers=(
            "Arm",
            "Iteration",
            "Train Acc.",
            "Train BA",
            "Train AUC",
            "Validation Acc.",
            "Validation BA",
            "Validation AUC",
            "Accuracy Gap",
            "Planner Decision",
        ),
        rows=tuple(rows),
        alignments=("l", "r", "r", "r", "r", "r", "r", "r", "r", "l"),
    )


def _metrics_cells(metrics: ClassificationMetrics | None) -> tuple[str, str, str]:
    if metrics is None:
        return (NA, NA, NA)
    return (
        _metric(metrics.accuracy),
        _metric(metrics.balanced_accuracy),
        _metric(metrics.roc_auc),
    )


def _table_2(study: StudyAggregateInput) -> _Table:
    rows: list[tuple[str, ...]] = []
    for arm in _ordered_arms(study):
        final = arm.final_evaluation
        validation = _metrics_cells(None if final is None else final.validation_metrics)
        held_out = _metrics_cells(None if final is None else final.test_metrics)
        regime = (
            "shared-data trained arm"
            if arm.fair_training_comparison_allowed
            else "trained arm; comparison not qualified"
        )
        rows.append(
            (
                _clean(arm.display_name),
                _clean(arm.status),
                regime,
                study.dataset.dataset_id,
                _millions(arm.parameter_count),
                NA,
                *validation,
                *held_out,
            )
        )
    if study.mriganka_zero_shot is not None:
        external = study.mriganka_zero_shot
        rows.append(
            (
                "Mriganka (reported separately)",
                external.status,
                "frozen external checkpoint zero-shot; NOT a fair training comparison",
                external.evaluation_dataset_id or NA,
                _millions(external.learned_parameter_count),
                _millions(external.materialized_state_element_count),
                NA,
                NA,
                NA,
                *_metrics_cells(external.metrics),
            )
        )
    caption = (
        "Final validation and held-out evaluation summary. Shared-data trained arms "
        f"use regime {study.search_protocol.comparison_regime.regime_id}; any Mriganka "
        "row is a separately reported frozen external zero-shot evaluation and is not "
        "a fair training comparison. N/A means the metric was not measured. "
        f"{_qualification_sentence(study)}"
    )
    return _Table(
        number=2,
        slug="final_performance",
        caption=caption,
        headers=(
            "Arm",
            "Status",
            "Comparison Regime",
            "Evaluation Dataset",
            "Learned Params. (M)",
            "Materialized State (M)",
            "Validation Acc.",
            "Validation BA",
            "Validation AUC",
            "Held-out/Eval Acc.",
            "Held-out/Eval BA",
            "Held-out/Eval AUC",
        ),
        rows=tuple(rows),
        alignments=("l", "l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r"),
    )


def _complete_token_total(stages: Sequence[AgentStageEvidenceV2]) -> int | None:
    if not stages or any(stage.token_usage is None for stage in stages):
        return None
    return sum(
        stage.token_usage.measured_total_tokens
        for stage in stages
        if stage.token_usage is not None
    )


def _complete_cost_total(stages: Sequence[AgentStageEvidenceV2]) -> Decimal | None:
    if not stages or any(
        stage.cost is None
        or stage.cost.status != "estimated"
        or stage.cost.total_estimated_cost_usd is None
        for stage in stages
    ):
        return None
    return sum(
        (
            stage.cost.total_estimated_cost_usd
            for stage in stages
            if stage.cost is not None
            and stage.cost.total_estimated_cost_usd is not None
        ),
        Decimal(0),
    )


def _model_labels(totals: Sequence[ModelStudyTotal]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for total in totals:
        counts[total.display_name] = counts.get(total.display_name, 0) + 1
    return {
        total.model_id: (
            total.display_name
            if counts[total.display_name] == 1
            else f"{total.display_name} [{total.model_id}]"
        )
        for total in totals
    }


def _table_3(study: StudyAggregateInput) -> _Table:
    totals = tuple(
        sorted(
            study.model_totals,
            key=lambda item: (
                item.provider.lower(),
                item.display_name.lower(),
                item.model_id,
            ),
        )
    )
    labels = _model_labels(totals)
    model_ids = tuple(total.model_id for total in totals)
    rows: list[tuple[str, ...]] = []
    for arm in _ordered_arms(study):
        arm_stages = [stage for stage in study.stage_runs if stage.arm_id == arm.arm_id]
        phase_a = [stage for stage in arm_stages if stage.phase == "phase_a"]
        phase_b = [stage for stage in arm_stages if stage.phase == "phase_b"]
        cost_cells = []
        for model_id in model_ids:
            model_stages = [stage for stage in arm_stages if stage.model_id == model_id]
            cost_cells.append(_cost(_complete_cost_total(model_stages)))
        rows.append(
            (
                _clean(arm.display_name),
                _clean(arm.architecture_family),
                "trained arm",
                _millions(arm.parameter_count),
                NA,
                str(arm.tuning_iterations),
                _count(_complete_token_total(phase_a)),
                _count(_complete_token_total(phase_b)),
                _count(_complete_token_total(arm_stages)),
                _seconds(sum(stage.wall_time_seconds for stage in arm_stages)),
                _seconds(arm.total_worker_seconds),
                _seconds(arm.total_remote_wall_seconds),
                *cost_cells,
            )
        )
    shared_stages = [stage for stage in study.stage_runs if stage.arm_id is None]
    if shared_stages:
        shared_phase_a = [
            stage for stage in shared_stages if stage.phase == "phase_a"
        ]
        shared_phase_b = [
            stage for stage in shared_stages if stage.phase == "phase_b"
        ]
        shared_costs = [
            _cost(
                _complete_cost_total(
                    [stage for stage in shared_stages if stage.model_id == model_id]
                )
            )
            for model_id in model_ids
        ]
        rows.append(
            (
                "Shared model overhead",
                "study-wide",
                "shared; not attributable to one arm",
                NA,
                NA,
                NA,
                _count(_complete_token_total(shared_phase_a)),
                _count(_complete_token_total(shared_phase_b)),
                _count(_complete_token_total(shared_stages)),
                _seconds(sum(stage.wall_time_seconds for stage in shared_stages)),
                NA,
                NA,
                *shared_costs,
            )
        )
    if study.mriganka_zero_shot is not None:
        external = study.mriganka_zero_shot
        rows.append(
            (
                "Mriganka (separate)",
                "external checkpoint",
                "zero-shot; NOT comparable",
                _millions(external.learned_parameter_count),
                _millions(external.materialized_state_element_count),
                "0",
                NA,
                NA,
                NA,
                NA,
                NA,
                NA,
                *(NA for _ in model_ids),
            )
        )
    model_headers = tuple(
        f"{labels[model_id]} Rate-based Est. USD" for model_id in model_ids
    )
    caption = (
        "Measured resource use and rate-based API cost estimates by architecture arm. "
        "Token columns include measured input (uncached, cache-write, and cache-read) "
        "plus output tokens. Dollar values use the frozen pricing snapshot and are "
        "estimates, not AWS invoices or total compute costs. N/A means a measurement "
        "or required rate was absent. "
        f"{_qualification_sentence(study)}"
    )
    headers = (
        "Architecture",
        "Family",
        "Regime",
        "Learned Params. (M)",
        "Materialized State (M)",
        "Tuning Iters.",
        "Phase A Measured Tokens",
        "Phase B Measured Tokens",
        "All Measured Tokens",
        "Agent Wall Time (s)",
        "Worker Time (s)",
        "Remote Wall Time (s)",
        *model_headers,
    )
    return _Table(
        number=3,
        slug="resources_and_rate_costs",
        caption=caption,
        headers=headers,
        rows=tuple(rows),
        alignments=("l", "l", "l") + ("r",) * (len(headers) - 3),
    )


def _table_4(study: StudyAggregateInput) -> _Table:
    rows: list[tuple[str, ...]] = []
    ordered = sorted(
        study.model_totals,
        key=lambda item: (
            item.total_rate_based_estimated_cost_usd is None,
            -item.total_rate_based_estimated_cost_usd
            if item.total_rate_based_estimated_cost_usd is not None
            else Decimal(0),
            item.display_name.lower(),
            item.model_id,
        ),
    )
    for total in ordered:
        invoked_rates = {
            rate.pricing_basis
            for stage in study.stage_runs
            if stage.model_id == total.model_id
            for rate in study.model_matrix.pricing.rates
            if rate.model_id == (stage.pricing_model_id or stage.model_id)
            and rate.region == stage.region
        }
        if not invoked_rates:
            pricing_basis = NA
        elif len(invoked_rates) > 1:
            pricing_basis = "mixed"
        else:
            pricing_basis = {
                "published_rate": "published rate",
                "account_specific_rate": "account-specific rate",
                "manual_rate_estimate": "manual proxy estimate",
                "no_api_charge": "no API charge",
            }[next(iter(invoked_rates))]
        average = None
        if (
            total.total_rate_based_estimated_cost_usd is not None
            and total.arms_invoked > 0
        ):
            average = total.total_rate_based_estimated_cost_usd / total.arms_invoked
        rows.append(
            (
                _clean(total.display_name),
                _clean(total.model_id),
                _clean(total.provider),
                _category(total.category),
                str(total.stage_run_count),
                _count(total.provider_request_count),
                str(total.known_provider_request_count),
                (
                    f"{total.succeeded_stage_run_count}/"
                    f"{total.failed_stage_run_count}"
                ),
                (
                    f"{total.requests_with_measured_tokens}/"
                    f"{total.known_provider_request_count}"
                ),
                total.provider_request_count_status.replace("_", " "),
                str(total.arms_invoked),
                str(total.uncached_input_tokens),
                str(total.cache_write_input_tokens),
                str(total.cache_read_input_tokens),
                str(total.output_tokens),
                pricing_basis,
                _cost(total.total_rate_based_estimated_cost_usd),
                _cost(average),
                total.cost_status.replace("_", " "),
            )
        )
    caption = (
        "Actual recorded model invocations and token totals across the architecture "
        "study. Success/failure counts are agent-stage runs; request and measured "
        "counts are provider requests and expose missing evidence. Provider request "
        "totals are N/A when a failed stage did not report its request count. "
        "Pricing basis exposes published, account-specific, manual proxy, no-charge, "
        "or mixed rates. Average cost uses only distinct invoked architecture arms. "
        "Costs are "
        "rate-based estimates from the frozen snapshot, not AWS invoices; N/A means "
        "the complete value cannot be calculated. "
        f"{_qualification_sentence(study)}"
    )
    return _Table(
        number=4,
        slug="model_cost_comparison",
        caption=caption,
        headers=(
            "Model Name",
            "Invoked Model ID",
            "Provider",
            "Category",
            "Stage Runs",
            "Provider Requests",
            "Known Provider Requests",
            "Successful/Failed Stage Runs",
            "Measured/Known Requests",
            "Request Count Status",
            "Arms Invoked",
            "Uncached Input",
            "Cache Write",
            "Cache Read",
            "Output",
            "Pricing Basis",
            "Total Rate-based Est. USD",
            "Avg Est. USD / Invoked Arm",
            "Cost Status",
        ),
        rows=tuple(rows),
        alignments=("l", "l", "l", "l")
        + ("r",) * 5
        + ("l",)
        + ("r",) * 5
        + ("l", "r", "r", "l"),
    )


def build_paper_tables(study: StudyAggregateInput) -> tuple[_Table, ...]:
    """Build the four in-memory tables in stable paper order."""

    return (_table_1(study), _table_2(study), _table_3(study), _table_4(study))


def _csv_bytes(table: _Table) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(table.headers)
    writer.writerows(table.rows)
    return stream.getvalue().encode("utf-8")


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value: str) -> str:
    return "".join(_LATEX_ESCAPES.get(character, character) for character in value)


def _latex_bytes(table: _Table) -> bytes:
    environment = "table*" if len(table.headers) > 7 else "table"
    column_spec = "".join(table.alignments)
    rows = [
        " & ".join(_latex_escape(cell) for cell in row) + r" \\" for row in table.rows
    ]
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(table.caption)}}}",
        rf"\label{{tab:ripple-{table.slug.replace('_', '-')}}}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(header) for header in table.headers) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\end{{{environment}}}",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _artifact(
    *,
    artifact_id: str,
    role: str,
    relative_path: str,
    media_type: str,
    payload: bytes,
    table_number: int | None = None,
    row_count: int | None = None,
) -> PaperTableArtifact:
    return PaperTableArtifact(
        artifact_id=artifact_id,
        role=role,
        relative_path=relative_path,
        media_type=media_type,
        sha256=_sha256(payload),
        byte_count=len(payload),
        table_number=table_number,
        row_count=row_count,
    )


def _preflight_and_write(directory: Path, payloads: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ImmutableArtifactError("paper-table output must be a real directory")
    for relative_path, expected in payloads.items():
        target = directory / relative_path
        if target.exists() and target.read_bytes() != expected:
            raise ImmutableArtifactError(
                f"refusing to overwrite different immutable artifact: {target}"
            )
    for relative_path, expected in payloads.items():
        target = directory / relative_path
        if target.exists():
            continue
        try:
            with target.open("xb") as stream:
                stream.write(expected)
        except FileExistsError:
            if target.read_bytes() != expected:
                raise ImmutableArtifactError(
                    f"concurrent writer created different artifact: {target}"
                )


def render_paper_tables(
    source: StudyAggregateInput | str | Path,
    output_directory: str | Path,
) -> RenderedPaperTables:
    """Render immutable JSON, CSV, LaTeX, and hash-manifest artifacts.

    Existing byte-identical artifacts are accepted, making retries idempotent.
    A differing file is never overwritten.
    """

    if isinstance(source, StudyAggregateInput):
        supplied_study = source
        supplied_bytes = _canonical_json_bytes(source)
    else:
        supplied_study, supplied_bytes = load_study_json(source)
    verify_reported_model_totals(supplied_study)
    study = apply_rate_based_costing(supplied_study)
    study_bytes = _canonical_json_bytes(study)

    payloads: dict[str, bytes] = {"study_aggregate.json": study_bytes}
    artifacts: list[PaperTableArtifact] = [
        _artifact(
            artifact_id="paper-study-snapshot",
            role="study_snapshot",
            relative_path="study_aggregate.json",
            media_type="application/json",
            payload=study_bytes,
        )
    ]
    for table in build_paper_tables(study):
        csv_name = f"table_{table.number}_{table.slug}.csv"
        latex_name = f"table_{table.number}_{table.slug}.tex"
        csv_payload = _csv_bytes(table)
        latex_payload = _latex_bytes(table)
        payloads[csv_name] = csv_payload
        payloads[latex_name] = latex_payload
        artifacts.extend(
            (
                _artifact(
                    artifact_id=f"paper-table-{table.number}-csv",
                    role="paper_table_csv",
                    relative_path=csv_name,
                    media_type="text/csv",
                    payload=csv_payload,
                    table_number=table.number,
                    row_count=len(table.rows),
                ),
                _artifact(
                    artifact_id=f"paper-table-{table.number}-latex",
                    role="paper_table_latex",
                    relative_path=latex_name,
                    media_type="application/x-latex",
                    payload=latex_payload,
                    table_number=table.number,
                    row_count=len(table.rows),
                ),
            )
        )

    manifest = PaperTableManifest(
        study_id=study.study_id,
        provided_input_sha256=_sha256(supplied_bytes),
        rendered_study_sha256=_sha256(study_bytes),
        pricing_snapshot_sha256=canonical_json_sha256(study.model_matrix.pricing),
        comparison_regime_id=study.search_protocol.comparison_regime.regime_id,
        scientific_evidence_level=study.qualification.evidence_level,
        scientific_performance_claim_allowed=(
            study.qualification.scientific_performance_claim_allowed
        ),
        artifacts=tuple(artifacts),
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    payloads["manifest.json"] = manifest_bytes
    payloads["manifest.sha256"] = f"{manifest_sha256}  manifest.json\n".encode(
        "ascii"
    )
    destination = Path(output_directory)
    _preflight_and_write(destination, payloads)
    verify_paper_table_bundle(destination)
    return RenderedPaperTables(
        output_directory=destination,
        manifest_path=destination / "manifest.json",
        manifest_sha256_path=destination / "manifest.sha256",
        manifest_sha256=manifest_sha256,
        artifact_paths=tuple(destination / item.relative_path for item in artifacts),
    )


def verify_paper_table_bundle(output_directory: str | Path) -> PaperTableManifest:
    """Verify the manifest sidecar and every content-addressed table artifact."""

    directory = Path(output_directory)
    manifest_path = directory / "manifest.json"
    sidecar_path = directory / "manifest.sha256"
    try:
        manifest_bytes = manifest_path.read_bytes()
        expected_sidecar = f"{_sha256(manifest_bytes)}  manifest.json\n".encode("ascii")
        if sidecar_path.read_bytes() != expected_sidecar:
            raise PaperTableError("paper-table manifest SHA-256 sidecar is invalid")
        manifest = PaperTableManifest.model_validate_json(manifest_bytes)
        for artifact in manifest.artifacts:
            payload = (directory / artifact.relative_path).read_bytes()
            if (
                len(payload) != artifact.byte_count
                or _sha256(payload) != artifact.sha256
            ):
                raise PaperTableError(
                    f"paper-table artifact digest mismatch: {artifact.relative_path}"
                )
        snapshot_artifact = next(
            artifact
            for artifact in manifest.artifacts
            if artifact.role == "study_snapshot"
        )
        if snapshot_artifact.sha256 != manifest.rendered_study_sha256:
            raise PaperTableError("manifest study hashes disagree")
        snapshot = StudyAggregateInput.model_validate_json(
            (directory / snapshot_artifact.relative_path).read_bytes()
        )
        if snapshot.study_id != manifest.study_id:
            raise PaperTableError("manifest references a different study ID")
        if (
            snapshot.search_protocol.comparison_regime.regime_id
            != manifest.comparison_regime_id
        ):
            raise PaperTableError("manifest comparison regime is inconsistent")
        if (
            snapshot.qualification.evidence_level
            != manifest.scientific_evidence_level
            or snapshot.qualification.scientific_performance_claim_allowed
            != manifest.scientific_performance_claim_allowed
        ):
            raise PaperTableError("manifest scientific qualification is inconsistent")
        if (
            canonical_json_sha256(snapshot.model_matrix.pricing)
            != manifest.pricing_snapshot_sha256
        ):
            raise PaperTableError("manifest pricing snapshot hash is inconsistent")
    except FileNotFoundError as exc:
        raise PaperTableError(
            f"paper-table bundle is incomplete: {exc.filename}"
        ) from exc
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render immutable paper tables from a StudyAggregateInput JSON file."
        )
    )
    parser.add_argument("--study-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    rendered = render_paper_tables(arguments.study_json, arguments.output_dir)
    print(
        json.dumps(
            {
                "manifest": str(rendered.manifest_path),
                "manifest_sha256": rendered.manifest_sha256,
                "output_directory": str(rendered.output_directory),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
