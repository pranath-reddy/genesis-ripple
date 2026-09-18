"""Versioned contracts for reproducible architecture/model comparison studies.

The records in this module separate measured observations (tokens, timings, and
metrics) from rate-based cost estimates.  In particular, an estimated API cost
is never represented as an AWS invoice or as total compute cost.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    FrozenModel,
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    canonical_json_sha256,
    validate_relative_artifact_path,
)


NONNEGATIVE_DECIMAL = Decimal("0")
TOKEN_RATE_DENOMINATOR = 1_000_000


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError("timestamp must use UTC")
    return value


class StudyDatasetIdentity(FrozenModel):
    """Identity and split boundary shared by every comparable trained arm."""

    schema_version: Literal["ripple.study-dataset-identity.v1"] = (
        "ripple.study-dataset-identity.v1"
    )
    dataset_id: str = Field(pattern=IDENTIFIER_PATTERN)
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    purpose: Literal["integration_smoke", "scientific_training"]
    data_origin: Literal["synthetic_simulation", "observational", "hybrid"]
    source: str = Field(min_length=1, max_length=512)
    bands: tuple[str, ...] = Field(min_length=1, max_length=32)
    sample_shape_chw: tuple[int, int, int]
    class_names: tuple[str, ...] = Field(min_length=2, max_length=32)
    split_counts: dict[Literal["train", "validation", "test"], int]
    scientific_use_allowed: bool
    supports_scientific_claims: bool
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_dataset(self) -> "StudyDatasetIdentity":
        if len(self.class_names) != len(set(self.class_names)):
            raise ValueError("dataset class names must be unique")
        if len(self.bands) != len(set(self.bands)):
            raise ValueError("dataset band names must be unique")
        if any(not band or len(band) > 64 for band in self.bands):
            raise ValueError("dataset band names must be non-empty and bounded")
        channels, height, width = self.sample_shape_chw
        if channels != len(self.bands):
            raise ValueError("dataset channel count must equal the band count")
        if channels < 1 or height < 16 or width < 16:
            raise ValueError("dataset sample shape is outside supported bounds")
        if set(self.split_counts) != {"train", "validation", "test"}:
            raise ValueError("dataset must freeze train, validation, and test splits")
        if any(count <= 0 for count in self.split_counts.values()):
            raise ValueError("every frozen dataset split must be non-empty")
        if self.supports_scientific_claims and not self.scientific_use_allowed:
            raise ValueError(
                "a dataset cannot support scientific claims while scientific use "
                "is disallowed"
            )
        return self


class ScientificQualification(FrozenModel):
    """Explicit boundary on claims that may be made from a study artifact."""

    evidence_level: Literal[
        "integration_smoke",
        "synthetic_benchmark_unqualified",
        "controlled_experiment",
        "scientific_training",
    ]
    scientific_performance_claim_allowed: bool
    fair_architecture_comparison_allowed: bool
    held_out_test_used_once_after_selection: bool
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _smoke_is_not_scientific(self) -> "ScientificQualification":
        if (
            self.evidence_level == "integration_smoke"
            and self.fair_architecture_comparison_allowed
        ):
            raise ValueError(
                "integration-smoke evidence cannot authorize fair comparisons"
            )
        if self.evidence_level in {
            "integration_smoke",
            "synthetic_benchmark_unqualified",
        } and self.scientific_performance_claim_allowed:
            raise ValueError(
                "unqualified evidence cannot authorize scientific performance claims"
            )
        return self


class ComparisonRegime(FrozenModel):
    """Rules defining which arms can be compared as a controlled experiment."""

    regime_id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: Literal["shared_data_equal_budget_from_scratch"] = (
        "shared_data_equal_budget_from_scratch"
    )
    shared_dataset_required: Literal[True] = True
    shared_split_required: Literal[True] = True
    equal_training_budget_required: Literal[True] = True
    held_out_test_policy: Literal["single_use_after_arm_selection"] = (
        "single_use_after_arm_selection"
    )
    external_checkpoint_policy: Literal[
        "report_separately_not_a_fair_training_comparison"
    ] = "report_separately_not_a_fair_training_comparison"
    model_comparison_scope: Literal[
        "all_models_all_arms",
        "observed_invocations_only",
    ] = "observed_invocations_only"
    description: str = Field(min_length=1, max_length=1600)


class SearchProtocol(FrozenModel):
    """Pre-registered search and selection policy for one study."""

    schema_version: Literal["ripple.search-protocol.v1"] = (
        "ripple.search-protocol.v1"
    )
    protocol_id: str = Field(pattern=IDENTIFIER_PATTERN)
    objective: Literal["binary_strong_lens_classification"] = (
        "binary_strong_lens_classification"
    )
    primary_selection_metric: Literal[
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_roc_auc",
    ]
    maximize_primary_metric: Literal[True] = True
    comparison_regime: ComparisonRegime
    phase_a_description: str = Field(min_length=1, max_length=1000)
    phase_b_description: str = Field(min_length=1, max_length=1000)
    random_seeds: tuple[int, ...] = Field(min_length=1)
    test_metrics_hidden_during_search: Literal[True] = True

    @field_validator("random_seeds")
    @classmethod
    def _unique_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) != len(set(value)):
            raise ValueError("search-protocol random seeds must be unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in value):
            raise ValueError("random seeds must fit an unsigned 32-bit integer")
        return value


class TuningPolicy(FrozenModel):
    """Bounded closed-loop tuning policy applied equally to eligible arms."""

    schema_version: Literal["ripple.tuning-policy.v1"] = (
        "ripple.tuning-policy.v1"
    )
    policy_id: str = Field(pattern=IDENTIFIER_PATTERN)
    maximum_iterations: int = Field(ge=1, le=100)
    allowed_decisions: tuple[str, ...] = Field(min_length=1, max_length=32)
    stop_decision: str = Field(min_length=1, max_length=128)
    generalization_gap_threshold: float = Field(ge=0.0, le=1.0)
    minimum_validation_improvement: float = Field(ge=0.0, le=1.0)
    per_iteration_epoch_budget: int = Field(ge=1, le=10_000)
    augmentation_policy: str = Field(min_length=1, max_length=1000)
    policy_description: str = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def _valid_decisions(self) -> "TuningPolicy":
        if len(self.allowed_decisions) != len(set(self.allowed_decisions)):
            raise ValueError("tuning decisions must be unique")
        if self.stop_decision not in self.allowed_decisions:
            raise ValueError("stop decision must be in the tuning decision allowlist")
        return self


class BedrockModelAvailabilityRecord(FrozenModel):
    """Account-, region-, and time-specific Bedrock discovery evidence."""

    schema_version: Literal["ripple.bedrock-model-availability.v1"] = (
        "ripple.bedrock-model-availability.v1"
    )
    availability_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(min_length=1, max_length=512)
    inference_profile_id: str | None = Field(default=None, max_length=512)
    display_name: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    region: str = Field(pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
    category: Literal["paid_api", "free_open_source", "external_api"]
    catalog_status: Literal["listed", "not_listed", "not_checked"]
    access_status: Literal[
        "enabled",
        "unavailable",
        "blocked_scp",
        "access_denied",
        "unknown",
    ]
    invocation_status: Literal["succeeded", "failed", "not_attempted"]
    supports_converse: bool | None = None
    checked_at_utc: datetime
    failure_code: str | None = Field(default=None, max_length=256)
    failure_message: str | None = Field(default=None, max_length=2000)

    _utc_timestamp = field_validator("checked_at_utc")(_require_utc)

    @model_validator(mode="after")
    def _consistent_availability(self) -> "BedrockModelAvailabilityRecord":
        if self.invocation_status == "succeeded" and self.access_status != "enabled":
            raise ValueError("a successful invocation proves enabled access")
        if self.invocation_status == "failed" and not self.failure_code:
            raise ValueError("a failed invocation must retain its failure code")
        if self.invocation_status != "failed" and (
            self.failure_code is not None or self.failure_message is not None
        ):
            raise ValueError("failure details are only valid for a failed invocation")
        return self


class BedrockTokenRate(FrozenModel):
    """One auditable USD-per-million-token pricing record."""

    rate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    model_id: str = Field(min_length=1, max_length=512)
    display_name: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    region: str = Field(pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
    category: Literal["paid_api", "free_open_source", "external_api"]
    pricing_basis: Literal[
        "published_rate",
        "account_specific_rate",
        "manual_rate_estimate",
        "no_api_charge",
    ]
    input_usd_per_million_tokens: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    output_usd_per_million_tokens: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    cache_write_usd_per_million_tokens: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    cache_read_usd_per_million_tokens: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    source_locator: str = Field(min_length=1, max_length=2000)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _valid_rates(self) -> "BedrockTokenRate":
        ordinary = (
            self.input_usd_per_million_tokens,
            self.output_usd_per_million_tokens,
        )
        if self.pricing_basis == "no_api_charge":
            values = (
                *ordinary,
                self.cache_write_usd_per_million_tokens,
                self.cache_read_usd_per_million_tokens,
            )
            if any(value not in {None, NONNEGATIVE_DECIMAL} for value in values):
                raise ValueError("no-api-charge records may only contain zero rates")
        elif any(value is None for value in ordinary):
            raise ValueError("priced models require ordinary input and output rates")
        return self


class BedrockPricingSnapshot(FrozenModel):
    """Frozen price inputs; subsequent AWS price changes cannot alter a run."""

    schema_version: Literal["ripple.bedrock-pricing-snapshot.v1"] = (
        "ripple.bedrock-pricing-snapshot.v1"
    )
    snapshot_id: str = Field(pattern=IDENTIFIER_PATTERN)
    currency: Literal["USD"] = "USD"
    token_rate_denominator: Literal[1_000_000] = TOKEN_RATE_DENOMINATOR
    effective_at_utc: datetime
    captured_at_utc: datetime
    rates: tuple[BedrockTokenRate, ...]
    qualification: Literal[
        "rates_are_inputs_to_estimates_not_aws_invoices"
    ] = "rates_are_inputs_to_estimates_not_aws_invoices"

    _effective_utc = field_validator("effective_at_utc")(_require_utc)
    _captured_utc = field_validator("captured_at_utc")(_require_utc)

    @model_validator(mode="after")
    def _unique_rates(self) -> "BedrockPricingSnapshot":
        keys = [(rate.model_id, rate.region) for rate in self.rates]
        ids = [rate.rate_id for rate in self.rates]
        if len(keys) != len(set(keys)):
            raise ValueError("pricing snapshot contains duplicate model/region rates")
        if len(ids) != len(set(ids)):
            raise ValueError("pricing rate IDs must be unique")
        return self


class BedrockModelMatrix(FrozenModel):
    """Model discovery results paired with the exact rate snapshot used."""

    schema_version: Literal["ripple.bedrock-model-matrix.v1"] = (
        "ripple.bedrock-model-matrix.v1"
    )
    matrix_id: str = Field(pattern=IDENTIFIER_PATTERN)
    account_fingerprint_sha256: str = Field(pattern=SHA256_PATTERN)
    availability: tuple[BedrockModelAvailabilityRecord, ...] = Field(min_length=1)
    pricing: BedrockPricingSnapshot

    @model_validator(mode="after")
    def _unique_availability(self) -> "BedrockModelMatrix":
        keys = [
            (item.model_id, item.inference_profile_id, item.region)
            for item in self.availability
        ]
        ids = [item.availability_id for item in self.availability]
        if len(keys) != len(set(keys)):
            raise ValueError("model matrix contains duplicate availability records")
        if len(ids) != len(set(ids)):
            raise ValueError("model availability IDs must be unique")
        return self


class TokenUsage(FrozenModel):
    """Measured token components with explicit cache arithmetic."""

    schema_version: Literal["ripple.measured-token-usage.v1"] = (
        "ripple.measured-token-usage.v1"
    )
    uncached_input_tokens: int = Field(ge=0)
    cache_write_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    measured_total_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    source: Literal["bedrock_usage", "provider_usage", "local_tokenizer_measurement"]

    @model_validator(mode="after")
    def _cache_arithmetic(self) -> "TokenUsage":
        expected = (
            self.uncached_input_tokens
            + self.cache_write_input_tokens
            + self.cache_read_input_tokens
        )
        if self.measured_total_input_tokens != expected:
            raise ValueError(
                "measured_total_input_tokens must equal uncached + cache-write + "
                "cache-read input tokens"
            )
        return self

    @property
    def measured_total_tokens(self) -> int:
        return self.measured_total_input_tokens + self.output_tokens


class RateBasedCostEstimate(FrozenModel):
    """Token-rate multiplication result, explicitly not a billing artifact."""

    schema_version: Literal["ripple.rate-based-cost-estimate.v1"] = (
        "ripple.rate-based-cost-estimate.v1"
    )
    estimation_method: Literal["rate_based_estimate"] = "rate_based_estimate"
    is_invoice: Literal[False] = False
    status: Literal["estimated", "unavailable"]
    currency: Literal["USD"] = "USD"
    invoked_model_id: str = Field(min_length=1, max_length=512)
    pricing_model_id: str = Field(min_length=1, max_length=512)
    region: str = Field(pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
    pricing_snapshot_id: str = Field(pattern=IDENTIFIER_PATTERN)
    pricing_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    rate_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    token_usage_sha256: str = Field(pattern=SHA256_PATTERN)
    uncached_input_cost_usd: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    cache_write_cost_usd: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    cache_read_cost_usd: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    output_cost_usd: Decimal | None = Field(default=None, ge=NONNEGATIVE_DECIMAL)
    total_estimated_cost_usd: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    unavailable_reasons: tuple[str, ...] = ()
    qualification: Literal[
        "rate_based_estimate_not_aws_invoice"
    ] = "rate_based_estimate_not_aws_invoice"

    @model_validator(mode="after")
    def _valid_estimate(self) -> "RateBasedCostEstimate":
        components = (
            self.uncached_input_cost_usd,
            self.cache_write_cost_usd,
            self.cache_read_cost_usd,
            self.output_cost_usd,
        )
        if self.status == "estimated":
            if self.rate_id is None or any(value is None for value in components):
                raise ValueError(
                    "an estimated cost requires a rate and every component"
                )
            expected = sum(
                (value for value in components if value is not None),
                Decimal(0),
            )
            if self.total_estimated_cost_usd != expected:
                raise ValueError("total estimated cost does not equal its components")
            if self.unavailable_reasons:
                raise ValueError("an estimated cost cannot have unavailable reasons")
        else:
            if self.total_estimated_cost_usd is not None:
                raise ValueError("an unavailable estimate cannot have a total")
            if not self.unavailable_reasons:
                raise ValueError("an unavailable estimate must explain why")
        return self


class AgentStageFailure(FrozenModel):
    error_type: str = Field(min_length=1, max_length=256)
    error_code: str | None = Field(default=None, max_length=256)
    message: str = Field(min_length=1, max_length=4000)
    retryable: bool


class AgentStageEvidenceV2(FrozenModel):
    """One exact agent run within one arm, round, phase, and agent stage."""

    schema_version: Literal["ripple.campaign-agent-stage.v2"] = (
        "ripple.campaign-agent-stage.v2"
    )
    evidence_id: str = Field(pattern=IDENTIFIER_PATTERN)
    request_id: str = Field(pattern=IDENTIFIER_PATTERN)
    arm_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    round_index: int = Field(ge=0)
    phase: Literal["phase_a", "phase_b", "other"]
    stage: str = Field(pattern=IDENTIFIER_PATTERN)
    attempt_index: int = Field(default=1, ge=1, le=100)
    provider_request_count: int | None = Field(default=None, ge=0)
    tool_call_count: int | None = Field(default=None, ge=0)
    model_id: str = Field(min_length=1, max_length=512)
    pricing_model_id: str | None = Field(default=None, max_length=512)
    inference_profile_id: str | None = Field(default=None, max_length=512)
    provider: str = Field(min_length=1, max_length=128)
    region: str = Field(pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
    status: Literal["succeeded", "failed"]
    started_at_utc: datetime
    completed_at_utc: datetime
    wall_time_seconds: float = Field(ge=0.0)
    token_usage: TokenUsage | None = None
    cost: RateBasedCostEstimate | None = None
    output_artifact_id: str | None = Field(default=None, pattern=IDENTIFIER_PATTERN)
    output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    called_tools: tuple[str, ...] = ()
    failure: AgentStageFailure | None = None

    _started_utc = field_validator("started_at_utc")(_require_utc)
    _completed_utc = field_validator("completed_at_utc")(_require_utc)

    @model_validator(mode="after")
    def _consistent_stage(self) -> "AgentStageEvidenceV2":
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("agent stage completed before it started")
        observed = (self.completed_at_utc - self.started_at_utc).total_seconds()
        tolerance = max(0.500, observed * 0.05)
        if not math.isclose(self.wall_time_seconds, observed, abs_tol=tolerance):
            raise ValueError("agent-stage wall time disagrees with UTC timestamps")
        if self.status == "succeeded":
            if self.provider_request_count is None or self.provider_request_count < 1:
                raise ValueError(
                    "a successful agent run needs its provider request count"
                )
            if self.tool_call_count is None:
                raise ValueError("a successful agent run needs its tool-call count")
            if self.failure is not None:
                raise ValueError("successful agent stage cannot retain a failure")
            if self.output_artifact_id is None or self.output_sha256 is None:
                raise ValueError("successful agent stage requires its persisted output")
        elif self.failure is None:
            raise ValueError(
                "failed agent stage must retain structured failure evidence"
            )
        if self.token_usage is not None and (
            self.provider_request_count is None or self.provider_request_count < 1
        ):
            raise ValueError("measured token usage requires a provider request")
        if (self.output_artifact_id is None) != (self.output_sha256 is None):
            raise ValueError("output artifact ID and digest must be present together")
        if self.cost is not None:
            if self.token_usage is None:
                raise ValueError("a cost estimate requires measured token usage")
            if self.cost.invoked_model_id != self.model_id:
                raise ValueError("cost estimate references a different invoked model")
            expected_pricing_model = self.pricing_model_id or self.model_id
            if self.cost.pricing_model_id != expected_pricing_model:
                raise ValueError("cost estimate references a different pricing model")
            if self.cost.region != self.region:
                raise ValueError("cost estimate references a different region")
            if self.cost.token_usage_sha256 != canonical_json_sha256(self.token_usage):
                raise ValueError("cost estimate is not bound to this token usage")
        return self


class ClassificationMetrics(FrozenModel):
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    balanced_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    loss: float | None = Field(default=None, ge=0.0)
    sample_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _at_least_one_metric(self) -> "ClassificationMetrics":
        if all(
            value is None
            for value in (
                self.accuracy,
                self.balanced_accuracy,
                self.roc_auc,
                self.loss,
            )
        ):
            raise ValueError("classification metrics must contain at least one metric")
        return self


class ArchitecturePlanCandidate(FrozenModel):
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    architecture_family: str = Field(pattern=IDENTIFIER_PATTERN)
    role: Literal["searched", "trained_baseline"]
    plan_order: int = Field(ge=0)
    proposed_by_evidence_id: str | None = Field(
        default=None, pattern=IDENTIFIER_PATTERN
    )
    specification_sha256: str = Field(pattern=SHA256_PATTERN)


class ArchitecturePlan(FrozenModel):
    schema_version: Literal["ripple.study-architecture-plan.v1"] = (
        "ripple.study-architecture-plan.v1"
    )
    plan_id: str = Field(pattern=IDENTIFIER_PATTERN)
    candidates: tuple[ArchitecturePlanCandidate, ...] = Field(min_length=1)
    selected_arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selection_evidence_id: str | None = Field(
        default=None, pattern=IDENTIFIER_PATTERN
    )

    @model_validator(mode="after")
    def _valid_plan(self) -> "ArchitecturePlan":
        for field_name, values in {
            "candidate IDs": [item.candidate_id for item in self.candidates],
            "arm IDs": [item.arm_id for item in self.candidates],
            "plan order": [item.plan_order for item in self.candidates],
        }.items():
            if len(values) != len(set(values)):
                raise ValueError(f"architecture plan has duplicate {field_name}")
        if self.selected_arm_id not in {item.arm_id for item in self.candidates}:
            raise ValueError("selected arm is absent from the architecture plan")
        orders = sorted(item.plan_order for item in self.candidates)
        if orders != list(range(len(self.candidates))):
            raise ValueError("architecture plan order must be contiguous from zero")
        selected = next(
            item for item in self.candidates if item.arm_id == self.selected_arm_id
        )
        if selected.role != "searched":
            raise ValueError("the selected architecture arm must be a searched arm")
        baselines = [
            item for item in self.candidates if item.role == "trained_baseline"
        ]
        if len(baselines) != 1:
            raise ValueError("architecture plan requires exactly one trained baseline")
        return self


class ArmTrajectoryPoint(FrozenModel):
    """One closed-loop observation and the decision made from it."""

    arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    iteration: int = Field(ge=0)
    training_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    train_metrics: ClassificationMetrics
    validation_metrics: ClassificationMetrics
    accuracy_generalization_gap: float | None = Field(default=None, ge=-1.0, le=1.0)
    planner_decision: str = Field(min_length=1, max_length=256)
    planner_rationale: str = Field(min_length=1, max_length=1600)
    stage_evidence_ids: tuple[str, ...] = ()
    worker_seconds: float = Field(ge=0.0)
    remote_wall_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _valid_gap(self) -> "ArmTrajectoryPoint":
        if (
            self.train_metrics.accuracy is not None
            and self.validation_metrics.accuracy is not None
        ):
            expected = self.train_metrics.accuracy - self.validation_metrics.accuracy
            if self.accuracy_generalization_gap is None or not math.isclose(
                self.accuracy_generalization_gap,
                expected,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "accuracy gap must equal train accuracy minus validation"
                )
        elif self.accuracy_generalization_gap is not None:
            raise ValueError("accuracy gap needs both train and validation accuracy")
        if len(self.stage_evidence_ids) != len(set(self.stage_evidence_ids)):
            raise ValueError("trajectory stage-evidence IDs must be unique")
        return self


class FinalEvaluationSummary(FrozenModel):
    """Frozen validation result and optional single-use held-out test result."""

    schema_version: Literal["ripple.study-final-evaluation.v1"] = (
        "ripple.study-final-evaluation.v1"
    )
    evaluation_id: str = Field(pattern=IDENTIFIER_PATTERN)
    arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    selected_training_run_id: str = Field(pattern=IDENTIFIER_PATTERN)
    checkpoint_sha256: str = Field(pattern=SHA256_PATTERN)
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics | None = None
    test_evaluated_at_utc: datetime | None = None

    @field_validator("test_evaluated_at_utc")
    @classmethod
    def _optional_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value)

    @model_validator(mode="after")
    def _test_fields_together(self) -> "FinalEvaluationSummary":
        if (self.test_metrics is None) != (self.test_evaluated_at_utc is None):
            raise ValueError(
                "test metrics and their evaluation time must be present together"
            )
        return self


class ArmStudyResult(FrozenModel):
    """Complete trained-arm trajectory and resource result."""

    schema_version: Literal["ripple.study-arm-result.v1"] = (
        "ripple.study-arm-result.v1"
    )
    arm_id: str = Field(pattern=IDENTIFIER_PATTERN)
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    architecture_family: str = Field(pattern=IDENTIFIER_PATTERN)
    status: Literal["succeeded", "failed", "blocked"]
    parameter_count: int | None = Field(default=None, gt=0)
    tuning_iterations: int = Field(ge=0)
    total_worker_seconds: float = Field(ge=0.0)
    total_remote_wall_seconds: float = Field(ge=0.0)
    trajectory: tuple[ArmTrajectoryPoint, ...] = ()
    final_evaluation: FinalEvaluationSummary | None = None
    failure_reason: str | None = Field(default=None, max_length=4000)
    fair_training_comparison_allowed: bool

    @model_validator(mode="after")
    def _valid_arm(self) -> "ArmStudyResult":
        if any(point.arm_id != self.arm_id for point in self.trajectory):
            raise ValueError("trajectory contains a point from another arm")
        iterations = [point.iteration for point in self.trajectory]
        if iterations != list(range(len(iterations))):
            raise ValueError(
                "trajectory iterations must be contiguous and start at zero"
            )
        if self.tuning_iterations != len(self.trajectory):
            raise ValueError("tuning_iterations must equal the trajectory length")
        if (
            self.final_evaluation is not None
            and self.final_evaluation.arm_id != self.arm_id
        ):
            raise ValueError("final evaluation belongs to another arm")
        if self.status == "succeeded":
            if not self.trajectory or self.parameter_count is None:
                raise ValueError(
                    "a successful arm needs a trajectory and parameter count"
                )
            if self.final_evaluation is None:
                raise ValueError("a successful arm needs a final evaluation summary")
            if self.failure_reason is not None:
                raise ValueError("a successful arm cannot retain a failure reason")
        elif not self.failure_reason:
            raise ValueError("failed or blocked arms must explain their status")
        if self.fair_training_comparison_allowed and self.status != "succeeded":
            raise ValueError(
                "only successful trained arms can enter the fair comparison"
            )
        if self.total_worker_seconds + 1e-9 < sum(
            point.worker_seconds for point in self.trajectory
        ):
            raise ValueError("arm worker total is below its trajectory total")
        if self.total_remote_wall_seconds + 1e-9 < sum(
            point.remote_wall_seconds for point in self.trajectory
        ):
            raise ValueError("arm remote total is below its trajectory total")
        return self


class MrigankaZeroShotSummary(FrozenModel):
    """Separately reported frozen external checkpoint; never a trained-arm peer."""

    schema_version: Literal["ripple.mriganka-zero-shot-summary.v1"] = (
        "ripple.mriganka-zero-shot-summary.v1"
    )
    name: Literal["mriganka"] = "mriganka"
    mode: Literal["frozen_external_checkpoint_zero_shot"] = (
        "frozen_external_checkpoint_zero_shot"
    )
    status: Literal["not_run", "succeeded", "failed", "blocked"]
    fair_training_comparison_allowed: Literal[False] = False
    checkpoint_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    learned_parameter_count: int | None = Field(default=None, gt=0)
    materialized_state_element_count: int | None = Field(default=None, gt=0)
    evaluation_dataset_id: str | None = Field(
        default=None, pattern=IDENTIFIER_PATTERN
    )
    metrics: ClassificationMetrics | None = None
    evaluated_at_utc: datetime | None = None
    reason: str = Field(min_length=1, max_length=4000)

    @field_validator("evaluated_at_utc")
    @classmethod
    def _evaluation_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _require_utc(value)

    @model_validator(mode="after")
    def _valid_zero_shot(self) -> "MrigankaZeroShotSummary":
        measured = (
            self.checkpoint_sha256,
            self.evaluation_dataset_id,
            self.metrics,
            self.evaluated_at_utc,
        )
        if self.status == "succeeded" and any(value is None for value in measured):
            raise ValueError(
                "successful zero-shot evaluation needs checkpoint and metrics"
            )
        if self.status != "succeeded" and self.metrics is not None:
            raise ValueError("an unsuccessful zero-shot status cannot expose metrics")
        return self


class ModelStudyTotal(FrozenModel):
    """Redundant, validated roll-up retained for convenient paper rendering."""

    model_id: str = Field(min_length=1, max_length=512)
    display_name: str = Field(min_length=1, max_length=256)
    provider: str = Field(min_length=1, max_length=128)
    category: Literal["paid_api", "free_open_source", "external_api"]
    stage_run_count: int = Field(ge=0)
    known_provider_request_count: int = Field(ge=0)
    provider_request_count: int | None = Field(default=None, ge=0)
    provider_request_count_status: Literal["complete", "partially_unavailable"]
    succeeded_stage_run_count: int = Field(ge=0)
    failed_stage_run_count: int = Field(ge=0)
    requests_with_measured_tokens: int = Field(ge=0)
    arms_invoked: int = Field(ge=0)
    uncached_input_tokens: int = Field(ge=0)
    cache_write_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    measured_total_input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_rate_based_estimated_cost_usd: Decimal | None = Field(
        default=None, ge=NONNEGATIVE_DECIMAL
    )
    cost_status: Literal["estimated", "partially_unavailable", "unavailable"]
    missing_cost_request_ids: tuple[str, ...] = ()
    cost_qualification: Literal[
        "rate_based_estimate_not_aws_invoice"
    ] = "rate_based_estimate_not_aws_invoice"

    @model_validator(mode="after")
    def _valid_total(self) -> "ModelStudyTotal":
        if (
            self.succeeded_stage_run_count + self.failed_stage_run_count
            != self.stage_run_count
        ):
            raise ValueError("model stage-run statuses do not sum to stage_run_count")
        if self.requests_with_measured_tokens > self.known_provider_request_count:
            raise ValueError("measured-token requests exceed known provider requests")
        if self.provider_request_count_status == "complete":
            if self.provider_request_count != self.known_provider_request_count:
                raise ValueError("complete provider request total is inconsistent")
        elif self.provider_request_count is not None:
            raise ValueError("a partial provider request total must be N/A")
        expected_input = (
            self.uncached_input_tokens
            + self.cache_write_input_tokens
            + self.cache_read_input_tokens
        )
        if self.measured_total_input_tokens != expected_input:
            raise ValueError("model total input-token arithmetic is invalid")
        if self.cost_status == "estimated":
            if self.total_rate_based_estimated_cost_usd is None:
                raise ValueError("estimated model total requires a value")
            if self.missing_cost_request_ids:
                raise ValueError("estimated model total cannot list missing requests")
        elif not self.missing_cost_request_ids:
            raise ValueError("unavailable model costs must identify affected requests")
        return self


class StudyAggregateInput(FrozenModel):
    """Single immutable JSON source for paper tables and manifest generation."""

    schema_version: Literal["ripple.architecture-model-study.v1"] = (
        "ripple.architecture-model-study.v1"
    )
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    title: str = Field(min_length=1, max_length=500)
    dataset: StudyDatasetIdentity
    qualification: ScientificQualification
    search_protocol: SearchProtocol
    tuning_policy: TuningPolicy
    model_matrix: BedrockModelMatrix
    architecture_plan: ArchitecturePlan
    stage_runs: tuple[AgentStageEvidenceV2, ...]
    arm_results: tuple[ArmStudyResult, ...] = Field(min_length=1)
    mriganka_zero_shot: MrigankaZeroShotSummary | None = None
    model_totals: tuple[ModelStudyTotal, ...] = ()
    started_at_utc: datetime
    completed_at_utc: datetime

    _study_started_utc = field_validator("started_at_utc")(_require_utc)
    _study_completed_utc = field_validator("completed_at_utc")(_require_utc)

    @model_validator(mode="after")
    def _consistent_study(self) -> "StudyAggregateInput":
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("study completed before it started")
        if (
            self.dataset.purpose == "integration_smoke"
            and self.qualification.evidence_level != "integration_smoke"
        ):
            raise ValueError("an integration-smoke dataset needs smoke qualification")
        if (
            not self.dataset.supports_scientific_claims
            and self.qualification.scientific_performance_claim_allowed
        ):
            raise ValueError("study qualification exceeds dataset claim support")
        if (
            not self.dataset.scientific_use_allowed
            and self.qualification.scientific_performance_claim_allowed
        ):
            raise ValueError("study claims are forbidden by the dataset record")
        if (
            self.dataset.data_origin == "synthetic_simulation"
            and not self.dataset.supports_scientific_claims
            and self.qualification.evidence_level
            not in {"integration_smoke", "synthetic_benchmark_unqualified"}
        ):
            raise ValueError("unqualified synthetic data need synthetic qualification")
        plan_by_arm = {
            candidate.arm_id: candidate
            for candidate in self.architecture_plan.candidates
        }
        result_by_arm = {result.arm_id: result for result in self.arm_results}
        if len(result_by_arm) != len(self.arm_results):
            raise ValueError("study contains duplicate arm results")
        if set(result_by_arm) != set(plan_by_arm):
            raise ValueError("arm results must cover the architecture plan exactly")
        for arm_id, result in result_by_arm.items():
            candidate = plan_by_arm[arm_id]
            if (
                result.candidate_id != candidate.candidate_id
                or result.architecture_family != candidate.architecture_family
            ):
                raise ValueError("arm result disagrees with its planned candidate")
            if result.tuning_iterations > self.tuning_policy.maximum_iterations:
                raise ValueError("arm exceeds the pre-registered tuning iteration cap")
            if any(
                point.planner_decision not in self.tuning_policy.allowed_decisions
                for point in result.trajectory
            ):
                raise ValueError(
                    "trajectory contains a non-allowlisted tuning decision"
                )
            if (
                result.status == "succeeded"
                and result.trajectory[-1].planner_decision
                != self.tuning_policy.stop_decision
            ):
                raise ValueError(
                    "a successful trajectory must end with the stop decision"
                )
            for point in result.trajectory:
                if (
                    point.train_metrics.sample_count
                    != self.dataset.split_counts["train"]
                ):
                    raise ValueError(
                        "trajectory train sample count disagrees with dataset"
                    )
                if (
                    point.validation_metrics.sample_count
                    != self.dataset.split_counts["validation"]
                ):
                    raise ValueError(
                        "trajectory validation sample count disagrees with dataset"
                    )
            if result.final_evaluation is not None:
                final = result.final_evaluation
                matching_points = [
                    point
                    for point in result.trajectory
                    if point.training_run_id == final.selected_training_run_id
                    and point.checkpoint_sha256 == final.checkpoint_sha256
                ]
                if len(matching_points) != 1:
                    raise ValueError(
                        "final evaluation is not bound to exactly one trajectory run"
                    )
                if final.validation_metrics != matching_points[0].validation_metrics:
                    raise ValueError(
                        "final validation metrics disagree with selected trajectory"
                    )
                if (
                    final.validation_metrics.sample_count
                    != self.dataset.split_counts["validation"]
                ):
                    raise ValueError(
                        "final validation sample count disagrees with dataset"
                    )
                if (
                    final.test_metrics is not None
                    and final.test_metrics.sample_count
                    != self.dataset.split_counts["test"]
                ):
                    raise ValueError("held-out sample count disagrees with dataset")
        selected = result_by_arm[self.architecture_plan.selected_arm_id]
        if selected.status != "succeeded":
            raise ValueError("the selected architecture arm must have succeeded")
        held_out_arm_ids = {
            self.architecture_plan.selected_arm_id,
            *(
                candidate.arm_id
                for candidate in self.architecture_plan.candidates
                if candidate.role == "trained_baseline"
            ),
        }
        actual_held_out_arm_ids = {
            result.arm_id
            for result in self.arm_results
            if result.final_evaluation is not None
            and result.final_evaluation.test_metrics is not None
        }
        if not actual_held_out_arm_ids <= held_out_arm_ids:
            raise ValueError(
                "held-out metrics are limited to selected searched arm and baseline"
            )
        eligible_count = sum(
            result.fair_training_comparison_allowed for result in self.arm_results
        )
        if (
            self.qualification.fair_architecture_comparison_allowed
            and eligible_count < 2
        ):
            raise ValueError("a fair architecture comparison needs two eligible arms")
        if any(
            result.fair_training_comparison_allowed
            and not self.qualification.fair_architecture_comparison_allowed
            for result in self.arm_results
        ):
            raise ValueError("arm comparison flag exceeds the study qualification")
        evidence_ids = [record.evidence_id for record in self.stage_runs]
        request_ids = [record.request_id for record in self.stage_runs]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("agent-stage evidence IDs must be unique")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("agent-stage request IDs must be unique")
        if any(
            record.started_at_utc < self.started_at_utc
            or record.completed_at_utc > self.completed_at_utc
            for record in self.stage_runs
        ):
            raise ValueError("agent-stage timing falls outside the study interval")
        unknown_stage_arms = {
            record.arm_id
            for record in self.stage_runs
            if record.arm_id is not None and record.arm_id not in plan_by_arm
        }
        if unknown_stage_arms:
            raise ValueError("agent-stage evidence references unknown arms")
        evidence_id_set = set(evidence_ids)
        for result in self.arm_results:
            for point in result.trajectory:
                if not set(point.stage_evidence_ids) <= evidence_id_set:
                    raise ValueError(
                        "trajectory references missing agent-stage evidence"
                    )
                linked = [
                    record
                    for record in self.stage_runs
                    if record.evidence_id in point.stage_evidence_ids
                ]
                if any(record.arm_id not in {None, result.arm_id} for record in linked):
                    raise ValueError("trajectory links agent evidence from another arm")
        proposed_ids = {
            candidate.proposed_by_evidence_id
            for candidate in self.architecture_plan.candidates
            if candidate.proposed_by_evidence_id is not None
        }
        if not proposed_ids <= evidence_id_set:
            raise ValueError("architecture plan references missing proposal evidence")
        if (
            self.architecture_plan.selection_evidence_id is not None
            and self.architecture_plan.selection_evidence_id not in evidence_id_set
        ):
            raise ValueError("architecture plan references missing selection evidence")
        model_total_ids = [total.model_id for total in self.model_totals]
        if len(model_total_ids) != len(set(model_total_ids)):
            raise ValueError("model totals must be unique by model ID")
        invoked_models = {record.model_id for record in self.stage_runs}
        if not set(model_total_ids) <= invoked_models:
            raise ValueError("model totals contain a model with no recorded invocation")
        held_out_present = any(
            result.final_evaluation is not None
            and result.final_evaluation.test_metrics is not None
            for result in self.arm_results
        )
        if (
            held_out_present
            != self.qualification.held_out_test_used_once_after_selection
        ):
            raise ValueError(
                "held-out test evidence disagrees with the scientific qualification"
            )
        return self


class PaperTableArtifact(FrozenModel):
    """Content digest for one deterministic paper-table output."""

    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    role: Literal["study_snapshot", "paper_table_csv", "paper_table_latex"]
    relative_path: str = Field(pattern=r"^[a-z0-9][a-z0-9._/-]*$")
    media_type: Literal["application/json", "text/csv", "application/x-latex"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=1)
    table_number: int | None = Field(default=None, ge=1, le=4)
    row_count: int | None = Field(default=None, ge=0)

    @field_validator("relative_path")
    @classmethod
    def _normalized_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @model_validator(mode="after")
    def _table_metadata(self) -> "PaperTableArtifact":
        is_table = self.role != "study_snapshot"
        if is_table != (self.table_number is not None and self.row_count is not None):
            raise ValueError("only table artifacts carry table number and row count")
        return self


class PaperTableManifest(FrozenModel):
    """Deterministic manifest; its own digest is written as a sidecar."""

    schema_version: Literal["ripple.paper-table-manifest.v1"] = (
        "ripple.paper-table-manifest.v1"
    )
    renderer_version: Literal["ripple-paper-tables-v1"] = "ripple-paper-tables-v1"
    study_id: str = Field(pattern=IDENTIFIER_PATTERN)
    provided_input_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_study_sha256: str = Field(pattern=SHA256_PATTERN)
    pricing_snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    comparison_regime_id: str = Field(pattern=IDENTIFIER_PATTERN)
    scientific_evidence_level: Literal[
        "integration_smoke",
        "synthetic_benchmark_unqualified",
        "controlled_experiment",
        "scientific_training",
    ]
    scientific_performance_claim_allowed: bool
    cost_qualification: Literal[
        "rate_based_estimate_not_aws_invoice"
    ] = "rate_based_estimate_not_aws_invoice"
    artifacts: tuple[PaperTableArtifact, ...] = Field(min_length=9, max_length=9)

    @model_validator(mode="after")
    def _complete_artifact_set(self) -> "PaperTableManifest":
        ids = [artifact.artifact_id for artifact in self.artifacts]
        paths = [artifact.relative_path for artifact in self.artifacts]
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("paper artifact IDs and paths must be unique")
        if sum(artifact.role == "study_snapshot" for artifact in self.artifacts) != 1:
            raise ValueError("paper manifest needs exactly one study snapshot")
        table_formats = {
            (artifact.table_number, artifact.role)
            for artifact in self.artifacts
            if artifact.role != "study_snapshot"
        }
        expected = {
            (number, role)
            for number in range(1, 5)
            for role in ("paper_table_csv", "paper_table_latex")
        }
        if table_formats != expected:
            raise ValueError("paper manifest must contain CSV and LaTeX for tables 1-4")
        return self
