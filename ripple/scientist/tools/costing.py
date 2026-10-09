"""Deterministic token-rate costing for Bedrock study evidence.

This module multiplies measured token components by a frozen pricing snapshot.
Its outputs are rate-based estimates only; they are not AWS invoices and do not
include GPU, storage, data-transfer, taxes, discounts, or support charges.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from ..schemas.common import canonical_json_sha256
from ..schemas.study import (
    AgentStageEvidenceV2,
    BedrockModelAvailabilityRecord,
    BedrockPricingSnapshot,
    BedrockTokenRate,
    ModelStudyTotal,
    RateBasedCostEstimate,
    StudyAggregateInput,
    TokenUsage,
)


USD_QUANTUM = Decimal("0.000000000001")
MILLION = Decimal(1_000_000)


class CostingError(ValueError):
    """Raised when pricing evidence is ambiguous or an aggregate is stale."""


def token_usage_from_bedrock_run(
    *,
    input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> TokenUsage:
    """Normalize Pydantic-AI Bedrock usage without double-counting cache tokens.

    Pydantic-AI's Bedrock adapter reports ``input_tokens`` as the sum of the
    Bedrock ordinary-input, cache-write, and cache-read fields.  The subtraction
    here recovers the mutually exclusive components used by the cost formula.
    """

    values = (input_tokens, cache_write_tokens, cache_read_tokens, output_tokens)
    if any(value < 0 for value in values):
        raise CostingError("Bedrock token counters cannot be negative")
    uncached = input_tokens - cache_write_tokens - cache_read_tokens
    if uncached < 0:
        raise CostingError(
            "Bedrock cache-token counts exceed the reported total input tokens"
        )
    return TokenUsage(
        uncached_input_tokens=uncached,
        cache_write_input_tokens=cache_write_tokens,
        cache_read_input_tokens=cache_read_tokens,
        measured_total_input_tokens=input_tokens,
        output_tokens=output_tokens,
        source="bedrock_usage",
    )


def _usd(value: Decimal) -> Decimal:
    return value.quantize(USD_QUANTUM, rounding=ROUND_HALF_UP)


def _matching_rate(
    snapshot: BedrockPricingSnapshot,
    *,
    model_id: str,
    region: str,
) -> BedrockTokenRate | None:
    matches = [
        rate
        for rate in snapshot.rates
        if rate.model_id == model_id and rate.region == region
    ]
    if len(matches) > 1:
        raise CostingError(
            f"pricing snapshot has ambiguous rates for {model_id!r} in {region!r}"
        )
    return matches[0] if matches else None


def _component_cost(
    *,
    tokens: int,
    rate: Decimal | None,
    component: str,
    no_api_charge: bool,
) -> tuple[Decimal | None, str | None]:
    if tokens == 0 or no_api_charge:
        return Decimal(0), None
    if rate is None:
        return None, f"{component} rate is unavailable for {tokens} measured tokens"
    return _usd(Decimal(tokens) * rate / MILLION), None


def estimate_token_cost(
    usage: TokenUsage,
    *,
    invoked_model_id: str,
    pricing_model_id: str,
    region: str,
    snapshot: BedrockPricingSnapshot,
) -> RateBasedCostEstimate:
    """Calculate one cache-aware rate estimate from measured token usage."""

    snapshot_sha256 = canonical_json_sha256(snapshot)
    usage_sha256 = canonical_json_sha256(usage)
    rate = _matching_rate(snapshot, model_id=pricing_model_id, region=region)
    if rate is None:
        return RateBasedCostEstimate(
            status="unavailable",
            invoked_model_id=invoked_model_id,
            pricing_model_id=pricing_model_id,
            region=region,
            pricing_snapshot_id=snapshot.snapshot_id,
            pricing_snapshot_sha256=snapshot_sha256,
            token_usage_sha256=usage_sha256,
            unavailable_reasons=(
                f"no frozen token rate for {pricing_model_id!r} in {region!r}",
            ),
        )

    no_api_charge = rate.pricing_basis == "no_api_charge"
    specifications = (
        (
            "uncached_input_cost_usd",
            usage.uncached_input_tokens,
            rate.input_usd_per_million_tokens,
            "ordinary input",
        ),
        (
            "cache_write_cost_usd",
            usage.cache_write_input_tokens,
            rate.cache_write_usd_per_million_tokens,
            "cache-write input",
        ),
        (
            "cache_read_cost_usd",
            usage.cache_read_input_tokens,
            rate.cache_read_usd_per_million_tokens,
            "cache-read input",
        ),
        (
            "output_cost_usd",
            usage.output_tokens,
            rate.output_usd_per_million_tokens,
            "output",
        ),
    )
    values: dict[str, Decimal | None] = {}
    reasons: list[str] = []
    for field_name, tokens, token_rate, label in specifications:
        value, reason = _component_cost(
            tokens=tokens,
            rate=token_rate,
            component=label,
            no_api_charge=no_api_charge,
        )
        values[field_name] = value
        if reason is not None:
            reasons.append(reason)

    total = None
    if not reasons:
        total = sum(
            (value for value in values.values() if value is not None), Decimal(0)
        )
    return RateBasedCostEstimate(
        status="unavailable" if reasons else "estimated",
        invoked_model_id=invoked_model_id,
        pricing_model_id=pricing_model_id,
        region=region,
        pricing_snapshot_id=snapshot.snapshot_id,
        pricing_snapshot_sha256=snapshot_sha256,
        rate_id=rate.rate_id,
        token_usage_sha256=usage_sha256,
        total_estimated_cost_usd=total,
        unavailable_reasons=tuple(reasons),
        **values,
    )


def attach_rate_based_cost(
    stage: AgentStageEvidenceV2,
    snapshot: BedrockPricingSnapshot,
) -> AgentStageEvidenceV2:
    """Return a stage copy carrying an estimate when token usage was measured."""

    if stage.token_usage is None:
        return stage.model_copy(update={"cost": None})
    pricing_model_id = stage.pricing_model_id or stage.model_id
    estimate = estimate_token_cost(
        stage.token_usage,
        invoked_model_id=stage.model_id,
        pricing_model_id=pricing_model_id,
        region=stage.region,
        snapshot=snapshot,
    )
    return stage.model_copy(update={"cost": estimate})


def attach_rate_based_costs(
    stages: Iterable[AgentStageEvidenceV2],
    snapshot: BedrockPricingSnapshot,
) -> tuple[AgentStageEvidenceV2, ...]:
    """Cost every stage without mutating or reordering the evidence stream."""

    return tuple(attach_rate_based_cost(stage, snapshot) for stage in stages)


def _availability_by_model(
    availability: Iterable[BedrockModelAvailabilityRecord],
) -> dict[str, BedrockModelAvailabilityRecord]:
    grouped: dict[str, list[BedrockModelAvailabilityRecord]] = defaultdict(list)
    for record in availability:
        grouped[record.model_id].append(record)
        if record.inference_profile_id is not None:
            grouped[record.inference_profile_id].append(record)
    output: dict[str, BedrockModelAvailabilityRecord] = {}
    for model_id, records in grouped.items():
        metadata = {
            (item.display_name, item.provider, item.category) for item in records
        }
        if len(metadata) > 1:
            raise CostingError(
                f"availability metadata is ambiguous for invoked model {model_id!r}"
            )
        output[model_id] = sorted(
            records,
            key=lambda item: (item.checked_at_utc, item.availability_id),
            reverse=True,
        )[0]
    return output


def aggregate_model_totals(
    stages: Iterable[AgentStageEvidenceV2],
    *,
    snapshot: BedrockPricingSnapshot,
    availability: Iterable[BedrockModelAvailabilityRecord] = (),
) -> tuple[ModelStudyTotal, ...]:
    """Aggregate actual recorded invocations by invoked model ID.

    An agent run with one or more provider requests and missing token usage or
    unavailable pricing prevents a complete model total. Known partial dollar
    values are deliberately not presented as a complete study cost.
    """

    costed = attach_rate_based_costs(stages, snapshot)
    grouped: dict[str, list[AgentStageEvidenceV2]] = defaultdict(list)
    for stage in costed:
        grouped[stage.model_id].append(stage)
    metadata = _availability_by_model(availability)

    totals: list[ModelStudyTotal] = []
    for model_id in sorted(grouped):
        records = grouped[model_id]
        measured = [record for record in records if record.token_usage is not None]
        usages = [record.token_usage for record in measured]
        missing_cost_ids = tuple(
            sorted(
                record.request_id
                for record in records
                if record.provider_request_count is None
                or (
                    record.provider_request_count > 0
                    and (record.cost is None or record.cost.status != "estimated")
                )
            )
        )
        known_costs = [
            record.cost.total_estimated_cost_usd
            for record in records
            if record.cost is not None
            and record.cost.status == "estimated"
            and record.cost.total_estimated_cost_usd is not None
        ]
        if not missing_cost_ids:
            cost_status = "estimated"
            total_cost: Decimal | None = sum(known_costs, Decimal(0))
        elif known_costs:
            cost_status = "partially_unavailable"
            total_cost = None
        else:
            cost_status = "unavailable"
            total_cost = None

        model_metadata = metadata.get(model_id)
        pricing_ids = {record.pricing_model_id or record.model_id for record in records}
        matching_rates = [
            rate for rate in snapshot.rates if rate.model_id in pricing_ids
        ]
        if model_metadata is not None:
            display_name = model_metadata.display_name
            provider = model_metadata.provider
            category = model_metadata.category
        elif matching_rates:
            distinct = {
                (rate.display_name, rate.provider, rate.category)
                for rate in matching_rates
            }
            if len(distinct) != 1:
                raise CostingError(f"pricing metadata is ambiguous for {model_id!r}")
            display_name, provider, category = distinct.pop()
        else:
            display_name = model_id
            provider = records[0].provider
            category = "paid_api"

        request_count_complete = all(
            record.provider_request_count is not None for record in records
        )
        known_request_count = sum(
            record.provider_request_count or 0 for record in records
        )

        totals.append(
            ModelStudyTotal(
                model_id=model_id,
                display_name=display_name,
                provider=provider,
                category=category,
                stage_run_count=len(records),
                known_provider_request_count=known_request_count,
                provider_request_count=(
                    known_request_count if request_count_complete else None
                ),
                provider_request_count_status=(
                    "complete" if request_count_complete else "partially_unavailable"
                ),
                succeeded_stage_run_count=sum(
                    record.status == "succeeded" for record in records
                ),
                failed_stage_run_count=sum(
                    record.status == "failed" for record in records
                ),
                requests_with_measured_tokens=sum(
                    record.provider_request_count or 0 for record in measured
                ),
                arms_invoked=len(
                    {
                        record.arm_id
                        for record in records
                        if record.arm_id is not None
                        and record.provider_request_count is not None
                        and record.provider_request_count > 0
                    }
                ),
                uncached_input_tokens=sum(
                    usage.uncached_input_tokens for usage in usages if usage is not None
                ),
                cache_write_input_tokens=sum(
                    usage.cache_write_input_tokens
                    for usage in usages
                    if usage is not None
                ),
                cache_read_input_tokens=sum(
                    usage.cache_read_input_tokens
                    for usage in usages
                    if usage is not None
                ),
                measured_total_input_tokens=sum(
                    usage.measured_total_input_tokens
                    for usage in usages
                    if usage is not None
                ),
                output_tokens=sum(
                    usage.output_tokens for usage in usages if usage is not None
                ),
                total_rate_based_estimated_cost_usd=total_cost,
                cost_status=cost_status,
                missing_cost_request_ids=missing_cost_ids,
            )
        )
    return tuple(totals)


def apply_rate_based_costing(study: StudyAggregateInput) -> StudyAggregateInput:
    """Return a study with reproducibly recomputed stage costs and model totals."""

    costed_stages = attach_rate_based_costs(
        study.stage_runs,
        study.model_matrix.pricing,
    )
    totals = aggregate_model_totals(
        costed_stages,
        snapshot=study.model_matrix.pricing,
        availability=study.model_matrix.availability,
    )
    payload = study.model_dump(mode="json")
    payload["stage_runs"] = [item.model_dump(mode="json") for item in costed_stages]
    payload["model_totals"] = [item.model_dump(mode="json") for item in totals]
    return StudyAggregateInput.model_validate(payload, strict=False)


def verify_reported_model_totals(study: StudyAggregateInput) -> None:
    """Reject duplicated aggregate values that disagree with invocation evidence."""

    if not study.model_totals:
        return
    expected = aggregate_model_totals(
        study.stage_runs,
        snapshot=study.model_matrix.pricing,
        availability=study.model_matrix.availability,
    )
    reported_by_model = {item.model_id: item for item in study.model_totals}
    expected_by_model = {item.model_id: item for item in expected}
    if reported_by_model != expected_by_model:
        raise CostingError(
            "reported model_totals do not match exact stage runs and pricing snapshot"
        )
