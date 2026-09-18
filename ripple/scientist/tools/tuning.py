"""Deterministic application of allowlisted tuning actions."""

from __future__ import annotations

from ..schemas.architecture import TrainingConfiguration
from ..schemas.tuning import TuningAction, TuningMeasurement


def eligible_tuning_actions(
    measurement: TuningMeasurement,
    *,
    minimum_iterations: int,
    maximum_iterations: int,
    minimum_validation_balanced_accuracy: float,
    maximum_generalization_gap: float,
) -> tuple[TuningAction, ...]:
    """Return the only actions an agent may select at this trajectory point."""

    if measurement.iteration + 1 >= maximum_iterations:
        return (TuningAction.STOP,)

    configuration = measurement.configuration
    actions: list[TuningAction] = []
    if not configuration.augment_d4:
        actions.append(TuningAction.ENABLE_D4)
    if configuration.dropout < 0.5:
        actions.append(TuningAction.INCREASE_DROPOUT)
    if configuration.weight_decay < 0.1:
        actions.append(TuningAction.INCREASE_WEIGHT_DECAY)
    if configuration.learning_rate > 1e-6:
        actions.append(TuningAction.REDUCE_LEARNING_RATE)

    stop_is_eligible = (
        measurement.iteration + 1 >= minimum_iterations
        and measurement.validation_balanced_accuracy
        >= minimum_validation_balanced_accuracy
        and measurement.generalization_gap <= maximum_generalization_gap
    )
    if stop_is_eligible or not actions:
        actions.append(TuningAction.STOP)
    return tuple(actions)


def apply_tuning_action(
    configuration: TrainingConfiguration,
    action: TuningAction,
) -> TrainingConfiguration:
    """Apply one bounded mutation and return a newly validated configuration."""

    if action == TuningAction.STOP:
        return configuration
    payload = configuration.model_dump(mode="python")
    if action == TuningAction.ENABLE_D4:
        if configuration.augment_d4:
            raise ValueError("D4 augmentation is already enabled")
        payload["augment_d4"] = True
    elif action == TuningAction.INCREASE_DROPOUT:
        if configuration.dropout >= 0.5:
            raise ValueError("dropout is already at the tuning ceiling")
        payload["dropout"] = min(0.5, configuration.dropout + 0.1)
    elif action == TuningAction.INCREASE_WEIGHT_DECAY:
        if configuration.weight_decay >= 0.1:
            raise ValueError("weight decay is already at the tuning ceiling")
        payload["weight_decay"] = min(
            0.1,
            max(1e-4, configuration.weight_decay * 10.0),
        )
    elif action == TuningAction.REDUCE_LEARNING_RATE:
        if configuration.learning_rate <= 1e-6:
            raise ValueError("learning rate is already at the tuning floor")
        payload["learning_rate"] = max(1e-6, configuration.learning_rate * 0.5)
    else:  # pragma: no cover - Enum exhaustiveness guard.
        raise ValueError("unsupported tuning action")
    return TrainingConfiguration.model_validate(payload, strict=True)


__all__ = ["apply_tuning_action", "eligible_tuning_actions"]
