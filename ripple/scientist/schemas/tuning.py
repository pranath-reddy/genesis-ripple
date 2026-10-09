"""Typed contracts for bounded, validation-only architecture tuning."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from .architecture import TrainingConfiguration
from .common import FrozenModel, IDENTIFIER_PATTERN


class TuningAction(str, Enum):
    ENABLE_D4 = "enable_d4"
    INCREASE_DROPOUT = "increase_dropout"
    INCREASE_WEIGHT_DECAY = "increase_weight_decay"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    STOP = "stop"


class TuningMeasurement(FrozenModel):
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    iteration: int = Field(ge=0, le=100)
    train_accuracy: float = Field(ge=0.0, le=1.0)
    validation_loss: float = Field(ge=0.0)
    validation_accuracy: float = Field(ge=0.0, le=1.0)
    validation_balanced_accuracy: float = Field(ge=0.0, le=1.0)
    validation_roc_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    generalization_gap: float = Field(ge=-1.0, le=1.0)
    configuration: TrainingConfiguration


class TuningDecision(FrozenModel):
    schema_version: Literal["ripple.tuning-decision.v1"] = (
        "ripple.tuning-decision.v1"
    )
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    iteration: int = Field(ge=0, le=100)
    action: TuningAction
    eligible_actions: tuple[TuningAction, ...] = Field(min_length=1, max_length=5)
    rationale: str = Field(min_length=1, max_length=1200)
    validation_only: Literal[True] = True
    test_split_inspected: Literal[False] = False
    scientific_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def _action_is_eligible(self) -> "TuningDecision":
        if len(self.eligible_actions) != len(set(self.eligible_actions)):
            raise ValueError("eligible tuning actions must be unique")
        if self.action not in self.eligible_actions:
            raise ValueError("tuning action was not in the code-owned allowlist")
        return self


__all__ = ["TuningAction", "TuningDecision", "TuningMeasurement"]
