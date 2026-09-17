"""Deterministic report assembly after the required evidence boundary."""

from __future__ import annotations

from ..schemas.common import utc_now
from ..schemas.lenscat import LensCatAttempt
from ..schemas.report import (
    CandidateEvidence,
    CandidateScientificReport,
    SyntheticTrainingSmokeEvidence,
    SyntheticTrainingTechnicalReport,
)


_CATALOG_LIMITATION = (
    "LensCat association is external catalog evidence only; it is not a classifier "
    "label and does not independently validate the model prediction."
)
_NO_MATCH_LIMITATION = (
    "No catalog match means only that no configured record was found inside the "
    "specified cone; it does not imply that the target is a non-lens."
)


def _candidate_summary(candidate: CandidateEvidence, attempt: LensCatAttempt) -> str:
    score = f"{candidate.lens_score:.6f}"
    threshold = f"{candidate.decision_threshold:.6f}"
    if attempt.status == "matched_confirmed":
        catalog_text = (
            "The final catalog step found one association whose exact mapped status "
            "is configured as confirmed."
        )
    elif attempt.status == "matched_candidate":
        catalog_text = (
            "The final catalog step found one association whose exact mapped status "
            "is configured as candidate."
        )
    elif attempt.status == "ambiguous":
        catalog_text = "The final catalog step found an ambiguous nearby association."
    elif attempt.status == "no_match":
        catalog_text = (
            "The final catalog step found no association in the configured cone; "
            "this is not a non-lens determination."
        )
    else:
        catalog_text = (
            "The final catalog step was attempted but the configured local catalog "
            "was unavailable."
        )
    return (
        f"Candidate {candidate.candidate_id} has classifier lens score {score} at "
        f"decision threshold {threshold}. {catalog_text}"
    )


def assemble_candidate_scientific_report(
    *,
    report_id: str,
    candidate: CandidateEvidence,
    lenscat_attempt: LensCatAttempt,
    additional_limitations: tuple[str, ...] = (),
) -> CandidateScientificReport:
    """Assemble a candidate report only after receiving its LensCat attempt.

    This is the candidate-report entry point: the required ``lenscat_attempt``
    argument and schema invariants prevent report assembly from silently skipping
    or substituting the final catalog step.
    """

    if candidate.candidate_id != lenscat_attempt.query.candidate_id:
        raise ValueError("LensCat attempt belongs to a different candidate")
    if candidate.coordinate != lenscat_attempt.query.coordinate:
        raise ValueError("LensCat attempt used a different sky coordinate")
    limitations = tuple(
        dict.fromkeys(
            (
                _CATALOG_LIMITATION,
                _NO_MATCH_LIMITATION,
                *additional_limitations,
            )
        )
    )
    return CandidateScientificReport(
        report_id=report_id,
        candidate=candidate,
        lenscat_attempt=lenscat_attempt,
        catalog_disposition=lenscat_attempt.status,
        final_pre_report_attempt_id=lenscat_attempt.attempt_id,
        summary=_candidate_summary(candidate, lenscat_attempt),
        limitations=limitations,
        assembled_at_utc=utc_now(),
    )


def assemble_synthetic_training_technical_report(
    *,
    report_id: str,
    evidence: SyntheticTrainingSmokeEvidence,
) -> SyntheticTrainingTechnicalReport:
    """Report synthetic wiring validation without pretending LensCat applies."""

    return SyntheticTrainingTechnicalReport(
        report_id=report_id,
        evidence=evidence,
        summary=(
            f"Synthetic smoke run {evidence.selected_training_run_id} processed "
            f"{evidence.total_samples} labeled simulation samples. This validates "
            "pipeline wiring only and does not support a scientific performance claim. "
            "LensCat is not applicable because these samples have no sky coordinates."
        ),
        assembled_at_utc=utc_now(),
    )
