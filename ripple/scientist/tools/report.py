"""Deterministic report assembly after the required evidence boundary."""

from __future__ import annotations

import math
import re

from ripple.dp2.package_models import Dp2CutoutPackage
from ripple.inference.contracts import (
    M3ToM4BridgeCompletionRecord,
    M3ToM4BridgeRecord,
    M4CompletionRecord,
    M4InferenceResult,
)
from ripple.modeling.service import PreprocessingCompletionRecord
from ripple.preprocessing.mriganka_enn.contracts import (
    MrigankaEnnThreeBandModelInputPackage,
    SourcePackageRef,
)

from ..schemas.common import SHA256_PATTERN, SkyCoordinate, utc_now
from ..schemas.lenscat import LensCatAttempt
from ..schemas.report import (
    MRIGANKA_DP2_SCIENTIFIC_BLOCKERS,
    CandidateEvidence,
    CandidateScientificReport,
    MrigankaDp2BridgeEvidence,
    MrigankaDp2M2BandEvidence,
    MrigankaDp2M3Evidence,
    MrigankaDp2M4Evidence,
    MrigankaDp2TechnicalReport,
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


def assemble_mriganka_dp2_technical_report(
    *,
    report_id: str,
    request_id: str,
    target: SkyCoordinate,
    m2_packages: tuple[
        Dp2CutoutPackage,
        Dp2CutoutPackage,
        Dp2CutoutPackage,
    ],
    m3_package: MrigankaEnnThreeBandModelInputPackage,
    m3_completion: PreprocessingCompletionRecord,
    m3_completion_sha256: str,
    bridge_record: M3ToM4BridgeRecord,
    bridge_completion: M3ToM4BridgeCompletionRecord,
    bridge_completion_sha256: str,
    m4_result: M4InferenceResult,
    m4_completion: M4CompletionRecord,
    m4_completion_sha256: str,
) -> MrigankaDp2TechnicalReport:
    """Bind verified M2 through M4 records into one non-scientific report.

    The caller supplies already reloaded typed records plus the raw SHA-256 of
    each last-written completion file. Cross-stage identities are independently
    checked here and again by :class:`MrigankaDp2TechnicalReport`.
    """

    _require_sha256(m3_completion_sha256, name="M3 completion")
    _require_sha256(bridge_completion_sha256, name="bridge completion")
    _require_sha256(m4_completion_sha256, name="M4 completion")
    if len(m2_packages) != 3:
        raise ValueError("technical report requires exactly three M2 packages")
    package_by_band = {package.dataset.band_name: package for package in m2_packages}
    if len(package_by_band) != 3 or set(package_by_band) != {"g", "r", "i"}:
        raise ValueError("technical report requires one M2 package for each g/r/i band")

    source_by_band = {source.band: source for source in m3_package.sources}
    if set(source_by_band) != {"g", "r", "i"}:
        raise ValueError("M3 provenance does not contain one source per g/r/i band")
    m2_evidence: list[MrigankaDp2M2BandEvidence] = []
    for band in ("g", "r", "i"):
        package = package_by_band[band]
        source = source_by_band[band]
        _validate_m2_source_binding(
            package=package,
            source=source,
            target=target,
        )
        m2_evidence.append(
            MrigankaDp2M2BandEvidence(
                band=band,
                package_manifest_sha256=source.manifest_sha256,
                fits_sha256=package.artifact.sha256,
                dataset_id=package.dataset.dataset_id,
                obs_id=package.dataset.obs_id,
                target=target,
                image_decoded_sha256=package.image.digest.sha256,
                mask_decoded_sha256=package.mask.digest.sha256,
                variance_decoded_sha256=package.variance.digest.sha256,
                celestial_wcs_sha256=package.celestial_wcs.plane_digests[0].sha256,
                psf_state=package.psf.state,
                authenticated_live_rubin_rsp=True,
                byte_preserved_before_m3=True,
            )
        )

    m3_evidence = MrigankaDp2M3Evidence(
        manifest_sha256=m3_completion.adapter_package_manifest.file_sha256,
        completion_sha256=m3_completion_sha256,
        package=m3_package,
        completion=m3_completion,
    )
    bridge_evidence = MrigankaDp2BridgeEvidence(
        manifest_sha256=bridge_completion.bridge_manifest.sha256,
        completion_sha256=bridge_completion_sha256,
        record=bridge_record,
        completion=bridge_completion,
    )
    m4_evidence = MrigankaDp2M4Evidence(
        result_sha256=m4_completion.inference_result.sha256,
        completion_sha256=m4_completion_sha256,
        result=m4_result,
        completion=m4_completion,
    )
    logits_text = ", ".join(f"{value:.10g}" for value in m4_result.logits)
    scores_text = ", ".join(f"{value:.10g}" for value in m4_result.scores)
    return MrigankaDp2TechnicalReport(
        report_id=report_id,
        request_id=request_id,
        target=target,
        m2_bands=(m2_evidence[0], m2_evidence[1], m2_evidence[2]),
        m3=m3_evidence,
        bridge=bridge_evidence,
        m4=m4_evidence,
        raw_logits=m4_result.logits,
        uncalibrated_softmax_components=m4_result.scores,
        unresolved_scientific_blockers=MRIGANKA_DP2_SCIENTIFIC_BLOCKERS,
        summary=(
            "Authenticated Rubin DP2 g/r/i retrieval, audited three-band M3 "
            "preprocessing, byte-preserving M3-to-M4 bridging, and checkpoint-bound "
            f"M4 execution completed for request {request_id}. Raw logits are "
            f"({logits_text}); softmax components are ({scores_text}). The softmax "
            "components are explicitly uncalibrated and are not lens probabilities. "
            "All six scientific blockers remain unresolved; no threshold, candidate "
            "decision, scientific classification, candidate report, or LensCat "
            "association was produced."
        ),
        assembled_at_utc=utc_now(),
    )


def _require_sha256(value: str, *, name: str) -> None:
    if re.fullmatch(SHA256_PATTERN, value) is None:
        raise ValueError(f"{name} SHA-256 is invalid")


def _validate_m2_source_binding(
    *,
    package: Dp2CutoutPackage,
    source: SourcePackageRef,
    target: SkyCoordinate,
) -> None:
    if (
        package.dataset.band_name != source.band
        or package.artifact.sha256 != source.fits_sha256
        or package.dataset.dataset_id != source.dataset_id
        or package.dataset.obs_id != source.obs_id
        or package.dataset.tract != source.tract
        or package.dataset.patch != source.patch
        or package.psf.state != source.psf_state
        or not package.retrieval.authenticated
        or not package.preservation.payload_preserved_byte_for_byte
        or not math.isclose(
            package.request.ra_deg,
            target.ra_deg,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or not math.isclose(
            package.request.dec_deg,
            target.dec_deg,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or not math.isclose(
            source.ra_deg,
            target.ra_deg,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        or not math.isclose(
            source.dec_deg,
            target.dec_deg,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("M2 package does not match its M3 source provenance")


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
