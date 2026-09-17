"""Deterministic, read-only checks for a proposed model contract.

The validator in this module is deliberately narrower than scientific model
qualification.  It checks whether a typed onboarding draft is internally
consistent, locally pinned, and bound to a code-owned adapter allowlist.  It
does not register, freeze, promote, preprocess with, or execute the proposed
manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .onboarding import (
    DeterministicQualificationReport,
    ModelContractDraft,
    QualificationFinding,
)
from .registry import ModelAdapterRegistry, RegistryError
from .source_reader import SourceExcerpt, SourceExcerptError, read_source_excerpt
from .source_inventory import inventory_source_tree_sha256


_VALIDATOR_ID = "ripple-deterministic-draft-validator"
_VALIDATOR_VERSION = "1.0.0"


def qualify_model_contract_draft(
    draft: ModelContractDraft,
    *,
    registry: ModelAdapterRegistry | None = None,
    repository_root: Path | None = None,
    source_excerpts: tuple[SourceExcerpt, ...] = (),
    completed_at_utc: datetime | None = None,
) -> DeterministicQualificationReport:
    """Return a non-executing consistency report for ``draft``.

    ``registry`` is optional so an absent runtime allowlist can be represented
    as a blocking finding instead of causing an exception.  Supplying a
    registry never mutates it: only its immutable adapter-identity snapshots
    are read.

    The optional timestamp makes byte-for-byte reproducible reports possible
    in callers and tests.  When omitted, the current UTC time is recorded.
    """

    if not isinstance(draft, ModelContractDraft):
        raise TypeError("draft must be a validated ModelContractDraft")
    if registry is not None and not isinstance(registry, ModelAdapterRegistry):
        raise TypeError("registry must be a ModelAdapterRegistry or None")
    if not isinstance(source_excerpts, tuple) or any(
        not isinstance(excerpt, SourceExcerpt) for excerpt in source_excerpts
    ):
        raise TypeError("source_excerpts must be a tuple of SourceExcerpt records")
    try:
        revalidated_draft = ModelContractDraft.model_validate_json(
            draft.model_dump_json(exclude_none=False)
        )
        revalidated_excerpts = tuple(
            SourceExcerpt.model_validate_json(
                excerpt.model_dump_json(exclude_none=False)
            )
            for excerpt in source_excerpts
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise TypeError(
            f"draft inputs failed independent strict revalidation ({type(exc).__name__})"
        ) from None
    if revalidated_draft != draft or revalidated_excerpts != source_excerpts:
        raise TypeError("draft inputs changed during independent strict revalidation")
    draft = revalidated_draft
    source_excerpts = revalidated_excerpts

    findings = [
        _passed(
            finding_id="schema-valid",
            category="schema",
            code="typed-draft-schema-valid",
            message=(
                "The input is a validated ModelContractDraft; its strict schema "
                "and draft-only invariants hold."
            ),
            paths=("schema_version", "status", "proposal_only"),
        ),
        _passed(
            finding_id="links-valid",
            category="evidence_links",
            code="typed-cross-links-valid",
            message=(
                "Model, request, inventory, evidence-claim, artifact, finding, "
                "conflict, and unresolved-path links satisfy the draft model validators."
            ),
            paths=(
                "onboarding_request",
                "source_inventory",
                "proposed_manifest",
                "requirement_findings",
                "conflicts",
            ),
        ),
        _artifact_integrity_finding(draft, repository_root),
        _source_identity_finding(draft),
        _checkpoint_identity_finding(draft),
        _evidence_value_binding_finding(
            draft,
            repository_root=repository_root,
            source_excerpts=source_excerpts,
        ),
        _required_field_coverage_finding(draft),
        _requirement_completeness_finding(draft),
        _conflict_finding(draft),
        _adapter_binding_finding(draft, registry),
        _execution_gate_finding(draft),
    ]

    has_blocker = any(
        finding.outcome == "failed" and finding.blocks_approval_freeze_request
        for finding in findings
    )
    draft_sha256 = draft.canonical_sha256()
    return DeterministicQualificationReport(
        report_id=f"qualification-{draft_sha256[:24]}",
        draft_id=draft.draft_id,
        draft_sha256=draft_sha256,
        completed_at_utc=completed_at_utc or datetime.now(timezone.utc),
        validator_id=_VALIDATOR_ID,
        validator_version=_VALIDATOR_VERSION,
        findings=tuple(findings),
        outcome="failed" if has_blocker else "passed",
        approval_freeze_request_allowed=not has_blocker,
    )


def _artifact_integrity_finding(
    draft: ModelContractDraft,
    repository_root: Path | None,
) -> QualificationFinding:
    paths = ("source_inventory.artifacts", "source_inventory.evidence_sources")
    if repository_root is None:
        return _failed(
            finding_id="artifact-integrity",
            category="source_integrity",
            code="repository-root-not-supplied",
            message=(
                "No explicit local repository root was supplied, so inventoried source and "
                "checkpoint bytes cannot be reverified."
            ),
            paths=paths,
        )
    if not isinstance(repository_root, Path) or not repository_root.is_absolute():
        return _failed(
            finding_id="artifact-integrity",
            category="source_integrity",
            code="repository-root-invalid",
            message="The repository root must be an explicit absolute pathlib.Path.",
            paths=paths,
        )

    root = Path(os.path.abspath(os.fspath(repository_root)))
    try:
        root_state = os.lstat(root)
    except OSError:
        return _failed(
            finding_id="artifact-integrity",
            category="source_integrity",
            code="repository-root-unavailable",
            message="The supplied repository root is unavailable.",
            paths=paths,
        )
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        return _failed(
            finding_id="artifact-integrity",
            category="source_integrity",
            code="repository-root-unsafe",
            message="The supplied repository root must be a non-symlink directory.",
            paths=paths,
        )

    failures: list[str] = []
    for artifact in draft.source_inventory.artifacts:
        relative = PurePosixPath(artifact.repository_relative_path)
        target = root.joinpath(*relative.parts)
        current = root
        unsafe = False
        for part in relative.parts:
            current = current / part
            try:
                state = os.lstat(current)
            except OSError:
                failures.append(f"{artifact.artifact_id} is missing")
                unsafe = True
                break
            if stat.S_ISLNK(state.st_mode):
                failures.append(f"{artifact.artifact_id} contains a symlink")
                unsafe = True
                break
        if unsafe:
            continue
        try:
            state = os.lstat(target)
            if not stat.S_ISREG(state.st_mode):
                failures.append(f"{artifact.artifact_id} is not a regular file")
                continue
            if state.st_size != artifact.byte_count:
                failures.append(f"{artifact.artifact_id} has a different byte count")
                continue
            digest = hashlib.sha256()
            byte_count = 0
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    byte_count += len(chunk)
                    digest.update(chunk)
            final_state = os.lstat(target)
        except OSError:
            failures.append(f"{artifact.artifact_id} could not be reverified")
            continue
        if (
            byte_count != artifact.byte_count
            or digest.hexdigest() != artifact.sha256
            or (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
            != (
                final_state.st_dev,
                final_state.st_ino,
                final_state.st_size,
                final_state.st_mtime_ns,
            )
        ):
            failures.append(f"{artifact.artifact_id} no longer matches its inventory")

    if failures:
        return _failed(
            finding_id="artifact-integrity",
            category="source_integrity",
            code="inventoried-artifacts-unverified",
            message=_summarize_problems("Local artifact verification failed", failures),
            paths=paths,
        )
    return _passed(
        finding_id="artifact-integrity",
        category="source_integrity",
        code="inventoried-artifacts-reverified",
        message="Every inventoried artifact still matches its local byte count and SHA-256.",
        paths=paths,
    )


def _source_identity_finding(draft: ModelContractDraft) -> QualificationFinding:
    inventory = draft.source_inventory
    implementation = draft.proposed_manifest.implementation
    source_artifacts = tuple(
        artifact for artifact in inventory.artifacts if artifact.kind == "python_source"
    )
    problems: list[str] = []

    if not inventory.inventory_complete_for_declared_scope:
        problems.append("the declared source inventory is incomplete")
    if not source_artifacts:
        problems.append("the inventory contains no hashed local Python source artifact")
    if implementation.source_sha256 is None:
        problems.append("the implementation has no digest-pinned source identity")

    source_locator = PurePosixPath(implementation.source_locator)
    matching_locator_artifacts = tuple(
        artifact
        for artifact in source_artifacts
        if PurePosixPath(artifact.repository_relative_path) == source_locator
        or source_locator in PurePosixPath(artifact.repository_relative_path).parents
    )
    if not matching_locator_artifacts:
        problems.append(
            "the implementation source locator is not represented by local source"
        )

    tree_sha256 = None
    if ":" not in implementation.source_locator.split("/", 1)[0]:
        tree_sha256 = inventory_source_tree_sha256(
            inventory,
            source_locator=implementation.source_locator,
        )
    valid_source_digests = {artifact.sha256 for artifact in matching_locator_artifacts}
    if tree_sha256 is not None:
        valid_source_digests.add(tree_sha256)
    if (
        implementation.source_sha256 is not None
        and implementation.source_sha256 not in valid_source_digests
    ):
        problems.append(
            "the implementation source digest matches neither a local file nor its source-tree digest"
        )

    remote_source_locators = tuple(
        locator
        for locator in draft.onboarding_request.source_locators
        if ":" in locator.split("/", 1)[0]
    )
    if remote_source_locators:
        problems.append(
            "a requested non-local source locator has no explicit local resolution"
        )
    requested_local_sources = tuple(
        PurePosixPath(locator)
        for locator in draft.onboarding_request.source_locators
        if ":" not in locator.split("/", 1)[0]
    )
    if requested_local_sources and not any(
        source_locator == requested or requested in source_locator.parents
        for requested in requested_local_sources
    ):
        problems.append(
            "the implementation source locator was not selected by the request"
        )
    for requested in requested_local_sources:
        if not any(
            PurePosixPath(artifact.repository_relative_path) == requested
            or requested in PurePosixPath(artifact.repository_relative_path).parents
            for artifact in source_artifacts
        ):
            problems.append(
                "a requested local source locator is absent from the inventory"
            )
            break

    if implementation.source_revision is not None:
        sources_by_id = {
            source.source_id: source for source in inventory.evidence_sources
        }
        artifact_revisions = {
            sources_by_id[artifact.evidence_source_id].revision
            for artifact in source_artifacts
            if artifact.evidence_source_id in sources_by_id
            and sources_by_id[artifact.evidence_source_id].revision is not None
        }
        revision_is_local = (
            inventory.repository_revision == implementation.source_revision
            or implementation.source_revision in artifact_revisions
        )
        if not revision_is_local:
            problems.append(
                "the implementation source revision is not represented by the local inventory"
            )

    if problems:
        return _failed(
            finding_id="source-identity",
            category="source_integrity",
            code="local-source-identity-unverified",
            message=_summarize_problems(
                "Local pinned source identity is not established",
                problems,
            ),
            paths=(
                "source_inventory.artifacts",
                "source_inventory.inventory_complete_for_declared_scope",
                "proposed_manifest.implementation.source_revision",
                "proposed_manifest.implementation.source_sha256",
            ),
        )

    return _passed(
        finding_id="source-identity",
        category="source_integrity",
        code="local-source-identity-verified",
        message=(
            "The complete local inventory contains hashed Python source, and the "
            "implementation is bound to a deterministic local file or source-tree digest."
        ),
        paths=(
            "source_inventory.artifacts",
            "proposed_manifest.implementation.source_revision",
            "proposed_manifest.implementation.source_sha256",
        ),
    )


def _checkpoint_identity_finding(draft: ModelContractDraft) -> QualificationFinding:
    implementation = draft.proposed_manifest.implementation
    checkpoint_artifacts = tuple(
        artifact
        for artifact in draft.source_inventory.artifacts
        if artifact.kind == "checkpoint"
    )
    checkpoint_digest = implementation.checkpoint_sha256
    requested_checkpoint_locators = set(draft.onboarding_request.checkpoint_locators)

    if implementation.checkpoint_locator is None or checkpoint_digest is None:
        return _failed(
            finding_id="checkpoint-identity",
            category="checkpoint_identity",
            code="checkpoint-identity-missing",
            message=(
                "The proposed implementation does not identify a checkpoint with both "
                "a locator and SHA-256 digest."
            ),
            paths=(
                "proposed_manifest.implementation.checkpoint_locator",
                "proposed_manifest.implementation.checkpoint_sha256",
            ),
        )

    if any(
        ":" in locator.split("/", 1)[0] for locator in requested_checkpoint_locators
    ):
        return _failed(
            finding_id="checkpoint-identity",
            category="checkpoint_identity",
            code="remote-checkpoint-resolution-missing",
            message=(
                "A requested non-local checkpoint locator has no explicit local artifact "
                "resolution."
            ),
            paths=(
                "onboarding_request.checkpoint_locators",
                "source_inventory.artifacts",
            ),
        )
    if implementation.checkpoint_locator not in requested_checkpoint_locators:
        return _failed(
            finding_id="checkpoint-identity",
            category="checkpoint_identity",
            code="checkpoint-not-requested",
            message=(
                "The proposed implementation checkpoint is not one of the explicit "
                "checkpoint locators in the onboarding request."
            ),
            paths=(
                "onboarding_request.checkpoint_locators",
                "proposed_manifest.implementation.checkpoint_locator",
            ),
        )

    if not any(
        artifact.sha256 == checkpoint_digest
        and artifact.repository_relative_path == implementation.checkpoint_locator
        for artifact in checkpoint_artifacts
    ):
        return _failed(
            finding_id="checkpoint-identity",
            category="checkpoint_identity",
            code="local-checkpoint-identity-unverified",
            message=(
                "The proposed checkpoint locator and digest do not match the same hashed "
                "local checkpoint artifact in the source inventory."
            ),
            paths=(
                "source_inventory.artifacts",
                "proposed_manifest.implementation.checkpoint_locator",
                "proposed_manifest.implementation.checkpoint_sha256",
            ),
        )

    return _passed(
        finding_id="checkpoint-identity",
        category="checkpoint_identity",
        code="local-checkpoint-identity-verified",
        message=(
            "The checkpoint locator and digest are present, and the digest matches a "
            "hashed local checkpoint artifact."
        ),
        paths=(
            "source_inventory.artifacts",
            "proposed_manifest.implementation.checkpoint_locator",
            "proposed_manifest.implementation.checkpoint_sha256",
        ),
    )


def _evidence_value_binding_finding(
    draft: ModelContractDraft,
    *,
    repository_root: Path | None,
    source_excerpts: tuple[SourceExcerpt, ...],
) -> QualificationFinding:
    manifest_payload = draft.proposed_manifest.model_dump(
        mode="json", exclude_none=False
    )
    manifest_claims = {
        claim.claim_id: claim for claim in draft.proposed_manifest.evidence_claims
    }
    problems: list[str] = []
    referenced_claim_ids: set[str] = set()
    excerpt_by_identity: dict[tuple[str, str], SourceExcerpt] = {}
    for excerpt in source_excerpts:
        excerpt_key = (excerpt.artifact_id, excerpt.excerpt_sha256)
        if excerpt_key in excerpt_by_identity:
            problems.append("a duplicate source excerpt identity was supplied")
            continue
        if excerpt.inventory_id != draft.source_inventory.inventory_id:
            problems.append("a source excerpt belongs to a different inventory")
            continue
        if repository_root is None:
            problems.append(
                "source excerpts cannot be reverified without a repository root"
            )
            continue
        try:
            reloaded = read_source_excerpt(
                draft.source_inventory,
                repository_root=repository_root,
                artifact_id=excerpt.artifact_id,
                start_line=excerpt.start_line,
                end_line=excerpt.end_line,
            )
        except SourceExcerptError:
            problems.append("a source excerpt failed independent local reverification")
            continue
        if reloaded != excerpt:
            problems.append(
                "a source excerpt changed during independent reverification"
            )
            continue
        excerpt_by_identity[excerpt_key] = excerpt

    for finding in draft.requirement_findings:
        referenced_claim_ids.update(finding.evidence_claim_ids)
        try:
            actual = _field_path_value(manifest_payload, finding.field_path)
        except KeyError:
            problems.append(f"{finding.field_path} is not a manifest field")
            continue
        if not _json_identical(actual, finding.proposed_value):
            problems.append(
                f"{finding.field_path} proposed value differs from the manifest"
            )
        if finding.status == "directly_supported":
            required_source_ids = {
                source_id
                for claim_id in finding.evidence_claim_ids
                for source_id in manifest_claims[claim_id].source_ids
            }
            verified_source_ids: set[str] = set()
            for location in finding.evidence_locations:
                if location.location_kind != "source_lines":
                    problems.append(
                        f"{finding.field_path} uses a direct evidence type not yet deterministically verified"
                    )
                    continue
                excerpt = excerpt_by_identity.get(
                    (location.artifact_id or "", location.excerpt_sha256 or "")
                )
                if (
                    excerpt is None
                    or excerpt.artifact_id != location.artifact_id
                    or excerpt.evidence_source_id != location.source_id
                ):
                    problems.append(
                        f"{finding.field_path} cites a source excerpt that was not reverified"
                    )
                    continue
                expected_locator = (
                    f"{excerpt.artifact_id}:{excerpt.start_line}-{excerpt.end_line}"
                )
                if location.locator != expected_locator:
                    problems.append(
                        f"{finding.field_path} cites a non-canonical source-line locator"
                    )
                    continue
                verified_source_ids.add(location.source_id)
            if not required_source_ids <= verified_source_ids:
                problems.append(
                    f"{finding.field_path} lacks a verified excerpt for every direct claim source"
                )

    claim_ids = {claim.claim_id for claim in draft.proposed_manifest.evidence_claims}
    unreferenced = sorted(claim_ids - referenced_claim_ids)
    if unreferenced:
        problems.append(
            "one or more manifest evidence claims are not used by a finding"
        )

    if problems:
        return _failed(
            finding_id="evidence-value-binding",
            category="evidence_links",
            code="evidence-values-not-bound",
            message=_summarize_problems(
                "Evidence-to-manifest binding failed",
                problems,
            ),
            paths=("requirement_findings", "proposed_manifest.evidence_claims"),
        )
    return _passed(
        finding_id="evidence-value-binding",
        category="evidence_links",
        code="evidence-values-bound",
        message=(
            "Every finding identifies a real manifest field with a type-sensitive exact "
            "value match, and every manifest claim is used."
        ),
        paths=("requirement_findings", "proposed_manifest.evidence_claims"),
    )


def _required_field_coverage_finding(
    draft: ModelContractDraft,
) -> QualificationFinding:
    required = {
        "implementation.source_locator",
        "implementation.source_revision",
        "implementation.source_sha256",
        "implementation.architecture_entrypoint",
        "implementation.checkpoint_locator",
        "implementation.checkpoint_sha256",
        "observation",
        "preprocessing",
        "tensor",
        "output",
    }
    observed = {finding.field_path for finding in draft.requirement_findings}
    missing = tuple(sorted(required - observed))
    if missing:
        return _failed(
            finding_id="required-field-coverage",
            category="requirement_completeness",
            code="required-contract-fields-uncovered",
            message=(
                "The evidence proposal does not cover every execution-critical manifest "
                "field: " + ", ".join(missing) + "."
            ),
            paths=missing,
        )
    return _passed(
        finding_id="required-field-coverage",
        category="requirement_completeness",
        code="required-contract-fields-covered",
        message="Every execution-critical manifest field has an exact evidence finding.",
        paths=tuple(sorted(required)),
    )


def _requirement_completeness_finding(
    draft: ModelContractDraft,
) -> QualificationFinding:
    blocking_findings = tuple(
        finding
        for finding in draft.requirement_findings
        if finding.status == "unresolved" or finding.blocks_model_execution
    )
    unresolved_claims = tuple(
        claim
        for claim in draft.proposed_manifest.evidence_claims
        if claim.support == "unresolved"
    )
    qualification_unresolved = (
        draft.proposed_manifest.qualification.unresolved_requirements
    )
    if (
        draft.unresolved_field_paths
        or blocking_findings
        or unresolved_claims
        or qualification_unresolved
    ):
        related_paths = set(draft.unresolved_field_paths)
        related_paths.update(finding.field_path for finding in blocking_findings)
        related_paths.update(claim.field_path for claim in unresolved_claims)
        if qualification_unresolved:
            related_paths.add("proposed_manifest.qualification.unresolved_requirements")
        return _failed(
            finding_id="requirements-complete",
            category="requirement_completeness",
            code="unresolved-requirements-remain",
            message=(
                "Unresolved evidence or execution-blocking requirement findings remain "
                "in the draft."
            ),
            paths=tuple(sorted(related_paths)),
        )

    return _passed(
        finding_id="requirements-complete",
        category="requirement_completeness",
        code="no-unresolved-requirements",
        message="The draft contains no unresolved or execution-blocking requirements.",
        paths=("requirement_findings", "unresolved_field_paths"),
    )


def _conflict_finding(draft: ModelContractDraft) -> QualificationFinding:
    conflicting_claims = tuple(
        claim
        for claim in draft.proposed_manifest.evidence_claims
        if claim.support == "conflicting"
    )
    if draft.conflicts or conflicting_claims:
        related_paths = {conflict.field_path for conflict in draft.conflicts}
        related_paths.update(claim.field_path for claim in conflicting_claims)
        return _failed(
            finding_id="conflicts-clear",
            category="conflict_check",
            code="evidence-conflicts-remain",
            message=(
                "The draft still contains requirement conflicts or conflicting "
                "evidence claims."
            ),
            paths=tuple(sorted(related_paths)),
        )

    return _passed(
        finding_id="conflicts-clear",
        category="conflict_check",
        code="no-evidence-conflicts",
        message="The draft contains no requirement conflicts or conflicting evidence claims.",
        paths=("conflicts", "proposed_manifest.evidence_claims"),
    )


def _adapter_binding_finding(
    draft: ModelContractDraft,
    registry: ModelAdapterRegistry | None,
) -> QualificationFinding:
    preprocessing = draft.proposed_manifest.preprocessing
    paths = (
        "proposed_manifest.model_id",
        "proposed_manifest.preprocessing.adapter_id",
        "proposed_manifest.preprocessing.adapter_version",
        "proposed_manifest.preprocessing.steps",
    )
    if registry is None:
        return _failed(
            finding_id="adapter-binding",
            category="adapter_allowlist",
            code="adapter-registry-not-supplied",
            message=(
                "No ModelAdapterRegistry was supplied, so the proposed adapter binding "
                "cannot be checked against the code-owned allowlist."
            ),
            paths=paths,
        )

    identities = {
        identity.adapter_id: identity for identity in registry.adapter_identities()
    }
    identity = identities.get(preprocessing.adapter_id)
    problems: list[str] = []
    if identity is None:
        problems.append("the adapter ID is not allowlisted")
    else:
        if preprocessing.adapter_version != identity.adapter_version:
            problems.append(
                "the adapter version does not match the allowlisted version"
            )
        if (
            identity.supported_model_ids
            and draft.proposed_manifest.model_id not in identity.supported_model_ids
        ):
            problems.append("the model ID is not supported by the allowlisted adapter")
        requested_implementations = {
            step.implementation_id for step in preprocessing.steps
        }
        if not requested_implementations <= set(identity.implementation_ids):
            problems.append("one or more transform implementations are not allowlisted")
        try:
            contract_issues = registry.validate_manifest_contract(
                draft.proposed_manifest
            )
        except RegistryError as exc:
            problems.append(f"code-owned adapter validation failed ({exc.code})")
        else:
            error_codes = tuple(
                issue.code for issue in contract_issues if issue.severity == "error"
            )
            if error_codes:
                problems.append(
                    "the exact adapter contract rejected the manifest: "
                    + ", ".join(error_codes)
                )

    if problems:
        return _failed(
            finding_id="adapter-binding",
            category="adapter_allowlist",
            code="adapter-binding-invalid",
            message="The proposed adapter binding is invalid: "
            + "; ".join(problems)
            + ".",
            paths=paths,
        )

    return _passed(
        finding_id="adapter-binding",
        category="adapter_allowlist",
        code="adapter-binding-verified",
        message=(
            "The adapter ID, version, model support, transform implementations, and exact "
            "scientific behavior match the current code-owned adapter contract."
        ),
        paths=paths,
    )


def _execution_gate_finding(draft: ModelContractDraft) -> QualificationFinding:
    request = draft.onboarding_request
    inventory = draft.source_inventory
    qualification = draft.proposed_manifest.qualification
    all_closed = (
        not any(
            (
                request.preprocessing_execution_authorized,
                request.model_execution_authorized,
                request.scientific_use_authorized,
                inventory.execution_authorized,
                draft.preprocessing_execution_authorized,
                draft.model_execution_authorized,
                draft.scientific_use_authorized,
                qualification.preprocessing_execution_allowed,
                qualification.model_execution_allowed,
                qualification.scientific_use_allowed,
                *(
                    finding.execution_authorized
                    for finding in draft.requirement_findings
                ),
                *(conflict.execution_authorized for conflict in draft.conflicts),
            )
        )
        and qualification.state == "draft"
    )
    paths = (
        "onboarding_request.preprocessing_execution_authorized",
        "onboarding_request.model_execution_authorized",
        "onboarding_request.scientific_use_authorized",
        "source_inventory.execution_authorized",
        "preprocessing_execution_authorized",
        "model_execution_authorized",
        "scientific_use_authorized",
        "proposed_manifest.qualification",
    )
    if not all_closed:
        return _failed(
            finding_id="execution-gates",
            category="execution_gate",
            code="execution-gate-open",
            message=(
                "At least one onboarding, draft, finding, conflict, or manifest execution "
                "gate is open, or the proposed manifest is not in draft state."
            ),
            paths=paths,
        )

    return _passed(
        finding_id="execution-gates",
        category="execution_gate",
        code="all-execution-gates-closed",
        message=(
            "Every preprocessing, model-execution, and scientific-use gate remains "
            "closed; this report grants no execution authority."
        ),
        paths=paths,
    )


def _field_path_value(payload: dict[str, object], field_path: str) -> object:
    current: object = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(field_path)
        current = current[part]
    return current


def _summarize_problems(
    prefix: str, problems: list[str], *, maximum: int = 1000
) -> str:
    """Build a bounded human message while retaining deterministic failure codes."""

    if not problems:
        return prefix + "."
    rendered = prefix + ": "
    included = 0
    for problem in problems:
        suffix = ("; " if included else "") + problem
        remaining_note = (
            f"; plus {len(problems) - included - 1} more"
            if len(problems) > included + 1
            else ""
        )
        if len(rendered) + len(suffix) + len(remaining_note) + 1 > maximum:
            if included == 0:
                room = maximum - len(rendered) - len(remaining_note) - 2
                rendered += problem[: max(room, 1)]
                included = 1
            break
        rendered += suffix
        included += 1
    omitted = len(problems) - included
    if omitted:
        rendered += f"; plus {omitted} more"
    return rendered[: maximum - 1] + "."


def _json_identical(left: object, right: object) -> bool:
    def encode(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    return encode(left) == encode(right)


def _passed(
    *,
    finding_id: str,
    category: str,
    code: str,
    message: str,
    paths: tuple[str, ...],
) -> QualificationFinding:
    return QualificationFinding(
        finding_id=finding_id,
        category=category,
        code=code,
        outcome="passed",
        severity="info",
        message=message,
        related_field_paths=paths,
        blocks_approval_freeze_request=False,
    )


def _failed(
    *,
    finding_id: str,
    category: str,
    code: str,
    message: str,
    paths: tuple[str, ...],
) -> QualificationFinding:
    return QualificationFinding(
        finding_id=finding_id,
        category=category,
        code=code,
        outcome="failed",
        severity="error",
        message=message,
        related_field_paths=paths,
        blocks_approval_freeze_request=True,
    )


__all__ = ["qualify_model_contract_draft"]
