"""Explicit in-memory allowlist for model manifests and preprocessing adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

from .adapters import (
    AdapterIdentity,
    CompatibilityIssue,
    CompatibilityResult,
    DeterministicPreprocessingAdapter,
    PreprocessingInvocation,
    PreprocessingRunEnvelope,
)
from .contracts import ModelManifest, ModelManifestRef


QualificationGate = Literal["preprocessing", "model_execution", "scientific_use"]


class RegistryError(RuntimeError):
    """Base class for bounded, machine-readable registry failures."""

    def __init__(self, *, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


class DuplicateRegistrationError(RegistryError):
    pass


class UnknownRegistrationError(RegistryError):
    pass


class ManifestMismatchError(RegistryError):
    pass


class QualificationGateError(RegistryError):
    pass


class IncompatibleObservationError(RegistryError):
    def __init__(self, result: CompatibilityResult) -> None:
        self.result = result
        super().__init__(
            code="observation_incompatible",
            message="The registered preprocessing adapter rejected the observation.",
        )


class RegistryFrozenError(RegistryError):
    pass


@dataclass(frozen=True)
class ResolvedRegistration:
    """One manifest bound to the exact code-owned adapter selected for it."""

    manifest: ModelManifest
    adapter: DeterministicPreprocessingAdapter[Any, BaseModel]
    adapter_identity: AdapterIdentity


@dataclass(frozen=True)
class _RegisteredAdapter:
    identity: AdapterIdentity
    implementation: DeterministicPreprocessingAdapter[Any, BaseModel]


class ModelAdapterRegistry:
    """Mutable-at-setup, optionally frozen, explicit runtime allowlist.

    Only adapter objects supplied by trusted application code can be registered.
    Manifest fields are compared with those objects; they are never passed to
    ``importlib`` or used to locate executable Python code.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, _RegisteredAdapter] = {}
        self._manifests: dict[str, ModelManifest] = {}
        self._model_versions: dict[tuple[str, str], str] = {}
        self._frozen = False
        self._lock = RLock()

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def freeze(self) -> None:
        """Prevent later changes to the process-local allowlist."""

        with self._lock:
            self._frozen = True

    def register_adapter(
        self,
        adapter: DeterministicPreprocessingAdapter[Any, BaseModel],
    ) -> AdapterIdentity:
        """Register one concrete adapter object; duplicate IDs are forbidden."""

        if not isinstance(adapter, DeterministicPreprocessingAdapter):
            raise RegistryError(
                code="invalid_adapter_interface",
                message="The adapter does not implement the deterministic adapter interface.",
            )
        identity = adapter.identity
        if not isinstance(identity, AdapterIdentity):
            raise RegistryError(
                code="invalid_adapter_identity",
                message="The adapter identity did not satisfy the typed adapter contract.",
            )
        with self._lock:
            self._require_mutable()
            if identity.adapter_id in self._adapters:
                raise DuplicateRegistrationError(
                    code="duplicate_adapter_id",
                    message="An adapter with this ID is already registered.",
                )
            self._adapters[identity.adapter_id] = _RegisteredAdapter(
                identity=identity,
                implementation=adapter,
            )
        return identity

    def register_manifest(self, manifest: ModelManifest) -> ModelManifestRef:
        """Register one manifest only after its adapter binding is allowlisted."""

        if not isinstance(manifest, ModelManifest):
            raise RegistryError(
                code="invalid_model_manifest",
                message="The model manifest did not satisfy the typed manifest contract.",
            )
        try:
            revalidated = ModelManifest.model_validate_json(
                manifest.model_dump_json(exclude_none=False)
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise RegistryError(
                code="invalid_model_manifest",
                message=(
                    "The model manifest failed an independent strict round trip "
                    f"({type(exc).__name__})."
                ),
            ) from None
        if revalidated != manifest:
            raise RegistryError(
                code="model_manifest_round_trip_mismatch",
                message="The model manifest changed during strict revalidation.",
            )
        manifest = revalidated
        with self._lock:
            self._require_mutable()
            if manifest.manifest_id in self._manifests:
                raise DuplicateRegistrationError(
                    code="duplicate_manifest_id",
                    message="A model manifest with this ID is already registered.",
                )
            model_key = (manifest.model_id, manifest.model_version)
            if model_key in self._model_versions:
                raise DuplicateRegistrationError(
                    code="duplicate_model_version",
                    message="This model ID and version already have a registered manifest.",
                )
            registered_adapter = self._adapters.get(manifest.preprocessing.adapter_id)
            if registered_adapter is None:
                raise UnknownRegistrationError(
                    code="unregistered_manifest_adapter",
                    message="The manifest references an adapter that is not registered.",
                )
            self._require_stable_adapter(registered_adapter)
            self._validate_binding(manifest, registered_adapter.identity)
            issues = registered_adapter.implementation.validate_manifest_contract(
                manifest=manifest
            )
            self._validate_manifest_contract_issues(issues)
            if any(issue.severity == "error" for issue in issues):
                raise ManifestMismatchError(
                    code="adapter_manifest_contract_mismatch",
                    message="The code-owned adapter rejected the manifest's scientific contract.",
                )
            self._manifests[manifest.manifest_id] = manifest
            self._model_versions[model_key] = manifest.manifest_id
        return ModelManifestRef.from_manifest(manifest)

    def adapter_identities(self) -> tuple[AdapterIdentity, ...]:
        with self._lock:
            return tuple(
                self._adapters[adapter_id].identity
                for adapter_id in sorted(self._adapters)
            )

    def manifest_references(self) -> tuple[ModelManifestRef, ...]:
        with self._lock:
            return tuple(
                ModelManifestRef.from_manifest(self._manifests[manifest_id])
                for manifest_id in sorted(self._manifests)
            )

    def resolve(
        self,
        reference: ModelManifestRef,
        *,
        gate: QualificationGate,
    ) -> ResolvedRegistration:
        """Resolve an exact reference, enforce its gate, and bind its adapter."""

        resolved = self.inspect(reference)
        self._enforce_gate(resolved.manifest, gate)
        return resolved

    def inspect(self, reference: ModelManifestRef) -> ResolvedRegistration:
        """Resolve and validate an exact binding without authorizing execution."""

        with self._lock:
            manifest = self._manifests.get(reference.manifest_id)
            if manifest is None:
                raise UnknownRegistrationError(
                    code="unknown_manifest_id",
                    message="The requested model manifest is not registered.",
                )
            actual_reference = ModelManifestRef.from_manifest(manifest)
            if actual_reference != reference:
                raise ManifestMismatchError(
                    code="model_manifest_reference_mismatch",
                    message="The requested model identity, version, or digest does not match the registry.",
                )
            mapped_manifest_id = self._model_versions.get(
                (reference.model_id, reference.model_version)
            )
            if mapped_manifest_id != reference.manifest_id:
                raise ManifestMismatchError(
                    code="model_manifest_mapping_mismatch",
                    message="The requested model ID and version resolve to a different manifest.",
                )
            registered_adapter = self._adapters.get(manifest.preprocessing.adapter_id)
            if registered_adapter is None:
                raise UnknownRegistrationError(
                    code="registered_manifest_adapter_missing",
                    message="The manifest's preprocessing adapter is no longer registered.",
                )
            self._require_stable_adapter(registered_adapter)
            self._validate_binding(manifest, registered_adapter.identity)
            issues = registered_adapter.implementation.validate_manifest_contract(
                manifest=manifest
            )
            self._validate_manifest_contract_issues(issues)
            if any(issue.severity == "error" for issue in issues):
                raise ManifestMismatchError(
                    code="registered_manifest_contract_mismatch",
                    message="The registered manifest no longer matches its code-owned adapter.",
                )
            return ResolvedRegistration(
                manifest=manifest,
                adapter=registered_adapter.implementation,
                adapter_identity=registered_adapter.identity,
            )

    def resolve_model(
        self,
        *,
        model_id: str,
        model_version: str,
        manifest_id: str,
        manifest_sha256: str,
        gate: QualificationGate,
    ) -> ResolvedRegistration:
        """Resolve caller-supplied identities without accepting a loose manifest."""

        return self.resolve(
            ModelManifestRef(
                manifest_id=manifest_id,
                model_id=model_id,
                model_version=model_version,
                sha256=manifest_sha256,
            ),
            gate=gate,
        )

    def validate_manifest_contract(
        self,
        manifest: ModelManifest,
    ) -> tuple[CompatibilityIssue, ...]:
        """Check a proposal against its code-owned adapter without executing it."""

        if not isinstance(manifest, ModelManifest):
            raise RegistryError(
                code="invalid_model_manifest",
                message="The model manifest did not satisfy the typed manifest contract.",
            )
        try:
            revalidated = ModelManifest.model_validate_json(
                manifest.model_dump_json(exclude_none=False)
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise RegistryError(
                code="invalid_model_manifest",
                message=(
                    "The model manifest failed an independent strict round trip "
                    f"({type(exc).__name__})."
                ),
            ) from None
        if revalidated != manifest:
            raise RegistryError(
                code="model_manifest_round_trip_mismatch",
                message="The model manifest changed during strict revalidation.",
            )
        manifest = revalidated
        with self._lock:
            registered = self._adapters.get(manifest.preprocessing.adapter_id)
            if registered is None:
                raise UnknownRegistrationError(
                    code="unregistered_manifest_adapter",
                    message="The manifest references an adapter that is not registered.",
                )
            self._require_stable_adapter(registered)
            self._validate_binding(manifest, registered.identity)
            issues = registered.implementation.validate_manifest_contract(
                manifest=manifest
            )
        self._validate_manifest_contract_issues(issues)
        return issues

    def run_preprocessing(
        self,
        reference: ModelManifestRef,
        *,
        observation: Any,
        output_directory: Path,
        invocation: PreprocessingInvocation,
    ) -> PreprocessingRunEnvelope[BaseModel]:
        """Compatibility-check and execute one registered deterministic adapter."""

        resolved = self.resolve(reference, gate="preprocessing")
        compatibility = resolved.adapter.check_compatibility(
            manifest=resolved.manifest,
            observation=observation,
        )
        self._validate_compatibility_result(resolved, compatibility, reference)
        if not compatibility.compatible:
            raise IncompatibleObservationError(compatibility)
        package = resolved.adapter.preprocess(
            manifest=resolved.manifest,
            observation=observation,
            output_directory=Path(output_directory),
        )
        if not isinstance(package, BaseModel):
            raise RegistryError(
                code="invalid_adapter_package",
                message="The preprocessing adapter did not return a Pydantic package.",
            )
        return cast(
            PreprocessingRunEnvelope[BaseModel],
            PreprocessingRunEnvelope(
                manifest=reference,
                adapter=resolved.adapter_identity,
                recipe_id=resolved.manifest.preprocessing.recipe_id,
                invocation=invocation,
                compatibility=compatibility,
                package=package,
            ),
        )

    def _require_mutable(self) -> None:
        if self._frozen:
            raise RegistryFrozenError(
                code="registry_frozen",
                message="The adapter and model-manifest registry is frozen.",
            )

    @staticmethod
    def _require_stable_adapter(registered: _RegisteredAdapter) -> None:
        if registered.implementation.identity != registered.identity:
            raise RegistryError(
                code="adapter_identity_changed",
                message="A registered adapter changed identity after registration.",
            )

    @staticmethod
    def _validate_binding(manifest: ModelManifest, identity: AdapterIdentity) -> None:
        preprocessing = manifest.preprocessing
        if preprocessing.adapter_id != identity.adapter_id:
            raise ManifestMismatchError(
                code="manifest_adapter_id_mismatch",
                message="The manifest adapter ID does not match the registered adapter.",
            )
        if preprocessing.adapter_version != identity.adapter_version:
            raise ManifestMismatchError(
                code="manifest_adapter_version_mismatch",
                message="The manifest adapter version does not match the registered adapter.",
            )
        if (
            identity.supported_model_ids
            and manifest.model_id not in identity.supported_model_ids
        ):
            raise ManifestMismatchError(
                code="adapter_model_mismatch",
                message="The registered adapter does not allowlist this model ID.",
            )
        requested_implementations = {
            step.implementation_id for step in preprocessing.steps
        }
        allowed_implementations = set(identity.implementation_ids)
        if not requested_implementations <= allowed_implementations:
            raise ManifestMismatchError(
                code="unregistered_transform_implementation",
                message="The manifest references a transform implementation not allowlisted by the adapter.",
            )

    @staticmethod
    def _validate_manifest_contract_issues(
        issues: tuple[CompatibilityIssue, ...],
    ) -> None:
        if not isinstance(issues, tuple) or any(
            not isinstance(issue, CompatibilityIssue) for issue in issues
        ):
            raise RegistryError(
                code="invalid_manifest_contract_result",
                message="The adapter returned an invalid manifest-contract result.",
            )
        codes = tuple(issue.code for issue in issues)
        if len(codes) != len(set(codes)):
            raise RegistryError(
                code="duplicate_manifest_contract_issue",
                message="The adapter returned duplicate manifest-contract issue codes.",
            )

    @staticmethod
    def _enforce_gate(manifest: ModelManifest, gate: QualificationGate) -> None:
        qualification = manifest.qualification
        allowed = {
            "preprocessing": qualification.preprocessing_execution_allowed,
            "model_execution": qualification.model_execution_allowed,
            "scientific_use": qualification.scientific_use_allowed,
        }[gate]
        if not allowed:
            raise QualificationGateError(
                code=f"{gate}_not_allowed",
                message="The manifest qualification record blocks the requested execution gate.",
            )

    @staticmethod
    def _validate_compatibility_result(
        resolved: ResolvedRegistration,
        result: CompatibilityResult,
        reference: ModelManifestRef,
    ) -> None:
        if not isinstance(result, CompatibilityResult):
            raise RegistryError(
                code="invalid_compatibility_result",
                message="The preprocessing adapter returned an invalid compatibility result.",
            )
        if result.manifest != reference:
            raise ManifestMismatchError(
                code="compatibility_manifest_mismatch",
                message="The adapter compatibility result identifies a different manifest.",
            )
        if result.adapter != resolved.adapter_identity:
            raise ManifestMismatchError(
                code="compatibility_adapter_mismatch",
                message="The adapter compatibility result identifies a different adapter.",
            )
