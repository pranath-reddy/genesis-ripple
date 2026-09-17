"""Typed boundary for deterministic, allowlisted preprocessing adapters.

An adapter is registered by application code as a Python object.  A model
manifest may select that registered object by its stable identifier, but it can
never name a module, callable, or import path for this module to load.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ModelManifest, ModelManifestRef


_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,127}$"
_VERSION_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,63}$"
_FIELD_PATH_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$"
_ENTRYPOINT_PATTERN = r"^[a-z][a-z0-9_.-]{2,255}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_PREPROCESSING_ENTRYPOINTS = {
    "modeling_cli": "ripple.modeling.cli",
    "mriganka_cli": "ripple.preprocessing.cli",
    "python_api": "ripple.modeling.service.run_registered_preprocessing",
    "agent_tool": "ripple.scientist.router.run_pipeline_route",
}


class _ImmutableAdapterModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class AdapterIdentity(_ImmutableAdapterModel):
    """Code-owned identity and allowlist declared by one adapter."""

    adapter_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    adapter_version: str = Field(pattern=_VERSION_PATTERN)
    deterministic: Literal[True] = True
    implementation_ids: tuple[str, ...] = Field(min_length=1)
    supported_model_ids: tuple[str, ...] = ()

    @field_validator("implementation_ids", "supported_model_ids")
    @classmethod
    def _unique_identifiers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("adapter identifier lists must not contain duplicates")
        for identifier in value:
            if re.fullmatch(_IDENTIFIER_PATTERN, identifier) is None:
                raise ValueError(
                    "adapter identifier lists contain an invalid identifier"
                )
        return value


class CompatibilityIssue(_ImmutableAdapterModel):
    """One machine-readable reason for accepting or rejecting an observation."""

    code: str = Field(pattern=_IDENTIFIER_PATTERN)
    severity: Literal["warning", "error"]
    message: str = Field(min_length=1, max_length=512)
    manifest_field: str | None = Field(default=None, pattern=_FIELD_PATH_PATTERN)

    @field_validator("message")
    @classmethod
    def _safe_message(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("compatibility messages must be trimmed printable text")
        return value


class CompatibilityResult(_ImmutableAdapterModel):
    """Auditable result of checking one manifest against one observation."""

    manifest: ModelManifestRef
    adapter: AdapterIdentity
    compatible: bool
    issues: tuple[CompatibilityIssue, ...] = ()

    @model_validator(mode="after")
    def _validate_outcome(self) -> "CompatibilityResult":
        codes = tuple(issue.code for issue in self.issues)
        if len(codes) != len(set(codes)):
            raise ValueError("compatibility issue codes must be unique")
        has_error = any(issue.severity == "error" for issue in self.issues)
        if self.compatible == has_error:
            raise ValueError(
                "compatible must be true exactly when no error issue is present"
            )
        return self

    @classmethod
    def accepted(
        cls,
        *,
        manifest: ModelManifest,
        adapter: AdapterIdentity,
        warnings: tuple[CompatibilityIssue, ...] = (),
    ) -> "CompatibilityResult":
        if any(issue.severity != "warning" for issue in warnings):
            raise ValueError("accepted compatibility results may contain warnings only")
        return cls(
            manifest=ModelManifestRef.from_manifest(manifest),
            adapter=adapter,
            compatible=True,
            issues=warnings,
        )

    @classmethod
    def rejected(
        cls,
        *,
        manifest: ModelManifest,
        adapter: AdapterIdentity,
        issues: tuple[CompatibilityIssue, ...],
    ) -> "CompatibilityResult":
        if not issues or not any(issue.severity == "error" for issue in issues):
            raise ValueError(
                "rejected compatibility results require at least one error"
            )
        return cls(
            manifest=ModelManifestRef.from_manifest(manifest),
            adapter=adapter,
            compatible=False,
            issues=issues,
        )


class VerifiedInputPackageRef(_ImmutableAdapterModel):
    """Identity of an M2 package after verification, never copied from raw CLI text."""

    run_relative_manifest_path: str = Field(min_length=1, max_length=1024)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    fits_sha256: str = Field(pattern=_SHA256_PATTERN)
    band: str = Field(min_length=1, max_length=32)
    dataset_id: str = Field(min_length=1, max_length=1024)

    @field_validator("run_relative_manifest_path")
    @classmethod
    def _safe_manifest_path(cls, value: str) -> str:
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or ".." in parsed.parts
            or parsed.as_posix() != value
            or parsed.name != "package.json"
            or not parsed.parts
            or parsed.parts[0] != "inputs"
        ):
            raise ValueError(
                "input manifest path must be normalized beneath the run inputs"
            )
        return value

    @field_validator("band", "dataset_id")
    @classmethod
    def _safe_identity_text(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("input package identities must be trimmed printable text")
        return value


class PreprocessingInvocation(_ImmutableAdapterModel):
    """Structured record of the model-aware interface that requested the run."""

    interface: Literal["modeling_cli", "mriganka_cli", "python_api", "agent_tool"]
    entrypoint: str = Field(pattern=_ENTRYPOINT_PATTERN)
    inputs: tuple[VerifiedInputPackageRef, ...] = Field(min_length=1, max_length=32)
    require_aligned_shapes: bool

    @model_validator(mode="after")
    def _entrypoint_matches_interface(self) -> "PreprocessingInvocation":
        expected = _PREPROCESSING_ENTRYPOINTS[self.interface]
        if self.entrypoint != expected:
            raise ValueError(
                f"entrypoint must be {expected!r} for interface {self.interface!r}"
            )
        return self

    @field_validator("inputs")
    @classmethod
    def _unique_inputs(
        cls, value: tuple[VerifiedInputPackageRef, ...]
    ) -> tuple[VerifiedInputPackageRef, ...]:
        paths = tuple(item.run_relative_manifest_path for item in value)
        if len(paths) != len(set(paths)):
            raise ValueError("invocation input package paths must be unique")
        return value


ObservationT_contra = TypeVar("ObservationT_contra", contravariant=True)
PackageT_co = TypeVar("PackageT_co", bound=BaseModel, covariant=True)
PackageT = TypeVar("PackageT", bound=BaseModel)


@runtime_checkable
class DeterministicPreprocessingAdapter(Protocol[ObservationT_contra, PackageT_co]):
    """Structural interface implemented by reviewed preprocessing adapters.

    Implementations must derive their scientific operations from their own
    reviewed code.  ``manifest.preprocessing.steps`` is evidence and selection
    metadata; it is never interpreted as Python code.
    """

    @property
    def identity(self) -> AdapterIdentity:
        """Return the code-owned adapter identity and operation allowlist."""

        ...

    def check_compatibility(
        self,
        *,
        manifest: ModelManifest,
        observation: ObservationT_contra,
    ) -> CompatibilityResult:
        """Check the observation without changing pixels or writing artifacts."""

        ...

    def validate_manifest_contract(
        self,
        *,
        manifest: ModelManifest,
    ) -> tuple[CompatibilityIssue, ...]:
        """Validate code-owned scientific behavior without reading observations."""

        ...

    def preprocess(
        self,
        *,
        manifest: ModelManifest,
        observation: ObservationT_contra,
        output_directory: Path,
    ) -> PackageT_co:
        """Execute the adapter's deterministic, reviewed transformation."""

        ...

    def load_package(self, *, output_directory: Path) -> PackageT_co:
        """Reload and verify adapter-owned artifacts from a completed run."""

        ...


class PreprocessingRunEnvelope(_ImmutableAdapterModel, Generic[PackageT]):
    """Generic success envelope around an adapter-specific Pydantic package."""

    schema_version: Literal["ripple.preprocessing.run-envelope.v2"] = (
        "ripple.preprocessing.run-envelope.v2"
    )
    status: Literal["success"] = "success"
    manifest: ModelManifestRef
    adapter: AdapterIdentity
    recipe_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    invocation: PreprocessingInvocation
    compatibility: CompatibilityResult
    package: PackageT

    @model_validator(mode="after")
    def _cross_validate(self) -> "PreprocessingRunEnvelope[PackageT]":
        if not self.compatibility.compatible:
            raise ValueError("a success envelope requires a compatible observation")
        if self.compatibility.manifest != self.manifest:
            raise ValueError("compatibility and run manifest references disagree")
        if self.compatibility.adapter != self.adapter:
            raise ValueError("compatibility and run adapter identities disagree")
        source = getattr(self.package, "source", None)
        if source is not None and all(
            hasattr(source, field)
            for field in (
                "run_relative_manifest_path",
                "manifest_sha256",
                "fits_sha256",
                "band",
                "dataset_id",
            )
        ):
            expected_input = VerifiedInputPackageRef(
                run_relative_manifest_path=source.run_relative_manifest_path,
                manifest_sha256=source.manifest_sha256,
                fits_sha256=source.fits_sha256,
                band=source.band,
                dataset_id=source.dataset_id,
            )
            if self.invocation.inputs != (expected_input,):
                raise ValueError(
                    "invocation inputs do not match the adapter package source provenance"
                )
        return self

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", exclude_none=False, by_alias=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
