"""Model-aware deterministic preprocessing orchestration and publication.

This module is the execution boundary a future PydanticAI coordinator may
call. It contains no LLM. A run is complete only after adapter artifacts and
the two model-aware records have been independently reloaded and a final
``completion.json`` integrity record has been published.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ripple.dp2.service import create_private_run_directory

from .adapters import (
    AdapterIdentity,
    PreprocessingInvocation,
    PreprocessingRunEnvelope,
    VerifiedInputPackageRef,
)
from .contracts import ModelManifest, ModelManifestRef
from .manifest_io import load_builtin_manifests, load_model_manifest
from .mriganka_adapter import Mriganka64Adapter
from .mriganka_enn_adapter import MrigankaEnnThreeBandAdapter
from .observation import ObservationBundle, load_observation_bundle
from .registry import ModelAdapterRegistry, UnknownRegistrationError

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$"
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_MAX_COMPLETION_BYTES = 2 * 1024 * 1024


class ModelingServiceError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class _ImmutableServiceModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class CompletionFileRef(_ImmutableServiceModel):
    """Raw-byte identity for one required publication record."""

    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    byte_count: int = Field(gt=0, le=_MAX_RECORD_BYTES)
    file_sha256: str = Field(pattern=_SHA256_PATTERN)


class PreprocessingCompletionRecord(_ImmutableServiceModel):
    """Last-written integrity commit for one model-aware preprocessing run."""

    schema_version: Literal["ripple.preprocessing.completion.v2"] = (
        "ripple.preprocessing.completion.v2"
    )
    status: Literal["complete"] = "complete"
    completed_at_utc: datetime
    run_directory_name: str = Field(pattern=_IDENTIFIER_PATTERN)
    manifest: ModelManifestRef
    adapter: AdapterIdentity
    recipe_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    selected_model_manifest: CompletionFileRef
    run_envelope: CompletionFileRef
    adapter_package_manifest: CompletionFileRef
    classifier_execution_performed: Literal[False] = False
    scientific_use_authorized: Literal[False] = False

    @field_validator("completed_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completion timestamp must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("completion timestamp must use UTC")
        return value

    @model_validator(mode="after")
    def _fixed_record_names(self) -> "PreprocessingCompletionRecord":
        observed = (
            self.selected_model_manifest.filename,
            self.run_envelope.filename,
            self.adapter_package_manifest.filename,
        )
        expected = (
            "selected-model-manifest.json",
            "run-envelope.json",
            "manifest.json",
        )
        if observed != expected:
            raise ValueError(
                "completion record filenames do not match the publication contract"
            )
        return self


@dataclass(frozen=True)
class CompletedPreprocessingRun:
    run_directory: Path
    selected_manifest_path: Path
    envelope_path: Path
    completion_path: Path
    envelope: PreprocessingRunEnvelope[BaseModel]
    completion: PreprocessingCompletionRecord


def build_default_registry() -> ModelAdapterRegistry:
    """Build and freeze the explicit adapters/manifests shipped with RIPPLe."""

    registry = ModelAdapterRegistry()
    registry.register_adapter(Mriganka64Adapter())
    registry.register_adapter(MrigankaEnnThreeBandAdapter())
    for manifest in load_builtin_manifests():
        registry.register_manifest(manifest)
    registry.freeze()
    return registry


def find_manifest_reference(
    registry: ModelAdapterRegistry,
    manifest_id: str,
) -> ModelManifestRef:
    matches = tuple(
        reference
        for reference in registry.manifest_references()
        if reference.manifest_id == manifest_id
    )
    if len(matches) != 1:
        raise UnknownRegistrationError(
            code="unknown_manifest_id",
            message="The requested model manifest is not registered.",
        )
    return matches[0]


def run_registered_preprocessing(
    *,
    registry: ModelAdapterRegistry,
    manifest_id: str,
    package_paths: Path | str | tuple[Path | str, ...],
    output_root: Path,
    require_aligned_shapes: bool = False,
    invocation_interface: Literal[
        "modeling_cli",
        "mriganka_cli",
        "python_api",
        "agent_tool",
    ] = "python_api",
) -> CompletedPreprocessingRun:
    """Run one registered adapter and publish a last-written completion record."""

    reference = find_manifest_reference(registry, manifest_id)
    resolved = registry.resolve(reference, gate="preprocessing")
    source_observation = load_observation_bundle(
        package_paths,
        require_aligned_shapes=require_aligned_shapes,
    )
    run_directory = create_private_run_directory(Path(output_root))
    observation = _bundle_observation(
        source_observation,
        run_directory=run_directory,
        require_aligned_shapes=require_aligned_shapes,
    )
    invocation = _build_invocation(
        observation,
        run_directory=run_directory,
        interface=invocation_interface,
        require_aligned_shapes=require_aligned_shapes,
    )
    envelope = registry.run_preprocessing(
        reference,
        observation=observation,
        output_directory=run_directory,
        invocation=invocation,
    )

    verified_package = resolved.adapter.load_package(output_directory=run_directory)
    if verified_package != envelope.package:
        raise ModelingServiceError(
            code="adapter_package_round_trip_mismatch",
            message="The adapter package did not round-trip exactly before publication.",
        )

    selected_manifest_path = run_directory / "selected-model-manifest.json"
    _write_json_record(selected_manifest_path, resolved.manifest)
    reloaded_manifest = load_model_manifest(selected_manifest_path)
    if (
        reloaded_manifest != resolved.manifest
        or ModelManifestRef.from_manifest(reloaded_manifest) != reference
    ):
        raise ModelingServiceError(
            code="selected_manifest_round_trip_mismatch",
            message="The selected model manifest did not round-trip exactly.",
        )

    envelope_path = run_directory / "run-envelope.json"
    _write_json_record(envelope_path, envelope)
    reloaded_envelope = _load_concrete_envelope(
        envelope_path,
        package_type=verified_package.__class__,
    )
    if reloaded_envelope != envelope:
        raise ModelingServiceError(
            code="run_envelope_round_trip_mismatch",
            message="The preprocessing run envelope did not round-trip exactly.",
        )

    completion = PreprocessingCompletionRecord(
        completed_at_utc=datetime.now(timezone.utc),
        run_directory_name=run_directory.name,
        manifest=reference,
        adapter=resolved.adapter_identity,
        recipe_id=resolved.manifest.preprocessing.recipe_id,
        selected_model_manifest=_completion_file_ref(selected_manifest_path),
        run_envelope=_completion_file_ref(envelope_path),
        adapter_package_manifest=_completion_file_ref(run_directory / "manifest.json"),
    )
    completion_path = run_directory / "completion.json"
    _write_json_record(completion_path, completion, maximum_bytes=_MAX_COMPLETION_BYTES)

    return load_completed_preprocessing_run(
        run_directory=run_directory,
        registry=registry,
    )


def load_completed_preprocessing_run(
    *,
    run_directory: Path,
    registry: ModelAdapterRegistry,
) -> CompletedPreprocessingRun:
    """Load a run only if its final marker and adapter artifacts all verify."""

    directory = _require_private_directory(Path(run_directory))
    completion_path = directory / "completion.json"
    encoded_completion = _read_private_file(
        completion_path,
        maximum_bytes=_MAX_COMPLETION_BYTES,
    )
    try:
        json.loads(encoded_completion)
        completion = PreprocessingCompletionRecord.model_validate_json(
            encoded_completion
        )
    except Exception as exc:
        raise ModelingServiceError(
            code="invalid_completion_record",
            message=f"The preprocessing completion record is invalid ({type(exc).__name__}).",
        ) from None
    if completion.run_directory_name != directory.name:
        raise ModelingServiceError(
            code="completion_directory_mismatch",
            message="The completion record belongs to a different run directory.",
        )

    selected_manifest_path = directory / completion.selected_model_manifest.filename
    envelope_path = directory / completion.run_envelope.filename
    adapter_manifest_path = directory / completion.adapter_package_manifest.filename
    encoded_manifest = _verify_completion_file(
        selected_manifest_path,
        completion.selected_model_manifest,
    )
    encoded_envelope = _verify_completion_file(envelope_path, completion.run_envelope)
    encoded_adapter_manifest = _verify_completion_file(
        adapter_manifest_path,
        completion.adapter_package_manifest,
    )

    try:
        json.loads(encoded_manifest)
        manifest = ModelManifest.model_validate_json(encoded_manifest)
    except Exception as exc:
        raise ModelingServiceError(
            code="invalid_selected_manifest",
            message=f"The selected model manifest is invalid ({type(exc).__name__}).",
        ) from None
    manifest_ref = ModelManifestRef.from_manifest(manifest)
    if manifest_ref != completion.manifest:
        raise ModelingServiceError(
            code="completion_manifest_mismatch",
            message="The completed run identifies a different model manifest.",
        )
    resolved = registry.inspect(manifest_ref)
    if (
        resolved.manifest != manifest
        or resolved.adapter_identity != completion.adapter
        or manifest.preprocessing.recipe_id != completion.recipe_id
    ):
        raise ModelingServiceError(
            code="completion_registry_binding_mismatch",
            message="The completed run no longer matches the code-owned registry binding.",
        )

    package = resolved.adapter.load_package(output_directory=directory)
    if (
        _read_private_file(
            adapter_manifest_path,
            maximum_bytes=_MAX_RECORD_BYTES,
        )
        != encoded_adapter_manifest
    ):
        raise ModelingServiceError(
            code="adapter_manifest_changed_during_load",
            message="The adapter package manifest changed while the package was loading.",
        )
    envelope = _parse_concrete_envelope(
        encoded_envelope,
        package_type=package.__class__,
    )
    if (
        _read_private_file(selected_manifest_path, maximum_bytes=_MAX_RECORD_BYTES)
        != encoded_manifest
        or _read_private_file(envelope_path, maximum_bytes=_MAX_RECORD_BYTES)
        != encoded_envelope
    ):
        raise ModelingServiceError(
            code="completed_records_changed_during_load",
            message="A completed preprocessing record changed while the run was loading.",
        )
    if (
        envelope.package != package
        or envelope.manifest != manifest_ref
        or envelope.adapter != resolved.adapter_identity
        or envelope.recipe_id != manifest.preprocessing.recipe_id
    ):
        raise ModelingServiceError(
            code="completed_envelope_binding_mismatch",
            message="The completed envelope does not match its verified package and registry binding.",
        )

    return CompletedPreprocessingRun(
        run_directory=directory,
        selected_manifest_path=selected_manifest_path,
        envelope_path=envelope_path,
        completion_path=completion_path,
        envelope=cast(PreprocessingRunEnvelope[BaseModel], envelope),
        completion=completion,
    )


def _build_invocation(
    observation: ObservationBundle,
    *,
    run_directory: Path,
    interface: Literal[
        "modeling_cli",
        "mriganka_cli",
        "python_api",
        "agent_tool",
    ],
    require_aligned_shapes: bool,
) -> PreprocessingInvocation:
    run_root = _require_private_directory(Path(run_directory)).resolve(strict=True)
    inputs: list[VerifiedInputPackageRef] = []
    for plane in observation.planes:
        try:
            source = plane.source_manifest_path.resolve(strict=True)
            relative = source.relative_to(run_root).as_posix()
        except (FileNotFoundError, ValueError):
            raise ModelingServiceError(
                code="invocation_input_outside_run",
                message="A verified input package must be bundled beneath its preprocessing run.",
            ) from None
        encoded_source_manifest = _read_private_file(
            source,
            maximum_bytes=_MAX_RECORD_BYTES,
        )
        manifest_sha256 = hashlib.sha256(encoded_source_manifest).hexdigest()
        inputs.append(
            VerifiedInputPackageRef(
                run_relative_manifest_path=relative,
                manifest_sha256=manifest_sha256,
                fits_sha256=plane.package.artifact.sha256,
                band=plane.band,
                dataset_id=plane.package.dataset.dataset_id,
            )
        )
    entrypoint = {
        "modeling_cli": "ripple.modeling.cli",
        "mriganka_cli": "ripple.preprocessing.cli",
        "python_api": "ripple.modeling.service.run_registered_preprocessing",
        "agent_tool": "ripple.scientist.router.run_pipeline_route",
    }[interface]
    return PreprocessingInvocation(
        interface=interface,
        entrypoint=entrypoint,
        inputs=tuple(inputs),
        require_aligned_shapes=require_aligned_shapes,
    )


def _bundle_observation(
    observation: ObservationBundle,
    *,
    run_directory: Path,
    require_aligned_shapes: bool,
) -> ObservationBundle:
    """Copy verified M2 packages into the immutable M3 run for portable replay."""

    run_root = _require_private_directory(Path(run_directory))
    inputs_root = run_root / "inputs"
    inputs_root.mkdir(mode=0o700)
    staged_manifests: list[Path] = []
    for index, plane in enumerate(observation.planes, start=1):
        input_directory = inputs_root / f"input-{index:02d}"
        input_directory.mkdir(mode=0o700)
        source_manifest = plane.source_manifest_path
        source_fits = source_manifest.parent / plane.package.artifact.filename
        manifest_bytes = _read_private_file(
            source_manifest,
            maximum_bytes=_MAX_RECORD_BYTES,
        )
        target_manifest = input_directory / "package.json"
        target_fits = input_directory / plane.package.artifact.filename
        _copy_private_file(
            source_manifest,
            target_manifest,
            expected_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            expected_bytes=len(manifest_bytes),
        )
        _copy_private_file(
            source_fits,
            target_fits,
            expected_sha256=plane.package.artifact.sha256,
            expected_bytes=plane.package.artifact.byte_count,
        )
        staged_manifests.append(target_manifest)

    staged = load_observation_bundle(
        tuple(staged_manifests),
        require_aligned_shapes=require_aligned_shapes,
    )
    if tuple(item.package for item in staged.planes) != tuple(
        item.package for item in observation.planes
    ):
        raise ModelingServiceError(
            code="bundled_input_mismatch",
            message="A bundled M2 package differs from its verified source package.",
        )
    return staged


def _copy_private_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Copy one private regular file without following a final-component symlink."""

    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        source_state = os.lstat(source)
        if stat.S_ISLNK(source_state.st_mode) or not stat.S_ISREG(source_state.st_mode):
            raise ModelingServiceError(
                code="invalid_input_package_file",
                message="A verified input package component is not a regular file.",
            )
        source_descriptor = os.open(source, source_flags)
        opened_state = os.fstat(source_descriptor)
        if not _same_file_state(source_state, opened_state):
            raise ModelingServiceError(
                code="input_package_changed_before_copy",
                message="A verified input package component changed before bundling.",
            )
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                view = view[written:]
        os.fsync(destination_descriptor)
        final_state = os.fstat(source_descriptor)
        if not _same_file_state(opened_state, final_state):
            raise ModelingServiceError(
                code="input_package_changed_during_copy",
                message="A verified input package component changed while bundling.",
            )
        if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ModelingServiceError(
                code="bundled_input_digest_mismatch",
                message="A bundled input component did not match its verified identity.",
            )
    except ModelingServiceError:
        raise
    except (FileExistsError, OSError) as exc:
        raise ModelingServiceError(
            code="input_package_copy_failed",
            message=f"A verified input package could not be bundled ({type(exc).__name__}).",
        ) from None
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if source_descriptor is not None:
            os.close(source_descriptor)


def _load_concrete_envelope(
    path: Path,
    *,
    package_type: type[BaseModel],
) -> PreprocessingRunEnvelope[BaseModel]:
    encoded = _read_private_file(path, maximum_bytes=_MAX_RECORD_BYTES)
    return _parse_concrete_envelope(encoded, package_type=package_type)


def _parse_concrete_envelope(
    encoded: bytes,
    *,
    package_type: type[BaseModel],
) -> PreprocessingRunEnvelope[BaseModel]:
    try:
        json.loads(encoded)
        concrete_type = PreprocessingRunEnvelope[package_type]  # type: ignore[valid-type]
        return cast(
            PreprocessingRunEnvelope[BaseModel],
            concrete_type.model_validate_json(encoded),
        )
    except Exception as exc:
        raise ModelingServiceError(
            code="invalid_run_envelope",
            message=f"The preprocessing run envelope is invalid ({type(exc).__name__}).",
        ) from None


def _completion_file_ref(path: Path) -> CompletionFileRef:
    encoded = _read_private_file(path, maximum_bytes=_MAX_RECORD_BYTES)
    return CompletionFileRef(
        filename=path.name,
        byte_count=len(encoded),
        file_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _verify_completion_file(path: Path, reference: CompletionFileRef) -> bytes:
    encoded = _read_private_file(path, maximum_bytes=_MAX_RECORD_BYTES)
    if (
        len(encoded) != reference.byte_count
        or hashlib.sha256(encoded).hexdigest() != reference.file_sha256
    ):
        raise ModelingServiceError(
            code="completion_file_digest_mismatch",
            message="A completed preprocessing record no longer matches its recorded digest.",
        )
    return encoded


def _require_private_directory(path: Path) -> Path:
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise ModelingServiceError(
            code="run_directory_not_found",
            message="The preprocessing run directory does not exist.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ModelingServiceError(
            code="invalid_run_directory",
            message="The preprocessing run path must be a non-symlink directory.",
        )
    if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise ModelingServiceError(
            code="run_directory_not_private",
            message="The preprocessing run directory must be private to the current user.",
        )
    return path


def _read_private_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise ModelingServiceError(
            code="required_run_record_missing",
            message="A required completed-run record is missing.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise ModelingServiceError(
            code="invalid_run_record_file",
            message="A completed-run record must be a regular non-symlink file.",
        )
    if (
        details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) & 0o077
        or details.st_size <= 0
        or details.st_size > maximum_bytes
    ):
        raise ModelingServiceError(
            code="run_record_not_private_or_bounded",
            message="A completed-run record is not private or is outside its size bound.",
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not _same_file_state(details, opened):
            raise ModelingServiceError(
                code="run_record_changed_before_read",
                message="A completed-run record changed before it could be read.",
            )
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1))
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise ModelingServiceError(
                    code="run_record_size_changed",
                    message="A completed-run record exceeded its byte bound while reading.",
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if not _same_file_state(opened, final) or byte_count != final.st_size:
            raise ModelingServiceError(
                code="run_record_size_changed",
                message="A completed-run record changed while it was being read.",
            )
        encoded = b"".join(chunks)
    except ModelingServiceError:
        raise
    except OSError as exc:
        raise ModelingServiceError(
            code="run_record_read_failed",
            message=f"A completed-run record could not be read ({type(exc).__name__}).",
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return encoded


def _same_file_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_uid,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_uid,
        right.st_size,
        right.st_mtime_ns,
    )


def _write_json_record(
    path: Path,
    model: BaseModel,
    *,
    maximum_bytes: int = _MAX_RECORD_BYTES,
) -> None:
    """Publish one immutable JSON record without a check/replace race."""

    encoded = (
        model.model_dump_json(indent=2, exclude_none=False).encode("utf-8") + b"\n"
    )
    if len(encoded) > maximum_bytes:
        raise ModelingServiceError(
            code="record_size_out_of_bounds",
            message="A model-aware preprocessing record exceeded its size bound.",
        )
    if path.exists() or path.is_symlink():
        raise ModelingServiceError(
            code="record_already_exists",
            message="A model-aware preprocessing record cannot overwrite an existing path.",
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o600)
        os.link(temporary_path, path, follow_symlinks=False)
        path.chmod(0o600)
        temporary_path.unlink()
        temporary_path = None
    except FileExistsError:
        raise ModelingServiceError(
            code="record_already_exists",
            message="A model-aware preprocessing record cannot overwrite an existing path.",
        ) from None
    except Exception as exc:
        raise ModelingServiceError(
            code="record_write_failed",
            message=f"A model-aware preprocessing record could not be written ({type(exc).__name__}).",
        ) from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "CompletedPreprocessingRun",
    "CompletionFileRef",
    "ModelingServiceError",
    "PreprocessingCompletionRecord",
    "build_default_registry",
    "find_manifest_reference",
    "load_completed_preprocessing_run",
    "run_registered_preprocessing",
]
