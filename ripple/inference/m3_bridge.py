"""Audited mechanical handoff from a completed three-band M3 run to M4."""

from __future__ import annotations

import hashlib
import io
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ripple.scientist.artifacts.store import ArtifactStore
from ripple.scientist.paths import (
    UnsafePathError,
    checked_real_directory,
    checked_real_file,
)

from .contracts import (
    ConfiguredChannelMapping,
    COrderPayloadIntegrity,
    FileIdentity,
    M3BchwArrayEvidence,
    M3BridgeSourceProvenance,
    M3ToM4BridgeCompletionRecord,
    M3ToM4BridgeRecord,
    M4BridgeProvenance,
    M4ChwArrayEvidence,
)

_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ARRAY_BYTES = 4 * 1024 * 1024


class M3ToM4BridgeError(RuntimeError):
    """One fail-closed bridge validation or publication error."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class _VerifiedM3Source:
    provenance: M3BridgeSourceProvenance
    channel_mapping: ConfiguredChannelMapping
    model_input_bchw: np.ndarray

    def __post_init__(self) -> None:
        immutable = np.frombuffer(
            np.ascontiguousarray(self.model_input_bchw).tobytes(order="C"),
            dtype=np.float32,
        ).reshape((1, 3, 64, 64))
        object.__setattr__(self, "model_input_bchw", immutable)


@dataclass(frozen=True)
class LoadedM3ToM4Bridge:
    run_directory: Path
    model_input_path: Path
    bridge_path: Path
    completion_path: Path
    bridge_identity: FileIdentity
    completion_identity: FileIdentity
    record: M3ToM4BridgeRecord
    completion: M3ToM4BridgeCompletionRecord
    model_input_chw: np.ndarray

    def __post_init__(self) -> None:
        immutable = np.frombuffer(
            np.ascontiguousarray(self.model_input_chw).tobytes(order="C"),
            dtype=np.float32,
        ).reshape((3, 64, 64))
        object.__setattr__(self, "model_input_chw", immutable)

    @property
    def m4_provenance(self) -> M4BridgeProvenance:
        return M4BridgeProvenance(
            bridge_manifest=self.bridge_identity,
            bridge_completion=self.completion_identity,
            record=self.record,
        )


def run_m3_to_m4_bridge(
    *,
    m3_run_directory: Path,
    output_root: Path,
) -> LoadedM3ToM4Bridge:
    """Reverify one M3 run and publish its exact singleton-batch CHW payload."""

    source = _load_verified_m3_source(Path(m3_run_directory))
    source_array = source.model_input_bchw
    output_array = np.ascontiguousarray(source_array[0], dtype=np.float32)
    _validate_bchw(source_array)
    _validate_chw(output_array)
    if not np.array_equal(output_array, source_array[0]):
        raise M3ToM4BridgeError(
            code="bridge_array_changed",
            message="Selecting batch index zero changed the M3 numeric payload.",
        )

    source_payload_sha256 = _c_order_payload_sha256(source_array)
    output_payload_sha256 = _c_order_payload_sha256(output_array)
    payload_integrity = COrderPayloadIntegrity(
        source_c_order_payload_sha256=source_payload_sha256,
        output_c_order_payload_sha256=output_payload_sha256,
    )

    run_id, run_directory = _create_run_directory(Path(output_root))
    store = ArtifactStore(run_directory, create=False)
    output_buffer = io.BytesIO()
    np.save(output_buffer, output_array, allow_pickle=False)
    model_input_path = store.write_bytes(
        "model_input_chw.npy",
        output_buffer.getvalue(),
    )
    output_identity = _file_identity(model_input_path, maximum_bytes=_MAX_ARRAY_BYTES)
    output_evidence = M4ChwArrayEvidence(
        file=output_identity,
        decoded_sha256=_decoded_array_sha256(output_array),
        dtype="float32",
        shape=(3, 64, 64),
        axes=("channel", "y", "x"),
        minimum=float(output_array.min()),
        maximum=float(output_array.max()),
    )
    record = M3ToM4BridgeRecord(
        run_id=run_id,
        created_at_utc=datetime.now(timezone.utc),
        source_m3=source.provenance,
        channel_mapping=source.channel_mapping,
        model_input_chw=output_evidence,
        payload_integrity=payload_integrity,
        implementation_source_sha256=_implementation_source_hashes(),
    )
    bridge_path = store.write_json("bridge.json", record)
    bridge_identity = _file_identity(bridge_path, maximum_bytes=_MAX_JSON_BYTES)
    completion = M3ToM4BridgeCompletionRecord(
        run_id=run_id,
        model_input_chw=output_identity,
        bridge_manifest=bridge_identity,
    )
    store.write_json("completion.json", completion)

    loaded = load_m3_to_m4_bridge(bridge_path)
    if loaded.record != record or loaded.completion != completion:
        raise M3ToM4BridgeError(
            code="bridge_publication_round_trip_mismatch",
            message="The completed bridge changed during strict reload.",
        )
    return loaded


def load_m3_to_m4_bridge(bridge_manifest_path: Path) -> LoadedM3ToM4Bridge:
    """Load a completed bridge, reverify M3, and replay the batch selection."""

    try:
        bridge_path = checked_real_file(Path(bridge_manifest_path))
    except UnsafePathError:
        raise M3ToM4BridgeError(
            code="bridge_manifest_unavailable",
            message="The M3-to-M4 bridge manifest is unavailable or unsafe.",
        ) from None
    if bridge_path.name != "bridge.json":
        raise M3ToM4BridgeError(
            code="bridge_manifest_name_invalid",
            message="The M3-to-M4 bridge manifest must be named bridge.json.",
        )
    encoded_bridge, bridge_identity = _read_regular_file(
        bridge_path,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    try:
        record = M3ToM4BridgeRecord.model_validate_json(encoded_bridge, strict=True)
    except (ValueError, ValidationError):
        raise M3ToM4BridgeError(
            code="bridge_manifest_invalid",
            message="The M3-to-M4 bridge manifest failed its strict contract.",
        ) from None

    try:
        run_directory = checked_real_directory(bridge_path.parent)
    except UnsafePathError:
        raise M3ToM4BridgeError(
            code="bridge_run_directory_unavailable",
            message="The M3-to-M4 bridge run directory is unavailable or unsafe.",
        ) from None
    if run_directory.name != record.run_id:
        raise M3ToM4BridgeError(
            code="bridge_run_directory_mismatch",
            message="The bridge record belongs to a different run directory.",
        )
    completion_path = run_directory / "completion.json"
    encoded_completion, completion_identity = _read_regular_file(
        completion_path,
        maximum_bytes=_MAX_JSON_BYTES,
    )
    try:
        completion = M3ToM4BridgeCompletionRecord.model_validate_json(
            encoded_completion,
            strict=True,
        )
    except (ValueError, ValidationError):
        raise M3ToM4BridgeError(
            code="bridge_completion_invalid",
            message="The M3-to-M4 bridge completion record failed its strict contract.",
        ) from None
    if (
        completion.run_id != record.run_id
        or completion.bridge_manifest != bridge_identity
    ):
        raise M3ToM4BridgeError(
            code="bridge_completion_binding_mismatch",
            message="The bridge completion record does not bind this bridge manifest.",
        )

    source = _load_verified_m3_source(Path(record.source_m3.run_directory))
    if (
        source.provenance != record.source_m3
        or source.channel_mapping != record.channel_mapping
    ):
        raise M3ToM4BridgeError(
            code="bridge_source_provenance_mismatch",
            message="The reverified M3 source no longer matches the bridge record.",
        )

    model_input_path = run_directory / record.model_input_chw.file.filename
    encoded_array, output_identity = _read_regular_file(
        model_input_path,
        maximum_bytes=_MAX_ARRAY_BYTES,
    )
    if (
        output_identity != record.model_input_chw.file
        or output_identity != completion.model_input_chw
    ):
        raise M3ToM4BridgeError(
            code="bridge_output_file_mismatch",
            message="The bridge CHW file does not match its manifest and completion record.",
        )
    try:
        loaded = np.load(io.BytesIO(encoded_array), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - reject any unsafe NPY failure
        raise M3ToM4BridgeError(
            code="bridge_output_array_invalid",
            message=f"The bridge CHW array could not be loaded ({type(exc).__name__}).",
        ) from None
    if not isinstance(loaded, np.ndarray):
        raise M3ToM4BridgeError(
            code="bridge_output_not_array",
            message="The bridge CHW artifact did not contain one NumPy array.",
        )
    output_array = np.ascontiguousarray(loaded)
    _validate_chw(output_array)
    expected_output = np.ascontiguousarray(source.model_input_bchw[0], dtype=np.float32)
    if not np.array_equal(output_array, expected_output):
        raise M3ToM4BridgeError(
            code="bridge_output_replay_mismatch",
            message="The bridge CHW payload differs from M3 batch index zero.",
        )
    observed_evidence = M4ChwArrayEvidence(
        file=output_identity,
        decoded_sha256=_decoded_array_sha256(output_array),
        dtype="float32",
        shape=(3, 64, 64),
        axes=("channel", "y", "x"),
        minimum=float(output_array.min()),
        maximum=float(output_array.max()),
    )
    source_payload_sha256 = _c_order_payload_sha256(source.model_input_bchw)
    output_payload_sha256 = _c_order_payload_sha256(output_array)
    observed_integrity = COrderPayloadIntegrity(
        source_c_order_payload_sha256=source_payload_sha256,
        output_c_order_payload_sha256=output_payload_sha256,
    )
    if (
        observed_evidence != record.model_input_chw
        or observed_integrity != record.payload_integrity
    ):
        raise M3ToM4BridgeError(
            code="bridge_output_evidence_mismatch",
            message="The bridge CHW evidence does not match the replayed output.",
        )
    if _implementation_source_hashes() != record.implementation_source_sha256:
        raise M3ToM4BridgeError(
            code="bridge_implementation_changed",
            message="Current bridge source does not match the recorded implementation.",
        )

    return LoadedM3ToM4Bridge(
        run_directory=run_directory,
        model_input_path=model_input_path,
        bridge_path=bridge_path,
        completion_path=completion_path,
        bridge_identity=bridge_identity,
        completion_identity=completion_identity,
        record=record,
        completion=completion,
        model_input_chw=output_array,
    )


def _load_verified_m3_source(run_directory: Path) -> _VerifiedM3Source:
    try:
        from ripple.modeling.service import (
            build_default_registry,
            load_completed_preprocessing_run,
        )
        from ripple.preprocessing.mriganka_enn.artifact_io import (
            load_three_band_model_input_package,
        )
        from ripple.preprocessing.mriganka_enn.contracts import (
            MrigankaEnnThreeBandModelInputPackage,
        )

        registry = build_default_registry()
        source_run_directory = checked_real_directory(run_directory)
        completed = load_completed_preprocessing_run(
            run_directory=source_run_directory,
            registry=registry,
        )
        verified_run_directory = completed.run_directory
        concrete = load_three_band_model_input_package(
            verified_run_directory / "manifest.json"
        )
    except Exception as exc:  # noqa: BLE001 - collapse nested verification failures
        raise M3ToM4BridgeError(
            code="m3_source_verification_failed",
            message=f"The completed three-band M3 run could not be verified ({type(exc).__name__}).",
        ) from None

    package = concrete.package
    if (
        not isinstance(package, MrigankaEnnThreeBandModelInputPackage)
        or completed.envelope.package != package
    ):
        raise M3ToM4BridgeError(
            code="m3_package_type_mismatch",
            message="The completed run is not the exact registered three-band M3 package.",
        )
    completion = completed.completion
    expected_binding = (
        "mriganka-enn-three-band-dp2-provisional-v1",
        "deeplense.mriganka.enn-sda",
        "sda-epoch-20-iteration-0-dp2-provisional-v1",
        "mriganka-enn-native64-three-band",
        "v1",
        "mriganka-enn-dp2-native64-three-band-minmax-v1",
        "ripple.preprocessing.mriganka-enn-three-band-model-input.v1",
    )
    observed_binding = (
        completion.manifest.manifest_id,
        completion.manifest.model_id,
        completion.manifest.model_version,
        completion.adapter.adapter_id,
        completion.adapter.adapter_version,
        completion.recipe_id,
        package.schema_version,
    )
    if observed_binding != expected_binding:
        raise M3ToM4BridgeError(
            code="m3_registry_binding_mismatch",
            message="The completed M3 run does not match the exact bridge allowlist.",
        )
    bands = tuple(package.recipe.channel_bands)
    if (
        package.recipe.physical_band_mapping_status != "configured_unverified"
        or len(bands) != 3
        or set(bands) != {"g", "r", "i"}
    ):
        raise M3ToM4BridgeError(
            code="m3_channel_mapping_mismatch",
            message="The M3 channel mapping is not one configured-unverified g/r/i permutation.",
        )

    model_input = np.ascontiguousarray(concrete.model_input)
    _validate_bchw(model_input)
    model_input_path = verified_run_directory / package.model_input.filename
    model_input_identity = _file_identity(
        model_input_path,
        maximum_bytes=_MAX_ARRAY_BYTES,
    )
    if (
        model_input_identity.byte_count != package.model_input.byte_count
        or model_input_identity.sha256 != package.model_input.file_sha256
        or _decoded_array_sha256(model_input)
        != package.model_input.decoded_array_sha256
    ):
        raise M3ToM4BridgeError(
            code="m3_model_input_identity_mismatch",
            message="The reverified M3 BCHW array does not match its package identity.",
        )

    source = M3BridgeSourceProvenance(
        run_directory=str(verified_run_directory),
        completion_record=_file_identity(
            completed.completion_path,
            maximum_bytes=_MAX_JSON_BYTES,
        ),
        run_envelope=_file_identity(
            completed.envelope_path,
            maximum_bytes=_MAX_JSON_BYTES,
        ),
        selected_model_manifest=_file_identity(
            completed.selected_manifest_path,
            maximum_bytes=_MAX_JSON_BYTES,
        ),
        adapter_package_manifest=_file_identity(
            verified_run_directory / "manifest.json",
            maximum_bytes=_MAX_JSON_BYTES,
        ),
        registry_manifest_id=completion.manifest.manifest_id,
        registry_manifest_sha256=completion.manifest.sha256,
        model_id=completion.manifest.model_id,
        model_version=completion.manifest.model_version,
        adapter_id=completion.adapter.adapter_id,
        adapter_version=completion.adapter.adapter_version,
        recipe_id=completion.recipe_id,
        package_schema_version=package.schema_version,
        model_input_bchw=M3BchwArrayEvidence(
            file=model_input_identity,
            decoded_sha256=package.model_input.decoded_array_sha256,
            dtype="float32",
            shape=(1, 3, 64, 64),
            axes=("batch", "channel", "y", "x"),
            minimum=float(model_input.min()),
            maximum=float(model_input.max()),
        ),
    )
    mapping = ConfiguredChannelMapping(configured_bands=bands)
    return _VerifiedM3Source(
        provenance=source,
        channel_mapping=mapping,
        model_input_bchw=model_input,
    )


def _validate_bchw(array: np.ndarray) -> None:
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype("float32")
        or not array.dtype.isnative
        or tuple(array.shape) != (1, 3, 64, 64)
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
    ):
        raise M3ToM4BridgeError(
            code="m3_bchw_contract_mismatch",
            message="M3 must provide native contiguous finite float32 BCHW 1x3x64x64 in [0,1].",
        )


def _validate_chw(array: np.ndarray) -> None:
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.dtype("float32")
        or not array.dtype.isnative
        or tuple(array.shape) != (3, 64, 64)
        or not array.flags.c_contiguous
        or not np.isfinite(array).all()
        or float(array.min()) < 0.0
        or float(array.max()) > 1.0
    ):
        raise M3ToM4BridgeError(
            code="m4_chw_contract_mismatch",
            message="The bridge must publish native contiguous finite float32 CHW 3x64x64 in [0,1].",
        )


def _create_run_directory(output_root: Path) -> tuple[str, Path]:
    try:
        root = checked_real_directory(output_root, create=True)
    except UnsafePathError:
        raise M3ToM4BridgeError(
            code="unsafe_bridge_output_root",
            message="The bridge output root is unavailable or traverses a symlink.",
        ) from None
    for _ in range(8):
        prefix = datetime.now(timezone.utc).strftime("bridge-%Y%m%dt%H%M%S")
        run_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
        run_directory = root / run_id
        try:
            run_directory.mkdir(mode=0o700, exist_ok=False)
            return run_id, checked_real_directory(run_directory)
        except FileExistsError:
            continue
        except (OSError, UnsafePathError) as exc:
            raise M3ToM4BridgeError(
                code="bridge_run_directory_creation_failed",
                message=f"The bridge run directory could not be created ({type(exc).__name__}).",
            ) from None
    raise M3ToM4BridgeError(
        code="bridge_run_id_collision",
        message="A unique bridge run directory could not be allocated.",
    )


def _read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
) -> tuple[bytes, FileIdentity]:
    try:
        candidate = checked_real_file(path)
    except UnsafePathError:
        raise M3ToM4BridgeError(
            code="bridge_file_unavailable",
            message="A required bridge file is unavailable or unsafe.",
        ) from None
    before = os.lstat(candidate)
    if before.st_size <= 0 or before.st_size > maximum_bytes:
        raise M3ToM4BridgeError(
            code="bridge_file_size_out_of_bounds",
            message="A required bridge file is outside its fixed size bound.",
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(opened):
            raise M3ToM4BridgeError(
                code="bridge_file_changed_before_read",
                message="A required bridge file changed before it was read.",
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1)):
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise M3ToM4BridgeError(
                    code="bridge_file_size_changed",
                    message="A required bridge file exceeded its fixed size bound.",
                )
            chunks.append(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        if (
            _stable_file_identity(opened) != _stable_file_identity(final)
            or byte_count != final.st_size
        ):
            raise M3ToM4BridgeError(
                code="bridge_file_changed_during_read",
                message="A required bridge file changed while it was read.",
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return b"".join(chunks), FileIdentity(
        filename=candidate.name,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _file_identity(path: Path, *, maximum_bytes: int) -> FileIdentity:
    _, identity = _read_regular_file(path, maximum_bytes=maximum_bytes)
    return identity


def _decoded_array_sha256(array: np.ndarray) -> str:
    target_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(axis) for axis in canonical.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _c_order_payload_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _implementation_source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        filename: _sha256_file(root / filename)
        for filename in ("contracts.py", "m3_bridge.py")
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
    )


__all__ = [
    "LoadedM3ToM4Bridge",
    "M3ToM4BridgeError",
    "load_m3_to_m4_bridge",
    "run_m3_to_m4_bridge",
]
