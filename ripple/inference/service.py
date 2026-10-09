"""Deterministic M4 execution and immutable evidence publication."""

from __future__ import annotations

import hashlib
import io
import os
import platform
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydantic
import torch
from pydantic import ValidationError

from ripple.scientist.artifacts.store import ArtifactStore
from ripple.scientist.paths import (
    UnsafePathError,
    checked_real_directory,
    checked_real_file,
)

from .checkpoint_io import load_checkpoint_bundle
from .contracts import (
    ArrayArtifact,
    FileIdentity,
    InputArrayEvidence,
    M4CompletionRecord,
    M4InferenceResult,
    MrigankaEnnBundleManifest,
    PublishedFile,
    ReferenceCaseSpec,
    ReferenceVerification,
    RuntimeIdentity,
)
from .m3_bridge import (
    LoadedM3ToM4Bridge,
    M3ToM4BridgeError,
    load_m3_to_m4_bridge,
)
from .mriganka_enn import MrigankaENN

_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_INPUT_BYTES = 1024 * 1024


class M4InferenceError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


@dataclass(frozen=True)
class CompletedM4Inference:
    run_directory: Path
    manifest_path: Path
    embedding_path: Path
    result_path: Path
    completion_path: Path
    result: M4InferenceResult
    completion: M4CompletionRecord


def builtin_bundle_manifest_path() -> Path:
    return (
        Path(__file__).parent
        / "manifests"
        / "mriganka-enn-sda-epoch20.hsc-eval.v1.json"
    )


def load_bundle_manifest(path: Path) -> MrigankaEnnBundleManifest:
    encoded, _ = _read_regular_file(path, maximum_bytes=_MAX_MANIFEST_BYTES)
    try:
        return MrigankaEnnBundleManifest.model_validate_json(encoded, strict=True)
    except (ValueError, ValidationError):
        raise M4InferenceError(
            code="invalid_bundle_manifest",
            message="The M4 bundle manifest failed its strict contract.",
        ) from None


def run_m4_inference(
    *,
    manifest_path: Path,
    checkpoint_root: Path,
    input_path: Path,
    output_root: Path,
    bridge_manifest_path: Path | None = None,
) -> CompletedM4Inference:
    manifest = load_bundle_manifest(manifest_path)
    bridge = _load_bridge(bridge_manifest_path)
    input_array, input_evidence = _load_input_array(input_path)
    reference_case: ReferenceCaseSpec | None = None
    if bridge is not None:
        _verify_bridge_input(
            input_path=input_path,
            input_array=input_array,
            input_evidence=input_evidence,
            bridge=bridge,
        )
    else:
        reference_case = _require_pinned_reference_input(
            manifest=manifest,
            input_evidence=input_evidence,
        )
    checkpoints = load_checkpoint_bundle(
        bundle_root=checkpoint_root,
        encoder_spec=manifest.encoder,
        classifier_spec=manifest.classifier,
    )

    torch.use_deterministic_algorithms(True)
    model = MrigankaENN(
        checkpoints.encoder.state_dict,
        checkpoints.classifier.state_dict,
    )
    model.eval()
    writable_input = np.array(
        input_array,
        dtype=np.float32,
        order="C",
        copy=True,
    )
    tensor = torch.from_numpy(writable_input).unsqueeze(0)
    with torch.inference_mode():
        embedding = model.encode(tensor)
        logits = model.classifier(embedding)
        scores = torch.softmax(logits, dim=1)
    if embedding.shape != (1, 256) or logits.shape != (1, 2) or scores.shape != (1, 2):
        raise M4InferenceError(
            code="unexpected_model_output_shape",
            message="The checkpoint graph returned an unexpected output shape.",
        )
    if not all(
        torch.isfinite(value).all().item() for value in (embedding, logits, scores)
    ):
        raise M4InferenceError(
            code="nonfinite_model_output",
            message="The checkpoint graph returned a non-finite value.",
        )

    embedding_array = np.ascontiguousarray(
        embedding.detach().cpu().numpy()[0], dtype=np.float32
    )
    logits_tuple = tuple(float(value) for value in logits.detach().cpu().numpy()[0])
    scores_tuple = tuple(float(value) for value in scores.detach().cpu().numpy()[0])
    if len(logits_tuple) != 2 or len(scores_tuple) != 2:
        raise M4InferenceError(
            code="unexpected_model_output_length",
            message="The checkpoint graph did not return exactly two classes.",
        )
    if bridge is None:
        if reference_case is None:  # pragma: no cover - guarded before execution
            raise M4InferenceError(
                code="reference_preflight_missing",
                message="The pinned HSC reference preflight result is missing.",
            )
        reference_verification = _verify_reference_case(
            case=reference_case,
            scores=(scores_tuple[0], scores_tuple[1]),
        )
    else:
        reference_verification = ReferenceVerification(status="not_applicable")

    run_id, run_directory = _create_run_directory(output_root)
    store = ArtifactStore(run_directory, create=False)
    selected_manifest_path = store.write_json("bundle_manifest.json", manifest)

    embedding_buffer = io.BytesIO()
    np.save(embedding_buffer, embedding_array, allow_pickle=False)
    embedding_path = store.write_bytes("embedding.npy", embedding_buffer.getvalue())
    embedding_file = _published_file(embedding_path)
    embedding_artifact = ArrayArtifact(
        filename=embedding_file.filename,
        byte_count=embedding_file.byte_count,
        file_sha256=embedding_file.sha256,
        decoded_sha256=_decoded_array_sha256(embedding_array),
        dtype="float32",
        shape=(256,),
    )

    result = M4InferenceResult(
        run_id=run_id,
        completed_at_utc=datetime.now(timezone.utc),
        bundle_id=manifest.bundle_id,
        bundle_manifest_sha256=manifest.canonical_sha256(),
        architecture_id=manifest.architecture_id,
        input=input_evidence,
        encoder_checkpoint=checkpoints.encoder.identity,
        classifier_checkpoint=checkpoints.classifier.identity,
        embedding=embedding_artifact,
        logits=(logits_tuple[0], logits_tuple[1]),
        scores=(scores_tuple[0], scores_tuple[1]),
        lens_score=scores_tuple[1],
        reference_verification=reference_verification,
        runtime=RuntimeIdentity(
            python=platform.python_version(),
            numpy=np.__version__,
            pydantic=pydantic.__version__,
            torch=torch.__version__,
        ),
        implementation_source_sha256=_implementation_source_hashes(),
        execution_scope_used=(
            "hsc_reference_path"
            if bridge is None
            else "unqualified_dp2_technical_integration"
        ),
        bridge_provenance=None if bridge is None else bridge.m4_provenance,
    )
    result_path = store.write_json("inference.json", result)

    completion = M4CompletionRecord(
        run_id=run_id,
        bundle_manifest=_published_file(selected_manifest_path),
        embedding=embedding_file,
        inference_result=_published_file(result_path),
    )
    completion_path = store.write_json("completion.json", completion)
    _verify_publication(
        manifest_path=selected_manifest_path,
        embedding_path=embedding_path,
        result_path=result_path,
        completion_path=completion_path,
        expected_result=result,
        expected_completion=completion,
    )
    return CompletedM4Inference(
        run_directory=run_directory,
        manifest_path=selected_manifest_path,
        embedding_path=embedding_path,
        result_path=result_path,
        completion_path=completion_path,
        result=result,
        completion=completion,
    )


def _load_bridge(bridge_manifest_path: Path | None) -> LoadedM3ToM4Bridge | None:
    if bridge_manifest_path is None:
        return None
    try:
        return load_m3_to_m4_bridge(Path(bridge_manifest_path))
    except M3ToM4BridgeError as exc:
        raise M4InferenceError(
            code=f"bridge_{exc.code}",
            message="The supplied M3-to-M4 bridge could not be verified.",
        ) from None


def _verify_bridge_input(
    *,
    input_path: Path,
    input_array: np.ndarray,
    input_evidence: InputArrayEvidence,
    bridge: LoadedM3ToM4Bridge,
) -> None:
    try:
        candidate = checked_real_file(input_path)
    except UnsafePathError:
        raise M4InferenceError(
            code="bridge_input_unavailable",
            message="The supplied M4 input is unavailable or unsafe.",
        ) from None
    if candidate != bridge.model_input_path:
        raise M4InferenceError(
            code="bridge_input_path_mismatch",
            message="M4 requires the exact model_input_chw.npy published by the bridge.",
        )
    expected = bridge.record.model_input_chw
    if (
        input_evidence.file != expected.file
        or input_evidence.decoded_sha256 != expected.decoded_sha256
        or input_evidence.dtype != expected.dtype
        or input_evidence.shape != expected.shape
        or input_evidence.minimum != expected.minimum
        or input_evidence.maximum != expected.maximum
        or not np.array_equal(input_array, bridge.model_input_chw)
    ):
        raise M4InferenceError(
            code="bridge_input_identity_mismatch",
            message="The supplied M4 input does not exactly match the verified bridge output.",
        )


def _load_input_array(path: Path) -> tuple[np.ndarray, InputArrayEvidence]:
    encoded, file_identity = _read_regular_file(path, maximum_bytes=_MAX_INPUT_BYTES)
    try:
        loaded = np.load(io.BytesIO(encoded), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - reject any unsafe NPY failure
        raise M4InferenceError(
            code="invalid_input_array",
            message=f"The M4 input is not a safe NPY array ({type(exc).__name__}).",
        ) from None
    if not isinstance(loaded, np.ndarray):
        raise M4InferenceError(
            code="input_not_ndarray",
            message="The M4 input did not decode to one NumPy array.",
        )
    if loaded.dtype != np.dtype("float32") or tuple(loaded.shape) != (3, 64, 64):
        raise M4InferenceError(
            code="input_tensor_contract_mismatch",
            message="M4 requires one float32 array with shape 3x64x64.",
        )
    if not loaded.dtype.isnative:
        raise M4InferenceError(
            code="input_byte_order_unsupported",
            message="M4 requires a native-byte-order float32 input array.",
        )
    array = np.ascontiguousarray(loaded)
    if not np.isfinite(array).all():
        raise M4InferenceError(
            code="input_not_finite",
            message="The M4 input contains a non-finite value.",
        )
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        raise M4InferenceError(
            code="input_value_range_mismatch",
            message="The M4 input contains values outside the closed interval [0,1].",
        )
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(
        (3, 64, 64)
    )
    evidence = InputArrayEvidence(
        file=file_identity,
        decoded_sha256=_decoded_array_sha256(immutable),
        dtype="float32",
        shape=(3, 64, 64),
        minimum=minimum,
        maximum=maximum,
    )
    return immutable, evidence


def _require_pinned_reference_input(
    *,
    manifest: MrigankaEnnBundleManifest,
    input_evidence: InputArrayEvidence,
) -> ReferenceCaseSpec:
    same_name = tuple(
        case
        for case in manifest.reference_cases
        if case.input_filename == input_evidence.file.filename
    )
    exact = tuple(
        case
        for case in same_name
        if case.input_file_sha256 == input_evidence.file.sha256
        and case.input_decoded_sha256 == input_evidence.decoded_sha256
    )
    if len(exact) == 1:
        return exact[0]
    if same_name:
        raise M4InferenceError(
            code="reference_input_identity_mismatch",
            message="The named M4 reference input does not match its pinned identity.",
        )
    raise M4InferenceError(
        code="unqualified_unbridged_input",
        message=(
            "Bridge-free M4 execution is restricted to an exactly pinned "
            "HSC reference input."
        ),
    )


def _verify_reference_case(
    *,
    case: ReferenceCaseSpec,
    scores: tuple[float, float],
) -> ReferenceVerification:
    error = max(
        abs(observed - expected)
        for observed, expected in zip(scores, case.expected_scores, strict=True)
    )
    if error > case.maximum_absolute_error:
        raise M4InferenceError(
            code="reference_output_mismatch",
            message="M4 did not reproduce the pinned reference output within tolerance.",
        )
    return ReferenceVerification(
        status="reproduced",
        case_id=case.case_id,
        maximum_absolute_error=error,
        tolerance=case.maximum_absolute_error,
    )


def _create_run_directory(output_root: Path) -> tuple[str, Path]:
    try:
        root = checked_real_directory(output_root, create=True)
    except UnsafePathError:
        raise M4InferenceError(
            code="unsafe_output_root",
            message="The M4 output root is unavailable or traverses a symlink.",
        ) from None
    for _ in range(8):
        prefix = datetime.now(timezone.utc).strftime("m4-%Y%m%dt%H%M%S")
        run_id = f"{prefix}-{uuid.uuid4().hex[:12]}"
        run_directory = root / run_id
        try:
            run_directory.mkdir(mode=0o700, exist_ok=False)
            return run_id, checked_real_directory(run_directory)
        except FileExistsError:
            continue
        except (OSError, UnsafePathError) as exc:
            raise M4InferenceError(
                code="run_directory_creation_failed",
                message=f"The private M4 run directory could not be created ({type(exc).__name__}).",
            ) from None
    raise M4InferenceError(
        code="run_id_collision",
        message="A unique M4 run directory could not be allocated.",
    )


def _read_regular_file(path: Path, *, maximum_bytes: int) -> tuple[bytes, FileIdentity]:
    try:
        candidate = checked_real_file(path)
    except UnsafePathError:
        raise M4InferenceError(
            code="input_file_unavailable",
            message="A required M4 input file is unavailable or unsafe.",
        ) from None
    before = os.lstat(candidate)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise M4InferenceError(
            code="input_file_size_out_of_bounds",
            message="A required M4 input file is outside its fixed size bound.",
        )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _stable_file_identity(before) != _stable_file_identity(opened):
            raise M4InferenceError(
                code="input_file_changed_before_read",
                message="A required M4 input changed before it was read.",
            )
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1)):
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise M4InferenceError(
                    code="input_file_size_changed",
                    message="A required M4 input exceeded its fixed size bound.",
                )
            chunks.append(chunk)
            digest.update(chunk)
        final = os.fstat(descriptor)
        if (
            _stable_file_identity(opened) != _stable_file_identity(final)
            or byte_count != final.st_size
        ):
            raise M4InferenceError(
                code="input_file_changed_during_read",
                message="A required M4 input changed while it was read.",
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return b"".join(chunks), FileIdentity(
        filename=candidate.name,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


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


def _implementation_source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        filename: _sha256_file(root / filename)
        for filename in (
            "checkpoint_io.py",
            "contracts.py",
            "mriganka_enn.py",
            "service.py",
        )
    }


def _published_file(path: Path) -> PublishedFile:
    details = os.lstat(path)
    if not stat.S_ISREG(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise M4InferenceError(
            code="published_artifact_invalid",
            message="An M4 output artifact is missing or unsafe.",
        )
    return PublishedFile(
        filename=path.name,
        byte_count=details.st_size,
        sha256=_sha256_file(path),
    )


def _verify_publication(
    *,
    manifest_path: Path,
    embedding_path: Path,
    result_path: Path,
    completion_path: Path,
    expected_result: M4InferenceResult,
    expected_completion: M4CompletionRecord,
) -> None:
    try:
        reloaded_result = M4InferenceResult.model_validate_json(
            result_path.read_bytes(), strict=True
        )
        reloaded_completion = M4CompletionRecord.model_validate_json(
            completion_path.read_bytes(), strict=True
        )
    except (ValueError, ValidationError):
        raise M4InferenceError(
            code="published_record_reload_failed",
            message="The completed M4 records could not be reloaded strictly.",
        ) from None
    if reloaded_result != expected_result or reloaded_completion != expected_completion:
        raise M4InferenceError(
            code="published_record_changed",
            message="A completed M4 record changed during publication.",
        )
    observed = (
        _published_file(manifest_path),
        _published_file(embedding_path),
        _published_file(result_path),
    )
    expected = (
        expected_completion.bundle_manifest,
        expected_completion.embedding,
        expected_completion.inference_result,
    )
    if observed != expected:
        raise M4InferenceError(
            code="published_artifact_digest_mismatch",
            message="An M4 output artifact does not match its completion record.",
        )


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
    "CompletedM4Inference",
    "M4InferenceError",
    "builtin_bundle_manifest_path",
    "load_bundle_manifest",
    "run_m4_inference",
]
