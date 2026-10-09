"""Private, non-overwriting M3 artifact publication and verified reloading."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ripple.dp2.errors import Dp2Error
from ripple.dp2.package_service import load_cutout_package

from .contracts import (
    ArrayArtifactRef,
    M3FailureEvidence,
    MrigankaModelInputPackage,
)
from .errors import PreprocessingArtifactError
from .mriganka import build_mriganka64_input

_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_MAX_ARRAY_BYTES = 1024 * 1024
_MAX_PREVIEW_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class LoadedMrigankaModelInput:
    """Verified runtime view of the M3 manifest and its array artifacts."""

    package: MrigankaModelInputPackage
    model_input: np.ndarray
    native_crop: np.ndarray
    mask_crop: np.ndarray
    variance_crop: np.ndarray
    preview_path: Path

    def __post_init__(self) -> None:
        for field_name in (
            "model_input",
            "native_crop",
            "mask_crop",
            "variance_crop",
        ):
            array = getattr(self, field_name)
            immutable = np.frombuffer(
                np.ascontiguousarray(array).tobytes(order="C"),
                dtype=array.dtype,
            ).reshape(array.shape)
            object.__setattr__(self, field_name, immutable)


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def decoded_array_digest(array: np.ndarray) -> str:
    """Hash dtype, shape, and little-endian C-order bytes, independent of NPY headers."""

    target_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(axis) for axis in canonical.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def write_npy_new(path: Path, array: np.ndarray) -> ArrayArtifactRef:
    """Write one private NPY without replacing any existing file."""

    path = Path(path)
    _require_private_run_directory(path.parent)
    role_by_name = {
        "model_input.npy": (
            "model_input",
            "dimensionless",
            ("batch", "channel", "y", "x"),
            "float32",
            (1, 1, 64, 64),
        ),
        "native_crop_njy.npy": ("native_image", "nJy", ("y", "x"), "float32", (64, 64)),
        "mask_crop.npy": ("mask", "bitfield", ("y", "x"), "int32", (64, 64)),
        "variance_crop_njy2.npy": (
            "variance",
            "nJy2",
            ("y", "x"),
            "float32",
            (64, 64),
        ),
    }
    try:
        role, unit, axes, expected_dtype, expected_shape = role_by_name[path.name]
    except KeyError:
        raise PreprocessingArtifactError(
            stage="artifact_write",
            code="artifact_name_not_allowed",
            message="The M3 array filename was outside the artifact allowlist.",
        ) from None
    if str(array.dtype) != expected_dtype or tuple(array.shape) != expected_shape:
        raise PreprocessingArtifactError(
            stage="artifact_write",
            code="array_contract_mismatch",
            message="M3 refused to write an array outside its fixed shape and dtype contract.",
        )
    try:
        with path.open("xb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)
    except FileExistsError:
        raise PreprocessingArtifactError(
            stage="artifact_write",
            code="artifact_exists",
            message="M3 refused to replace an existing artifact.",
        ) from None
    except PreprocessingArtifactError:
        raise
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="artifact_write",
            code="array_write_failed",
            message=f"An M3 array artifact could not be written ({type(exc).__name__}).",
        ) from None

    byte_count, sha256 = file_digest(path)
    return ArrayArtifactRef(
        filename=path.name,
        role=role,
        dtype=str(array.dtype),
        shape=tuple(int(axis) for axis in array.shape),
        axes=axes,
        unit=unit,
        byte_count=byte_count,
        file_sha256=sha256,
        decoded_array_sha256=decoded_array_digest(array),
    )


def write_manifest_json_atomic(
    path: Path,
    model: MrigankaModelInputPackage | M3FailureEvidence,
) -> None:
    """Write the approved M3 success/failure model privately and atomically."""

    path = Path(path)
    _require_private_run_directory(path.parent)
    expected_name = (
        "manifest.json"
        if isinstance(model, MrigankaModelInputPackage)
        else "failure.json"
    )
    if path.name != expected_name:
        raise PreprocessingArtifactError(
            stage="manifest_write",
            code="manifest_name_not_allowed",
            message="The M3 JSON filename was outside the serializer allowlist.",
        )
    if os.path.lexists(path):
        raise PreprocessingArtifactError(
            stage="manifest_write",
            code="manifest_exists",
            message="M3 refused to replace an existing manifest.",
        )
    encoded = model.model_dump_json(indent=2).encode("utf-8") + b"\n"
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise PreprocessingArtifactError(
            stage="manifest_write",
            code="manifest_too_large",
            message="The M3 manifest exceeded its local size bound.",
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.link(temporary, path, follow_symlinks=False)
        path.chmod(0o600)
    except FileExistsError:
        raise PreprocessingArtifactError(
            stage="manifest_write",
            code="manifest_exists",
            message="M3 refused to replace an existing manifest.",
        ) from None
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="manifest_write",
            code="manifest_write_failed",
            message=f"The M3 manifest could not be published ({type(exc).__name__}).",
        ) from None
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model_input_package(manifest_path: Path) -> LoadedMrigankaModelInput:
    """Reload and independently verify all M3 artifacts."""

    manifest_path = Path(manifest_path)
    _require_private_run_directory(manifest_path.parent)
    if manifest_path.name != "manifest.json":
        raise PreprocessingArtifactError(
            stage="manifest_reload",
            code="invalid_manifest_filename",
            message="The M3 success manifest must be named manifest.json.",
        )
    _require_private_regular_file(manifest_path, maximum_bytes=_MAX_MANIFEST_BYTES)
    encoded = manifest_path.read_bytes()
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise PreprocessingArtifactError(
            stage="manifest_reload",
            code="manifest_too_large",
            message="The M3 manifest exceeded its local size bound.",
        )
    try:
        json.loads(encoded)
        package = MrigankaModelInputPackage.model_validate_json(encoded)
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="manifest_reload",
            code="invalid_manifest",
            message=f"The M3 manifest could not be validated ({type(exc).__name__}).",
        ) from None

    return verify_model_input_artifacts(manifest_path.parent, package)


def verify_model_input_artifacts(
    directory: Path,
    package: MrigankaModelInputPackage,
) -> LoadedMrigankaModelInput:
    """Verify an in-memory package against its artifacts before or after publication."""

    directory = Path(directory)
    _require_private_run_directory(directory)
    _verify_implementation_sources(package)
    arrays: dict[str, np.ndarray] = {}
    for artifact in (
        package.model_input,
        package.native_crop,
        package.mask_crop,
        package.variance_crop,
    ):
        arrays[artifact.role] = _load_array_artifact(directory, artifact)

    _verify_source_and_derived_semantics(directory, package, arrays)
    model_input = arrays["model_input"]
    preview_path = directory / package.preview.filename
    _require_private_regular_file(preview_path, maximum_bytes=_MAX_PREVIEW_BYTES)
    byte_count, sha256 = file_digest(preview_path)
    if (
        byte_count != package.preview.byte_count
        or sha256 != package.preview.file_sha256
    ):
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="preview_digest_mismatch",
            message="The QA preview no longer matches the M3 manifest.",
        )
    try:
        with Image.open(preview_path) as image:
            image.verify()
        with Image.open(preview_path) as image:
            pixel_size = tuple(int(value) for value in image.size)
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="invalid_preview_png",
            message=f"The QA preview could not be verified ({type(exc).__name__}).",
        ) from None
    if pixel_size != package.preview.pixel_size_xy:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="preview_size_mismatch",
            message="The QA preview dimensions do not match the M3 manifest.",
        )

    return LoadedMrigankaModelInput(
        package=package,
        model_input=model_input,
        native_crop=arrays["native_image"],
        mask_crop=arrays["mask"],
        variance_crop=arrays["variance"],
        preview_path=preview_path,
    )


def _load_array_artifact(directory: Path, artifact: ArrayArtifactRef) -> np.ndarray:
    path = directory / artifact.filename
    _require_private_regular_file(path, maximum_bytes=_MAX_ARRAY_BYTES)
    byte_count, sha256 = file_digest(path)
    if byte_count != artifact.byte_count or sha256 != artifact.file_sha256:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="array_file_digest_mismatch",
            message="An M3 array file no longer matches its manifest.",
        )
    try:
        mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="array_load_failed",
            message=f"An M3 array artifact could not be loaded ({type(exc).__name__}).",
        ) from None
    if not isinstance(mapped, np.ndarray):
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="invalid_array_payload",
            message="An M3 NPY artifact did not contain an array.",
        )
    if tuple(mapped.shape) != artifact.shape or str(mapped.dtype) != artifact.dtype:
        if isinstance(mapped, np.memmap) and mapped._mmap is not None:
            mapped._mmap.close()
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="array_contract_mismatch",
            message="An M3 array shape or dtype no longer matches its manifest.",
        )
    array = np.array(mapped, copy=True, order="C")
    if isinstance(mapped, np.memmap) and mapped._mmap is not None:
        mapped._mmap.close()
    if decoded_array_digest(array) != artifact.decoded_array_sha256:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="decoded_array_digest_mismatch",
            message="Decoded M3 array content no longer matches its manifest.",
        )
    array.setflags(write=False)
    return array


def _verify_implementation_sources(package: MrigankaModelInputPackage) -> None:
    repository = Path(__file__).resolve().parents[2]
    for source in package.implementation_sources:
        try:
            _, sha256 = file_digest(repository / source.relative_path)
        except Exception:
            raise PreprocessingArtifactError(
                stage="implementation_replay",
                code="implementation_source_unavailable",
                message="A recorded M3 implementation source is unavailable for replay.",
            ) from None
        if sha256 != source.sha256:
            raise PreprocessingArtifactError(
                stage="implementation_replay",
                code="implementation_source_digest_mismatch",
                message="Current M3 source code does not match the recorded implementation.",
            )


def _verify_source_and_derived_semantics(
    directory: Path,
    package: MrigankaModelInputPackage,
    arrays: dict[str, np.ndarray],
) -> None:
    """Reopen M2 and independently replay the declared crop and normalization."""

    source_path = directory / package.source.run_relative_manifest_path
    try:
        source_manifest = _read_private_regular_file(
            source_path,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
        source_manifest_sha256 = hashlib.sha256(source_manifest).hexdigest()
    except Exception:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_manifest_unavailable",
            message="The source M2 manifest required for M3 replay is unavailable.",
        ) from None
    if source_manifest_sha256 != package.source.manifest_sha256:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_manifest_digest_mismatch",
            message="The source M2 manifest no longer matches the M3 provenance record.",
        )
    try:
        loaded = load_cutout_package(source_path)
    except Dp2Error as exc:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_m2_verification_failed",
            message=f"The source M2 package could not be reverified ({exc.code}).",
        ) from None
    try:
        reloaded_source_manifest = _read_private_regular_file(
            source_path,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except Exception:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_manifest_unavailable",
            message="The source M2 manifest required for M3 replay became unavailable.",
        ) from None
    if reloaded_source_manifest != source_manifest:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_manifest_changed_during_replay",
            message="The source M2 manifest changed while M3 replay was in progress.",
        )
    source = loaded.package
    observed_source_identity = (
        source.artifact.sha256,
        source.schema_version,
        source.dataset.dataset_id,
        source.dataset.obs_id,
        source.dataset.band_name,
        source.request.ra_deg,
        source.request.dec_deg,
    )
    declared_source_identity = (
        package.source.fits_sha256,
        package.source.m2_schema_version,
        package.source.dataset_id,
        package.source.obs_id,
        package.source.band,
        package.source.ra_deg,
        package.source.dec_deg,
    )
    if observed_source_identity != declared_source_identity:
        raise PreprocessingArtifactError(
            stage="source_replay",
            code="source_identity_mismatch",
            message="The reverified M2 identity does not match the M3 provenance record.",
        )
    try:
        expected = build_mriganka64_input(loaded, package.recipe)
    except Exception as exc:
        raise PreprocessingArtifactError(
            stage="semantic_replay",
            code="preprocessing_replay_failed",
            message=f"The declared M3 transform could not be replayed ({type(exc).__name__}).",
        ) from None
    if expected.crop != package.crop or expected.quality != package.quality:
        raise PreprocessingArtifactError(
            stage="semantic_replay",
            code="derived_metadata_mismatch",
            message="Recomputed M3 geometry or quality metadata does not match the manifest.",
        )
    expected_arrays = {
        "model_input": expected.model_input,
        "native_image": expected.native_crop,
        "mask": expected.mask_crop,
        "variance": expected.variance_crop,
    }
    if any(
        not np.array_equal(arrays[role], value)
        for role, value in expected_arrays.items()
    ):
        raise PreprocessingArtifactError(
            stage="semantic_replay",
            code="derived_array_mismatch",
            message="A reloaded M3 array does not match the replayed source transformation.",
        )


def _require_private_run_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise PreprocessingArtifactError(
            stage="local_output",
            code="missing_run_directory",
            message="The private M3 run directory does not exist.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise PreprocessingArtifactError(
            stage="local_output",
            code="invalid_run_directory",
            message="The M3 run path is not a real directory.",
        )
    if details.st_uid != os.geteuid() or details.st_mode & 0o077:
        raise PreprocessingArtifactError(
            stage="local_output",
            code="insecure_run_directory",
            message="The M3 run directory must be owned by this user with mode 0700.",
        )


def _require_private_regular_file(path: Path, *, maximum_bytes: int) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="missing_artifact",
            message="A required M3 artifact is missing.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="invalid_artifact_type",
            message="An M3 artifact is not a regular non-symlink file.",
        )
    if details.st_uid != os.geteuid() or details.st_mode & 0o077:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="insecure_artifact_permissions",
            message="M3 artifacts must be private files owned by this user.",
        )
    if details.st_size <= 0 or details.st_size > maximum_bytes:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="artifact_size_out_of_bounds",
            message="An M3 artifact size was outside its fixed replay bound.",
        )


def _read_private_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one bounded private file without following its final symlink."""

    path = Path(path)
    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="artifact_open_failed",
            message=f"An M3 artifact could not be opened safely ({type(exc).__name__}).",
        ) from None
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or details.st_mode & 0o077
        ):
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="insecure_artifact_file",
                message="M3 artifacts must be private regular files owned by this user.",
            )
        if details.st_size <= 0 or details.st_size > maximum_bytes:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="artifact_size_out_of_bounds",
                message="An M3 artifact size was outside its fixed replay bound.",
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum_bytes + 1)
        if len(payload) != details.st_size or len(payload) > maximum_bytes:
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="artifact_changed_during_read",
                message="An M3 artifact changed while it was being read.",
            )
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reject_symlink_ancestors(path: Path) -> None:
    """Reject lexical traversal and symlinks anywhere in a local artifact path."""

    if ".." in path.parts:
        raise PreprocessingArtifactError(
            stage="artifact_reload",
            code="parent_traversal_forbidden",
            message="Parent traversal is forbidden in M3 artifact paths.",
        )
    absolute = path.absolute()
    for component in list(reversed(absolute.parents)) + [absolute]:
        if component == Path(component.anchor):
            continue
        try:
            mode = os.lstat(component).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise PreprocessingArtifactError(
                stage="artifact_reload",
                code="symlinked_artifact_path",
                message="Symlinks are forbidden in M3 artifact paths.",
            )
