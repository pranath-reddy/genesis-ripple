"""Immutable publication and semantic replay for three-band M3 artifacts."""

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
from pydantic import ValidationError

from ripple.dp2.errors import Dp2Error
from ripple.modeling.observation import ObservationBundleError, load_observation_bundle
from ripple.preprocessing.errors import PreprocessingArtifactError

from .contracts import (
    BandArrayArtifactRef,
    BandName,
    ImplementationSourceRef,
    ModelInputArtifactRef,
    MrigankaEnnThreeBandModelInputPackage,
    SourcePackageRef,
)
from .transform import build_mriganka_enn_three_band_input

_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ARRAY_BYTES = 1024 * 1024
_MAX_PREVIEW_BYTES = 30 * 1024 * 1024


@dataclass(frozen=True)
class LoadedBandChannel:
    band: BandName
    native_crop: np.ndarray
    mask_crop: np.ndarray
    variance_crop: np.ndarray

    def __post_init__(self) -> None:
        for field_name in ("native_crop", "mask_crop", "variance_crop"):
            object.__setattr__(
                self,
                field_name,
                _hard_readonly_copy(getattr(self, field_name)),
            )


@dataclass(frozen=True)
class LoadedMrigankaEnnThreeBandInput:
    package: MrigankaEnnThreeBandModelInputPackage
    model_input: np.ndarray
    channels: tuple[LoadedBandChannel, LoadedBandChannel, LoadedBandChannel]
    preview_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_input", _hard_readonly_copy(self.model_input))


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def decoded_array_digest(array: np.ndarray) -> str:
    """Hash dtype, shape, and little-endian C-order values independent of NPY headers."""

    target_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(target_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(axis) for axis in canonical.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def write_model_input_new(path: Path, array: np.ndarray) -> ModelInputArtifactRef:
    path = Path(path)
    if path.name != "model_input_bchw.npy":
        raise _artifact_error(
            "artifact_write",
            "artifact_name_not_allowed",
            "The three-band model input filename is outside its fixed allowlist.",
        )
    if array.dtype != np.dtype("float32") or tuple(array.shape) != (1, 3, 64, 64):
        raise _artifact_error(
            "artifact_write",
            "array_contract_mismatch",
            "The three-band model input does not have shape 1x3x64x64 and float32 dtype.",
        )
    byte_count, file_sha256 = _write_npy_new(path, array)
    return ModelInputArtifactRef(
        byte_count=byte_count,
        file_sha256=file_sha256,
        decoded_array_sha256=decoded_array_digest(array),
    )


def write_band_array_new(
    path: Path,
    array: np.ndarray,
    *,
    band: BandName,
    role: str,
) -> BandArrayArtifactRef:
    path = Path(path)
    expected = {
        "native_image": (f"native_crop_{band}_njy.npy", np.dtype("float32"), "nJy"),
        "mask": (f"mask_crop_{band}.npy", np.dtype("int32"), "bitfield"),
        "variance": (
            f"variance_crop_{band}_njy2.npy",
            np.dtype("float32"),
            "nJy2",
        ),
    }
    if role not in expected:
        raise _artifact_error(
            "artifact_write",
            "artifact_role_not_allowed",
            "The three-band array role is outside its fixed allowlist.",
        )
    expected_name, expected_dtype, unit = expected[role]
    if (
        path.name != expected_name
        or array.dtype != expected_dtype
        or tuple(array.shape) != (64, 64)
    ):
        raise _artifact_error(
            "artifact_write",
            "array_contract_mismatch",
            "A three-band channel artifact did not match its filename, shape, or dtype contract.",
        )
    byte_count, file_sha256 = _write_npy_new(path, array)
    return BandArrayArtifactRef(
        filename=path.name,
        band=band,
        role=role,
        dtype=str(array.dtype),
        unit=unit,
        byte_count=byte_count,
        file_sha256=file_sha256,
        decoded_array_sha256=decoded_array_digest(array),
    )


def write_manifest_json_atomic(
    path: Path,
    package: MrigankaEnnThreeBandModelInputPackage,
) -> None:
    path = Path(path)
    _require_private_run_directory(path.parent)
    if path.name != "manifest.json":
        raise _artifact_error(
            "manifest_write",
            "manifest_name_not_allowed",
            "The three-band package manifest must be named manifest.json.",
        )
    if os.path.lexists(path):
        raise _artifact_error(
            "manifest_write",
            "manifest_exists",
            "The three-band adapter refused to replace an existing manifest.",
        )
    encoded = package.model_dump_json(indent=2).encode("utf-8") + b"\n"
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise _artifact_error(
            "manifest_write",
            "manifest_too_large",
            "The three-band package manifest exceeded its fixed byte bound.",
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".manifest.json.",
        suffix=".tmp",
        dir=path.parent,
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
        raise _artifact_error(
            "manifest_write",
            "manifest_exists",
            "The three-band adapter refused to replace an existing manifest.",
        ) from None
    except PreprocessingArtifactError:
        raise
    except OSError as exc:
        raise _artifact_error(
            "manifest_write",
            "manifest_write_failed",
            f"The three-band package manifest could not be written ({type(exc).__name__}).",
        ) from None
    finally:
        temporary.unlink(missing_ok=True)


def load_three_band_model_input_package(
    manifest_path: Path,
) -> LoadedMrigankaEnnThreeBandInput:
    """Strictly reload one completed adapter package and replay its transform."""

    manifest_path = Path(manifest_path)
    _require_private_run_directory(manifest_path.parent)
    if manifest_path.name != "manifest.json":
        raise _artifact_error(
            "manifest_reload",
            "invalid_manifest_filename",
            "The three-band package manifest must be named manifest.json.",
        )
    encoded = _read_private_regular_file(
        manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
    )
    try:
        json.loads(encoded)
        package = MrigankaEnnThreeBandModelInputPackage.model_validate_json(
            encoded,
            strict=True,
        )
    except (ValueError, ValidationError) as exc:
        raise _artifact_error(
            "manifest_reload",
            "invalid_manifest",
            f"The three-band package manifest failed strict validation ({type(exc).__name__}).",
        ) from None
    return verify_three_band_model_input_artifacts(manifest_path.parent, package)


def verify_three_band_model_input_artifacts(
    directory: Path,
    package: MrigankaEnnThreeBandModelInputPackage,
) -> LoadedMrigankaEnnThreeBandInput:
    directory = Path(directory)
    _require_private_run_directory(directory)
    _verify_implementation_sources(package)

    model_input = _load_array_artifact(directory, package.model_input)
    loaded_channels: list[LoadedBandChannel] = []
    for channel in package.channels:
        loaded_channels.append(
            LoadedBandChannel(
                band=channel.band,
                native_crop=_load_array_artifact(directory, channel.native_crop),
                mask_crop=_load_array_artifact(directory, channel.mask_crop),
                variance_crop=_load_array_artifact(directory, channel.variance_crop),
            )
        )
    typed_channels = (
        loaded_channels[0],
        loaded_channels[1],
        loaded_channels[2],
    )

    preview_path = directory / package.preview.filename
    _require_private_regular_file(preview_path, maximum_bytes=_MAX_PREVIEW_BYTES)
    byte_count, sha256 = file_digest(preview_path)
    if (
        byte_count != package.preview.byte_count
        or sha256 != package.preview.file_sha256
    ):
        raise _artifact_error(
            "artifact_reload",
            "preview_digest_mismatch",
            "The three-band QA preview no longer matches its manifest.",
        )
    try:
        with Image.open(preview_path) as image:
            image.verify()
        with Image.open(preview_path) as image:
            pixel_size = tuple(int(value) for value in image.size)
    except Exception as exc:  # noqa: BLE001 - isolate third-party image decoder errors
        raise _artifact_error(
            "artifact_reload",
            "invalid_preview_png",
            f"The three-band QA preview could not be verified ({type(exc).__name__}).",
        ) from None
    if pixel_size != package.preview.pixel_size_xy:
        raise _artifact_error(
            "artifact_reload",
            "preview_size_mismatch",
            "The three-band QA preview size does not match its manifest.",
        )

    _verify_source_and_derived_semantics(
        directory,
        package,
        model_input,
        typed_channels,
    )
    return LoadedMrigankaEnnThreeBandInput(
        package=package,
        model_input=model_input,
        channels=typed_channels,
        preview_path=preview_path,
    )


def _write_npy_new(path: Path, array: np.ndarray) -> tuple[int, str]:
    _require_private_run_directory(path.parent)
    try:
        with path.open("xb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o600)
    except FileExistsError:
        raise _artifact_error(
            "artifact_write",
            "artifact_exists",
            "The three-band adapter refused to replace an existing artifact.",
        ) from None
    except (OSError, ValueError) as exc:
        raise _artifact_error(
            "artifact_write",
            "array_write_failed",
            f"A three-band array artifact could not be written ({type(exc).__name__}).",
        ) from None
    return file_digest(path)


def _load_array_artifact(
    directory: Path,
    artifact: ModelInputArtifactRef | BandArrayArtifactRef,
) -> np.ndarray:
    path = directory / artifact.filename
    _require_private_regular_file(path, maximum_bytes=_MAX_ARRAY_BYTES)
    byte_count, sha256 = file_digest(path)
    if byte_count != artifact.byte_count or sha256 != artifact.file_sha256:
        raise _artifact_error(
            "artifact_reload",
            "array_file_digest_mismatch",
            "A three-band array file no longer matches its manifest.",
        )
    try:
        mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise _artifact_error(
            "artifact_reload",
            "array_load_failed",
            f"A three-band NPY artifact could not be loaded ({type(exc).__name__}).",
        ) from None
    try:
        if not isinstance(mapped, np.ndarray):
            raise _artifact_error(
                "artifact_reload",
                "invalid_array_payload",
                "A three-band NPY artifact did not contain an array.",
            )
        if tuple(mapped.shape) != artifact.shape or str(mapped.dtype) != artifact.dtype:
            raise _artifact_error(
                "artifact_reload",
                "array_contract_mismatch",
                "A three-band array shape or dtype no longer matches its manifest.",
            )
        array = np.array(mapped, copy=True, order="C")
    finally:
        if isinstance(mapped, np.memmap) and mapped._mmap is not None:
            mapped._mmap.close()
    if decoded_array_digest(array) != artifact.decoded_array_sha256:
        raise _artifact_error(
            "artifact_reload",
            "decoded_array_digest_mismatch",
            "Decoded three-band array values no longer match the manifest.",
        )
    array.setflags(write=False)
    return array


def _verify_implementation_sources(
    package: MrigankaEnnThreeBandModelInputPackage,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    for source in package.implementation_sources:
        try:
            _, sha256 = file_digest(repository / source.relative_path)
        except OSError:
            raise _artifact_error(
                "implementation_replay",
                "implementation_source_unavailable",
                "A recorded three-band implementation source is unavailable.",
            ) from None
        if sha256 != source.sha256:
            raise _artifact_error(
                "implementation_replay",
                "implementation_source_digest_mismatch",
                "Current three-band source code does not match the recorded implementation.",
            )


def _verify_source_and_derived_semantics(
    directory: Path,
    package: MrigankaEnnThreeBandModelInputPackage,
    model_input: np.ndarray,
    loaded_channels: tuple[
        LoadedBandChannel,
        LoadedBandChannel,
        LoadedBandChannel,
    ],
) -> None:
    source_paths: list[Path] = []
    for source in package.sources:
        source_path = directory / source.run_relative_manifest_path
        encoded = _read_private_regular_file(
            source_path,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
        if hashlib.sha256(encoded).hexdigest() != source.manifest_sha256:
            raise _artifact_error(
                "source_replay",
                "source_manifest_digest_mismatch",
                "A bundled M2 manifest no longer matches the three-band provenance record.",
            )
        source_paths.append(source_path)
    try:
        observation = load_observation_bundle(tuple(source_paths))
    except (Dp2Error, ObservationBundleError) as exc:
        code = getattr(exc, "code", "invalid_observation_bundle")
        raise _artifact_error(
            "source_replay",
            "source_m2_verification_failed",
            f"A source M2 package could not be reverified ({code}).",
        ) from None

    for source, plane in zip(package.sources, observation.planes, strict=True):
        observed = _source_ref_from_plane(
            source.run_relative_manifest_path,
            source.manifest_sha256,
            plane,
        )
        if observed != source:
            raise _artifact_error(
                "source_replay",
                "source_identity_mismatch",
                "A reverified M2 identity does not match the three-band provenance record.",
            )
    try:
        expected = build_mriganka_enn_three_band_input(observation, package.recipe)
    except Exception as exc:  # noqa: BLE001 - sanitize deterministic replay boundary
        raise _artifact_error(
            "semantic_replay",
            "preprocessing_replay_failed",
            f"The three-band transform could not be replayed ({type(exc).__name__}).",
        ) from None
    if expected.cross_band_wcs != package.cross_band_wcs:
        raise _artifact_error(
            "semantic_replay",
            "wcs_evidence_mismatch",
            "Recomputed cross-band WCS evidence does not match the manifest.",
        )
    if not np.array_equal(model_input, expected.model_input):
        raise _artifact_error(
            "semantic_replay",
            "model_input_mismatch",
            "The saved three-band tensor does not match deterministic replay.",
        )
    channel_evidence_by_band = {channel.band: channel for channel in package.channels}
    loaded_by_band = {channel.band: channel for channel in loaded_channels}
    for expected_channel in expected.channels:
        evidence = channel_evidence_by_band[expected_channel.band]
        loaded = loaded_by_band[expected_channel.band]
        if (
            evidence.crop != expected_channel.crop
            or evidence.quality != expected_channel.quality
            or evidence.normalization_nonfinite_replacement_count
            != expected_channel.normalization_nonfinite_replacement_count
        ):
            raise _artifact_error(
                "semantic_replay",
                "derived_metadata_mismatch",
                "Recomputed channel geometry or quality evidence does not match the manifest.",
            )
        if not (
            np.array_equal(loaded.native_crop, expected_channel.native_crop)
            and np.array_equal(loaded.mask_crop, expected_channel.mask_crop)
            and np.array_equal(loaded.variance_crop, expected_channel.variance_crop)
        ):
            raise _artifact_error(
                "semantic_replay",
                "derived_array_mismatch",
                "A saved per-band artifact does not match deterministic replay.",
            )


def _source_ref_from_plane(
    relative_path: str,
    manifest_sha256: str,
    plane: object,
) -> SourcePackageRef:
    package = plane.package
    return SourcePackageRef(
        run_relative_manifest_path=relative_path,
        manifest_sha256=manifest_sha256,
        fits_sha256=package.artifact.sha256,
        m2_schema_version=package.schema_version,
        product_kind=package.product_kind,
        release=package.retrieval.release,
        observation_collection=package.dataset.observation_collection,
        product_subtype=package.dataset.product_subtype,
        dataset_id=package.dataset.dataset_id,
        obs_id=package.dataset.obs_id,
        band=package.dataset.band_name,
        tract=package.dataset.tract,
        patch=package.dataset.patch,
        ra_deg=package.request.ra_deg,
        dec_deg=package.request.dec_deg,
        image_unit=package.image.canonical_unit,
        variance_unit=package.variance.canonical_unit,
        psf_state=package.psf.state,
    )


def source_ref_from_plane(
    *,
    run_relative_manifest_path: str,
    manifest_sha256: str,
    plane: object,
) -> SourcePackageRef:
    """Construct source evidence from an already verified staged observation plane."""

    return _source_ref_from_plane(run_relative_manifest_path, manifest_sha256, plane)


def implementation_source_ref(
    path: Path, *, repository: Path
) -> ImplementationSourceRef:
    relative = path.resolve(strict=True).relative_to(repository.resolve(strict=True))
    _, sha256 = file_digest(path)
    return ImplementationSourceRef(relative_path=relative.as_posix(), sha256=sha256)


def _hard_readonly_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)
    immutable.setflags(write=False)
    return immutable


def _require_private_run_directory(path: Path) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise _artifact_error(
            "local_output",
            "missing_run_directory",
            "The private three-band run directory does not exist.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise _artifact_error(
            "local_output",
            "invalid_run_directory",
            "The three-band run path is not a real directory.",
        )
    if details.st_uid != os.geteuid() or details.st_mode & 0o077:
        raise _artifact_error(
            "local_output",
            "insecure_run_directory",
            "The three-band run directory must be owned by this user with mode 0700.",
        )


def _require_private_regular_file(path: Path, *, maximum_bytes: int) -> None:
    _reject_symlink_ancestors(path)
    try:
        details = os.lstat(path)
    except FileNotFoundError:
        raise _artifact_error(
            "artifact_reload",
            "missing_artifact",
            "A required three-band artifact is missing.",
        ) from None
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
        raise _artifact_error(
            "artifact_reload",
            "invalid_artifact_type",
            "A three-band artifact is not a regular non-symlink file.",
        )
    if details.st_uid != os.geteuid() or details.st_mode & 0o077:
        raise _artifact_error(
            "artifact_reload",
            "insecure_artifact_permissions",
            "Three-band artifacts must be private files owned by this user.",
        )
    if details.st_size <= 0 or details.st_size > maximum_bytes:
        raise _artifact_error(
            "artifact_reload",
            "artifact_size_out_of_bounds",
            "A three-band artifact size is outside its fixed replay bound.",
        )


def _read_private_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    _require_private_regular_file(path, maximum_bytes=maximum_bytes)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        byte_count = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1)):
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise _artifact_error(
                    "artifact_reload",
                    "artifact_size_changed",
                    "A three-band artifact exceeded its byte bound while reading.",
                )
            chunks.append(chunk)
        final = os.fstat(descriptor)
        if _file_state(opened) != _file_state(final) or byte_count != final.st_size:
            raise _artifact_error(
                "artifact_reload",
                "artifact_changed_during_read",
                "A three-band artifact changed while it was being read.",
            )
        return b"".join(chunks)
    except PreprocessingArtifactError:
        raise
    except OSError as exc:
        raise _artifact_error(
            "artifact_reload",
            "artifact_open_failed",
            f"A three-band artifact could not be read safely ({type(exc).__name__}).",
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_symlink_ancestors(path: Path) -> None:
    if ".." in path.parts:
        raise _artifact_error(
            "artifact_reload",
            "parent_traversal_forbidden",
            "Parent traversal is forbidden in three-band artifact paths.",
        )
    absolute = path.absolute()
    for component in tuple(reversed(absolute.parents)) + (absolute,):
        if component == Path(component.anchor):
            continue
        try:
            mode = os.lstat(component).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise _artifact_error(
                "artifact_reload",
                "symlinked_artifact_path",
                "Symlinks are forbidden in three-band artifact paths.",
            )


def _file_state(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
    )


def _artifact_error(stage: str, code: str, message: str) -> PreprocessingArtifactError:
    return PreprocessingArtifactError(stage=stage, code=code, message=message)


__all__ = [
    "LoadedBandChannel",
    "LoadedMrigankaEnnThreeBandInput",
    "decoded_array_digest",
    "file_digest",
    "implementation_source_ref",
    "load_three_band_model_input_package",
    "source_ref_from_plane",
    "verify_three_band_model_input_artifacts",
    "write_band_array_new",
    "write_manifest_json_atomic",
    "write_model_input_new",
]
