"""Validate SLSim artifacts and freeze leakage-safe classification splits."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactStore, sha256_file
from ..schemas.common import SourceIdentity, canonical_json_sha256
from ..schemas.dataset import (
    DatasetManifest,
    FrozenSplits,
    SampleRecord,
    SplitConfiguration,
)
from ..schemas.simulation import (
    SimulationDatasetRecord,
    canonical_model_bytes,
    canonical_spec_sha256,
)


class DatasetBuildError(RuntimeError):
    pass


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ImportError:
        raise DatasetBuildError("NumPy is required to assemble the dataset") from None
    return np


def _resolve_safe(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise DatasetBuildError("simulation artifact escaped its dataset root")
    if not path.is_file() or path.is_symlink():
        raise DatasetBuildError("simulation artifact is missing or is a symlink")
    return path


def _split_class(
    sample_ids: list[str], configuration: SplitConfiguration, *, class_salt: str
) -> tuple[list[str], list[str], list[str]]:
    if len(sample_ids) < 3:
        raise DatasetBuildError(
            "each class needs at least three samples for frozen splits"
        )
    ordering = sorted(
        sample_ids,
        key=lambda item: hashlib.sha256(
            f"{configuration.seed}:{class_salt}:{item}".encode("utf-8")
        ).digest(),
    )
    train_count = max(1, int(len(ordering) * configuration.train_fraction))
    validation_count = max(1, int(len(ordering) * configuration.validation_fraction))
    if train_count + validation_count >= len(ordering):
        train_count = len(ordering) - 2
        validation_count = 1
    return (
        ordering[:train_count],
        ordering[train_count : train_count + validation_count],
        ordering[train_count + validation_count :],
    )


def build_dataset_manifest(
    *,
    simulation_root: str | os.PathLike[str],
    split_configuration: SplitConfiguration | None = None,
) -> DatasetManifest:
    """Reverify every numeric sample and persist one immutable split manifest."""

    np = _load_numpy()
    root = Path(simulation_root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise DatasetBuildError("simulation root must be a real directory")
    simulation_manifest_path = _resolve_safe(root, "simulation_manifest.json")
    simulation = SimulationDatasetRecord.model_validate_json(
        simulation_manifest_path.read_bytes(), strict=True
    )
    expected_spec_sha256 = canonical_spec_sha256(simulation.spec)
    if simulation.spec_sha256 != expected_spec_sha256:
        # Compatibility is deliberately reader-only: completed campaigns written
        # before the canonical digest was unified hashed the same compact JSON
        # with one trailing newline.  New producers always emit the canonical
        # semantic digest above.
        legacy_spec_sha256 = hashlib.sha256(
            canonical_model_bytes(simulation.spec)
        ).hexdigest()
        if simulation.spec_sha256 != legacy_spec_sha256:
            raise DatasetBuildError(
                "simulation specification digest does not match its manifest"
            )
    if simulation.supports_scientific_claims:
        raise DatasetBuildError(
            "smoke backend unexpectedly authorized scientific claims"
        )

    records: list[SampleRecord] = []
    by_class: dict[str, list[str]] = {"non_lens": [], "lens": []}
    expected_shape = (
        len(simulation.spec.rendering.bands),
        simulation.spec.rendering.num_pix,
        simulation.spec.rendering.num_pix,
    )
    for sample in simulation.samples:
        image_path = _resolve_safe(root, sample.image_artifact.relative_path)
        if sha256_file(image_path) != sample.image_artifact.sha256:
            raise DatasetBuildError(f"sample digest changed: {sample.sample_id}")
        if image_path.stat().st_size != sample.image_artifact.byte_count:
            raise DatasetBuildError(f"sample byte count changed: {sample.sample_id}")
        image = np.load(image_path, allow_pickle=False)
        if image.dtype != np.float32 or tuple(image.shape) != expected_shape:
            raise DatasetBuildError(
                f"sample tensor contract failed: {sample.sample_id}"
            )
        if not bool(np.isfinite(image).all()):
            raise DatasetBuildError(f"sample has non-finite pixels: {sample.sample_id}")
        record = SampleRecord(
            sample_id=sample.sample_id,
            label_name=sample.class_name,
            label_index=sample.numeric_label,
            seed=sample.sample_seed,
            relative_path=sample.image_artifact.relative_path,
            sha256=sample.image_artifact.sha256,
            shape=expected_shape,
            dtype="float32",
            finite=True,
            minimum=float(image.min()),
            maximum=float(image.max()),
            mean=float(image.mean()),
            standard_deviation=float(image.std()),
            simulator_parameters_sha256=canonical_json_sha256(sample.parameters),
        )
        records.append(record)
        by_class[sample.class_name].append(sample.sample_id)

    configuration = split_configuration or SplitConfiguration()
    non_lens = _split_class(by_class["non_lens"], configuration, class_salt="non_lens")
    lens = _split_class(by_class["lens"], configuration, class_salt="lens")
    splits = FrozenSplits(
        seed=configuration.seed,
        train_sample_ids=tuple(non_lens[0] + lens[0]),
        validation_sample_ids=tuple(non_lens[1] + lens[1]),
        test_sample_ids=tuple(non_lens[2] + lens[2]),
    )
    manifest = DatasetManifest(
        dataset_id=simulation.dataset_id,
        purpose=simulation.spec.purpose,
        simulator=SourceIdentity(
            name="slsim",
            revision=simulation.spec.provenance.slsim_source.git_commit,
            repository=simulation.spec.provenance.slsim_source.repository_relative_path,
        ),
        simulation_spec_sha256=expected_spec_sha256,
        class_names=("non_lens", "lens"),
        bands=simulation.spec.rendering.bands,
        samples=tuple(records),
        splits=splits,
        scientific_use_allowed=False,
        qualification_notes=(
            simulation.qualification_boundary,
            "Frozen split membership is for integration plumbing only.",
        ),
    )
    ArtifactStore(root).write_json("dataset_manifest.json", manifest)
    return manifest


def load_dataset_manifest(path: str | os.PathLike[str]) -> DatasetManifest:
    manifest_path = Path(path)
    return DatasetManifest.model_validate_json(manifest_path.read_bytes(), strict=True)
