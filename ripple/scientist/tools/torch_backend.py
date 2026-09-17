"""Safe CUDA training and evaluation backend owned by RIPPLe.

Architectures are constructed by the self-contained
``ripple.scientist.tools.model_builders`` module.  The separate DeepLense AI
Scientist checkout is source precedent only and is neither imported nor needed
at runtime.  Numeric arrays use ``allow_pickle=False`` and model weights use
Safetensors rather than Python pickle.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from ..artifacts import sha256_file
from ..schemas.architecture import BoundArchitecture, TrainingConfiguration
from ..schemas.dataset import DatasetManifest
from ..schemas.training import (
    BinaryMetrics,
    EpochRecord,
    FinalEvaluationRecord,
    NormalizationRecord,
    PredictionRecord,
    TrainingRunRecord,
)
from .model_builders import build_model


_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
# This module imports Torch lazily. Set the documented deterministic CuBLAS
# workspace before that import so CUDA matrix operations can obey the strict
# deterministic-algorithm policy below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)


class TrainingBackendError(RuntimeError):
    pass


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise TrainingBackendError(
            "training requires numpy, torch, and safetensors"
        ) from exc
    return np, torch, load_file, save_file


def _enable_deterministic_torch(torch: Any) -> None:
    if torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise TrainingBackendError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise TrainingBackendError("dataset path escaped its root")
    if not path.is_file() or path.is_symlink():
        raise TrainingBackendError("dataset sample is missing or unsafe")
    return path


def _load_split(
    np: Any,
    *,
    root: Path,
    manifest: DatasetManifest,
    split: Literal["train", "validation", "test"],
) -> tuple[Any, Any, tuple[str, ...]]:
    sample_ids = getattr(manifest.splits, f"{split}_sample_ids")
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    images: list[Any] = []
    labels: list[int] = []
    for sample_id in sample_ids:
        sample = by_id[sample_id]
        path = _safe_path(root, sample.relative_path)
        if sha256_file(path) != sample.sha256:
            raise TrainingBackendError(f"sample digest changed: {sample_id}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or tuple(array.shape) != sample.shape:
            raise TrainingBackendError(f"sample tensor contract failed: {sample_id}")
        if not bool(np.isfinite(array).all()):
            raise TrainingBackendError(
                f"sample contains non-finite pixels: {sample_id}"
            )
        images.append(array)
        labels.append(sample.label_index)
    return (
        np.stack(images).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.int64),
        tuple(sample_ids),
    )


def _average_ranks(np: Any, values: Any) -> Any:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _binary_metrics(
    np: Any, labels: Any, probabilities: Any, loss: float
) -> BinaryMetrics:
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    tp = int(((labels == 1) & (predictions == 1)).sum())
    recall_non_lens = tn / (tn + fp) if tn + fp else 0.0
    recall_lens = tp / (tp + fn) if tp + fn else 0.0
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    auc: float | None = None
    if positives and negatives:
        ranks = _average_ranks(np, probabilities[:, 1])
        rank_sum = float(ranks[labels == 1].sum())
        auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return BinaryMetrics(
        loss=float(loss),
        accuracy=float((predictions == labels).mean()),
        balanced_accuracy=float((recall_non_lens + recall_lens) / 2.0),
        roc_auc=auc,
        confusion_matrix=((tn, fp), (fn, tp)),
        sample_count=int(len(labels)),
    )


def _evaluate(
    np: Any,
    torch: Any,
    model: Any,
    x: Any,
    y: Any,
    *,
    device: Any,
    mean: Any,
    std: Any,
) -> tuple[BinaryMetrics, Any]:
    from torch import nn

    model.eval()
    tensor_x = torch.from_numpy((x - mean) / std).to(device)
    tensor_y = torch.from_numpy(y).to(device)
    with torch.no_grad():
        logits = model(tensor_x)
        loss = float(nn.functional.cross_entropy(logits, tensor_y).detach().cpu())
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return _binary_metrics(np, y, probabilities, loss), probabilities


def train_candidate(
    *,
    dataset_root: str | os.PathLike[str],
    manifest: DatasetManifest,
    manifest_sha256: str,
    architecture: BoundArchitecture,
    configuration: TrainingConfiguration,
    output_dir: str | os.PathLike[str],
) -> TrainingRunRecord:
    """Train one shortlisted candidate without reading the final test split."""

    np, torch, _, save_file = _dependencies()
    root = Path(dataset_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise TrainingBackendError("training output directory already exists")
    destination.mkdir(parents=True)
    progress_path = destination / "progress.json"

    if architecture.input_shape != manifest.samples[0].shape[1:]:
        raise TrainingBackendError("architecture image shape disagrees with dataset")
    if architecture.channels != manifest.samples[0].shape[0]:
        raise TrainingBackendError("architecture channel count disagrees with dataset")
    if architecture.num_classes != len(manifest.class_names):
        raise TrainingBackendError("architecture class count disagrees with dataset")

    train_x, train_y, _ = _load_split(np, root=root, manifest=manifest, split="train")
    validation_x, validation_y, _ = _load_split(
        np, root=root, manifest=manifest, split="validation"
    )
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-8, np.float32(1.0), std).astype(np.float32)

    random.seed(configuration.seed)
    np.random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configuration.seed)
    _enable_deterministic_torch(torch)
    if configuration.require_cuda and not torch.cuda.is_available():
        raise TrainingBackendError("CUDA was required but is not available")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(architecture, dropout=configuration.dropout).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configuration.epochs
        )
        if configuration.lr_scheduler == "cosine"
        else None
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(configuration.seed)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy((train_x - mean) / std), torch.from_numpy(train_y)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(configuration.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.monotonic()
    history: list[EpochRecord] = []
    best_epoch = 0
    best_score = -1.0
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    for epoch_index in range(configuration.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if configuration.augment_d4:
                rotations = int(torch.randint(0, 4, (1,), generator=generator))
                batch_x = torch.rot90(batch_x, rotations, dims=(2, 3))
                if int(torch.randint(0, 2, (1,), generator=generator)):
                    batch_x = torch.flip(batch_x, dims=(3,))
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = torch.nn.functional.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_y)
            correct += int((logits.argmax(dim=1) == batch_y).sum().detach().cpu())
            seen += len(batch_y)
        if scheduler is not None:
            scheduler.step()
        validation, _ = _evaluate(
            np,
            torch,
            model,
            validation_x,
            validation_y,
            device=device,
            mean=mean,
            std=std,
        )
        record = EpochRecord(
            epoch=epoch_index + 1,
            train_loss=loss_sum / seen,
            train_accuracy=correct / seen,
            validation=validation,
        )
        history.append(record)
        if validation.balanced_accuracy > best_score or (
            validation.balanced_accuracy == best_score and validation.loss < best_loss
        ):
            best_score = validation.balanced_accuracy
            best_loss = validation.loss
            best_epoch = record.epoch
            best_state = {
                key: value.detach().cpu().contiguous().clone()
                for key, value in model.state_dict().items()
            }
        _atomic_json(
            progress_path,
            {
                "candidate_id": architecture.candidate.candidate_id,
                "epoch": record.epoch,
                "epochs": configuration.epochs,
                "validation_balanced_accuracy": validation.balanced_accuracy,
                "status": "running",
            },
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    if best_state is None:
        raise TrainingBackendError("training completed without a checkpoint state")
    model.load_state_dict(best_state)
    weights_path = destination / "model.safetensors"
    save_file(best_state, str(weights_path))
    normalization = NormalizationRecord(
        mean=tuple(float(item) for item in mean.reshape(-1)),
        standard_deviation=tuple(float(item) for item in std.reshape(-1)),
    )
    metadata_payload: dict[str, object] = {
        "schema_version": "ripple.safe-checkpoint-metadata.v1",
        "dataset_id": manifest.dataset_id,
        "dataset_manifest_sha256": manifest_sha256,
        "architecture": architecture.model_dump(mode="json"),
        "training_configuration": configuration.model_dump(mode="json"),
        "normalization": normalization.model_dump(mode="json"),
        "class_names": list(manifest.class_names),
        "weights_sha256": sha256_file(weights_path),
        "format": "safetensors",
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    metadata_path = destination / "checkpoint.json"
    _atomic_json(metadata_path, metadata_payload)
    run_id = (
        f"train-{architecture.candidate.candidate_id}-{sha256_file(weights_path)[:12]}"
    )
    result = TrainingRunRecord(
        run_id=run_id,
        dataset_id=manifest.dataset_id,
        dataset_manifest_sha256=manifest_sha256,
        architecture=architecture,
        configuration=configuration,
        normalization=normalization,
        epochs=tuple(history),
        best_epoch=best_epoch,
        best_validation=history[best_epoch - 1].validation,
        weights_relative_path="model.safetensors",
        weights_sha256=sha256_file(weights_path),
        checkpoint_metadata_relative_path="checkpoint.json",
        checkpoint_metadata_sha256=sha256_file(metadata_path),
        parameter_count=parameter_count,
        device=str(device),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda)
        if torch.version.cuda is not None
        else None,
        elapsed_seconds=elapsed,
        smoke_only=manifest.purpose == "integration_smoke",
        scientific_performance_claim_allowed=False,
    )
    _atomic_json(destination / "training_record.json", result.model_dump(mode="json"))
    _atomic_json(
        progress_path,
        {
            "candidate_id": architecture.candidate.candidate_id,
            "epoch": configuration.epochs,
            "epochs": configuration.epochs,
            "validation_balanced_accuracy": result.best_validation.balanced_accuracy,
            "status": "complete",
        },
    )
    for artifact in destination.iterdir():
        if artifact.is_file() and artifact.name != "progress.json":
            artifact.chmod(0o444)
    return result


def evaluate_selected_checkpoint(
    *,
    dataset_root: str | os.PathLike[str],
    manifest: DatasetManifest,
    manifest_sha256: str,
    training_dir: str | os.PathLike[str],
    training_record: TrainingRunRecord,
    output_path: str | os.PathLike[str],
) -> FinalEvaluationRecord:
    """Open the frozen test split once for the code-selected checkpoint."""

    np, torch, load_file, _ = _dependencies()
    _enable_deterministic_torch(torch)
    root = Path(dataset_root).expanduser().resolve()
    train_root = Path(training_dir).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    consumed_marker = destination.with_suffix(destination.suffix + ".consumed")
    if destination.exists() or consumed_marker.exists():
        raise TrainingBackendError("final test evaluation has already been consumed")
    if manifest_sha256 != training_record.dataset_manifest_sha256:
        raise TrainingBackendError("checkpoint and test manifest identities differ")
    weights_path = _safe_path(train_root, training_record.weights_relative_path)
    metadata_path = _safe_path(
        train_root, training_record.checkpoint_metadata_relative_path
    )
    if sha256_file(weights_path) != training_record.weights_sha256:
        raise TrainingBackendError("checkpoint weight digest changed")
    if sha256_file(metadata_path) != training_record.checkpoint_metadata_sha256:
        raise TrainingBackendError("checkpoint metadata digest changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_manifest_sha256") != manifest_sha256:
        raise TrainingBackendError("checkpoint metadata is bound to another dataset")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if training_record.configuration.require_cuda and device.type != "cuda":
        raise TrainingBackendError("CUDA was required for final evaluation")
    model = build_model(
        training_record.architecture,
        dropout=training_record.configuration.dropout,
    ).to(device)
    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    test_x, test_y, sample_ids = _load_split(
        np, root=root, manifest=manifest, split="test"
    )
    mean = np.asarray(training_record.normalization.mean, dtype=np.float32).reshape(
        1, -1, 1, 1
    )
    std = np.asarray(
        training_record.normalization.standard_deviation, dtype=np.float32
    ).reshape(1, -1, 1, 1)
    metrics, probabilities = _evaluate(
        np,
        torch,
        model,
        test_x,
        test_y,
        device=device,
        mean=mean,
        std=std,
    )
    predictions = tuple(
        PredictionRecord(
            sample_id=sample_id,
            true_label=int(label),
            predicted_label=int(probability.argmax()),
            non_lens_score=float(probability[0]),
            lens_score=float(probability[1]),
        )
        for sample_id, label, probability in zip(
            sample_ids, test_y, probabilities, strict=True
        )
    )
    evaluation = FinalEvaluationRecord(
        evaluation_id=f"test-{training_record.run_id}",
        selected_training_run_id=training_record.run_id,
        dataset_manifest_sha256=manifest_sha256,
        split="test",
        metrics=metrics,
        predictions=predictions,
        checkpoint_sha256=training_record.weights_sha256,
        smoke_only=manifest.purpose == "integration_smoke",
        scientific_performance_claim_allowed=False,
    )
    _atomic_json(destination, evaluation.model_dump(mode="json"))
    _atomic_json(
        consumed_marker,
        {
            "evaluation_sha256": sha256_file(destination),
            "checkpoint_sha256": training_record.weights_sha256,
        },
    )
    destination.chmod(0o444)
    consumed_marker.chmod(0o444)
    return evaluation
