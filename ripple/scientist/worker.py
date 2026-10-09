"""Offline deterministic worker entrypoint for simulation and CUDA execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

_PROCESS_STARTED = time.monotonic()

# Must be set before the worker lazily imports Torch. The training backend also
# validates this value before any CUDA training begins.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

from .artifacts import sha256_file  # noqa: E402
from .schemas.architecture import (  # noqa: E402
    BoundArchitecture,
    TrainingConfiguration,
)
from .schemas.training import TrainingRunRecord  # noqa: E402
from .tools.dataset_builder import (  # noqa: E402
    build_dataset_manifest,
    load_dataset_manifest,
)
from .tools.slsim_backend import (  # noqa: E402
    SlsimBackend,
    load_slsim_spec,
)
from .tools.torch_backend import (  # noqa: E402
    evaluate_selected_checkpoint,
    train_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ripple-scientist-worker")
    commands = parser.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("environment")
    environment.add_argument("--require-cuda", action="store_true")
    environment.add_argument("--require-scientific-stack", action="store_true")

    simulate = commands.add_parser("simulate")
    simulate.add_argument("--spec", required=True)
    simulate.add_argument("--output-dir", required=True)

    dataset = commands.add_parser("build-dataset")
    dataset.add_argument("--simulation-root", required=True)

    train = commands.add_parser("train")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--architecture", required=True)
    train.add_argument("--training-config", required=True)
    train.add_argument("--output-dir", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--training-dir", required=True)
    evaluate.add_argument("--training-record", required=True)
    evaluate.add_argument("--output", required=True)
    return parser


def _environment(
    *, require_cuda: bool, require_scientific_stack: bool
) -> dict[str, object]:
    import platform

    import numpy
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required but unavailable")
    result: dict[str, object] = {
        "worker_source": str(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "timeout_available": shutil.which("timeout") is not None,
    }
    if require_scientific_stack:
        import astropy
        import jaxtronomy
        import lenstronomy
        import pydantic
        import safetensors
        import slsim

        result["scientific_stack"] = {
            "astropy": astropy.__version__,
            "jaxtronomy_source": str(Path(jaxtronomy.__file__).resolve()),
            "lenstronomy": lenstronomy.__version__,
            "pydantic": pydantic.__version__,
            "safetensors": safetensors.__version__,
            "slsim_source": str(Path(slsim.__file__).resolve()),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    source_manifest_sha256 = os.environ.get(
        "RIPPLE_VERIFIED_SOURCE_MANIFEST_SHA256", ""
    )
    if re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None:
        raise RuntimeError("worker source manifest was not verified by the controller")
    arguments = _parser().parse_args(argv)
    if arguments.command == "environment":
        result = _environment(
            require_cuda=arguments.require_cuda,
            require_scientific_stack=arguments.require_scientific_stack,
        )
    elif arguments.command == "simulate":
        record = SlsimBackend().generate(
            spec=load_slsim_spec(arguments.spec),
            output_dir=arguments.output_dir,
        )
        result = {
            "dataset_id": record.dataset_id,
            "manifest": str(Path(arguments.output_dir) / record.manifest_relative_path),
            "sample_count": record.total_count,
            "supports_scientific_claims": record.supports_scientific_claims,
        }
    elif arguments.command == "build-dataset":
        manifest = build_dataset_manifest(simulation_root=arguments.simulation_root)
        path = Path(arguments.simulation_root) / "dataset_manifest.json"
        result = {
            "dataset_id": manifest.dataset_id,
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "split_counts": {
                "train": len(manifest.splits.train_sample_ids),
                "validation": len(manifest.splits.validation_sample_ids),
                "test": len(manifest.splits.test_sample_ids),
            },
        }
    elif arguments.command == "train":
        manifest_path = Path(arguments.manifest)
        manifest = load_dataset_manifest(manifest_path)
        architecture = BoundArchitecture.model_validate_json(
            Path(arguments.architecture).read_bytes(), strict=True
        )
        configuration = TrainingConfiguration.model_validate_json(
            Path(arguments.training_config).read_bytes(), strict=True
        )
        record = train_candidate(
            dataset_root=arguments.dataset_root,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            architecture=architecture,
            configuration=configuration,
            output_dir=arguments.output_dir,
        )
        result = {
            "run_id": record.run_id,
            "record": str(Path(arguments.output_dir) / "training_record.json"),
            "weights_sha256": record.weights_sha256,
            "validation_balanced_accuracy": record.best_validation.balanced_accuracy,
            "smoke_only": record.smoke_only,
        }
    elif arguments.command == "evaluate":
        manifest_path = Path(arguments.manifest)
        manifest = load_dataset_manifest(manifest_path)
        training_record = TrainingRunRecord.model_validate_json(
            Path(arguments.training_record).read_bytes(), strict=True
        )
        evaluation_started = time.monotonic()
        evaluation = evaluate_selected_checkpoint(
            dataset_root=arguments.dataset_root,
            manifest=manifest,
            manifest_sha256=sha256_file(manifest_path),
            training_dir=arguments.training_dir,
            training_record=training_record,
            output_path=arguments.output,
        )
        evaluation_elapsed_seconds = time.monotonic() - evaluation_started
        result = {
            "evaluation_id": evaluation.evaluation_id,
            "output": arguments.output,
            "accuracy": evaluation.metrics.accuracy,
            "balanced_accuracy": evaluation.metrics.balanced_accuracy,
            "elapsed_seconds": evaluation_elapsed_seconds,
            "smoke_only": evaluation.smoke_only,
        }
    else:  # pragma: no cover - argparse prevents this path.
        raise RuntimeError("unknown worker command")
    result["verified_source_manifest_sha256"] = source_manifest_sha256
    result["worker_elapsed_seconds"] = time.monotonic() - _PROCESS_STARTED
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
