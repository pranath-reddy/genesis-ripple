"""Lazy exports for deterministic scientific tools.

The offline worker imports individual numerical modules. Keeping this package
initializer lazy prevents control-plane-only repository and provider modules
from becoming worker runtime dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_IMPORTS = {
    "SlsimDependencyError": (".slsim_backend", "SlsimDependencyError"),
    "SlsimGenerationError": (".slsim_backend", "SlsimGenerationError"),
    "SlsimSmokeBackend": (".slsim_backend", "SlsimSmokeBackend"),
    "load_smoke_spec": (".slsim_backend", "load_smoke_spec"),
    "DatasetBuildError": (".dataset_builder", "DatasetBuildError"),
    "build_dataset_manifest": (".dataset_builder", "build_dataset_manifest"),
    "load_dataset_manifest": (".dataset_builder", "load_dataset_manifest"),
    "TrainingBackendError": (".torch_backend", "TrainingBackendError"),
    "evaluate_selected_checkpoint": (
        ".torch_backend",
        "evaluate_selected_checkpoint",
    ),
    "train_candidate": (".torch_backend", "train_candidate"),
    "RemoteExecutionError": (".remote_executor", "RemoteExecutionError"),
    "SourceTreeDigest": (".remote_executor", "SourceTreeDigest"),
    "VerifiedRemoteSources": (".remote_executor", "VerifiedRemoteSources"),
    "copy_remote_output": (".remote_executor", "copy_remote_output"),
    "copy_to_remote_run": (".remote_executor", "copy_to_remote_run"),
    "run_remote_worker": (".remote_executor", "run_remote_worker"),
    "sync_worker_source": (".remote_executor", "sync_worker_source"),
    "RepositoryAcquisitionError": (
        ".repository_intake",
        "RepositoryAcquisitionError",
    ),
    "RepositoryIntegrityError": (
        ".repository_intake",
        "RepositoryIntegrityError",
    ),
    "RepositoryIntakeError": (".repository_intake", "RepositoryIntakeError"),
    "RepositoryPolicyError": (".repository_intake", "RepositoryPolicyError"),
    "create_repository_intake": (
        ".repository_intake",
        "create_repository_intake",
    ),
    "list_repository_files": (".repository_intake", "list_repository_files"),
    "load_repository_intake": (".repository_intake", "load_repository_intake"),
    "read_repository_excerpt": (
        ".repository_intake",
        "read_repository_excerpt",
    ),
    "search_repository": (".repository_intake", "search_repository"),
    "IntakeRepositorySnapshot": (
        ".repository_snapshot",
        "IntakeRepositorySnapshot",
    ),
    "RepositorySnapshotBinding": (
        ".repository_snapshot",
        "RepositorySnapshotBinding",
    ),
    "build_intake_repository_snapshot": (
        ".repository_snapshot",
        "build_intake_repository_snapshot",
    ),
}

__all__ = sorted(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
