"""Compatibility facade for the model contract and adapter registry.

New code should import the typed contracts from :mod:`ripple.modeling`. This
module keeps the original ``ModelInterface`` import useful without implying
that an unqualified classifier is available.
"""

from __future__ import annotations

from pathlib import Path

from ripple.modeling.contracts import ModelManifestRef
from ripple.modeling.registry import ModelAdapterRegistry, ResolvedRegistration
from ripple.modeling.service import (
    CompletedPreprocessingRun,
    build_default_registry,
    find_manifest_reference,
    run_registered_preprocessing,
)


class ModelInterface:
    """Read manifests and run only registered, qualified preprocessing adapters."""

    def __init__(self, registry: ModelAdapterRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()

    @property
    def registry(self) -> ModelAdapterRegistry:
        return self._registry

    def manifests(self) -> tuple[ModelManifestRef, ...]:
        return self._registry.manifest_references()

    def inspect(self, manifest_id: str) -> ResolvedRegistration:
        reference = find_manifest_reference(self._registry, manifest_id)
        return self._registry.inspect(reference)

    def preprocess(
        self,
        *,
        manifest_id: str,
        package_paths: Path | str | tuple[Path | str, ...],
        output_root: Path,
        require_aligned_shapes: bool = False,
    ) -> CompletedPreprocessingRun:
        return run_registered_preprocessing(
            registry=self._registry,
            manifest_id=manifest_id,
            package_paths=package_paths,
            output_root=output_root,
            require_aligned_shapes=require_aligned_shapes,
            invocation_interface="python_api",
        )


__all__ = ["ModelInterface"]
