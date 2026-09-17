"""RIPPLe: Rubin Image Preparation and Processing Lensing engine.

The legacy Butler implementation depends on the LSST Science Pipelines.  Those
imports are intentionally lazy so lightweight clients, such as the external DP2
API adapter, can run in an ordinary Python environment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "1.0.0-dev"
__author__ = "Kartik Mandar"
__email__ = "kartik4321mandar@gmail.com"

__all__ = [
    "LsstDataFetcher",
    "Preprocessor",
    "PipelineOrchestrator",
    "ModelInterface",
]


_LAZY_IMPORTS = {
    "LsstDataFetcher": (".data_access", "LsstDataFetcher"),
    "Preprocessor": (".preprocessing", "Preprocessor"),
    "PipelineOrchestrator": (".pipeline", "PipelineOrchestrator"),
    "ModelInterface": (".models", "ModelInterface"),
}


def __getattr__(name: str) -> Any:
    """Load LSST-dependent legacy objects only when they are requested."""
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
