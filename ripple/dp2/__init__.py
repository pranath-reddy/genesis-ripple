"""Lightweight external Rubin DP2 access for RIPPLe.

This package has no dependency on the LSST Science Pipelines and no LLM/API-key
dependency.  DP2 retrieval requires only an RSP token supplied through
``RSP_TOKEN``.
"""

from .client import Dp2Client
from .models import (
    DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND,
    DP2_SIA_BAND_EDGES_M_BY_BAND,
    Dp2Band,
    Dp2ClientConfig,
    Dp2CutoutRequest,
    Dp2SmokeEvidence,
)
from .package_models import Dp2CutoutPackage
from .package_service import Dp2PackageService, LoadedDp2Cutout, load_cutout_package
from .service import Dp2SmokeService

__all__ = [
    "DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND",
    "DP2_SIA_BAND_EDGES_M_BY_BAND",
    "Dp2Band",
    "Dp2Client",
    "Dp2ClientConfig",
    "Dp2CutoutPackage",
    "Dp2CutoutRequest",
    "Dp2PackageService",
    "Dp2SmokeEvidence",
    "Dp2SmokeService",
    "LoadedDp2Cutout",
    "load_cutout_package",
]
