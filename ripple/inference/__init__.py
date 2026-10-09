"""Runtime inference implementations with optional heavy dependencies.

Submodules are intentionally not imported here.  Importing ``ripple.inference``
therefore does not require PyTorch; callers that need the Mriganka ENN runtime
must import :mod:`ripple.inference.mriganka_enn` explicitly.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
