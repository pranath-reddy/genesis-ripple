"""Safe, typed failures for external Rubin DP2 access."""

from __future__ import annotations


class Dp2Error(RuntimeError):
    """Base error whose public message is safe to write to evidence artifacts."""

    def __init__(
        self,
        *,
        stage: str,
        code: str,
        message: str,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.safe_message = message
        self.http_status = http_status


class Dp2ConfigurationError(Dp2Error):
    """The local, secret-free client configuration is invalid."""


class Dp2AuthenticationError(Dp2Error):
    """Rubin rejected or could not authorize the supplied credential."""


class Dp2NoMatchError(Dp2Error):
    """The bounded SIA query returned no eligible image."""


class Dp2SelectionError(Dp2Error):
    """The SIA results could not be selected deterministically."""


class Dp2ProtocolError(Dp2Error):
    """A SIA, DataLink, or SODA response violated the expected protocol."""


class Dp2DownloadError(Dp2Error):
    """The cutout could not be downloaded safely."""


class Dp2FitsValidationError(Dp2Error):
    """The downloaded artifact was not an acceptable FITS cutout."""
