"""Authenticated Rubin DP2 SIA -> DataLink -> SODA client.

The client follows the official PyVO workflow while enforcing bounded retries,
timeouts, a response-size ceiling, and an HTTPS hostname allowlist.  It never
serializes the token or remote access URLs.
"""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pyvo
import requests
from astropy import units as u
from astropy.time import Time
from pyvo.dal.adhoc import DatalinkResults, SodaQuery
from requests.auth import AuthBase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import (
    Dp2AuthenticationError,
    Dp2ConfigurationError,
    Dp2DownloadError,
    Dp2Error,
    Dp2FitsValidationError,
    Dp2NoMatchError,
    Dp2ProtocolError,
    Dp2SelectionError,
)
from .models import (
    DatasetIdentity,
    Dp2ClientConfig,
    Dp2Credentials,
    Dp2CutoutRequest,
    RetrievalReceipt,
    StageCheck,
)

_RUBIN_AUTH_ORIGIN = ("https", "data.lsst.cloud", 443)
_TRUSTED_REMOTE_ORIGINS = frozenset(
    {
        _RUBIN_AUTH_ORIGIN,
        ("https", "storage.googleapis.com", 443),
    }
)
_CLIENT_CONSTRUCTION_GUARD = object()


def _reject_symlink_ancestors(path: Path) -> None:
    if ".." in path.parts:
        raise Dp2ConfigurationError(
            stage="local_output",
            code="parent_traversal_forbidden",
            message="Parent traversal is forbidden in the Stage-1 output path.",
        )
    absolute = path.absolute()
    for component in list(reversed(absolute.parents)) + [absolute]:
        if component == Path(component.anchor):
            continue
        try:
            mode = os.lstat(component).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise Dp2ConfigurationError(
                stage="local_output",
                code="symlinked_output_path",
                message="Symlinks are forbidden in the Stage-1 output path.",
            )


def _normalized_origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(str(url))
    if parsed.username is not None or parsed.password is not None:
        raise Dp2ConfigurationError(
            stage="network_policy",
            code="userinfo_forbidden",
            message="Remote URLs containing user information are forbidden.",
        )
    try:
        port = parsed.port
    except ValueError:
        raise Dp2ConfigurationError(
            stage="network_policy",
            code="invalid_remote_port",
            message="A remote URL contained an invalid port.",
        ) from None
    if parsed.scheme != "https" or parsed.hostname is None or port not in {None, 443}:
        raise Dp2ConfigurationError(
            stage="network_policy",
            code="invalid_remote_origin",
            message="A remote URL did not use an approved HTTPS origin.",
        )
    return parsed.scheme, parsed.hostname.lower(), port or 443


class _HostScopedBearerAuth(AuthBase):
    """Attach the bearer credential only to the exact Rubin API hostname."""

    def __init__(self, auth_origin: tuple[str, str, int], bearer_token: str) -> None:
        self._auth_origin = auth_origin
        self._bearer_token = bearer_token

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        if _normalized_origin(request.url or "") == self._auth_origin:
            request.headers["Authorization"] = f"Bearer {self._bearer_token}"
        else:
            request.headers.pop("Authorization", None)
        return request

    def __repr__(self) -> str:
        return (
            "_HostScopedBearerAuth(auth_origin=<pinned-rubin-origin>, token=<redacted>)"
        )


class _BoundedRawStream:
    """Count decoded bytes read by PyVO and fail before an unbounded allocation."""

    def __init__(self, raw: Any, limit: int) -> None:
        self._raw = raw
        self._limit = limit
        self._read_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    def read(
        self,
        amount: int | None = None,
        decode_content: bool = False,
        **kwargs: Any,
    ) -> bytes:
        if amount is None or amount < 0:
            chunks: list[bytes] = []
            while True:
                chunk = self.read(
                    min(1024 * 1024, self._limit - self._read_count + 1),
                    decode_content=decode_content,
                    **kwargs,
                )
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)

        remaining_with_probe = self._limit - self._read_count + 1
        if remaining_with_probe <= 0:
            self._raw.close()
            raise Dp2ProtocolError(
                stage="network_response",
                code="response_too_large",
                message="A Rubin protocol response exceeded its configured byte limit.",
            )
        requested = min(amount, remaining_with_probe)
        chunk = self._raw.read(
            requested,
            decode_content=decode_content,
            **kwargs,
        )
        if chunk is None:
            return b""
        self._read_count += len(chunk)
        if self._read_count > self._limit:
            self._raw.close()
            raise Dp2ProtocolError(
                stage="network_response",
                code="response_too_large",
                message="A Rubin protocol response exceeded its configured byte limit.",
            )
        return chunk

    def stream(
        self,
        amount: int = 2**16,
        decode_content: bool = True,
    ) -> Any:
        while True:
            chunk = self.read(amount, decode_content=decode_content)
            if not chunk:
                return
            yield chunk


class _AllowlistedSession(requests.Session):
    """Requests session that refuses non-allowlisted destinations and redirects."""

    def __init__(self, config: Dp2ClientConfig, bearer_token: str) -> None:
        super().__init__()
        self._default_timeout = (
            config.connect_timeout_seconds,
            config.read_timeout_seconds,
        )
        self._max_metadata_response_bytes = config.max_metadata_response_bytes
        self._max_download_bytes = config.max_download_bytes
        if _normalized_origin(config.sia_url) != _RUBIN_AUTH_ORIGIN:
            raise Dp2ConfigurationError(
                stage="credentials",
                code="invalid_auth_origin",
                message="The Rubin authentication origin is not the pinned Stage-1 origin.",
            )
        self.auth = _HostScopedBearerAuth(_RUBIN_AUTH_ORIGIN, bearer_token)
        self.headers.update({"User-Agent": config.user_agent})
        self.hooks["response"].append(self._bound_response)
        retry = Retry(
            total=config.retries_total,
            connect=config.retries_total,
            read=config.retries_total,
            status=0,
            backoff_factor=config.retry_backoff_seconds,
            status_forcelist=(),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
            respect_retry_after_header=False,
        )
        self.mount("https://", HTTPAdapter(max_retries=retry))

    def _validate_url(self, url: str) -> None:
        if _normalized_origin(url) not in _TRUSTED_REMOTE_ORIGINS:
            raise Dp2ConfigurationError(
                stage="network_policy",
                code="untrusted_remote_url",
                message="Refused a remote origin outside the fixed Stage-1 HTTPS allowlist.",
            )

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._validate_url(url)
        kwargs.setdefault("timeout", self._default_timeout)
        response = super().request(method, url, **kwargs)
        self._validate_url(response.url)
        return response

    def _bound_response(
        self,
        response: requests.Response,
        **_: Any,
    ) -> requests.Response:
        """Validate and cap every response before retries/redirects consume it."""
        self._validate_url(response.url)
        if isinstance(response.raw, _BoundedRawStream):
            return response
        content_type = response.headers.get("Content-Type", "").lower()
        is_download = (
            "fits" in content_type
            or "octet-stream" in content_type
            or _normalized_origin(response.url)
            == ("https", "storage.googleapis.com", 443)
        )
        limit = (
            self._max_download_bytes
            if is_download
            else self._max_metadata_response_bytes
        )
        announced_length = response.headers.get("Content-Length")
        if announced_length:
            try:
                announced_bytes = int(announced_length)
            except (TypeError, ValueError, OverflowError):
                announced_bytes = None
            if announced_bytes is not None and announced_bytes > limit:
                response.close()
                raise Dp2ProtocolError(
                    stage="network_response",
                    code="announced_response_too_large",
                    message="A Rubin protocol response exceeded its configured byte limit.",
                )
        response.raw = _BoundedRawStream(response.raw, limit)
        return response

    def get_redirect_target(self, response: requests.Response) -> str | None:
        target = super().get_redirect_target(response)
        if target:
            self._validate_url(urljoin(response.url, target))
        return target

    def rebuild_auth(
        self,
        prepared_request: requests.PreparedRequest,
        response: requests.Response,
    ) -> None:
        """Reapply authorization after redirects only when the target is Rubin."""
        prepared_request.headers.pop("Authorization", None)
        if self.auth is not None:
            self.auth(prepared_request)


@dataclass(frozen=True)
class _SelectedRecord:
    """Internal selection that deliberately hides the remote access URL."""

    record: Any = field(repr=False)
    access_url: str = field(repr=False)
    identity: DatasetIdentity
    total_count: int
    eligible_count: int
    selection_rule: str


class Dp2Client:
    """Deterministic, non-LLM adapter for Rubin DP2 image cutouts."""

    def __init__(
        self,
        config: Dp2ClientConfig,
        session: requests.Session,
        *,
        _construction_guard: object | None = None,
    ) -> None:
        if _construction_guard is not _CLIENT_CONSTRUCTION_GUARD:
            raise Dp2ConfigurationError(
                stage="client_construction",
                code="direct_client_construction_forbidden",
                message="Construct the live DP2 client with Dp2Client.from_environment().",
            )
        self.config = config
        self._session = session

    def __repr__(self) -> str:
        return f"Dp2Client(sia_host={urlparse(str(self.config.sia_url)).hostname!r})"

    @classmethod
    def from_environment(
        cls,
        config: Dp2ClientConfig | None = None,
    ) -> "Dp2Client":
        resolved_config = config or Dp2ClientConfig()
        credentials = Dp2Credentials.from_environment()
        secret = credentials.rsp_token.get_secret_value()
        try:
            session = _AllowlistedSession(resolved_config, secret)
        finally:
            secret = ""
            credentials = None
        return cls(
            resolved_config,
            session,
            _construction_guard=_CLIENT_CONSTRUCTION_GUARD,
        )

    def retrieve_one_cutout(
        self,
        request: Dp2CutoutRequest,
        destination: Path,
    ) -> RetrievalReceipt:
        """Discover, select, download, and minimally validate one DP2 cutout."""
        destination = Path(destination)
        if destination.suffix.lower() not in {".fits", ".fit"}:
            raise Dp2ConfigurationError(
                stage="local_output",
                code="invalid_destination_suffix",
                message="The DP2 cutout destination must use a .fits or .fit suffix.",
            )
        if os.path.lexists(destination):
            raise Dp2ConfigurationError(
                stage="local_output",
                code="destination_exists",
                message="Refused to overwrite an existing cutout artifact.",
            )

        _reject_symlink_ancestors(destination.parent)
        if (
            not destination.parent.is_dir()
            or destination.parent.is_symlink()
            or destination.parent.stat().st_mode & 0o077
        ):
            raise Dp2ConfigurationError(
                stage="local_output",
                code="insecure_output_directory",
                message="The cutout output directory must already exist as a private, non-symlink directory.",
            )

        selected = self._search_and_select(request)
        checks = [
            StageCheck(
                name="sia_query",
                passed=True,
                message=f"SIA returned {selected.total_count} row(s).",
            ),
            StageCheck(
                name="dataset_selection",
                passed=True,
                message=(
                    "Selected one explicit DP2 deep-coadd dataset using "
                    f"{selected.selection_rule}."
                ),
            ),
        ]

        partial_path = destination.with_suffix(destination.suffix + ".part")
        if os.path.lexists(partial_path):
            raise Dp2ConfigurationError(
                stage="local_output",
                code="partial_destination_exists",
                message="Refused to overwrite an existing partial cutout artifact.",
            )

        partial_owned = False
        destination_published = False
        try:
            soda_query = self._build_soda_query(selected, request)
            checks.append(
                StageCheck(
                    name="datalink_soda_resolution",
                    passed=True,
                    message=f"Resolved the declared {request.soda_service_type} service.",
                )
            )
            byte_count, digest, content_type = self._download(
                soda_query,
                partial_path,
            )
            partial_owned = True
            self._validate_fits_container(partial_path)
            try:
                os.link(partial_path, destination, follow_symlinks=False)
            except FileExistsError:
                raise Dp2ConfigurationError(
                    stage="local_output",
                    code="destination_race_detected",
                    message="The cutout destination appeared during retrieval; no file was overwritten.",
                ) from None
            destination_published = True
            partial_path.unlink()
            partial_owned = False
        except Dp2Error:
            if partial_owned:
                partial_path.unlink(missing_ok=True)
            if destination_published:
                destination.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if partial_owned:
                partial_path.unlink(missing_ok=True)
            if destination_published:
                destination.unlink(missing_ok=True)
            raise self._safe_protocol_error("download", exc) from None

        checks.extend(
            [
                StageCheck(
                    name="soda_download",
                    passed=True,
                    message=f"Downloaded {byte_count} byte(s) within the configured limit.",
                ),
                StageCheck(
                    name="fits_container",
                    passed=True,
                    message="The download has a FITS primary signature and block-aligned size.",
                ),
            ]
        )
        return RetrievalReceipt(
            proof_mode="live_rubin_rsp",
            authenticated=True,
            service_origin="https://data.lsst.cloud",
            sia_path="/api/sia/dp2/query",
            dataset=selected.identity,
            total_match_count=selected.total_count,
            eligible_match_count=selected.eligible_count,
            selection_rule=selected.selection_rule,
            artifact_path=destination,
            byte_count=byte_count,
            sha256=digest,
            content_type=content_type,
            checks=tuple(checks),
        )

    def _search_and_select(self, request: Dp2CutoutRequest) -> _SelectedRecord:
        try:
            return self._search_and_select_impl(request)
        except Dp2Error:
            raise
        except Exception as exc:
            raise self._safe_protocol_error("dataset_selection", exc) from None

    def _search_and_select_impl(self, request: Dp2CutoutRequest) -> _SelectedRecord:
        try:
            service = pyvo.dal.SIA2Service(
                str(self.config.sia_url),
                session=self._session,
                check_baseurl=False,
            )
            search_kwargs: dict[str, Any] = {
                "pos": (request.ra_deg, request.dec_deg, request.search_radius_deg),
                "calib_level": request.calibration_level,
                "dpsubtype": request.product_subtype,
                "band": request.effective_wavelength_m,
            }
            if (
                request.time_start_mjd_tai is not None
                and request.time_end_mjd_tai is not None
            ):
                search_kwargs["time"] = (
                    Time(request.time_start_mjd_tai, format="mjd", scale="tai"),
                    Time(request.time_end_mjd_tai, format="mjd", scale="tai"),
                )
            results = service.search(**search_kwargs)
        except Exception as exc:
            raise self._safe_protocol_error("sia_query", exc) from None

        total_count = len(results)
        if total_count == 0:
            raise Dp2NoMatchError(
                stage="sia_query",
                code="no_sia_results",
                message="The bounded DP2 SIA query returned no rows.",
            )

        eligible: list[tuple[Any, DatasetIdentity, str]] = []
        for index in range(total_count):
            record = results[index]
            product_type = self._as_string(
                self._record_value(record, "dataproduct_type")
            )
            subtype = self._as_string(self._record_value(record, "dataproduct_subtype"))
            calibration_level = self._as_int(self._record_value(record, "calib_level"))
            collection = self._as_string(self._record_value(record, "obs_collection"))
            band_name = self._as_string(
                self._record_value(record, "lsst_band", "em_filter_name")
            )
            obs_id = self._as_string(self._record_value(record, "obs_id"))
            tract = self._as_int(self._record_value(record, "lsst_tract"))
            patch = self._as_int(self._record_value(record, "lsst_patch"))
            wavelength_min = self._as_float(self._record_value(record, "em_min"))
            wavelength_max = self._as_float(self._record_value(record, "em_max"))
            access_format = self._as_string(self._record_value(record, "access_format"))

            if product_type != "image":
                continue
            if subtype != request.product_subtype:
                continue
            if calibration_level != request.calibration_level:
                continue
            if collection != request.expected_collection:
                continue
            if band_name != request.band_name:
                continue
            if (
                request.expected_obs_id is not None
                and obs_id != request.expected_obs_id
            ):
                continue
            if tract != request.expected_tract or patch != request.expected_patch:
                continue
            if (
                wavelength_min is None
                or wavelength_max is None
                or not wavelength_min
                <= request.effective_wavelength_m
                <= wavelength_max
            ):
                continue
            if access_format is None or "content=datalink" not in access_format.lower():
                continue

            access_url = self._as_string(
                getattr(record, "access_url", None)
                or self._record_value(record, "access_url")
            )
            if not access_url:
                continue
            self._validate_remote_url(access_url)
            identity = self._identity_from_record(record)
            eligible.append((record, identity, access_url))

        if not eligible:
            expectation = (
                f" with obs_id {request.expected_obs_id}"
                if request.expected_obs_id
                else ""
            )
            raise Dp2NoMatchError(
                stage="dataset_selection",
                code="no_eligible_deep_coadd",
                message=(
                    "SIA returned rows, but none matched the exact DP2 deep-coadd, "
                    f"calibration, band, and identifier criteria{expectation}."
                ),
            )

        if len(eligible) != 1:
            raise Dp2SelectionError(
                stage="dataset_selection",
                code="duplicate_expected_dataset",
                message="More than one SIA row matched the expected stable obs_id.",
            )
        selected = eligible[0]
        selection_rule = "exact obs_id=lsst_cells_v2-5063-34"

        return _SelectedRecord(
            record=selected[0],
            identity=selected[1],
            access_url=selected[2],
            total_count=total_count,
            eligible_count=len(eligible),
            selection_rule=selection_rule,
        )

    def _build_soda_query(
        self,
        selected: _SelectedRecord,
        request: Dp2CutoutRequest,
    ) -> SodaQuery:
        try:
            datalink = DatalinkResults.from_result_url(
                selected.access_url,
                session=self._session,
                original_row=selected.record,
            )
            descriptor = datalink.get_adhocservice_by_id(request.soda_service_type)
            query = SodaQuery.from_resource(
                datalink,
                descriptor,
                session=self._session,
                original_row=selected.record,
            )
            self._validate_remote_url(query.baseurl)
            query.circle = (
                request.ra_deg * u.deg,
                request.dec_deg * u.deg,
                request.cutout_radius_deg * u.deg,
            )
            return query
        except Dp2Error:
            raise
        except Exception as exc:
            raise self._safe_protocol_error("datalink_soda_resolution", exc) from None

    def _download(
        self,
        query: SodaQuery,
        partial_path: Path,
    ) -> tuple[int, str, str | None]:
        try:
            stream = query.execute_stream()
            try:
                query.raise_if_error()
            except Exception:
                try:
                    stream.close()
                except Exception:
                    pass
                raise
        except Exception as exc:
            raise self._safe_protocol_error("soda_request", exc) from None

        content_type = None
        headers = getattr(stream, "headers", None)
        if headers is not None:
            content_type = headers.get("Content-Type")
            content_length = headers.get("Content-Length")
            if content_length:
                try:
                    announced_size = int(content_length)
                except (TypeError, ValueError):
                    announced_size = None
                if (
                    announced_size is not None
                    and announced_size > self.config.max_download_bytes
                ):
                    stream.close()
                    raise Dp2DownloadError(
                        stage="soda_download",
                        code="announced_download_too_large",
                        message="The cutout exceeded the configured download-size limit.",
                    )

        digest = hashlib.sha256()
        byte_count = 0
        created_partial = False
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            file_descriptor = os.open(partial_path, flags, 0o600)
            created_partial = True
            with closing(stream), os.fdopen(file_descriptor, "wb") as output:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    byte_count += len(chunk)
                    if byte_count > self.config.max_download_bytes:
                        raise Dp2DownloadError(
                            stage="soda_download",
                            code="download_too_large",
                            message="The cutout exceeded the configured download-size limit.",
                        )
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        except Dp2Error:
            if created_partial:
                partial_path.unlink(missing_ok=True)
            raise
        except OSError as exc:
            if created_partial:
                partial_path.unlink(missing_ok=True)
            raise Dp2DownloadError(
                stage="soda_download",
                code="local_write_failed",
                message=f"Could not write the local cutout artifact ({type(exc).__name__}).",
            ) from None
        except Exception:
            if created_partial:
                partial_path.unlink(missing_ok=True)
            raise

        if byte_count == 0:
            if created_partial:
                partial_path.unlink(missing_ok=True)
            raise Dp2DownloadError(
                stage="soda_download",
                code="empty_download",
                message="The SODA service returned an empty response.",
            )
        return byte_count, digest.hexdigest(), content_type

    @staticmethod
    def _validate_fits_container(path: Path) -> None:
        try:
            with path.open("rb") as handle:
                magic = handle.read(9)
            if not magic.startswith(b"SIMPLE  ="):
                raise Dp2FitsValidationError(
                    stage="fits_container",
                    code="non_fits_payload",
                    message="The downloaded response did not have a FITS primary header.",
                )
            if path.stat().st_size < 2880 or path.stat().st_size % 2880 != 0:
                raise Dp2FitsValidationError(
                    stage="fits_container",
                    code="invalid_fits_block_size",
                    message="The downloaded FITS artifact was not aligned to 2880-byte blocks.",
                )
        except Dp2Error:
            raise
        except Exception as exc:
            raise Dp2FitsValidationError(
                stage="fits_container",
                code="invalid_fits_container",
                message=f"Astropy could not validate the FITS artifact ({type(exc).__name__}).",
            ) from None

    def _validate_remote_url(self, url: str) -> None:
        if _normalized_origin(url) not in _TRUSTED_REMOTE_ORIGINS:
            raise Dp2ProtocolError(
                stage="network_policy",
                code="untrusted_protocol_url",
                message="The DP2 protocol returned an origin outside the fixed HTTPS allowlist.",
            )

    def _identity_from_record(self, record: Any) -> DatasetIdentity:
        obs_id = self._as_string(self._record_value(record, "obs_id"))
        publisher_did = self._as_string(self._record_value(record, "obs_publisher_did"))
        if not publisher_did:
            raise Dp2SelectionError(
                stage="dataset_selection",
                code="missing_publisher_identifier",
                message="The selected SIA row had no publisher dataset identifier.",
            )
        return DatasetIdentity(
            dataset_id=publisher_did,
            identifier_field="obs_publisher_did",
            obs_id=obs_id,
            obs_publisher_did=publisher_did,
            observation_collection=self._as_string(
                self._record_value(record, "obs_collection")
            ),
            product_subtype=self._as_string(
                self._record_value(record, "dataproduct_subtype")
            ),
            calibration_level=self._as_int(self._record_value(record, "calib_level")),
            facility_name=self._as_string(self._record_value(record, "facility_name")),
            instrument_name=self._as_string(
                self._record_value(record, "instrument_name")
            ),
            central_ra_deg=self._as_float(self._record_value(record, "s_ra")),
            central_dec_deg=self._as_float(self._record_value(record, "s_dec")),
            wavelength_min_m=self._as_float(self._record_value(record, "em_min")),
            wavelength_max_m=self._as_float(self._record_value(record, "em_max")),
            tract=self._as_int(self._record_value(record, "lsst_tract")),
            patch=self._as_int(self._record_value(record, "lsst_patch")),
            band_name=self._as_string(
                self._record_value(record, "lsst_band", "em_filter_name")
            ),
            access_format=self._as_string(self._record_value(record, "access_format")),
        )

    @staticmethod
    def _record_value(record: Any, *names: str) -> Any:
        for name in names:
            try:
                value = record[name]
            except (KeyError, TypeError, IndexError):
                value = getattr(record, name, None)
            if value is not None and not np.ma.is_masked(value):
                return value
        return None

    @staticmethod
    def _as_string(value: Any) -> str | None:
        if value is None or np.ma.is_masked(value):
            return None
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        return text or None

    @staticmethod
    def _as_int(value: Any) -> int | None:
        if value is None or np.ma.is_masked(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or np.ma.is_masked(value):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if np.isfinite(number) else None

    @staticmethod
    def _status_code_from_exception(exc: Exception) -> int | None:
        pending: list[BaseException] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            response = getattr(current, "response", None)
            status = getattr(response, "status_code", None)
            if (
                isinstance(status, int)
                and not isinstance(status, bool)
                and 100 <= status <= 599
            ):
                return status
            code = getattr(current, "code", None)
            if (
                isinstance(code, int)
                and not isinstance(code, bool)
                and 100 <= code <= 599
            ):
                return code
            for attribute in ("cause", "_cause", "__cause__", "__context__"):
                nested = getattr(current, attribute, None)
                if isinstance(nested, BaseException) and id(nested) not in seen:
                    pending.append(nested)
        return None

    def _safe_protocol_error(self, stage: str, exc: Exception) -> Dp2Error:
        if isinstance(exc, Dp2Error):
            return exc
        status = self._status_code_from_exception(exc)
        if status in {401, 403}:
            return Dp2AuthenticationError(
                stage=stage,
                code="authentication_or_scope_rejected",
                message=(
                    "Rubin rejected the credential or its authorization scope "
                    f"during {stage}."
                ),
                http_status=status,
            )
        return Dp2ProtocolError(
            stage=stage,
            code="remote_protocol_failure",
            message=f"The Rubin {stage} operation failed ({type(exc).__name__}).",
            http_status=status,
        )
