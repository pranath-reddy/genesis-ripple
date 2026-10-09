"""Deterministic local-catalog adapter and spherical LensCat matching."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
from typing import Any, Mapping

from ..schemas.common import SkyCoordinate, canonical_json_sha256, utc_now
from ..schemas.lenscat import (
    CatalogRecordClassification,
    LensCatAttempt,
    LensCatAttemptStatus,
    LensCatCatalogProvenance,
    LensCatMatch,
    LensCatQuery,
    LocalLensCatalogSpec,
)


class LensCatAdapterError(RuntimeError):
    """Raised when supplied local catalog evidence cannot be loaded safely."""


def great_circle_separation_arcsec(
    first: SkyCoordinate,
    second: SkyCoordinate,
) -> float:
    """Return the great-circle separation using a stable haversine expression."""

    ra1 = math.radians(first.ra_deg)
    dec1 = math.radians(first.dec_deg)
    ra2 = math.radians(second.ra_deg)
    dec2 = math.radians(second.dec_deg)
    delta_ra = (ra2 - ra1 + math.pi) % (2.0 * math.pi) - math.pi
    delta_dec = dec2 - dec1
    haversine = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    angle = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))
    return math.degrees(angle) * 3600.0


def _read_csv_rows(payload: bytes) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LensCatAdapterError("CSV catalog is not valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise LensCatAdapterError("CSV catalog has no header")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise LensCatAdapterError("CSV catalog contains duplicate column names")
    try:
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise LensCatAdapterError(f"CSV catalog parse failed: {exc}") from exc
    return tuple(reader.fieldnames), rows


def _read_parquet_rows(
    payload: bytes,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise LensCatAdapterError(
            "Parquet catalog requires the optional pyarrow dependency"
        ) from exc
    try:
        table = pq.read_table(pa.BufferReader(payload))
    except Exception as exc:  # pyarrow exposes multiple format-specific errors
        raise LensCatAdapterError(f"Parquet catalog parse failed: {exc}") from exc
    if len(table.column_names) != len(set(table.column_names)):
        raise LensCatAdapterError("Parquet catalog contains duplicate column names")
    return tuple(table.column_names), table.to_pylist()


def _load_catalog(
    spec: LocalLensCatalogSpec,
) -> tuple[list[dict[str, Any]], LensCatCatalogProvenance]:
    try:
        path = Path(spec.path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LensCatAdapterError("configured local catalog does not exist") from exc
    if not path.is_file():
        raise LensCatAdapterError("configured local catalog is not a regular file")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LensCatAdapterError("unable to inspect local catalog") from exc
    if size > spec.max_bytes:
        raise LensCatAdapterError(
            f"catalog size {size} exceeds configured maximum {spec.max_bytes}"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise LensCatAdapterError("unable to read local catalog") from exc
    if len(payload) != size:
        raise LensCatAdapterError("catalog changed while it was being read")

    content_sha256 = hashlib.sha256(payload).hexdigest()
    if spec.expected_sha256 is not None and content_sha256 != spec.expected_sha256:
        raise LensCatAdapterError("catalog content SHA-256 does not match expectation")

    if spec.file_format == "csv":
        columns, rows = _read_csv_rows(payload)
    else:
        columns, rows = _read_parquet_rows(payload)

    required = {
        spec.columns.identifier,
        spec.columns.ra_deg,
        spec.columns.dec_deg,
        spec.columns.status,
    }
    missing = sorted(required - set(columns))
    if missing:
        raise LensCatAdapterError(
            "catalog is missing caller-mapped columns: " + ", ".join(missing)
        )
    mapping_payload = {
        "columns": spec.columns.model_dump(mode="json"),
        "statuses": spec.statuses.model_dump(mode="json"),
    }
    provenance = LensCatCatalogProvenance(
        catalog_id=spec.catalog_id,
        catalog_name=spec.catalog_name,
        catalog_release=spec.catalog_release,
        resolved_local_path=str(path),
        file_format=spec.file_format,
        content_sha256=content_sha256,
        byte_count=len(payload),
        row_count=len(rows),
        mapping_sha256=canonical_json_sha256(mapping_payload),
        loaded_at_utc=utc_now(),
    )
    return rows, provenance


def _required_text(row: Mapping[str, Any], column: str, row_number: int) -> str:
    raw = row.get(column)
    if raw is None:
        raise LensCatAdapterError(
            f"catalog row {row_number} has no value for mapped column {column!r}"
        )
    value = str(raw)
    if not value:
        raise LensCatAdapterError(
            f"catalog row {row_number} has an empty value for mapped column {column!r}"
        )
    return value


def _required_coordinate(
    row: Mapping[str, Any],
    spec: LocalLensCatalogSpec,
    row_number: int,
) -> SkyCoordinate:
    try:
        ra_deg = float(row[spec.columns.ra_deg])
        dec_deg = float(row[spec.columns.dec_deg])
    except (KeyError, TypeError, ValueError) as exc:
        raise LensCatAdapterError(
            f"catalog row {row_number} has a non-numeric sky coordinate"
        ) from exc
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg):
        raise LensCatAdapterError(
            f"catalog row {row_number} has a non-finite sky coordinate"
        )
    try:
        return SkyCoordinate(ra_deg=ra_deg, dec_deg=dec_deg)
    except ValueError as exc:
        raise LensCatAdapterError(
            f"catalog row {row_number} has an out-of-range sky coordinate"
        ) from exc


def _classify_status(
    raw_status: str,
    spec: LocalLensCatalogSpec,
) -> CatalogRecordClassification:
    if raw_status in spec.statuses.confirmed_values:
        return "confirmed"
    if raw_status in spec.statuses.candidate_values:
        return "candidate"
    return "unrecognized"


def _outcome(matches: tuple[LensCatMatch, ...]) -> LensCatAttemptStatus:
    if not matches:
        return "no_match"
    if len(matches) != 1 or matches[0].classification == "unrecognized":
        return "ambiguous"
    if matches[0].classification == "confirmed":
        return "matched_confirmed"
    return "matched_candidate"


def _message(
    status: LensCatAttemptStatus,
    match_count: int,
    radius_arcsec: float,
) -> str:
    if status == "no_match":
        return (
            f"No configured catalog record was found within {radius_arcsec:g} arcsec. "
            "This is not evidence that the target is a non-lens."
        )
    if status == "matched_confirmed":
        return "One nearby record maps to the caller-configured confirmed status."
    if status == "matched_candidate":
        return "One nearby record maps to the caller-configured candidate status."
    return (
        f"The cone contains {match_count} record(s), but the association is ambiguous "
        "because identities compete or a status value is unrecognized."
    )


def match_local_lens_catalog(
    *,
    attempt_id: str,
    spec: LocalLensCatalogSpec,
    query: LensCatQuery,
) -> LensCatAttempt:
    """Load exact local bytes and perform an inclusive spherical cone match."""

    rows, provenance = _load_catalog(spec)
    seen_identifiers: set[str] = set()
    matches: list[LensCatMatch] = []
    for row_number, row in enumerate(rows, start=2 if spec.file_format == "csv" else 1):
        object_id = _required_text(row, spec.columns.identifier, row_number)
        if object_id in seen_identifiers:
            raise LensCatAdapterError(
                f"catalog object identifier {object_id!r} is duplicated"
            )
        seen_identifiers.add(object_id)
        coordinate = _required_coordinate(row, spec, row_number)
        raw_status = _required_text(row, spec.columns.status, row_number)
        separation = great_circle_separation_arcsec(query.coordinate, coordinate)
        if separation <= query.cone_radius_arcsec:
            matches.append(
                LensCatMatch(
                    catalog_object_id=object_id,
                    coordinate=coordinate,
                    separation_arcsec=separation,
                    raw_catalog_status=raw_status,
                    classification=_classify_status(raw_status, spec),
                )
            )
    ordered = tuple(
        sorted(
            matches, key=lambda item: (item.separation_arcsec, item.catalog_object_id)
        )
    )
    status = _outcome(ordered)
    reason_codes: tuple[str, ...]
    if status == "no_match":
        reason_codes = ("no_catalog_record_within_cone",)
    elif status == "ambiguous":
        reason_codes = ("catalog_association_ambiguous",)
    else:
        reason_codes = ()
    return LensCatAttempt(
        attempt_id=attempt_id,
        query=query,
        status=status,
        requested_catalog_id=spec.catalog_id,
        provenance=provenance,
        matches=ordered,
        reason_codes=reason_codes,
        message=_message(status, len(ordered), query.cone_radius_arcsec),
        completed_at_utc=utc_now(),
    )


def unavailable_lenscat_attempt(
    *,
    attempt_id: str,
    query: LensCatQuery,
    requested_catalog_id: str,
    reason_code: str,
    message: str,
) -> LensCatAttempt:
    """Represent inability to perform the required association without fabrication."""

    return LensCatAttempt(
        attempt_id=attempt_id,
        query=query,
        status="unavailable",
        requested_catalog_id=requested_catalog_id,
        provenance=None,
        matches=(),
        reason_codes=(reason_code,),
        message=message,
        completed_at_utc=utc_now(),
    )


def attempt_local_lens_catalog_match(
    *,
    attempt_id: str,
    spec: LocalLensCatalogSpec,
    query: LensCatQuery,
) -> LensCatAttempt:
    """Orchestrator-facing adapter that records load failures as unavailable."""

    try:
        return match_local_lens_catalog(
            attempt_id=attempt_id,
            spec=spec,
            query=query,
        )
    except LensCatAdapterError as exc:
        return unavailable_lenscat_attempt(
            attempt_id=attempt_id,
            query=query,
            requested_catalog_id=spec.catalog_id,
            reason_code="local_catalog_unavailable",
            message=str(exc),
        )
