"""Typed contracts for local LensCat association attempts.

LensCat is treated as catalog evidence, never as a classifier or ground-truth
label.  A catalog source and its semantic column mapping must be supplied by
the caller; this module does not assume a remote service or a particular
catalog release.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import (
    FrozenModel,
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    SkyCoordinate,
)


LensCatAttemptStatus = Literal[
    "matched_confirmed",
    "matched_candidate",
    "no_match",
    "ambiguous",
    "unavailable",
]
CatalogRecordClassification = Literal["confirmed", "candidate", "unrecognized"]


class LensCatColumnMapping(FrozenModel):
    """Exact catalog-column names supplied by the caller."""

    identifier: str = Field(min_length=1, max_length=256)
    ra_deg: str = Field(min_length=1, max_length=256)
    dec_deg: str = Field(min_length=1, max_length=256)
    status: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _columns_are_distinct(self) -> "LensCatColumnMapping":
        values = (self.identifier, self.ra_deg, self.dec_deg, self.status)
        if len(values) != len(set(values)):
            raise ValueError("LensCat semantic columns must map to distinct columns")
        return self


class LensCatStatusVocabulary(FrozenModel):
    """Exact values that the selected catalog uses for scientific status."""

    confirmed_values: tuple[str, ...] = ()
    candidate_values: tuple[str, ...] = ()

    @field_validator("confirmed_values", "candidate_values")
    @classmethod
    def _nonempty_unique_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("catalog status vocabulary cannot contain empty values")
        if len(value) != len(set(value)):
            raise ValueError("catalog status vocabulary values must be unique")
        return value

    @model_validator(mode="after")
    def _vocabularies_do_not_overlap(self) -> "LensCatStatusVocabulary":
        if not self.confirmed_values and not self.candidate_values:
            raise ValueError("at least one LensCat status value must be configured")
        overlap = set(self.confirmed_values) & set(self.candidate_values)
        if overlap:
            raise ValueError("confirmed and candidate status values cannot overlap")
        return self


class LocalLensCatalogSpec(FrozenModel):
    """One explicitly identified local CSV or Parquet catalog."""

    catalog_id: str = Field(pattern=IDENTIFIER_PATTERN)
    catalog_name: str = Field(min_length=1, max_length=256)
    catalog_release: str = Field(min_length=1, max_length=256)
    path: str = Field(min_length=1, max_length=4096)
    file_format: Literal["csv", "parquet"]
    columns: LensCatColumnMapping
    statuses: LensCatStatusVocabulary
    expected_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    max_bytes: int = Field(default=512 * 1024**2, ge=1, le=10 * 1024**3)

    @field_validator("path")
    @classmethod
    def _local_path_only(cls, value: str) -> str:
        if "://" in value:
            raise ValueError("LensCat adapter accepts local paths only")
        return value


class LensCatQuery(FrozenModel):
    candidate_id: str = Field(pattern=IDENTIFIER_PATTERN)
    coordinate: SkyCoordinate
    cone_radius_arcsec: float = Field(gt=0.0, le=3600.0)


class LensCatCatalogProvenance(FrozenModel):
    """Identity of the exact bytes and mapping used by one attempt."""

    catalog_id: str = Field(pattern=IDENTIFIER_PATTERN)
    catalog_name: str = Field(min_length=1, max_length=256)
    catalog_release: str = Field(min_length=1, max_length=256)
    resolved_local_path: str = Field(min_length=1, max_length=4096)
    file_format: Literal["csv", "parquet"]
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=0)
    row_count: int = Field(ge=0)
    mapping_sha256: str = Field(pattern=SHA256_PATTERN)
    loaded_at_utc: datetime

    @field_validator("loaded_at_utc")
    @classmethod
    def _loaded_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("catalog load time must be UTC")
        return value


class LensCatMatch(FrozenModel):
    catalog_object_id: str = Field(min_length=1, max_length=512)
    coordinate: SkyCoordinate
    separation_arcsec: float = Field(ge=0.0, le=3600.0)
    raw_catalog_status: str = Field(min_length=1, max_length=512)
    classification: CatalogRecordClassification


class LensCatAttempt(FrozenModel):
    """Auditable outcome of the mandatory pre-report catalog step."""

    schema_version: Literal["ripple.lenscat-attempt.v1"] = "ripple.lenscat-attempt.v1"
    attempt_id: str = Field(pattern=IDENTIFIER_PATTERN)
    query: LensCatQuery
    status: LensCatAttemptStatus
    requested_catalog_id: str = Field(pattern=IDENTIFIER_PATTERN)
    provenance: LensCatCatalogProvenance | None = None
    matches: tuple[LensCatMatch, ...] = ()
    reason_codes: tuple[str, ...] = ()
    message: str = Field(min_length=1, max_length=2048)
    completed_at_utc: datetime
    association_evidence_only: Literal[True] = True
    no_match_implies_non_lens: Literal[False] = False

    @field_validator("completed_at_utc")
    @classmethod
    def _completed_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("LensCat attempt time must be UTC")
        return value

    @field_validator("reason_codes")
    @classmethod
    def _reason_codes_are_stable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("LensCat reason codes must be unique")
        return value

    @model_validator(mode="after")
    def _status_matches_evidence(self) -> "LensCatAttempt":
        if self.provenance is not None:
            if self.provenance.catalog_id != self.requested_catalog_id:
                raise ValueError("catalog provenance does not match requested catalog")
            if any(
                match.separation_arcsec > self.query.cone_radius_arcsec
                for match in self.matches
            ):
                raise ValueError(
                    "LensCat result contains a match outside the query cone"
                )

        if self.status == "unavailable":
            if self.provenance is not None or self.matches:
                raise ValueError(
                    "unavailable attempt cannot claim loaded catalog evidence"
                )
            if not self.reason_codes:
                raise ValueError("unavailable attempt must record a reason code")
            return self

        if self.provenance is None:
            raise ValueError("available LensCat outcome requires catalog provenance")

        if self.status == "no_match":
            if self.matches:
                raise ValueError("no_match outcome cannot contain catalog matches")
        elif self.status == "matched_confirmed":
            if len(self.matches) != 1 or self.matches[0].classification != "confirmed":
                raise ValueError(
                    "matched_confirmed requires one confirmed catalog match"
                )
        elif self.status == "matched_candidate":
            if len(self.matches) != 1 or self.matches[0].classification != "candidate":
                raise ValueError(
                    "matched_candidate requires one candidate catalog match"
                )
        elif self.status == "ambiguous":
            if not self.matches:
                raise ValueError(
                    "ambiguous outcome requires at least one nearby record"
                )
            if (
                len(self.matches) == 1
                and self.matches[0].classification != "unrecognized"
            ):
                raise ValueError(
                    "a single recognized catalog record is not an ambiguous result"
                )
        return self
