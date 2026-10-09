"""Bounded primary-literature discovery for model onboarding.

Crossref metadata is used only to discover stable DOI records.  Search results
are context, not evidence for a preprocessing method; method claims must later
be extracted from a pinned primary source and reviewed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import quote

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_CROSSREF_WORKS_ENDPOINT = "https://api.crossref.org/works"
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


class _ImmutableLiteratureModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
        validate_default=True,
    )


class LiteratureSearchRequest(_ImmutableLiteratureModel):
    query: str = Field(min_length=3, max_length=512)
    maximum_results: int = Field(default=5, ge=1, le=10)
    from_publication_year: int | None = Field(default=None, ge=1900, le=2200)
    through_publication_year: int | None = Field(default=None, ge=1900, le=2200)

    @field_validator("query")
    @classmethod
    def _safe_query(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("literature query must be trimmed printable text")
        return value

    @model_validator(mode="after")
    def _ordered_years(self) -> "LiteratureSearchRequest":
        if (
            self.from_publication_year is not None
            and self.through_publication_year is not None
            and self.from_publication_year > self.through_publication_year
        ):
            raise ValueError("literature publication-year bounds are reversed")
        return self


class LiteratureAuthor(_ImmutableLiteratureModel):
    given: str | None = Field(default=None, max_length=256)
    family: str | None = Field(default=None, max_length=256)
    orcid: str | None = Field(default=None, max_length=256)


class LiteratureCandidate(_ImmutableLiteratureModel):
    provider: Literal["crossref"] = "crossref"
    doi: str = Field(min_length=7, max_length=512)
    title: str = Field(min_length=1, max_length=2048)
    authors: tuple[LiteratureAuthor, ...]
    publication_year: int | None = Field(default=None, ge=1000, le=2200)
    work_type: str = Field(min_length=1, max_length=128)
    doi_url: str = Field(min_length=1, max_length=1024)
    relevance_score: float | None = Field(default=None, ge=0)
    metadata_only: Literal[True] = True
    eligible_as_methods_evidence: Literal[False] = False

    @field_validator("doi")
    @classmethod
    def _valid_doi(cls, value: str) -> str:
        normalized = value.strip().lower()
        if _DOI_PATTERN.fullmatch(normalized) is None or any(
            ord(character) < 33 for character in normalized
        ):
            raise ValueError("invalid DOI")
        return normalized

    @field_validator("title", "work_type")
    @classmethod
    def _printable_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("literature metadata contains invalid text")
        return normalized


class LiteratureSearchResult(_ImmutableLiteratureModel):
    schema_version: Literal["ripple.literature-search.v1"] = (
        "ripple.literature-search.v1"
    )
    request: LiteratureSearchRequest
    provider: Literal["crossref"] = "crossref"
    endpoint: Literal["https://api.crossref.org/works"] = _CROSSREF_WORKS_ENDPOINT
    retrieved_at_utc: datetime
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: tuple[LiteratureCandidate, ...]
    proof_boundary: Literal[
        "Bibliographic discovery metadata only; no paper methods were extracted and no preprocessing requirement was established."
    ] = (
        "Bibliographic discovery metadata only; no paper methods were extracted and no "
        "preprocessing requirement was established."
    )

    @field_validator("retrieved_at_utc")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _unique_dois(self) -> "LiteratureSearchResult":
        dois = tuple(candidate.doi for candidate in self.candidates)
        if len(dois) != len(set(dois)):
            raise ValueError("literature candidates must have unique DOIs")
        if len(dois) > self.request.maximum_results:
            raise ValueError("literature result exceeds the requested bound")
        return self


class LiteratureSearchError(RuntimeError):
    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class CrossrefLiteratureClient:
    """Small fixed-origin Crossref client; no API credential is accepted."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        contact_email: str | None = None,
    ) -> None:
        self._session = session or requests.Session()
        self._contact_email = contact_email

    def search(self, request: LiteratureSearchRequest) -> LiteratureSearchResult:
        params: dict[str, str | int] = {
            "query.bibliographic": request.query,
            "rows": request.maximum_results,
        }
        filters: list[str] = []
        if request.from_publication_year is not None:
            filters.append(f"from-pub-date:{request.from_publication_year}-01-01")
        if request.through_publication_year is not None:
            filters.append(f"until-pub-date:{request.through_publication_year}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self._contact_email:
            params["mailto"] = self._contact_email

        try:
            response = self._session.get(
                _CROSSREF_WORKS_ENDPOINT,
                params=params,
                headers={"User-Agent": "RIPPLe-model-onboarding/0.1"},
                timeout=(5, 20),
                allow_redirects=False,
                stream=True,
            )
        except requests.RequestException as exc:
            raise LiteratureSearchError(
                code="crossref_request_failed",
                message=f"Crossref metadata search failed ({type(exc).__name__}).",
            ) from None
        try:
            if response.status_code != 200:
                raise LiteratureSearchError(
                    code="crossref_http_failure",
                    message="Crossref metadata search returned a non-success status.",
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" not in content_type:
                raise LiteratureSearchError(
                    code="crossref_content_type_mismatch",
                    message="Crossref metadata search did not return JSON.",
                )
            raw = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                raw.extend(chunk)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise LiteratureSearchError(
                        code="crossref_response_too_large",
                        message="Crossref metadata response exceeded the local size bound.",
                    )
        finally:
            response.close()

        try:
            payload = json.loads(bytes(raw))
            message = payload["message"]
            items = message["items"]
            if payload.get("status") != "ok" or not isinstance(items, list):
                raise ValueError("invalid Crossref result envelope")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LiteratureSearchError(
                code="invalid_crossref_response",
                message=f"Crossref metadata could not be validated ({type(exc).__name__}).",
            ) from None

        candidates: list[LiteratureCandidate] = []
        seen_dois: set[str] = set()
        for item in items:
            candidate = _candidate_from_crossref(item)
            if candidate is None or candidate.doi in seen_dois:
                continue
            candidates.append(candidate)
            seen_dois.add(candidate.doi)
            if len(candidates) >= request.maximum_results:
                break
        return LiteratureSearchResult(
            request=request,
            retrieved_at_utc=datetime.now(timezone.utc),
            raw_response_sha256=hashlib.sha256(raw).hexdigest(),
            candidates=tuple(candidates),
        )


def _candidate_from_crossref(item: object) -> LiteratureCandidate | None:
    if not isinstance(item, dict):
        return None
    doi = item.get("DOI")
    titles = item.get("title")
    work_type = item.get("type")
    if not isinstance(doi, str) or not isinstance(titles, list) or not titles:
        return None
    if not isinstance(titles[0], str) or not isinstance(work_type, str):
        return None

    authors: list[LiteratureAuthor] = []
    raw_authors = item.get("author", [])
    if isinstance(raw_authors, list):
        for raw_author in raw_authors[:100]:
            if not isinstance(raw_author, dict):
                continue
            given = (
                raw_author.get("given")
                if isinstance(raw_author.get("given"), str)
                else None
            )
            family = (
                raw_author.get("family")
                if isinstance(raw_author.get("family"), str)
                else None
            )
            orcid = (
                raw_author.get("ORCID")
                if isinstance(raw_author.get("ORCID"), str)
                else None
            )
            authors.append(LiteratureAuthor(given=given, family=family, orcid=orcid))

    publication_year = _crossref_publication_year(item)
    raw_score = item.get("score")
    score = float(raw_score) if isinstance(raw_score, (int, float)) else None
    normalized_doi = doi.strip().lower()
    try:
        return LiteratureCandidate(
            doi=normalized_doi,
            title=titles[0],
            authors=tuple(authors),
            publication_year=publication_year,
            work_type=work_type,
            doi_url=f"https://doi.org/{quote(normalized_doi, safe='/()')}",
            relevance_score=score,
        )
    except ValueError:
        return None


def _crossref_publication_year(item: dict[str, object]) -> int | None:
    for key in ("published", "published-print", "published-online", "issued"):
        value = item.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None


__all__ = [
    "CrossrefLiteratureClient",
    "LiteratureAuthor",
    "LiteratureCandidate",
    "LiteratureSearchError",
    "LiteratureSearchRequest",
    "LiteratureSearchResult",
]
