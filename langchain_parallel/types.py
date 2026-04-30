"""Common types for Parallel API."""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ExcerptSettings(BaseModel):
    """Settings for excerpt extraction."""

    max_chars_per_result: Optional[int] = Field(
        default=None,
        description=(
            "Optional upper bound on the total number of characters to include "
            "per url. Excerpts may contain fewer characters than this limit to "
            "maximize relevance and token efficiency."
        ),
    )


class FullContentSettings(BaseModel):
    """Settings for full content extraction."""

    max_chars_per_result: Optional[int] = Field(
        default=None,
        description=(
            "Optional limit on the number of characters to include in the full "
            "content for each url. Full content always starts at the beginning "
            "of the page and is truncated at the limit if necessary."
        ),
    )


class FetchPolicy(BaseModel):
    """Fetch policy for cache vs live content."""

    max_age_seconds: Optional[int] = Field(
        default=None,
        ge=600,
        description=(
            "If cached content is older than this, fetch fresh content from the "
            "source. Minimum 600 seconds (10 minutes); if not provided, the API "
            "uses a dynamic age policy."
        ),
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Timeout in seconds for fetching live content if unavailable in cache. "
            "If unspecified, dynamic timeout will be used (15-60 seconds)."
        ),
    )
    disable_cache_fallback: bool = Field(
        default=False,
        description=(
            "If false, fallback to cached content older than max-age if live "
            "fetch fails or times out. If true, returns an error instead."
        ),
    )


class SourcePolicy(BaseModel):
    """Domain allow/deny lists and freshness floor for web research.

    Apex-domain semantics: include `nature.com`, not `https://www.nature.com`.
    Wildcards permitted (e.g. `.org`).
    """

    include_domains: Optional[list[str]] = Field(
        default=None,
        max_length=200,
        description=(
            "If provided, only sources from these apex domains are returned. "
            "Combined include + exclude lists are capped at 200 domains."
        ),
    )
    exclude_domains: Optional[list[str]] = Field(
        default=None,
        max_length=200,
        description="If provided, sources from these apex domains are excluded.",
    )
    after_date: Optional[Union[_dt.date, str]] = Field(
        default=None,
        description=(
            "ISO date (YYYY-MM-DD). Only return sources published on or after "
            "this date."
        ),
    )

    @field_validator("after_date", mode="before")
    @classmethod
    def _parse_after_date(cls, v: object) -> object:
        if v is None or isinstance(v, _dt.date):
            return v
        if isinstance(v, str):
            try:
                return _dt.date.fromisoformat(v)
            except ValueError as e:
                msg = f"after_date must be ISO YYYY-MM-DD; got {v!r} ({e!s})."
                raise ValueError(msg) from e
        return v

    @model_validator(mode="after")
    def _check_domain_total(self) -> SourcePolicy:
        total = len(self.include_domains or []) + len(self.exclude_domains or [])
        if total > 200:
            msg = (
                f"Combined include_domains + exclude_domains has {total} entries; "
                f"the API caps the total at 200."
            )
            raise ValueError(msg)
        return self
