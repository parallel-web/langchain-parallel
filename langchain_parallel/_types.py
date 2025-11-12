"""Common types for Parallel API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


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
        description=(
            "Maximum age of cached content in seconds to trigger a live fetch. "
            "Minimum value 600 seconds (10 minutes). If not provided, dynamic "
            "age policy will be used."
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
