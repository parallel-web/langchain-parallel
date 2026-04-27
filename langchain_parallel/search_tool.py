"""ParallelWeb tools."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from parallel import AsyncParallel, Parallel
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_parallel_client, get_parallel_client
from ._types import ExcerptSettings, FetchPolicy, SourcePolicy

_LEGACY_MODE_MAP: dict[str, str] = {
    "fast": "basic",
    "one-shot": "basic",
    "agentic": "advanced",
}


def _normalize_mode(mode: Optional[str]) -> Optional[str]:
    """Translate legacy beta mode strings to the GA `basic` / `advanced` set."""
    if mode is None or mode in {"basic", "advanced"}:
        return mode
    if mode in _LEGACY_MODE_MAP:
        new_mode = _LEGACY_MODE_MAP[mode]
        warnings.warn(
            f"mode='{mode}' is a legacy beta value; mapping to '{new_mode}'. "
            f"Pass mode='{new_mode}' directly to silence this warning.",
            DeprecationWarning,
            stacklevel=3,
        )
        return new_mode
    msg = (
        f"Invalid mode '{mode}'. Expected one of: 'basic', 'advanced'. "
        f"(Legacy values 'fast', 'one-shot', 'agentic' are accepted with a warning.)"
    )
    raise ValueError(msg)


def _coerce_source_policy(
    source_policy: Optional[Union[SourcePolicy, dict[str, Any]]],
) -> Optional[dict[str, Any]]:
    """Accept a SourcePolicy model or a raw dict, return a dict for the SDK."""
    if source_policy is None:
        return None
    if isinstance(source_policy, SourcePolicy):
        return source_policy.model_dump(exclude_none=True)
    return {k: v for k, v in source_policy.items() if v is not None}


def _build_advanced_settings(
    *,
    excerpts: Optional[ExcerptSettings],
    fetch_policy: Optional[FetchPolicy],
    source_policy: Optional[Union[SourcePolicy, dict[str, Any]]],
    max_results: Optional[int],
    location: Optional[str],
) -> Optional[dict[str, Any]]:
    """Pack the user-facing flat fields into the GA `advanced_settings` envelope."""
    settings: dict[str, Any] = {}
    if excerpts is not None:
        settings["excerpt_settings"] = excerpts.model_dump(exclude_none=True)
    if fetch_policy is not None:
        settings["fetch_policy"] = fetch_policy.model_dump(exclude_none=True)
    sp = _coerce_source_policy(source_policy)
    if sp:
        settings["source_policy"] = sp
    if max_results is not None:
        settings["max_results"] = max_results
    if location is not None:
        settings["location"] = location
    return settings or None


class ParallelWebSearchInput(BaseModel):
    """Input schema for ParallelWeb search tool."""

    objective: Optional[str] = Field(
        default=None,
        description=(
            "Natural-language description of the research goal. Up to 5000 "
            "characters. Include any source or freshness guidance. Recommended "
            "alongside `search_queries` for best results."
        ),
    )
    search_queries: list[str] = Field(
        description=(
            "Required. 1-5 keyword search queries (3-6 words each, up to "
            "200 characters). Pair with an optional `objective` for best "
            "results."
        ),
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of search results to return (1 to 40).",
    )
    excerpts: Optional[ExcerptSettings] = Field(
        default=None,
        description=(
            "Per-result excerpt-size settings. "
            "Example: ExcerptSettings(max_chars_per_result=1500)."
        ),
    )
    max_chars_total: Optional[int] = Field(
        default=None,
        description=(
            "Upper bound on the total characters of excerpts across all results. "
            "Useful for capping context size when feeding results to an LLM."
        ),
    )
    mode: Optional[str] = Field(
        default=None,
        description=(
            "Search mode: 'basic' for low-latency searches, 'advanced' (default) "
            "for higher quality with more retrieval and compression. Legacy "
            "values 'fast', 'one-shot' (-> 'basic') and 'agentic' (-> 'advanced') "
            "are accepted with a deprecation warning."
        ),
    )
    source_policy: Optional[Union[SourcePolicy, dict[str, Any]]] = Field(
        default=None,
        description=(
            "Domain include/exclude lists and a freshness floor (after_date). "
            "Example: SourcePolicy(include_domains=['nature.com'], "
            "after_date='2024-01-01'). A raw dict is also accepted."
        ),
    )
    fetch_policy: Optional[FetchPolicy] = Field(
        default=None,
        description=(
            "Cache vs live-fetch policy. "
            "Example: FetchPolicy(max_age_seconds=86400, timeout_seconds=60)."
        ),
    )
    location: Optional[str] = Field(
        default=None,
        description=(
            "ISO 3166-1 alpha-2 country code (e.g., 'us', 'gb', 'de', 'jp') "
            "to geo-target results. Unsupported values are ignored with a "
            "warning by the API."
        ),
    )
    client_model: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the calling LLM, used by the API for model-specific "
            "result optimizations."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Group related Search and Extract calls made by the same agent task "
            "under a shared session id. The server returns one if not provided."
        ),
    )
    include_metadata: bool = Field(
        default=True,
        description=(
            "Whether to attach client-side timing/result metadata to the artifact."
        ),
    )
    timeout: Optional[int] = Field(
        default=None,
        description=(
            "Request timeout in seconds. If not specified, uses the SDK default."
        ),
    )


class ParallelWebSearchTool(BaseTool):
    """Parallel Search tool with web research capabilities.

    This tool calls Parallel's Search API, which streamlines the traditional
    search -> scrape -> extract pipeline into a single API call. It supports
    natural-language objectives, keyword queries, domain filters, two modes
    (`basic`, `advanced`), location targeting, and async usage.

    Setup:
        Install `langchain-parallel` and set environment variable
        `PARALLEL_API_KEY`.

        ```bash
        pip install -U langchain-parallel
        export PARALLEL_API_KEY="your-api-key"
        ```

    Key init args:
        api_key: Optional[SecretStr]
            Parallel API key. If not provided, will be read from
            PARALLEL_API_KEY env var.
        base_url: str
            Base URL for Parallel API. Defaults to "https://api.parallel.ai".

    Instantiation:
        ```python
        from langchain_parallel import ParallelWebSearchTool

        tool = ParallelWebSearchTool()
        ```

    Invocation:
        ```python
        result = tool.invoke({
            "objective": "Latest developments in AI agents",
            "search_queries": ["AI agents 2026", "autonomous LLM systems"],
            "mode": "advanced",
            "max_results": 5,
        })
        print(result["search_id"], len(result["results"]))
        ```

    Domain and freshness filters:
        ```python
        from langchain_parallel import SourcePolicy

        result = tool.invoke({
            "search_queries": ["climate research breakthroughs"],
            "source_policy": SourcePolicy(
                include_domains=["nature.com", "science.org"],
                after_date="2025-01-01",
            ),
            "location": "us",
        })
        ```

    Async:
        ```python
        result = await tool.ainvoke({"search_queries": ["..."]})
        ```

    Response shape:
        ```python
        {
            "search_id": "search_abc123",
            "session_id": "sess_...",
            "results": [
                {"url": "...", "title": "...", "publish_date": "...",
                 "excerpts": ["..."]},
                ...
            ],
            "warnings": [...],
            "usage": {...},
            "search_metadata": {  # added by this tool when include_metadata=True
                "search_duration_seconds": 2.451,
                "search_timestamp": "2026-04-27T10:30:00",
                "actual_results_returned": 5,
            }
        }
        ```

    """

    name: str = "parallel_web_search"
    """The name passed to the model when performing tool calling."""

    description: str = (
        "Search the web using Parallel's Search API. "
        "Provides real-time web information with compressed, structured excerpts "
        "optimized for LLM consumption. Supports natural-language objectives, "
        "keyword queries, domain filtering, and geo-targeting. Returns the "
        "structured search response as a dict."
    )
    """The description passed to the model when performing tool calling."""

    args_schema: type[BaseModel] = ParallelWebSearchInput
    """The schema passed to the model when performing tool calling."""

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. If not provided, will be read from
    PARALLEL_API_KEY env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

    _client: Optional[Parallel] = None
    """Synchronous Parallel SDK client (initialized after validation)."""

    _async_client: Optional[AsyncParallel] = None
    """Asynchronous Parallel SDK client (initialized after validation)."""

    @model_validator(mode="after")
    def validate_environment(self) -> ParallelWebSearchTool:
        """Validate the environment and initialize SDK clients."""
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self

    def _build_metadata(
        self,
        *,
        start_time: datetime,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """Build client-side timing/result metadata."""
        end_time = datetime.now()
        return {
            "search_duration_seconds": round(
                (end_time - start_time).total_seconds(),
                3,
            ),
            "search_timestamp": start_time.isoformat(),
            "actual_results_returned": len(response.get("results") or []),
        }

    def _build_call_kwargs(
        self,
        *,
        objective: Optional[str],
        search_queries: Optional[list[str]],
        mode: Optional[str],
        max_chars_total: Optional[int],
        client_model: Optional[str],
        session_id: Optional[str],
        excerpts: Optional[ExcerptSettings],
        fetch_policy: Optional[FetchPolicy],
        source_policy: Optional[Union[SourcePolicy, dict[str, Any]]],
        max_results: int,
        location: Optional[str],
        timeout: Optional[int],
    ) -> dict[str, Any]:
        """Resolve params into the GA `client.search(...)` kwargs shape."""
        if not search_queries:
            msg = (
                "search_queries is required (1-5 keyword strings, 3-6 words "
                "each). Pass at least one query; pair with an optional "
                "`objective` for best results. See "
                "https://docs.parallel.ai/search/search-migration-guide for "
                "migrating from the pre-GA objective-only call shape."
            )
            raise ValueError(msg)

        normalized_mode = _normalize_mode(mode)
        advanced_settings = _build_advanced_settings(
            excerpts=excerpts,
            fetch_policy=fetch_policy,
            source_policy=source_policy,
            max_results=max_results,
            location=location,
        )

        kwargs: dict[str, Any] = {"search_queries": list(search_queries)}
        if objective is not None:
            kwargs["objective"] = objective
        if normalized_mode is not None:
            kwargs["mode"] = normalized_mode
        if max_chars_total is not None:
            kwargs["max_chars_total"] = max_chars_total
        if client_model is not None:
            kwargs["client_model"] = client_model
        if session_id is not None:
            kwargs["session_id"] = session_id
        if advanced_settings is not None:
            kwargs["advanced_settings"] = advanced_settings
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _finalize_response(
        self,
        response_obj: Any,
        *,
        start_time: datetime,
        include_metadata: bool,
    ) -> dict[str, Any]:
        """Convert SDK response to dict and attach client-side metadata."""
        response: dict[str, Any] = response_obj.model_dump()
        if include_metadata:
            response["search_metadata"] = self._build_metadata(
                start_time=start_time,
                response=response,
            )
        return response

    @staticmethod
    def _start_text(
        objective: Optional[str], search_queries: Optional[list[str]], *, async_: bool
    ) -> str:
        """Build the run-manager start-of-search log message."""
        prefix = "Starting async web search" if async_ else "Starting web search"
        query_desc = objective or f"{len(search_queries or [])} search queries"
        return f"{prefix}: {query_desc}\n"

    @staticmethod
    def _completion_text(response: dict[str, Any], *, async_: bool) -> str:
        """Build the run-manager end-of-search log message."""
        prefix = "Async search completed" if async_ else "Search completed"
        count = len(response.get("results") or [])
        duration = response.get("search_metadata", {}).get("search_duration_seconds", 0)
        return f"{prefix}: {count} results in {duration}s\n"

    def _run(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[ExcerptSettings] = None,
        max_chars_total: Optional[int] = None,
        mode: Optional[str] = None,
        source_policy: Optional[Union[SourcePolicy, dict[str, Any]]] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        location: Optional[str] = None,
        client_model: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        include_metadata: bool = True,
        timeout: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute the search using Parallel's Search API."""
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)

        start_time = datetime.now()
        if run_manager:
            run_manager.on_text(
                self._start_text(objective, search_queries, async_=False),
                color="blue",
            )

        kwargs = self._build_call_kwargs(
            objective=objective,
            search_queries=search_queries,
            mode=mode,
            max_chars_total=max_chars_total,
            client_model=client_model,
            session_id=session_id,
            excerpts=excerpts,
            fetch_policy=fetch_policy,
            source_policy=source_policy,
            max_results=max_results,
            location=location,
            timeout=timeout,
        )

        try:
            response_obj = self._client.search(**kwargs)
        except Exception as e:
            if run_manager:
                run_manager.on_text(f"Search failed: {e!s}\n", color="red")
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e

        response = self._finalize_response(
            response_obj,
            start_time=start_time,
            include_metadata=include_metadata,
        )
        if run_manager:
            run_manager.on_text(
                self._completion_text(response, async_=False),
                color="green",
            )
        return response

    async def _arun(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[ExcerptSettings] = None,
        max_chars_total: Optional[int] = None,
        mode: Optional[str] = None,
        source_policy: Optional[Union[SourcePolicy, dict[str, Any]]] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        location: Optional[str] = None,
        client_model: Optional[str] = None,
        session_id: Optional[str] = None,
        *,
        include_metadata: bool = True,
        timeout: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Async execute the search using Parallel's Search API."""
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)

        start_time = datetime.now()
        if run_manager:
            await run_manager.on_text(
                self._start_text(objective, search_queries, async_=True),
                color="blue",
            )

        kwargs = self._build_call_kwargs(
            objective=objective,
            search_queries=search_queries,
            mode=mode,
            max_chars_total=max_chars_total,
            client_model=client_model,
            session_id=session_id,
            excerpts=excerpts,
            fetch_policy=fetch_policy,
            source_policy=source_policy,
            max_results=max_results,
            location=location,
            timeout=timeout,
        )

        try:
            response_obj = await self._async_client.search(**kwargs)
        except Exception as e:
            if run_manager:
                await run_manager.on_text(
                    f"Async search failed: {e!s}\n",
                    color="red",
                )
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e

        response = self._finalize_response(
            response_obj,
            start_time=start_time,
            include_metadata=include_metadata,
        )
        if run_manager:
            await run_manager.on_text(
                self._completion_text(response, async_=True),
                color="green",
            )
        return response


#: Forward-compat alias for :class:`ParallelWebSearchTool`.
#:
#: Prefer ParallelSearchTool in new code; ParallelWebSearchTool will
#: continue to work indefinitely as an alias for this class.
ParallelSearchTool = ParallelWebSearchTool
