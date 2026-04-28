"""Parallel Extract Tool for LangChain."""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from parallel import AsyncParallel, Parallel
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_parallel_client, get_parallel_client
from ._types import ExcerptSettings, FetchPolicy, FullContentSettings


def _coerce_full_content(
    full_content: Union[bool, FullContentSettings, dict[str, Any]],
    *,
    tool_max_chars: Optional[int],
) -> Union[bool, dict[str, Any]]:
    """Resolve the user-provided full_content arg + tool-level default.

    Precedence: an explicit FullContentSettings or dict wins over tool_max_chars,
    which only applies when full_content was passed as a plain True/False.
    """
    if isinstance(full_content, FullContentSettings):
        return full_content.model_dump(exclude_none=True)
    if isinstance(full_content, dict):
        return {k: v for k, v in full_content.items() if v is not None}
    if full_content is True and tool_max_chars is not None:
        return {"max_chars_per_result": tool_max_chars}
    return full_content


def _coerce_excerpts(
    excerpts: Optional[ExcerptSettings],
) -> Optional[dict[str, Any]]:
    """Convert an ExcerptSettings to a kwargs dict, or None.

    The GA Extract API always returns excerpts; this controls only per-result
    size. ``None`` means no override (API default).
    """
    if excerpts is None:
        return None
    return excerpts.model_dump(exclude_none=True)


def _build_advanced_settings(
    *,
    excerpts_settings: Optional[dict[str, Any]],
    full_content: Union[bool, dict[str, Any]],
    fetch_policy: Optional[FetchPolicy],
) -> Optional[dict[str, Any]]:
    """Pack the user-facing flat fields into the GA `advanced_settings` envelope."""
    settings: dict[str, Any] = {}
    if excerpts_settings is not None:
        settings["excerpt_settings"] = excerpts_settings
    if fetch_policy is not None:
        settings["fetch_policy"] = fetch_policy.model_dump(exclude_none=True)
    # full_content goes through whether True/False/dict — the API treats False
    # as "do not return full content" (default).
    if full_content is not False:
        settings["full_content"] = full_content
    return settings or None


class ParallelExtractInput(BaseModel):
    """Input schema for Parallel Extract Tool."""

    urls: list[str] = Field(
        min_length=1,
        max_length=20,
        description="List of URLs to extract content from. Up to 20 per request.",
    )
    search_objective: Optional[str] = Field(
        default=None,
        max_length=5000,
        description=(
            "Natural-language objective to focus extraction. Up to 5000 characters."
        ),
    )
    search_queries: Optional[list[str]] = Field(
        default=None,
        description="Keyword queries to focus extracted content.",
    )
    excerpts: Optional[ExcerptSettings] = Field(
        default=None,
        description=(
            "Per-result excerpt-size settings. The GA Extract API always "
            "returns excerpts; pass `ExcerptSettings(max_chars_per_result=…)` "
            "to control their size, or leave None for the API default."
        ),
    )
    full_content: Union[bool, FullContentSettings] = Field(
        default=False,
        description=(
            "Include full page content in addition to excerpts. "
            "Use FullContentSettings(max_chars_per_result=...) to cap size."
        ),
    )
    max_chars_total: Optional[int] = Field(
        default=None,
        description=(
            "Upper bound on total characters of excerpts across all results. "
            "Does not affect full_content."
        ),
    )
    fetch_policy: Optional[FetchPolicy] = Field(
        default=None,
        description="Policy for cached vs live content fetches.",
    )
    client_model: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the calling LLM, used by the API for "
            "model-specific result optimizations."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Group related Search and Extract calls made by the same agent task "
            "under a shared session id. The server returns one if not provided."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "Request timeout in seconds. If not specified, uses the SDK default."
        ),
    )


class ParallelExtractTool(BaseTool):
    """Parallel Extract Tool.

    Calls Parallel's Extract API to pull clean, structured content from web
    pages. Returns a compact summary string the LLM sees and the full
    structured response as a tool artifact.

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
        max_chars_per_extract: Optional[int]
            Tool-wide default cap on full_content size (per URL). Only applied
            when full_content is passed as ``True`` (a settings object always
            wins).

    Instantiation:
        ```python
        from langchain_parallel import ParallelExtractTool

        tool = ParallelExtractTool()
        ```

    Invocation:
        ```python
        result = tool.invoke({
            "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
            "search_objective": "Main applications of AI",
            "full_content": False,
        })
        for r in result:
            print(r["url"], r.get("title"))
        ```

    Async:
        ```python
        result = await tool.ainvoke({"urls": [...]})
        ```

    Response shape (``list[dict]``):
        Each item carries `url`, `title`, optional `publish_date`, and
        either `excerpts` (always present in v1) and/or `full_content`.
        Errors carry `error_type` and `http_status_code`.
    """

    name: str = "parallel_extract"
    description: str = (
        "Extract clean, structured content from web pages using Parallel's "
        "Extract API. Returns a list of per-URL records "
        "(url, title, excerpts, optional full_content)."
    )
    args_schema: type[BaseModel] = ParallelExtractInput

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. If not provided, will be read from env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

    max_chars_per_extract: Optional[int] = None
    """Tool-wide default cap on full_content size (per URL).
    Only applied when ``full_content=True`` is passed.
    """

    _client: Optional[Parallel] = None
    """Synchronous Parallel SDK client (initialized after validation)."""

    _async_client: Optional[AsyncParallel] = None
    """Asynchronous Parallel SDK client (initialized after validation)."""

    @model_validator(mode="after")
    def validate_environment(self) -> ParallelExtractTool:
        """Validate the environment and initialize SDK clients."""
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self

    def _format_response(
        self,
        extract_response: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format the extract API response into a per-URL list.

        Mirrors the v0.2 shape so existing consumers continue to work:
        - "content" stays populated (full_content if present, else joined excerpts)
        - error rows carry "error_type" and "http_status_code"
        """
        results = extract_response.get("results") or []
        errors = extract_response.get("errors") or []

        formatted: list[dict[str, Any]] = []
        for result in results:
            entry: dict[str, Any] = {
                "url": result.get("url"),
                "title": result.get("title"),
            }
            excerpts = result.get("excerpts")
            full_content = result.get("full_content")
            if excerpts is not None:
                entry["excerpts"] = excerpts
                entry["content"] = "\n\n".join(excerpts)
            if full_content is not None:
                entry["full_content"] = full_content
                entry["content"] = full_content
            if "publish_date" in result:
                entry["publish_date"] = result["publish_date"]
            formatted.append(entry)

        formatted.extend(
            [
                {
                    "url": error.get("url"),
                    "title": None,
                    "content": f"Error: {error.get('error_type', 'Unknown error')}",
                    "error_type": error.get("error_type"),
                    "http_status_code": error.get("http_status_code"),
                }
                for error in errors
            ],
        )
        return formatted

    def _build_call_kwargs(
        self,
        *,
        urls: list[str],
        search_objective: Optional[str],
        search_queries: Optional[list[str]],
        excerpts: Optional[ExcerptSettings],
        full_content: Union[bool, FullContentSettings, dict[str, Any]],
        fetch_policy: Optional[FetchPolicy],
        max_chars_total: Optional[int],
        client_model: Optional[str],
        session_id: Optional[str],
        timeout: Optional[float],
    ) -> dict[str, Any]:
        """Resolve params into the GA `client.extract(...)` shape."""
        if not urls:
            msg = "At least one URL must be provided."
            raise ValueError(msg)

        full_content_resolved = _coerce_full_content(
            full_content,
            tool_max_chars=self.max_chars_per_extract,
        )
        advanced_settings = _build_advanced_settings(
            excerpts_settings=_coerce_excerpts(excerpts),
            full_content=full_content_resolved,
            fetch_policy=fetch_policy,
        )

        kwargs: dict[str, Any] = {"urls": list(urls)}
        if search_objective is not None:
            kwargs["objective"] = search_objective
        if search_queries is not None:
            kwargs["search_queries"] = list(search_queries)
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

    @staticmethod
    def _start_text(urls: list[str], *, async_: bool) -> str:
        """Build the run-manager start-of-extraction log message."""
        prefix = (
            "Starting async content extraction from"
            if async_
            else "Starting content extraction from"
        )
        count = len(urls)
        return f"{prefix} {count} URL{'' if count == 1 else 's'}\n"

    @staticmethod
    def _completion_text(formatted: list[dict[str, Any]], *, async_: bool) -> str:
        """Build the run-manager end-of-extraction log message."""
        prefix = "Async extraction completed" if async_ else "Extraction completed"
        success = sum(1 for item in formatted if "error_type" not in item)
        errors = len(formatted) - success
        if errors:
            return f"{prefix}: {success} succeeded, {errors} failed\n"
        return f"{prefix}: {success} URL{'' if success == 1 else 's'} processed\n"

    def _run(
        self,
        urls: list[str],
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[ExcerptSettings] = None,
        full_content: Union[bool, FullContentSettings] = False,
        max_chars_total: Optional[int] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        client_model: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs."""
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)

        if run_manager:
            run_manager.on_text(self._start_text(urls, async_=False), color="blue")

        kwargs = self._build_call_kwargs(
            urls=urls,
            search_objective=search_objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            max_chars_total=max_chars_total,
            client_model=client_model,
            session_id=session_id,
            timeout=timeout,
        )

        try:
            response_obj = self._client.extract(**kwargs)
        except Exception as e:
            if run_manager:
                run_manager.on_text(f"Extraction failed: {e!s}\n", color="red")
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e

        envelope = response_obj.model_dump()
        formatted = self._format_response(envelope)
        if run_manager:
            for warning in envelope.get("warnings") or []:
                warning_msg = (
                    warning.get("message")
                    if isinstance(warning, dict)
                    else str(warning)
                )
                if warning_msg:
                    run_manager.on_text(
                        f"[warning] {warning_msg}\n",
                        color="yellow",
                    )
            run_manager.on_text(
                self._completion_text(formatted, async_=False),
                color="green",
            )
        return formatted

    async def _arun(
        self,
        urls: list[str],
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[ExcerptSettings] = None,
        full_content: Union[bool, FullContentSettings] = False,
        max_chars_total: Optional[int] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        client_model: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Async extract content from URLs."""
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)

        if run_manager:
            await run_manager.on_text(
                self._start_text(urls, async_=True),
                color="blue",
            )

        kwargs = self._build_call_kwargs(
            urls=urls,
            search_objective=search_objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            max_chars_total=max_chars_total,
            client_model=client_model,
            session_id=session_id,
            timeout=timeout,
        )

        try:
            response_obj = await self._async_client.extract(**kwargs)
        except Exception as e:
            if run_manager:
                await run_manager.on_text(
                    f"Async extraction failed: {e!s}\n",
                    color="red",
                )
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e

        envelope = response_obj.model_dump()
        formatted = self._format_response(envelope)
        if run_manager:
            for warning in envelope.get("warnings") or []:
                warning_msg = (
                    warning.get("message")
                    if isinstance(warning, dict)
                    else str(warning)
                )
                if warning_msg:
                    await run_manager.on_text(
                        f"[warning] {warning_msg}\n",
                        color="yellow",
                    )
            await run_manager.on_text(
                self._completion_text(formatted, async_=True),
                color="green",
            )
        return formatted
