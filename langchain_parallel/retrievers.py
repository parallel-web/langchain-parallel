"""LangChain retrievers backed by the Parallel Search API."""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from parallel import AsyncParallel, Parallel
from pydantic import Field, SecretStr, model_validator

from ._client import get_api_key, get_async_parallel_client, get_parallel_client
from .types import ExcerptSettings, FetchPolicy, SourcePolicy


def _join_excerpts(excerpts: Optional[list[str]]) -> str:
    """Combine the API's excerpt list into a single ``Document.page_content``."""
    return "\n\n".join(excerpts) if excerpts else ""


def _result_to_document(
    result: dict[str, Any],
    *,
    search_id: Optional[str],
    query: str,
) -> Document:
    """Convert a single Parallel Search result into a LangChain Document."""
    metadata: dict[str, Any] = {"query": query}
    for key in ("url", "title", "publish_date"):
        if (value := result.get(key)) is not None:
            metadata[key] = value
    if search_id is not None:
        metadata["search_id"] = search_id
    if (excerpts := result.get("excerpts")) is not None:
        metadata["excerpts"] = excerpts
    return Document(
        page_content=_join_excerpts(result.get("excerpts")),
        metadata=metadata,
    )


class ParallelSearchRetriever(BaseRetriever):
    """LangChain retriever that returns Parallel Search results as Documents.

    Drops in to any RAG pipeline that expects a `BaseRetriever`. Each
    `Document.page_content` is the joined excerpt list from one Parallel
    result; `metadata` carries `url`, `title`, `publish_date`, `search_id`,
    the original `excerpts` list, and the `query` that produced it.

    Setup:
        Install `langchain-parallel` and set environment variable
        `PARALLEL_API_KEY`.

    Key init args:
        api_key: Optional[SecretStr]
            Parallel API key. Reads `PARALLEL_API_KEY` if not provided.
        base_url: str
            Defaults to "https://api.parallel.ai".
        max_results: int
            Maximum results returned per query. 1-40 (default 5).
        mode: Optional[Literal["basic", "advanced"]]
            Search mode; defaults to the API default ("advanced").
        objective: Optional[str]
            Objective forwarded to the Search API on every call. Useful
            when the same retriever is used for a focused research task
            and the LangChain query is just one of several keyword forms.
        excerpts: Optional[ExcerptSettings]
            Per-result excerpt-size cap.
        max_chars_total: Optional[int]
            Cap on total excerpt characters across all results.
        source_policy: Optional[SourcePolicy | dict]
            Domain include/exclude lists and freshness floor.
        fetch_policy: Optional[FetchPolicy]
            Cache vs live-fetch policy.
        location: Optional[str]
            ISO 3166-1 alpha-2 country code for geo-targeting.
        client_model: Optional[str]
            Identifier of the calling LLM (e.g. ``"claude-opus-4-7"``).

    Instantiation and use:
        ```python
        from langchain_parallel import ParallelSearchRetriever

        retriever = ParallelSearchRetriever(max_results=5, mode="advanced")
        docs = retriever.invoke("What's new in renewable energy this month?")
        for doc in docs:
            print(doc.metadata["title"], doc.metadata["url"])
        ```
    """

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. Reads PARALLEL_API_KEY if not provided."""

    base_url: str = Field(default="https://api.parallel.ai")

    max_results: int = Field(default=5)
    """Number of results per query (1-40)."""

    mode: Optional[str] = Field(default=None)
    """``"basic"`` or ``"advanced"``."""

    objective: Optional[str] = Field(default=None)
    """Optional persistent objective forwarded on every call."""

    excerpts: Optional[ExcerptSettings] = Field(default=None)
    max_chars_total: Optional[int] = Field(default=None)
    source_policy: Optional[Union[SourcePolicy, dict[str, Any]]] = Field(default=None)
    fetch_policy: Optional[FetchPolicy] = Field(default=None)
    location: Optional[str] = Field(default=None)
    client_model: Optional[str] = Field(default=None)

    _client: Optional[Parallel] = None
    _async_client: Optional[AsyncParallel] = None

    @model_validator(mode="after")
    def _initialize_clients(self) -> ParallelSearchRetriever:
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self

    def _build_kwargs(self, query: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"search_queries": [query]}
        if self.objective is not None:
            kwargs["objective"] = self.objective
        if self.mode is not None:
            kwargs["mode"] = self.mode
        if self.max_chars_total is not None:
            kwargs["max_chars_total"] = self.max_chars_total
        if self.client_model is not None:
            kwargs["client_model"] = self.client_model

        advanced: dict[str, Any] = {"max_results": self.max_results}
        if self.excerpts is not None:
            advanced["excerpt_settings"] = self.excerpts.model_dump(exclude_none=True)
        if self.fetch_policy is not None:
            advanced["fetch_policy"] = self.fetch_policy.model_dump(exclude_none=True)
        if self.source_policy is not None:
            sp = (
                self.source_policy.model_dump(exclude_none=True)
                if isinstance(self.source_policy, SourcePolicy)
                else {k: v for k, v in self.source_policy.items() if v is not None}
            )
            if sp:
                advanced["source_policy"] = sp
        if self.location is not None:
            advanced["location"] = self.location
        kwargs["advanced_settings"] = advanced
        return kwargs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        try:
            response = self._client.search(**self._build_kwargs(query))
        except Exception as e:
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e
        payload = response.model_dump()
        return [
            _result_to_document(r, search_id=payload.get("search_id"), query=query)
            for r in payload.get("results") or []
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        try:
            response = await self._async_client.search(**self._build_kwargs(query))
        except Exception as e:
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e
        payload = response.model_dump()
        return [
            _result_to_document(r, search_id=payload.get("search_id"), query=query)
            for r in payload.get("results") or []
        ]
