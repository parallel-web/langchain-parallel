"""Client utilities for Parallel integration."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

import openai
from parallel import AsyncParallel, Parallel


def get_api_key(api_key: Optional[str] = None) -> str:
    """Retrieve the Parallel API key from argument or environment variables.

    Args:
        api_key: Optional API key string.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not found.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("PARALLEL_API_KEY")
    if env_key:
        return env_key

    msg = (
        "Parallel API key not found. Please pass it as an argument or set the "
        "PARALLEL_API_KEY environment variable."
    )
    raise ValueError(msg)


def get_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """Returns a configured sync OpenAI client for the Chat API."""
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_async_openai_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    """Returns a configured async OpenAI client for the Chat API."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


class ParallelSearchClient:
    """Synchronous client for Parallel Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK client
        self.client = Parallel(api_key=api_key, base_url=base_url)

    def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[dict[str, Any]] = None,
        mode: Optional[str] = None,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Perform a synchronous search using the Parallel Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        # Use default timeout if not provided
        if timeout is None:
            timeout = 30.0

        # Build kwargs, only including non-None values for optional params
        kwargs: dict[str, Any] = {
            "objective": objective,
            "search_queries": search_queries,
            "max_results": max_results,
            "timeout": timeout,
        }
        if excerpts is not None:
            kwargs["excerpts"] = excerpts
        if mode is not None:
            kwargs["mode"] = mode
        if source_policy is not None:
            kwargs["source_policy"] = source_policy
        if fetch_policy is not None:
            kwargs["fetch_policy"] = fetch_policy

        # Use the Parallel SDK's beta.search method
        search_response = self.client.beta.search(**kwargs)

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


class AsyncParallelSearchClient:
    """Asynchronous client for Parallel Search API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK async client
        self.client = AsyncParallel(api_key=api_key, base_url=base_url)

    async def search(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[dict[str, Any]] = None,
        mode: Optional[str] = None,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Perform an async search using the Parallel Search API via SDK."""
        if not objective and not search_queries:
            msg = "Either 'objective' or 'search_queries' must be provided"
            raise ValueError(msg)

        # Use default timeout if not provided
        if timeout is None:
            timeout = 30.0

        # Build kwargs, only including non-None values for optional params
        kwargs: dict[str, Any] = {
            "objective": objective,
            "search_queries": search_queries,
            "max_results": max_results,
            "timeout": timeout,
        }
        if excerpts is not None:
            kwargs["excerpts"] = excerpts
        if mode is not None:
            kwargs["mode"] = mode
        if source_policy is not None:
            kwargs["source_policy"] = source_policy
        if fetch_policy is not None:
            kwargs["fetch_policy"] = fetch_policy

        # Use the Parallel SDK's beta.search method
        search_response = await self.client.beta.search(**kwargs)

        # Convert the SDK response to a dictionary
        return search_response.model_dump()


def get_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelSearchClient:
    """Returns a configured sync Parallel Search client."""
    return ParallelSearchClient(api_key, base_url)


def get_async_search_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> AsyncParallelSearchClient:
    """Returns a configured async Parallel Search client."""
    return AsyncParallelSearchClient(api_key, base_url)


class ParallelExtractClient:
    """Synchronous client for Parallel Extract API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK client
        self.client = Parallel(api_key=api_key, base_url=base_url)

    def extract(
        self,
        urls: list[str],
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[Union[bool, dict[str, Any]]] = None,
        full_content: Optional[Union[bool, dict[str, Any]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Perform a synchronous extract using the Parallel Extract API via SDK."""
        if not urls:
            msg = "At least one URL must be provided"
            raise ValueError(msg)

        # Use default timeout if not provided (5 seconds per URL)
        if timeout is None:
            timeout = 5.0 * len(urls)

        # Use the Parallel SDK's beta.extract method
        extract_response = self.client.beta.extract(
            urls=urls,
            objective=objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return extract_response.model_dump()


class AsyncParallelExtractClient:
    """Asynchronous client for Parallel Extract API using the Parallel SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.parallel.ai",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        # Initialize the Parallel SDK async client
        self.client = AsyncParallel(api_key=api_key, base_url=base_url)

    async def extract(
        self,
        urls: list[str],
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Optional[Union[bool, dict[str, Any]]] = None,
        full_content: Optional[Union[bool, dict[str, Any]]] = None,
        fetch_policy: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Perform an async extract using the Parallel Extract API via SDK."""
        if not urls:
            msg = "At least one URL must be provided"
            raise ValueError(msg)

        # Use default timeout if not provided (5 seconds per URL)
        if timeout is None:
            timeout = 5.0 * len(urls)

        # Use the Parallel SDK's beta.extract method
        extract_response = await self.client.beta.extract(
            urls=urls,
            objective=objective,
            search_queries=search_queries,
            excerpts=excerpts,
            full_content=full_content,
            fetch_policy=fetch_policy,
            timeout=timeout,
        )

        # Convert the SDK response to a dictionary
        return extract_response.model_dump()


def get_extract_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> ParallelExtractClient:
    """Returns a configured sync Parallel Extract client."""
    return ParallelExtractClient(api_key, base_url)


def get_async_extract_client(
    api_key: str, base_url: str = "https://api.parallel.ai"
) -> AsyncParallelExtractClient:
    """Returns a configured async Parallel Extract client."""
    return AsyncParallelExtractClient(api_key, base_url)
