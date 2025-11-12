"""ParallelWeb tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_search_client, get_search_client
from ._types import ExcerptSettings, FetchPolicy


class ParallelWebSearchInput(BaseModel):
    """Input schema for ParallelWeb search tool."""

    objective: Optional[str] = Field(
        default=None,
        description="Natural-language description of what the web research goal is. "
        "Include any source or freshness guidance. Either this or search_queries "
        "must be provided.",
    )
    search_queries: Optional[list[str]] = Field(
        default=None,
        description="Optional list of search queries to guide the search. "
        "Maximum 5 queries, each up to 200 characters. Either this or objective "
        "must be provided.",
    )
    max_results: int = Field(
        default=10, description="Maximum number of search results to return (1 to 40)."
    )
    excerpts: Optional[ExcerptSettings] = Field(
        default=None,
        description=(
            "Optional excerpt settings for controlling excerpt length. "
            "Example: ExcerptSettings(max_chars_per_result=1500)"
        ),
    )
    mode: Optional[str] = Field(
        default=None,
        description=(
            "Search mode: 'one-shot' for comprehensive results with longer "
            "excerpts, 'agentic' for concise, token-efficient results. "
            "Defaults to 'one-shot'."
        ),
    )
    source_policy: Optional[dict[str, Union[str, list[str]]]] = Field(
        default=None,
        description=(
            "Optional source policy with 'include_domains' and/or "
            "'exclude_domains' lists. Example: "
            "{'include_domains': ['wikipedia.org'], 'exclude_domains': ['reddit.com']}"
        ),
    )
    fetch_policy: Optional[FetchPolicy] = Field(
        default=None,
        description=(
            "Optional fetch policy to control when to return cached vs live "
            "content. Example: FetchPolicy(max_age_seconds=86400, timeout_seconds=60)"
        ),
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in the response "
        "(search timing, result counts, etc.).",
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Request timeout in seconds. If not specified, uses default timeout.",  # noqa: E501
    )


class ParallelWebSearchTool(BaseTool):
    """Parallel Search tool with web research capabilities.

    This tool provides access to Parallel's Search API, which streamlines
    the traditional search → scrape → extract pipeline into a single API call.
    Features include domain filtering, multiple processors, async support,
    and metadata collection.

    Setup:
        Install ``langchain-parallel`` and set environment variable
        ``PARALLEL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-parallel
            export PARALLEL_API_KEY="your-api-key"

    Key init args:
        api_key: Optional[SecretStr]
            Parallel API key. If not provided, will be read from
            PARALLEL_API_KEY env var.
        base_url: str
            Base URL for Parallel API. Defaults to "https://api.parallel.ai".

    Instantiation:
        .. code-block:: python

            from langchain_parallel import ParallelWebSearchTool

            # Basic instantiation
            tool = ParallelWebSearchTool()

            # With custom API key
            tool = ParallelWebSearchTool(api_key="your-api-key")

    Basic Usage:
        .. code-block:: python

            # Simple objective-based search
            result = tool.invoke({
                "objective": "What are the latest developments in AI?"
            })

            # Query-based search with multiple queries
            result = tool.invoke({
                "search_queries": [
                    "latest AI developments 2024",
                    "machine learning breakthroughs",
                    "artificial intelligence news"
                ],
                "max_results": 10
            })

    Domain filtering and advanced options:
        .. code-block:: python

            # Domain filtering with fetch policy (using dict format)
            result = tool.invoke({
                "objective": "Recent climate change research",
                "source_policy": {
                    "include_domains": ["nature.com", "science.org"],
                    "exclude_domains": ["reddit.com", "twitter.com"]
                },
                "max_results": 15,
                "excerpts": {"max_chars_per_result": 2000},  # Auto-converted
                "mode": "one-shot",  # Use 'agentic' for token-efficient results
                "fetch_policy": {  # Auto-converted to FetchPolicy
                    "max_age_seconds": 86400,  # 1 day cache
                    "timeout_seconds": 60
                },
                "include_metadata": True
            })

            # Or use the types directly
            from langchain_parallel import ExcerptSettings, FetchPolicy

            result = tool.invoke({
                "objective": "Recent climate change research",
                "excerpts": ExcerptSettings(max_chars_per_result=2000),
                "fetch_policy": FetchPolicy(max_age_seconds=86400, timeout_seconds=60),
            })

    Async Usage:
        .. code-block:: python

            import asyncio

            async def search_async():
                result = await tool.ainvoke({
                    "objective": "Latest tech news"
                })
                return result

            result = asyncio.run(search_async())

    Response Format:
        .. code-block:: python

            {
                "search_id": "search_abc123...",
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Article Title",
                        "excerpts": [
                            "Relevant excerpt from the page...",
                            "Another important section..."
                        ]
                    }
                ],
                "search_metadata": {
                    "search_duration_seconds": 2.451,
                    "search_timestamp": "2024-01-15T10:30:00",
                    "max_results_requested": 10,
                    "actual_results_returned": 8,
                    "search_id": "search_abc123...",
                    "query_count": 3,
                    "queries_used": ["query1", "query2", "query3"],
                    "source_policy_applied": true,
                    "included_domains": ["nature.com"],
                    "excluded_domains": ["reddit.com"]
                }
            }

    Tool Calling Integration:
        .. code-block:: python

            # When used with LangChain agents or chat models with tool calling
            from langchain_core.messages import HumanMessage
            from langchain_parallel import ChatParallelWeb

            chat = ChatParallelWeb()
            chat_with_tools = chat.bind_tools([tool])

            response = chat_with_tools.invoke([
                HumanMessage(content="Search for the latest AI research papers")
            ])

    Best Practices:
        - Use specific objectives for better results
        - Apply domain filtering for focused searches
        - Include metadata for debugging and optimization
    """

    name: str = "parallel_web_search"
    """The name that is passed to the model when performing tool calling."""

    description: str = (
        "Search the web using Parallel's Search API. "
        "Provides real-time web information with compressed, structured excerpts "
        "optimized for LLM consumption. Supports domain filtering "
        "and metadata. Specify either an objective "
        "(natural language goal) or specific search queries for targeted results."
    )
    """The description that is passed to the model when performing tool calling."""

    args_schema: type[BaseModel] = ParallelWebSearchInput
    """The schema that is passed to the model when performing tool calling."""

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. If not provided, will be read from
    PARALLEL_API_KEY env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

    _client: Any = None
    """Synchronous search client (initialized after validation)."""

    _async_client: Any = None
    """Asynchronous search client (initialized after validation)."""

    @model_validator(mode="after")
    def validate_environment(self) -> ParallelWebSearchTool:
        """Validate the environment and initialize clients."""
        # Get API key from parameter or environment
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None
        )

        # Initialize both sync and async clients once
        self._client = get_search_client(api_key_str, self.base_url)
        self._async_client = get_async_search_client(api_key_str, self.base_url)

        return self

    def _create_response_metadata(
        self,
        start_time: datetime,
        search_params: dict[str, Any],
        response: dict[str, Any],
        *,
        include_metadata: bool,
    ) -> dict[str, Any]:
        """Create response metadata."""
        if not include_metadata:
            return {}

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        metadata = {
            "search_duration_seconds": round(duration, 3),
            "search_timestamp": start_time.isoformat(),
            "max_results_requested": search_params.get("max_results", 10),
            "actual_results_returned": len(response.get("results", [])),
            "search_id": response.get("search_id"),
        }

        if search_params.get("search_queries"):
            metadata["query_count"] = len(search_params["search_queries"])
            metadata["queries_used"] = search_params["search_queries"]

        if search_params.get("source_policy"):
            metadata["source_policy_applied"] = True
            policy = search_params["source_policy"]
            if "include_domains" in policy:
                metadata["included_domains"] = policy["include_domains"]
            if "exclude_domains" in policy:
                metadata["excluded_domains"] = policy["exclude_domains"]

        return metadata

    def _run(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[ExcerptSettings] = None,
        mode: Optional[str] = None,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        *,
        include_metadata: bool = True,
        timeout: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Execute the search using Parallel's Search API.

        Args:
            objective: Natural-language description of the research goal
            search_queries: List of specific search queries
            max_results: Maximum number of results (1-40)
            excerpts: Optional ExcerptSettings for controlling excerpt length
            mode: Search mode ('one-shot' or 'agentic')
            source_policy: Optional source policy for domain filtering
            fetch_policy: Optional FetchPolicy for cache vs live content
            include_metadata: Whether to include metadata
            timeout: Request timeout in seconds
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing search results with metadata
        """
        start_time = datetime.now()

        # Notify callback manager about search start
        if run_manager:
            query_desc = objective or f"{len(search_queries or [])} search queries"
            run_manager.on_text(f"Starting web search: {query_desc}\n", color="blue")

        # Convert ExcerptSettings and FetchPolicy to dict if provided
        excerpts_dict = excerpts.model_dump(exclude_none=True) if excerpts else None
        fetch_policy_dict = (
            fetch_policy.model_dump(exclude_none=True) if fetch_policy else None
        )

        search_params = {
            "objective": objective,
            "search_queries": search_queries,
            "max_results": max_results,
            "excerpts": excerpts_dict,
            "mode": mode,
            "source_policy": source_policy,
            "fetch_policy": fetch_policy_dict,
        }

        try:
            # Notify about search execution
            if run_manager:
                run_manager.on_text("Executing search...\n", color="yellow")

            # Perform search using pre-initialized client
            response = self._client.search(
                objective=objective,
                search_queries=search_queries,
                max_results=max_results,
                excerpts=excerpts_dict,
                mode=mode,
                source_policy=source_policy,
                fetch_policy=fetch_policy_dict,
                timeout=timeout,
            )

            # Create metadata
            metadata = self._create_response_metadata(
                start_time, search_params, response, include_metadata=include_metadata
            )
            if metadata:
                response["search_metadata"] = metadata

            # Notify callback manager about completion
            if run_manager:
                result_count = len(response.get("results", []))
                duration = metadata.get("search_duration_seconds", 0) if metadata else 0
                run_manager.on_text(
                    f"Search completed: {result_count} results in {duration}s\n",
                    color="green",
                )

            return response

        except Exception as e:
            # Notify callback manager about error
            if run_manager:
                run_manager.on_text(f"Search failed: {e!s}\n", color="red")
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e

    async def _arun(
        self,
        objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        max_results: int = 10,
        excerpts: Optional[ExcerptSettings] = None,
        mode: Optional[str] = None,
        source_policy: Optional[dict[str, Union[str, list[str]]]] = None,
        fetch_policy: Optional[FetchPolicy] = None,
        *,
        include_metadata: bool = True,
        timeout: Optional[int] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        """Async execute the search using Parallel's Search API.

        Args:
            objective: Natural-language description of the research goal
            search_queries: List of specific search queries
            max_results: Maximum number of results (1-40)
            excerpts: Optional ExcerptSettings for controlling excerpt length
            mode: Search mode ('one-shot' or 'agentic')
            source_policy: Optional source policy for domain filtering
            fetch_policy: Optional FetchPolicy for cache vs live content
            include_metadata: Whether to include metadata
            timeout: Request timeout in seconds
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing search results with metadata
        """
        start_time = datetime.now()

        # Notify callback manager about search start
        if run_manager:
            query_desc = objective or f"{len(search_queries or [])} search queries"
            await run_manager.on_text(
                f"Starting async web search: {query_desc}\n", color="blue"
            )

        # Convert ExcerptSettings and FetchPolicy to dict if provided
        excerpts_dict = excerpts.model_dump(exclude_none=True) if excerpts else None
        fetch_policy_dict = (
            fetch_policy.model_dump(exclude_none=True) if fetch_policy else None
        )

        search_params = {
            "objective": objective,
            "search_queries": search_queries,
            "max_results": max_results,
            "excerpts": excerpts_dict,
            "mode": mode,
            "source_policy": source_policy,
            "fetch_policy": fetch_policy_dict,
        }

        try:
            # Notify about search execution
            if run_manager:
                await run_manager.on_text(
                    "Executing async search...\n",
                    color="yellow",
                )

            # Use the pre-initialized async client for better performance
            response = await self._async_client.search(
                objective=objective,
                search_queries=search_queries,
                max_results=max_results,
                excerpts=excerpts_dict,
                mode=mode,
                source_policy=source_policy,
                fetch_policy=fetch_policy_dict,
                timeout=timeout,
            )

            # Create metadata
            metadata = self._create_response_metadata(
                start_time, search_params, response, include_metadata=include_metadata
            )
            if metadata:
                response["search_metadata"] = metadata

            # Notify callback manager about completion
            if run_manager:
                result_count = len(response.get("results", []))
                duration = metadata.get("search_duration_seconds", 0) if metadata else 0
                await run_manager.on_text(
                    f"Async search completed: {result_count} results in {duration}s\n",
                    color="green",
                )

            return response

        except Exception as e:
            # Notify callback manager about error
            if run_manager:
                await run_manager.on_text(f"Async search failed: {e!s}\n", color="red")
            msg = f"Error calling Parallel Search API: {e!s}"
            raise ValueError(msg) from e
