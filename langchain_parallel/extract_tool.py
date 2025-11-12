"""Parallel Extract Tool for LangChain."""

from __future__ import annotations

from typing import Any, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_extract_client, get_extract_client
from ._types import ExcerptSettings, FetchPolicy, FullContentSettings


class ParallelExtractInput(BaseModel):
    """Input schema for Parallel Extract Tool."""

    urls: list[str] = Field(description="List of URLs to extract content from")

    search_objective: Optional[str] = Field(
        default=None,
        description=(
            "If provided, focuses extracted content on the specified search objective"
        ),
    )

    search_queries: Optional[list[str]] = Field(
        default=None,
        description=(
            "If provided, focuses extracted content on the specified keyword search "
            "queries"
        ),
    )

    excerpts: Union[bool, ExcerptSettings] = Field(
        default=True,
        description=(
            "Include excerpts from each URL relevant to the search objective and "
            "queries. Can be boolean or ExcerptSettings object."
        ),
    )

    full_content: Union[bool, FullContentSettings] = Field(
        default=False,
        description=(
            "Include full content from each URL. Can be boolean or "
            "FullContentSettings object."
        ),
    )

    fetch_policy: Optional[FetchPolicy] = Field(
        default=None,
        description=(
            "Fetch policy: determines when to return content from the cache "
            "(faster) vs fetching live content (fresher)"
        ),
    )

    timeout: Optional[float] = Field(
        default=None,
        description=(
            "Request timeout in seconds. If not specified, uses default of "
            "5 seconds per URL."
        ),
    )


class ParallelExtractTool(BaseTool):
    """Parallel Extract Tool.

    This tool extracts clean, structured content from web pages using the
    Parallel Extract API.

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
        max_chars_per_extract: Optional[int]
            Maximum characters per extracted result.

    Instantiation:
        .. code-block:: python

            from langchain_parallel import ParallelExtractTool

            # Basic instantiation
            tool = ParallelExtractTool()

            # With custom API key and parameters
            tool = ParallelExtractTool(
                api_key="your-api-key",
                max_chars_per_extract=5000
            )

    Invocation:
        .. code-block:: python

            # Extract content from URLs
            result = tool.invoke({
                "urls": [
                    "https://example.com/article1",
                    "https://example.com/article2"
                ]
            })

            # Result is a list of dicts with url, title, and content
            for item in result:
                print(f"Title: {item['title']}")
                print(f"URL: {item['url']}")
                print(f"Content: {item['content'][:200]}...")

    Response Format:
        Returns a list of dictionaries, each containing:
        - url: The URL that was extracted
        - title: Title of the webpage
        - content: Full extracted content as markdown
        - publish_date: Publish date if available (optional)
    """

    name: str = "parallel_extract"
    description: str = (
        "Extract clean, structured content from web pages. "
        "Input should be a list of URLs to extract content from. "
        "Returns extracted content formatted as markdown."
    )
    args_schema: type[BaseModel] = ParallelExtractInput

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. If not provided, will be read from env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

    max_chars_per_extract: Optional[int] = None
    """Maximum characters per extracted result."""

    _client: Any = None
    """Synchronous extract client (initialized after validation)."""

    _async_client: Any = None
    """Asynchronous extract client (initialized after validation)."""

    @model_validator(mode="after")
    def validate_environment(self) -> ParallelExtractTool:
        """Validate the environment and initialize clients."""
        # Get API key from parameter or environment
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None
        )

        # Initialize both sync and async clients once
        self._client = get_extract_client(api_key_str, self.base_url)
        self._async_client = get_async_extract_client(api_key_str, self.base_url)

        return self

    def _prepare_extract_params(
        self,
        excerpts: Union[bool, ExcerptSettings],
        full_content: Union[bool, FullContentSettings],
        fetch_policy: Optional[FetchPolicy],
    ) -> tuple[Any, Any, Optional[dict[str, Any]]]:
        """Prepare parameters for extract API call.

        Args:
            excerpts: Include excerpts (boolean or ExcerptSettings)
            full_content: Include full content (boolean or FullContentSettings)
            fetch_policy: Optional fetch policy for cache vs live content

        Returns:
            Tuple of (excerpts_param, full_content_param, fetch_policy_param)
        """
        # Build full_content config
        full_content_param = full_content
        if self.max_chars_per_extract and isinstance(full_content, bool):
            # Use tool-level config if full_content is just a boolean
            full_content_param = {"max_chars_per_result": self.max_chars_per_extract}
        elif isinstance(full_content, FullContentSettings):
            full_content_param = full_content.model_dump(exclude_none=True)

        # Build excerpts config
        excerpts_param = excerpts
        if isinstance(excerpts, ExcerptSettings):
            excerpts_param = excerpts.model_dump(exclude_none=True)

        # Build fetch_policy config
        fetch_policy_param = None
        if fetch_policy:
            fetch_policy_param = fetch_policy.model_dump(exclude_none=True)

        return excerpts_param, full_content_param, fetch_policy_param

    def _format_extract_response(
        self, extract_response: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Format the extract API response.

        Args:
            extract_response: Raw response from the extract API

        Returns:
            List of formatted result dictionaries
        """
        results = extract_response.get("results", [])
        errors = extract_response.get("errors", [])

        # Format results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "title": result.get("title"),
            }

            # Add excerpts if present
            if "excerpts" in result and result["excerpts"] is not None:
                formatted_result["excerpts"] = result["excerpts"]
                # Combine excerpts into content field for backward compatibility
                # Excerpts are a list of strings, join them with newlines
                formatted_result["content"] = "\n\n".join(result["excerpts"])

            # Add full_content if present and not None
            # (overrides excerpts-based content)
            if "full_content" in result and result["full_content"] is not None:
                formatted_result["full_content"] = result["full_content"]
                # For backward compatibility, also set as "content"
                formatted_result["content"] = result["full_content"]

            # Add optional fields if present
            if "publish_date" in result:
                formatted_result["publish_date"] = result["publish_date"]

            formatted_results.append(formatted_result)

        # If there were errors, add them to the results with error info
        formatted_results.extend(
            [
                {
                    "url": error.get("url"),
                    "title": None,
                    "content": f"Error: {error.get('error_type', 'Unknown error')}",
                    "error_type": error.get("error_type"),
                    "http_status_code": error.get("http_status_code"),
                }
                for error in errors
            ]
        )

        return formatted_results

    def _run(
        self,
        urls: list[str],
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Union[bool, ExcerptSettings] = True,
        full_content: Union[bool, FullContentSettings] = False,
        fetch_policy: Optional[FetchPolicy] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs.

        Args:
            urls: List of URLs to extract content from
            search_objective: Optional search objective to focus extraction
            search_queries: Optional keyword search queries to focus extraction
            excerpts: Include excerpts (boolean or ExcerptSettings)
            full_content: Include full content (boolean or FullContentSettings)
            fetch_policy: Optional fetch policy for cache vs live content
            timeout: Request timeout in seconds (defaults to 5 seconds per URL)
            run_manager: Callback manager for the tool run

        Returns:
            List of dictionaries with extracted content
        """
        # Notify callback manager about extraction start
        if run_manager:
            url_count = len(urls)
            url_desc = f"{url_count} URL{'s' if url_count != 1 else ''}"
            run_manager.on_text(
                f"Starting content extraction from {url_desc}\n", color="blue"
            )

        try:
            # Prepare parameters for the extract API call
            excerpts_param, full_content_param, fetch_policy_param = (
                self._prepare_extract_params(excerpts, full_content, fetch_policy)
            )

            # Notify about extraction execution
            if run_manager:
                run_manager.on_text("Executing extraction...\n", color="yellow")

            # Extract content from URLs using the pre-initialized client
            extract_response = self._client.extract(
                urls=urls,
                objective=search_objective,
                search_queries=search_queries,
                excerpts=excerpts_param,
                full_content=full_content_param,
                fetch_policy=fetch_policy_param,
                timeout=timeout,
            )

            # Format and return the response
            result = self._format_extract_response(extract_response)

            # Notify callback manager about completion
            if run_manager:
                success_count = sum(1 for item in result if "error_type" not in item)
                error_count = len(result) - success_count
                if error_count > 0:
                    run_manager.on_text(
                        f"Extraction completed: {success_count} succeeded, "
                        f"{error_count} failed\n",
                        color="green",
                    )
                else:
                    url_text = "URL" if success_count == 1 else "URLs"
                    run_manager.on_text(
                        f"Extraction completed: {success_count} {url_text} processed\n",
                        color="green",
                    )

            return result

        except Exception as e:
            # Notify callback manager about error
            if run_manager:
                run_manager.on_text(f"Extraction failed: {e!s}\n", color="red")
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e

    async def _arun(
        self,
        urls: list[str],
        search_objective: Optional[str] = None,
        search_queries: Optional[list[str]] = None,
        excerpts: Union[bool, ExcerptSettings] = True,
        full_content: Union[bool, FullContentSettings] = False,
        fetch_policy: Optional[FetchPolicy] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list[dict[str, Any]]:
        """Extract content from URLs asynchronously.

        Args:
            urls: List of URLs to extract content from
            search_objective: Optional search objective to focus extraction
            search_queries: Optional keyword search queries to focus extraction
            excerpts: Include excerpts (boolean or ExcerptSettings)
            full_content: Include full content (boolean or FullContentSettings)
            fetch_policy: Optional fetch policy for cache vs live content
            timeout: Request timeout in seconds (defaults to 5 seconds per URL)
            run_manager: Async callback manager for the tool run

        Returns:
            List of dictionaries with extracted content
        """
        # Notify callback manager about extraction start
        if run_manager:
            url_count = len(urls)
            url_desc = f"{url_count} URL{'s' if url_count != 1 else ''}"
            await run_manager.on_text(
                f"Starting async content extraction from {url_desc}\n", color="blue"
            )

        try:
            # Prepare parameters for the extract API call
            excerpts_param, full_content_param, fetch_policy_param = (
                self._prepare_extract_params(excerpts, full_content, fetch_policy)
            )

            # Notify about extraction execution
            if run_manager:
                await run_manager.on_text(
                    "Executing async extraction...\n", color="yellow"
                )

            # Extract content from URLs using the pre-initialized async client
            extract_response = await self._async_client.extract(
                urls=urls,
                objective=search_objective,
                search_queries=search_queries,
                excerpts=excerpts_param,
                full_content=full_content_param,
                fetch_policy=fetch_policy_param,
                timeout=timeout,
            )

            # Format and return the response
            result = self._format_extract_response(extract_response)

            # Notify callback manager about completion
            if run_manager:
                success_count = sum(1 for item in result if "error_type" not in item)
                error_count = len(result) - success_count
                if error_count > 0:
                    await run_manager.on_text(
                        f"Async extraction completed: {success_count} succeeded, "
                        f"{error_count} failed\n",
                        color="green",
                    )
                else:
                    url_text = "URL" if success_count == 1 else "URLs"
                    await run_manager.on_text(
                        f"Async extraction completed: {success_count} {url_text} "
                        f"processed\n",
                        color="green",
                    )

            return result

        except Exception as e:
            # Notify callback manager about error
            if run_manager:
                await run_manager.on_text(
                    f"Async extraction failed: {e!s}\n", color="red"
                )
            msg = f"Error calling Parallel Extract API: {e!s}"
            raise ValueError(msg) from e
