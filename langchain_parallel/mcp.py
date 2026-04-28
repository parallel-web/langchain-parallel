"""LangChain wrappers for Parallel's hosted MCP servers.

Parallel hosts two Streamable-HTTP MCP servers:

- **Search MCP** at ``https://search.parallel.ai/mcp`` exposes
  ``web_search`` and ``web_fetch`` tools backed by the Search and Extract
  APIs.
- **Task MCP** at ``https://task-mcp.parallel.ai/mcp`` exposes
  ``createDeepResearch``, ``createTaskGroup``, ``getStatus``, and
  ``getResultMarkdown`` for deep-research workflows.

When you can — for the Python use case where this package is installed —
the native tool surfaces (``ParallelSearchTool``, ``ParallelExtractTool``,
``ParallelTaskRunTool``, ``ParallelDeepResearch``) are simpler, faster,
and don't require an extra dependency. Use this MCP toolkit when you
want to mix Parallel tools with other MCP servers in the same agent
runtime, or when you've standardized on MCP for cross-language reasons.

Requires ``langchain-mcp-adapters`` as an optional dependency::

    pip install "langchain-parallel[mcp]"
"""

from __future__ import annotations

from typing import Any, Optional

from ._client import get_api_key

SEARCH_MCP_URL = "https://search.parallel.ai/mcp"
SEARCH_MCP_OAUTH_URL = "https://search.parallel.ai/mcp-oauth"
TASK_MCP_URL = "https://task-mcp.parallel.ai/mcp"


def _import_mcp_adapter() -> Any:
    """Import langchain-mcp-adapters or raise with an install hint."""
    try:
        from langchain_mcp_adapters.client import (  # type: ignore[import-not-found]
            MultiServerMCPClient,
        )
    except ImportError as e:  # pragma: no cover - exercised at import time
        msg = (
            "parallel_mcp_toolkit requires the optional `langchain-mcp-adapters` "
            "dependency. Install it with:\n"
            '    pip install "langchain-parallel[mcp]"\n'
            "or directly:\n"
            "    pip install langchain-mcp-adapters"
        )
        raise ImportError(msg) from e
    return MultiServerMCPClient


async def parallel_mcp_toolkit(
    *,
    api_key: Optional[str] = None,
    include_search: bool = True,
    include_tasks: bool = True,
) -> list[Any]:
    """Return Parallel's hosted MCP tools as LangChain ``BaseTool``s.

    This is a *thin* wrapper around `MultiServerMCPClient.get_tools()` from
    `langchain-mcp-adapters`, pre-configured for Parallel's hosted Search
    and Task MCP endpoints. Both endpoints authenticate via the
    ``Authorization: Bearer <api_key>`` header.

    Args:
        api_key: Parallel API key. Reads ``PARALLEL_API_KEY`` if omitted.
        include_search: If True, attach the Search MCP server (web_search,
            web_fetch tools).
        include_tasks: If True, attach the Task MCP server (createDeepResearch,
            createTaskGroup, getStatus, getResultMarkdown tools).

    Returns:
        A list of LangChain ``BaseTool``s — one per remote MCP tool.
    """
    if not include_search and not include_tasks:
        msg = "At least one of include_search or include_tasks must be True."
        raise ValueError(msg)

    multi_server_mcp_client_cls = _import_mcp_adapter()
    resolved_key = get_api_key(api_key)
    auth_headers = {"Authorization": f"Bearer {resolved_key}"}

    connections: dict[str, dict[str, Any]] = {}
    if include_search:
        connections["parallel_search"] = {
            "url": SEARCH_MCP_OAUTH_URL,
            "transport": "streamable_http",
            "headers": auth_headers,
        }
    if include_tasks:
        connections["parallel_tasks"] = {
            "url": TASK_MCP_URL,
            "transport": "streamable_http",
            "headers": auth_headers,
        }

    client = multi_server_mcp_client_cls(connections)
    return await client.get_tools()
