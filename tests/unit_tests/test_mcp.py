"""Unit tests for the MCP toolkit factory."""

from __future__ import annotations

import asyncio
import builtins
import sys
from unittest.mock import patch

import pytest


def test_missing_optional_dependency_raises() -> None:
    """If `langchain-mcp-adapters` is not installed, the factory raises clearly."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("langchain_mcp_adapters"):
            msg = f"No module named '{name}'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", fake_import):
        sys.modules.pop("langchain_mcp_adapters", None)
        sys.modules.pop("langchain_mcp_adapters.client", None)
        from langchain_parallel.mcp import parallel_mcp_toolkit

        with pytest.raises(ImportError, match=r"langchain-parallel\[mcp\]"):
            asyncio.run(parallel_mcp_toolkit(api_key="k"))


def test_must_include_at_least_one_server() -> None:
    """Both flags False -> clear ValueError."""
    from langchain_parallel.mcp import parallel_mcp_toolkit

    with pytest.raises(ValueError, match="At least one of"):
        asyncio.run(
            parallel_mcp_toolkit(
                api_key="k",
                include_search=False,
                include_tasks=False,
            ),
        )
