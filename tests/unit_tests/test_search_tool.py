"""Unit tests for Parallel Search functionality."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_parallel.search_tool import ParallelWebSearchTool, _validate_mode
from langchain_parallel.types import ExcerptSettings, FetchPolicy, SourcePolicy


def _make_response(payload: dict) -> SimpleNamespace:
    """Build a mock SDK response with .model_dump()."""
    return SimpleNamespace(model_dump=lambda: dict(payload))


class TestParallelWebSearchTool:
    """Test cases for ParallelWebSearchTool."""

    def test_tool_initialization(self) -> None:
        """Test tool can be initialized."""
        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            assert tool.name == "parallel_web_search"
            assert "Search the web" in tool.description

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_run_uses_v1_endpoint_when_search_queries_provided(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """search_queries triggers the GA endpoint and returns a dict."""
        sync_client = Mock()
        sync_client.search.return_value = _make_response(
            {
                "search_id": "search-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test",
                        "excerpts": ["snippet"],
                    },
                ],
            },
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            result = tool._run(
                search_queries=["query 1"],
                max_results=3,
                mode="advanced",
            )
            sync_client.search.assert_called_once()
            sync_client.beta.search.assert_not_called()
            kwargs = sync_client.search.call_args.kwargs
            assert kwargs["search_queries"] == ["query 1"]
            assert kwargs["mode"] == "advanced"
            assert kwargs["advanced_settings"] == {"max_results": 3}
            assert isinstance(result, dict)
            assert result["search_id"] == "search-1"
            assert "search_duration_seconds" in result["search_metadata"]

    def test_run_requires_search_queries(self) -> None:
        """The v1beta-fallback path was removed in 0.4.0; missing queries raises now."""
        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            with pytest.raises(ValueError, match="search_queries is required"):
                tool._run(search_queries=[], objective="What is AI?")

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_run_rejects_legacy_mode(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Legacy mode strings now raise (the deprecation shim was removed in 0.4.0)."""
        sync_client = Mock()
        sync_client.search.return_value = _make_response(
            {"search_id": "s", "results": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            with pytest.raises(ValueError, match="Invalid mode"):
                tool._run(search_queries=["q"], mode="agentic")

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_advanced_settings_envelope_pydantic(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Pydantic models pack into `advanced_settings` correctly."""
        sync_client = Mock()
        sync_client.search.return_value = _make_response(
            {"search_id": "s", "results": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            tool._run(
                search_queries=["q"],
                excerpts=ExcerptSettings(max_chars_per_result=1500),
                fetch_policy=FetchPolicy(max_age_seconds=86400),
                source_policy=SourcePolicy(
                    include_domains=["nature.com"],
                    after_date="2025-01-01",
                ),
                location="us",
                max_results=15,
            )
            kwargs = sync_client.search.call_args.kwargs
            assert kwargs["advanced_settings"] == {
                "excerpt_settings": {"max_chars_per_result": 1500},
                "fetch_policy": {
                    "max_age_seconds": 86400,
                    "disable_cache_fallback": False,
                },
                "source_policy": {
                    "include_domains": ["nature.com"],
                    "after_date": __import__("datetime").date(2025, 1, 1),
                },
                "max_results": 15,
                "location": "us",
            }

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_advanced_settings_envelope_dict(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Raw-dict source_policy is accepted alongside the pydantic model."""
        sync_client = Mock()
        sync_client.search.return_value = _make_response(
            {"search_id": "s", "results": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            tool._run(
                search_queries=["q"],
                source_policy={"include_domains": ["nature.com"]},
                location="us",
                max_results=15,
            )
            kwargs = sync_client.search.call_args.kwargs
            assert kwargs["advanced_settings"] == {
                "source_policy": {"include_domains": ["nature.com"]},
                "max_results": 15,
                "location": "us",
            }

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_top_level_passthrough_fields(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """`max_chars_total`, `client_model`, `session_id` flow through verbatim."""
        sync_client = Mock()
        sync_client.search.return_value = _make_response(
            {"search_id": "s", "results": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            tool._run(
                search_queries=["q"],
                max_chars_total=42_000,
                client_model="claude-opus-4-7",
                session_id="sess-1",
            )
            kwargs = sync_client.search.call_args.kwargs
            assert kwargs["max_chars_total"] == 42_000
            assert kwargs["client_model"] == "claude-opus-4-7"
            assert kwargs["session_id"] == "sess-1"

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    def test_run_handles_api_error(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """API exceptions are wrapped as ValueError."""
        sync_client = Mock()
        sync_client.search.side_effect = Exception("API Error")
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            with pytest.raises(
                ValueError,
                match="Error calling Parallel Search API: API Error",
            ):
                tool._run(search_queries=["q"])

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    async def test_async_functionality(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Async path uses the async client."""
        async_client = Mock()
        async_client.search = AsyncMock(
            return_value=_make_response(
                {
                    "search_id": "async-1",
                    "results": [{"url": "https://example.com", "title": "Async"}],
                },
            ),
        )
        mock_async_factory.return_value = async_client
        mock_sync_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            result = await tool._arun(search_queries=["q"])
            assert isinstance(result, dict)
            assert result["search_id"] == "async-1"

    @patch("langchain_parallel.search_tool.get_parallel_client")
    @patch("langchain_parallel.search_tool.get_async_parallel_client")
    async def test_async_handles_api_error(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Async API exceptions are wrapped as ValueError."""
        async_client = Mock()
        async_client.search = AsyncMock(side_effect=Exception("Async API Error"))
        mock_async_factory.return_value = async_client
        mock_sync_factory.return_value = Mock()

        with patch(
            "langchain_parallel.search_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelWebSearchTool()
            with pytest.raises(
                ValueError,
                match="Error calling Parallel Search API: Async API Error",
            ):
                await tool._arun(search_queries=["q"])


def test_parallel_search_tool_is_alias_of_parallel_web_search_tool() -> None:
    """``ParallelSearchTool`` is the new canonical name; old name still works."""
    from langchain_parallel import ParallelSearchTool, ParallelWebSearchTool

    assert ParallelSearchTool is ParallelWebSearchTool


class TestValidateMode:
    def test_passthrough(self) -> None:
        assert _validate_mode("basic") == "basic"
        assert _validate_mode("advanced") == "advanced"
        assert _validate_mode(None) is None

    def test_legacy_now_raises(self) -> None:
        # The DeprecationWarning shim was removed in 0.4.0; legacy mode strings
        # now raise instead of being mapped.
        for legacy in ("one-shot", "agentic", "fast"):
            with pytest.raises(ValueError, match="Invalid mode"):
                _validate_mode(legacy)

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            _validate_mode("nonsense")
