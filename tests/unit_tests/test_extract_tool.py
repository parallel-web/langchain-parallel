"""Unit tests for Parallel Extract Tool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_parallel._types import ExcerptSettings, FetchPolicy, FullContentSettings
from langchain_parallel.extract_tool import ParallelExtractTool


def _make_response(payload: dict) -> SimpleNamespace:
    """Build a mock SDK response with .model_dump()."""
    return SimpleNamespace(model_dump=lambda: dict(payload))


class TestParallelExtractTool:
    """Test cases for ParallelExtractTool."""

    def test_extract_tool_initialization(self) -> None:
        """Test extract tool can be initialized."""
        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            assert tool.name == "parallel_extract"
            assert tool.base_url == "https://api.parallel.ai"
            assert tool.max_chars_per_extract is None

    def test_extract_tool_initialization_with_params(self) -> None:
        """Test extract tool initialization with custom parameters."""
        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool(max_chars_per_extract=3000)
            assert tool.max_chars_per_extract == 3000

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_single_url(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Single URL hits the GA endpoint and returns a list of dicts."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test Article",
                        "full_content": "This is the extracted content.",
                        "publish_date": "2024-01-01",
                    },
                ],
                "errors": [],
            },
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool._run(urls=["https://example.com"], full_content=True)

            sync_client.extract.assert_called_once()
            sync_client.beta.extract.assert_not_called()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["url"] == "https://example.com"
            assert result[0]["title"] == "Test Article"
            assert result[0]["content"] == "This is the extracted content."
            assert result[0]["publish_date"] == "2024-01-01"

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_multiple_urls(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Test extraction with multiple URLs."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example1.com",
                        "title": "Article 1",
                        "full_content": "Content 1",
                    },
                    {
                        "url": "https://example2.com",
                        "title": "Article 2",
                        "full_content": "Content 2",
                    },
                ],
                "errors": [],
            },
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool._run(
                urls=["https://example1.com", "https://example2.com"],
                full_content=True,
            )
            assert [r["content"] for r in result] == ["Content 1", "Content 2"]

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_with_errors(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Test extraction handles errors gracefully."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example1.com",
                        "title": "Article 1",
                        "full_content": "Content 1",
                    },
                ],
                "errors": [
                    {
                        "url": "https://example2.com",
                        "error_type": "http_error",
                        "http_status_code": 404,
                        "content": None,
                    },
                ],
            },
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool._run(
                urls=["https://example1.com", "https://example2.com"],
                full_content=True,
            )
            assert len(result) == 2
            assert result[0]["content"] == "Content 1"
            assert result[1]["error_type"] == "http_error"
            assert result[1]["http_status_code"] == 404
            assert "Error: http_error" in result[1]["content"]

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_full_content_precedence_tool_level_default(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Tool-level max_chars_per_extract applies when full_content=True."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test",
                        "full_content": "Short",
                    },
                ],
                "errors": [],
            },
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool(max_chars_per_extract=5000)
            tool._run(urls=["https://example.com"], full_content=True)
            kwargs = sync_client.extract.call_args.kwargs
            assert kwargs["advanced_settings"]["full_content"] == {
                "max_chars_per_result": 5000,
            }

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_full_content_precedence_explicit_settings_wins(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Explicit FullContentSettings beats the tool-level cap."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool(max_chars_per_extract=5000)
            tool._run(
                urls=["https://example.com"],
                full_content=FullContentSettings(max_chars_per_result=200),
            )
            kwargs = sync_client.extract.call_args.kwargs
            assert kwargs["advanced_settings"]["full_content"] == {
                "max_chars_per_result": 200,
            }

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_full_content_false_omits_key(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """full_content=False produces no full_content key in advanced_settings."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            tool._run(urls=["https://example.com"], full_content=False)
            kwargs = sync_client.extract.call_args.kwargs
            advanced = kwargs.get("advanced_settings") or {}
            assert "full_content" not in advanced

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_excerpts_default_is_no_op(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Default `excerpts=None` adds no excerpt_settings."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            tool._run(urls=["https://example.com"])
            advanced = sync_client.extract.call_args.kwargs.get("advanced_settings")
            assert advanced is None

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_excerpts_bool_now_rejected(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """The legacy `excerpts: bool` form was removed in 0.4.0."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            # The boolean form was removed in 0.4.0; both True and False now
            # fail pydantic validation on the typed `Optional[ExcerptSettings]`
            # field.
            from pydantic import ValidationError

            with pytest.raises(ValidationError):
                tool.invoke({"urls": ["https://example.com"], "excerpts": False})
            with pytest.raises(ValidationError):
                tool.invoke({"urls": ["https://example.com"], "excerpts": True})

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_advanced_settings_envelope(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """ExcerptSettings + FetchPolicy + full_content nest into advanced_settings."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            tool._run(
                urls=["https://example.com"],
                excerpts=ExcerptSettings(max_chars_per_result=2000),
                full_content=FullContentSettings(max_chars_per_result=8000),
                fetch_policy=FetchPolicy(max_age_seconds=86400),
            )
            kwargs = sync_client.extract.call_args.kwargs
            assert kwargs["advanced_settings"] == {
                "excerpt_settings": {"max_chars_per_result": 2000},
                "fetch_policy": {
                    "max_age_seconds": 86400,
                    "disable_cache_fallback": False,
                },
                "full_content": {"max_chars_per_result": 8000},
            }

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_top_level_passthrough_fields(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """`max_chars_total`, `client_model`, `session_id` flow through verbatim."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "e", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            tool._run(
                urls=["https://example.com"],
                max_chars_total=42_000,
                client_model="claude-opus-4-7",
                session_id="sess-1",
            )
            kwargs = sync_client.extract.call_args.kwargs
            assert kwargs["max_chars_total"] == 42_000
            assert kwargs["client_model"] == "claude-opus-4-7"
            assert kwargs["session_id"] == "sess-1"

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_handles_api_error(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Test extract tool wraps API errors as ValueError."""
        sync_client = Mock()
        sync_client.extract.side_effect = Exception("API Error")
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            with pytest.raises(
                ValueError, match="Error calling Parallel Extract API: API Error"
            ):
                tool._run(urls=["https://example.com"])

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    @pytest.mark.asyncio
    async def test_extract_async_functionality(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Async path uses async client."""
        async_client = Mock()
        async_client.extract = AsyncMock(
            return_value=_make_response(
                {
                    "extract_id": "extract-1",
                    "results": [
                        {
                            "url": "https://example.com",
                            "title": "Async",
                            "full_content": "Async content",
                        },
                    ],
                    "errors": [],
                },
            ),
        )
        mock_async_factory.return_value = async_client
        mock_sync_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = await tool._arun(urls=["https://example.com"])
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["content"] == "Async content"

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    @pytest.mark.asyncio
    async def test_extract_async_handles_api_error(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Async API exceptions are wrapped as ValueError."""
        async_client = Mock()
        async_client.extract = AsyncMock(side_effect=Exception("Async API Error"))
        mock_async_factory.return_value = async_client
        mock_sync_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            with pytest.raises(
                ValueError,
                match="Error calling Parallel Extract API: Async API Error",
            ):
                await tool._arun(urls=["https://example.com"])

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_empty_results(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Test extract tool handles empty results."""
        sync_client = Mock()
        sync_client.extract.return_value = _make_response(
            {"extract_id": "extract-1", "results": [], "errors": []},
        )
        mock_sync_factory.return_value = sync_client
        mock_async_factory.return_value = Mock()

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool._run(urls=["https://example.com"])
            assert result == []

    def test_extract_empty_urls_raises(self) -> None:
        """urls=[] raises ValueError."""
        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            with pytest.raises(ValueError, match="At least one URL"):
                tool._run(urls=[])
