"""Unit tests for Parallel Extract Tool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

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
            assert tool.response_format == "content_and_artifact"

    def test_extract_tool_initialization_with_params(self) -> None:
        """Test extract tool initialization with custom parameters."""
        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool(
                max_chars_per_extract=3000,
            )
            assert tool.max_chars_per_extract == 3000

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_single_url(
        self,
        mock_async_factory: Mock,
        mock_sync_factory: Mock,
    ) -> None:
        """Test extracting content from a single URL via the GA endpoint."""
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
            content, artifact = tool._run(
                urls=["https://example.com"], full_content=True
            )

            sync_client.extract.assert_called_once()
            sync_client.beta.extract.assert_not_called()
            assert len(artifact) == 1
            assert artifact[0]["url"] == "https://example.com"
            assert artifact[0]["title"] == "Test Article"
            assert artifact[0]["content"] == "This is the extracted content."
            assert artifact[0]["publish_date"] == "2024-01-01"
            assert "Test Article" in content

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
            _, artifact = tool._run(
                urls=["https://example1.com", "https://example2.com"],
                full_content=True,
            )
            assert [r["content"] for r in artifact] == ["Content 1", "Content 2"]

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
            _, artifact = tool._run(
                urls=["https://example1.com", "https://example2.com"],
                full_content=True,
            )
            assert len(artifact) == 2
            assert artifact[0]["content"] == "Content 1"
            assert artifact[1]["error_type"] == "http_error"
            assert artifact[1]["http_status_code"] == 404
            assert "Error: http_error" in artifact[1]["content"]

    @patch("langchain_parallel.extract_tool.get_parallel_client")
    @patch("langchain_parallel.extract_tool.get_async_parallel_client")
    def test_extract_max_chars_default(
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
                "max_chars_per_result": 5000
            }

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
            _, artifact = await tool._arun(urls=["https://example.com"])
            assert len(artifact) == 1
            assert artifact[0]["content"] == "Async content"

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
            _, artifact = tool._run(urls=["https://example.com"])
            assert artifact == []
