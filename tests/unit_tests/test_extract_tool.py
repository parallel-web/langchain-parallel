"""Unit tests for Parallel Extract Tool."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_parallel.extract_tool import ParallelExtractTool


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
            tool = ParallelExtractTool(
                max_chars_per_extract=3000,
            )
            assert tool.max_chars_per_extract == 3000

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_single_url(self, mock_get_extract_client: Mock) -> None:
        """Test extracting content from a single URL."""
        # Mock the extract client
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test Article",
                    "full_content": "This is the extracted content.",
                    "publish_date": "2024-01-01",
                }
            ],
            "errors": [],
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool.invoke({"urls": ["https://example.com"]})

            assert len(result) == 1
            assert result[0]["url"] == "https://example.com"
            assert result[0]["title"] == "Test Article"
            assert result[0]["content"] == "This is the extracted content."
            assert result[0]["publish_date"] == "2024-01-01"

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_multiple_urls(self, mock_get_extract_client: Mock) -> None:
        """Test extraction with multiple URLs."""
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
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
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool.invoke(
                {"urls": ["https://example1.com", "https://example2.com"]}
            )

            assert len(result) == 2
            assert result[0]["content"] == "Content 1"
            assert result[1]["content"] == "Content 2"

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_with_errors(self, mock_get_extract_client: Mock) -> None:
        """Test extraction handles errors gracefully."""
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
            "results": [
                {
                    "url": "https://example1.com",
                    "title": "Article 1",
                    "full_content": "Content 1",
                }
            ],
            "errors": [
                {
                    "url": "https://example2.com",
                    "error_type": "http_error",
                    "http_status_code": 404,
                    "content": None,
                }
            ],
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool.invoke(
                {"urls": ["https://example1.com", "https://example2.com"]}
            )

            assert len(result) == 2
            assert result[0]["content"] == "Content 1"
            assert result[1]["url"] == "https://example2.com"
            assert "Error: http_error" in result[1]["content"]
            assert result[1]["error_type"] == "http_error"
            assert result[1]["http_status_code"] == 404

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_with_max_chars(self, mock_get_extract_client: Mock) -> None:
        """Test extraction with max_chars_per_extract limit."""
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "full_content": "Short content",
                }
            ],
            "errors": [],
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool(max_chars_per_extract=5000)
            tool.invoke({"urls": ["https://example.com"]})

            # Verify extract was called with full_content config
            call_kwargs = mock_client.extract.call_args[1]
            assert call_kwargs["full_content"] == {"max_chars_per_result": 5000}

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_handles_api_error(self, mock_get_extract_client: Mock) -> None:
        """Test extract tool handles API errors gracefully."""
        mock_client = Mock()
        mock_client.extract.side_effect = Exception("API Error")
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()

            with pytest.raises(
                ValueError, match="Error calling Parallel Extract API: API Error"
            ):
                tool.invoke({"urls": ["https://example.com"]})

    @patch("langchain_parallel.extract_tool.get_async_extract_client")
    @patch("langchain_parallel.extract_tool.get_extract_client")
    @pytest.mark.asyncio
    async def test_extract_async_functionality(
        self, mock_get_extract_client: Mock, mock_get_async_extract_client: Mock
    ) -> None:
        """Test async extraction functionality."""
        # Mock sync client (needed for initialization)
        mock_sync_client = Mock()
        mock_get_extract_client.return_value = mock_sync_client

        # Mock async client
        mock_async_client = Mock()
        mock_async_client.extract = AsyncMock(
            return_value={
                "extract_id": "extract-123",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Test Article",
                        "full_content": "Async content",
                    }
                ],
                "errors": [],
            }
        )
        mock_get_async_extract_client.return_value = mock_async_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = await tool.ainvoke({"urls": ["https://example.com"]})

            assert len(result) == 1
            assert result[0]["content"] == "Async content"

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_metadata_fields(self, mock_get_extract_client: Mock) -> None:
        """Test that all metadata fields are properly extracted."""
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test Article",
                    "full_content": "Content",
                    "publish_date": "2024-01-01",
                }
            ],
            "errors": [],
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool.invoke({"urls": ["https://example.com"]})

            assert result[0]["url"] == "https://example.com"
            assert result[0]["title"] == "Test Article"
            assert result[0]["content"] == "Content"
            assert result[0].get("publish_date") == "2024-01-01"

    @patch("langchain_parallel.extract_tool.get_extract_client")
    def test_extract_empty_results(self, mock_get_extract_client: Mock) -> None:
        """Test extract tool handles empty results."""
        mock_client = Mock()
        mock_client.extract.return_value = {
            "extract_id": "extract-123",
            "results": [],
            "errors": [],
        }
        mock_get_extract_client.return_value = mock_client

        with patch(
            "langchain_parallel.extract_tool.get_api_key", return_value="test-key"
        ):
            tool = ParallelExtractTool()
            result = tool.invoke({"urls": ["https://example.com"]})

            assert len(result) == 0
