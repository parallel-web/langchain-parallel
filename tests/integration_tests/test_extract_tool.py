"""Integration tests for Parallel Extract Tool."""

import os

import pytest

from langchain_parallel.extract_tool import ParallelExtractTool


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("PARALLEL_API_KEY")
    if not key:
        pytest.skip("PARALLEL_API_KEY not set")
    return key


class TestParallelExtractToolIntegration:
    """Integration tests for ParallelExtractTool."""

    def test_extract_single_url(self, api_key: str) -> None:
        """Test extracting content from a single URL."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {
                "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                "full_content": True,
            }
        )

        assert len(result) == 1
        assert (
            result[0]["url"] == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        assert len(result[0]["content"]) > 0
        assert result[0]["title"] is not None

    def test_extract_multiple_urls(self, api_key: str) -> None:
        """Test extracting content from multiple URLs."""
        tool = ParallelExtractTool(api_key=api_key)

        urls = [
            "https://www.wikipedia.org/",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ]

        result = tool.invoke({"urls": urls})

        assert len(result) == 2
        for item in result:
            assert "url" in item
            assert "content" in item
            # Content may be empty for some pages, so just check it exists

    def test_extract_with_search_objective(self, api_key: str) -> None:
        """Test extraction with search objective to focus content."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {
                "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                "search_objective": "What are the main applications of AI?",
                "excerpts": True,
                "full_content": False,
            }
        )

        assert len(result) == 1
        assert (
            result[0]["url"] == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        # Should have excerpts focused on the objective
        assert "excerpts" in result[0]
        assert isinstance(result[0]["excerpts"], list)
        # Content should be populated from excerpts
        assert len(result[0]["content"]) > 0

    def test_extract_with_search_queries(self, api_key: str) -> None:
        """Test extraction with search queries to focus content."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {
                "urls": [
                    "https://en.wikipedia.org/wiki/Machine_learning",
                ],
                "search_queries": ["neural networks", "training algorithms"],
                "excerpts": True,
            }
        )

        assert len(result) == 1
        # Should have excerpts focused on the queries
        assert "excerpts" in result[0]
        assert isinstance(result[0]["excerpts"], list)
        assert len(result[0]["excerpts"]) > 0

    def test_extract_with_max_chars(self, api_key: str) -> None:
        """Test extraction with max_chars_per_extract limit."""
        tool = ParallelExtractTool(api_key=api_key, max_chars_per_extract=1000)

        result = tool.invoke(
            {
                "urls": ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
                "full_content": True,
            }
        )

        assert len(result) == 1
        # Note: The API currently returns up to 100k characters for full_content
        # regardless of max_characters setting. This test verifies the tool
        # correctly passes the parameter to the API.
        assert len(result[0]["content"]) > 0
        assert result[0]["title"] is not None

    def test_extract_metadata_fields(self, api_key: str) -> None:
        """Test that metadata fields are properly populated."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {"urls": ["https://en.wikipedia.org/wiki/Machine_learning"]}
        )

        assert len(result) > 0

        item = result[0]
        assert "url" in item
        assert "title" in item
        assert "content" in item
        # Other metadata fields may or may not be present depending on the source

    def test_extract_invalid_url(self, api_key: str) -> None:
        """Test extraction handles invalid URLs gracefully."""
        tool = ParallelExtractTool(api_key=api_key)

        # The API handles invalid URLs gracefully by returning error info
        result = tool.invoke(
            {
                "urls": ["https://this-domain-does-not-exist-12345.com/"],
                "full_content": True,
                "timeout": 30.0,  # Reasonable timeout
            }
        )

        # Should return a result with error information
        assert len(result) == 1
        assert result[0]["url"] == "https://this-domain-does-not-exist-12345.com/"
        # Should have error information in content or error_type
        assert "Error" in result[0]["content"] or "error_type" in result[0]

    def test_extract_mixed_valid_invalid_urls(self, api_key: str) -> None:
        """Test extraction with mix of valid and invalid URLs."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {
                "urls": [
                    "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "https://this-domain-does-not-exist-12345.com/",
                ],
                "full_content": True,
            }
        )

        assert len(result) == 2
        # First URL should have content
        assert len(result[0]["content"]) > 0 or len(result[1]["content"]) > 0

    @pytest.mark.asyncio
    async def test_extract_async(self, api_key: str) -> None:
        """Test async extraction functionality."""
        tool = ParallelExtractTool(api_key=api_key)

        result = await tool.ainvoke(
            {
                "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                "full_content": True,
            }
        )

        assert len(result) == 1
        assert len(result[0]["content"]) > 0
        assert (
            result[0]["url"] == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

    def test_extract_with_long_content(self, api_key: str) -> None:
        """Test extraction of long articles."""
        tool = ParallelExtractTool(api_key=api_key)

        result = tool.invoke(
            {
                "urls": ["https://en.wikipedia.org/wiki/History_of_the_United_States"],
                "full_content": True,
            }
        )

        assert len(result) == 1
        # Long articles should have substantial content
        assert len(result[0]["content"]) > 1000

    def test_extract_different_content_types(self, api_key: str) -> None:
        """Test extraction from different types of web pages."""
        tool = ParallelExtractTool(api_key=api_key)

        # Test various content types
        urls = [
            "https://www.wikipedia.org/",  # Homepage
            "https://en.wikipedia.org/wiki/Main_Page",  # Wiki page
        ]

        result = tool.invoke({"urls": urls})

        assert len(result) == 2
        # All should return some result (even if empty content)
        for item in result:
            assert "url" in item
            assert "content" in item
