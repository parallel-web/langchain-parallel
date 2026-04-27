"""Integration tests for Parallel Extract Tool."""

from __future__ import annotations

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


def _invoke(tool: ParallelExtractTool, args: dict) -> tuple[str, list[dict]]:
    """Invoke via the tool_call form so we get back a ToolMessage with .artifact.

    Returns ``(content, artifact)``.
    """
    msg = tool.invoke(
        {
            "args": args,
            "id": "1",
            "name": tool.name,
            "type": "tool_call",
        },
    )
    return msg.content, msg.artifact


class TestParallelExtractToolIntegration:
    """Integration tests for ParallelExtractTool."""

    def test_extract_single_url(self, api_key: str) -> None:
        """Test extracting content from a single URL."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                "full_content": True,
            },
        )

        assert len(artifact) == 1
        assert (
            artifact[0]["url"]
            == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        assert len(artifact[0]["content"]) > 0
        assert artifact[0]["title"] is not None

    def test_extract_multiple_urls(self, api_key: str) -> None:
        """Test extracting content from multiple URLs."""
        tool = ParallelExtractTool(api_key=api_key)

        urls = [
            "https://www.wikipedia.org/",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        ]

        _, artifact = _invoke(tool, {"urls": urls})

        assert len(artifact) == 2
        for item in artifact:
            assert "url" in item
            assert "content" in item

    def test_extract_with_search_objective(self, api_key: str) -> None:
        """Test extraction with search objective to focus content."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                "search_objective": "What are the main applications of AI?",
                "full_content": False,
            },
        )

        assert len(artifact) == 1
        assert (
            artifact[0]["url"]
            == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )
        assert "excerpts" in artifact[0]
        assert isinstance(artifact[0]["excerpts"], list)
        assert len(artifact[0]["content"]) > 0

    def test_extract_with_search_queries(self, api_key: str) -> None:
        """Test extraction with search queries to focus content."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://en.wikipedia.org/wiki/Machine_learning"],
                "search_queries": ["neural networks", "training algorithms"],
            },
        )

        assert len(artifact) == 1
        assert "excerpts" in artifact[0]
        assert isinstance(artifact[0]["excerpts"], list)
        assert len(artifact[0]["excerpts"]) > 0

    def test_extract_with_max_chars(self, api_key: str) -> None:
        """Test extraction with max_chars_per_extract limit."""
        tool = ParallelExtractTool(api_key=api_key, max_chars_per_extract=1000)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://en.wikipedia.org/wiki/Python_(programming_language)"],
                "full_content": True,
            },
        )

        assert len(artifact) == 1
        assert len(artifact[0]["content"]) > 0
        assert artifact[0]["title"] is not None

    def test_extract_metadata_fields(self, api_key: str) -> None:
        """Test that metadata fields are properly populated."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool, {"urls": ["https://en.wikipedia.org/wiki/Machine_learning"]}
        )

        assert len(artifact) > 0
        item = artifact[0]
        assert "url" in item
        assert "title" in item
        assert "content" in item

    def test_extract_invalid_url(self, api_key: str) -> None:
        """Test extraction handles invalid URLs gracefully."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://this-domain-does-not-exist-12345.com/"],
                "full_content": True,
                "timeout": 30.0,
            },
        )

        assert len(artifact) == 1
        assert artifact[0]["url"] == "https://this-domain-does-not-exist-12345.com/"
        assert "Error" in artifact[0]["content"] or "error_type" in artifact[0]

    def test_extract_mixed_valid_invalid_urls(self, api_key: str) -> None:
        """Test extraction with mix of valid and invalid URLs."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": [
                    "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "https://this-domain-does-not-exist-12345.com/",
                ],
                "full_content": True,
            },
        )

        assert len(artifact) == 2
        assert len(artifact[0]["content"]) > 0 or len(artifact[1]["content"]) > 0

    @pytest.mark.asyncio
    async def test_extract_async(self, api_key: str) -> None:
        """Test async extraction functionality."""
        tool = ParallelExtractTool(api_key=api_key)

        msg = await tool.ainvoke(
            {
                "args": {
                    "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
                    "full_content": True,
                },
                "id": "1",
                "name": tool.name,
                "type": "tool_call",
            },
        )
        artifact = msg.artifact

        assert len(artifact) == 1
        assert len(artifact[0]["content"]) > 0
        assert (
            artifact[0]["url"]
            == "https://en.wikipedia.org/wiki/Artificial_intelligence"
        )

    def test_extract_with_long_content(self, api_key: str) -> None:
        """Test extraction of long articles."""
        tool = ParallelExtractTool(api_key=api_key)

        _, artifact = _invoke(
            tool,
            {
                "urls": ["https://en.wikipedia.org/wiki/History_of_the_United_States"],
                "full_content": True,
            },
        )

        assert len(artifact) == 1
        assert len(artifact[0]["content"]) > 1000

    def test_extract_different_content_types(self, api_key: str) -> None:
        """Test extraction from different types of web pages."""
        tool = ParallelExtractTool(api_key=api_key)

        urls = [
            "https://www.wikipedia.org/",
            "https://en.wikipedia.org/wiki/Main_Page",
        ]

        _, artifact = _invoke(tool, {"urls": urls})

        assert len(artifact) == 2
        for item in artifact:
            assert "url" in item
            assert "content" in item
