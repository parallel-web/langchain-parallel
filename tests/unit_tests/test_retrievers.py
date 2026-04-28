"""Unit tests for ParallelSearchRetriever."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from langchain_parallel import ParallelSearchRetriever, SourcePolicy


def _resp(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: dict(payload))


@patch("langchain_parallel.retrievers.get_parallel_client")
@patch("langchain_parallel.retrievers.get_async_parallel_client")
def test_invoke_returns_documents(
    mock_async_factory: Mock,
    mock_sync_factory: Mock,
) -> None:
    sync_client = Mock()
    sync_client.search.return_value = _resp(
        {
            "search_id": "s-1",
            "results": [
                {
                    "url": "https://example.com/a",
                    "title": "A",
                    "publish_date": "2025-01-01",
                    "excerpts": ["snippet 1", "snippet 2"],
                },
            ],
        },
    )
    mock_sync_factory.return_value = sync_client
    mock_async_factory.return_value = Mock()

    with patch("langchain_parallel.retrievers.get_api_key", return_value="k"):
        retriever = ParallelSearchRetriever(
            max_results=5,
            mode="advanced",
            source_policy=SourcePolicy(include_domains=["example.com"]),
        )
        docs = retriever.invoke("test query")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "snippet 1\n\nsnippet 2"
    assert doc.metadata["url"] == "https://example.com/a"
    assert doc.metadata["title"] == "A"
    assert doc.metadata["publish_date"] == "2025-01-01"
    assert doc.metadata["search_id"] == "s-1"
    assert doc.metadata["query"] == "test query"
    assert doc.metadata["excerpts"] == ["snippet 1", "snippet 2"]

    # Verify advanced_settings envelope built correctly.
    kwargs = sync_client.search.call_args.kwargs
    assert kwargs["search_queries"] == ["test query"]
    assert kwargs["mode"] == "advanced"
    assert kwargs["advanced_settings"]["max_results"] == 5
    assert kwargs["advanced_settings"]["source_policy"] == {
        "include_domains": ["example.com"],
    }


@patch("langchain_parallel.retrievers.get_parallel_client")
@patch("langchain_parallel.retrievers.get_async_parallel_client")
@pytest.mark.asyncio
async def test_async_invoke(
    mock_async_factory: Mock,
    mock_sync_factory: Mock,
) -> None:
    async_client = Mock()
    async_client.search = AsyncMock(
        return_value=_resp(
            {
                "search_id": "s-2",
                "results": [
                    {"url": "https://x.com", "title": "X", "excerpts": ["a"]},
                ],
            }
        ),
    )
    mock_async_factory.return_value = async_client
    mock_sync_factory.return_value = Mock()

    with patch("langchain_parallel.retrievers.get_api_key", return_value="k"):
        retriever = ParallelSearchRetriever()
        docs = await retriever.ainvoke("q")
    assert len(docs) == 1
    assert docs[0].metadata["url"] == "https://x.com"


@patch("langchain_parallel.retrievers.get_parallel_client")
@patch("langchain_parallel.retrievers.get_async_parallel_client")
def test_persistent_objective_forwarded(
    mock_async_factory: Mock,
    mock_sync_factory: Mock,
) -> None:
    sync_client = Mock()
    sync_client.search.return_value = _resp({"search_id": "s", "results": []})
    mock_sync_factory.return_value = sync_client
    mock_async_factory.return_value = Mock()

    with patch("langchain_parallel.retrievers.get_api_key", return_value="k"):
        retriever = ParallelSearchRetriever(objective="academic AI research only")
        retriever.invoke("transformers")
    kwargs = sync_client.search.call_args.kwargs
    assert kwargs["objective"] == "academic AI research only"
    assert kwargs["search_queries"] == ["transformers"]


@patch("langchain_parallel.retrievers.get_parallel_client")
@patch("langchain_parallel.retrievers.get_async_parallel_client")
def test_api_error_wrapped(
    mock_async_factory: Mock,
    mock_sync_factory: Mock,
) -> None:
    sync_client = Mock()
    sync_client.search.side_effect = Exception("boom")
    mock_sync_factory.return_value = sync_client
    mock_async_factory.return_value = Mock()

    with patch("langchain_parallel.retrievers.get_api_key", return_value="k"):
        retriever = ParallelSearchRetriever()
        with pytest.raises(ValueError, match="Error calling Parallel Search API"):
            retriever.invoke("q")
