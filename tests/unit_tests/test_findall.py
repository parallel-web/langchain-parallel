"""Unit tests for ParallelFindAllTool."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from langchain_parallel import (
    FindAllExcludeEntry,
    FindAllMatchCondition,
    ParallelFindAllTool,
)


def _result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: dict(payload))


@patch("langchain_parallel.findall.get_parallel_client")
@patch("langchain_parallel.findall.get_async_parallel_client")
def test_findall_tool_run(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.beta.findall.create.return_value = SimpleNamespace(findall_id="fa-1")
    # `retrieve()` is polled until the run is no longer active.
    sync_client.beta.findall.retrieve.return_value = SimpleNamespace(
        status=SimpleNamespace(is_active=False, status="completed"),
    )
    sync_client.beta.findall.result.return_value = _result(
        {
            "findall_id": "fa-1",
            "candidates": [
                {"name": "Acme AI", "url": "https://acme.example.com"},
            ],
            "status": "completed",
        },
    )
    mock_async.return_value = Mock()
    mock_sync.return_value = sync_client

    with patch("langchain_parallel.findall.get_api_key", return_value="k"):
        tool = ParallelFindAllTool(generator="base")
        result = tool.invoke(
            {
                "objective": "AI startups",
                "entity_type": "company",
                "match_conditions": [
                    FindAllMatchCondition(name="ai", description="Builds AI?"),
                ],
                "match_limit": 5,
                "exclude_list": [
                    FindAllExcludeEntry(name="OpenAI", url="https://openai.com"),
                ],
            },
        )

    create_kwargs = sync_client.beta.findall.create.call_args.kwargs
    assert create_kwargs["objective"] == "AI startups"
    assert create_kwargs["entity_type"] == "company"
    assert create_kwargs["match_conditions"] == [
        {"name": "ai", "description": "Builds AI?"},
    ]
    assert create_kwargs["match_limit"] == 5
    assert create_kwargs["generator"] == "base"
    assert create_kwargs["exclude_list"] == [
        {"name": "OpenAI", "url": "https://openai.com"},
    ]
    assert result["candidates"][0]["name"] == "Acme AI"


@patch("langchain_parallel.findall.get_parallel_client")
@patch("langchain_parallel.findall.get_async_parallel_client")
def test_findall_wraps_errors(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.beta.findall.create.side_effect = Exception("nope")
    mock_async.return_value = Mock()
    mock_sync.return_value = sync_client

    with patch("langchain_parallel.findall.get_api_key", return_value="k"):
        tool = ParallelFindAllTool()
        with pytest.raises(ValueError, match="Error calling Parallel FindAll API"):
            tool.invoke(
                {
                    "objective": "x",
                    "entity_type": "company",
                    "match_conditions": [
                        FindAllMatchCondition(name="a", description="?"),
                    ],
                    "match_limit": 1,
                },
            )
