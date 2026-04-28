"""Unit tests for the Task API surfaces."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from langchain_parallel import (
    McpServer,
    ParallelDeepResearch,
    ParallelTaskGroup,
    ParallelTaskRunTool,
    verify_webhook,
)


def _result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: dict(payload))


# --- verify_webhook ---


def _sign(payload: bytes, webhook_id: str, ts: str, secret: str) -> str:
    import base64
    import hashlib
    import hmac

    signed = f"{webhook_id}.{ts}.{payload.decode()}"
    digest = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def test_verify_webhook_success() -> None:
    import time

    payload = b'{"run_id":"abc"}'
    secret = "sek"  # noqa: S105 - test fixture
    wid = "msg_1"
    ts = str(int(time.time()))
    sig = _sign(payload, wid, ts, secret)
    assert (
        verify_webhook(
            payload,
            webhook_id=wid,
            webhook_timestamp=ts,
            webhook_signature=f"v1,{sig}",
            secret=secret,
        )
        is True
    )


def test_verify_webhook_multiple_signatures() -> None:
    """Header carrying multiple `v1,<sig>` entries: match if any matches."""
    import time

    payload = b'{"x":1}'
    secret = "sek"  # noqa: S105
    wid, ts = "msg", str(int(time.time()))
    good = _sign(payload, wid, ts, secret)
    header = f"v1,deadbeef v1,{good}"
    assert (
        verify_webhook(
            payload,
            webhook_id=wid,
            webhook_timestamp=ts,
            webhook_signature=header,
            secret=secret,
        )
        is True
    )


def test_verify_webhook_replay_rejected() -> None:
    """Timestamps outside tolerance are rejected."""
    payload = b"{}"
    secret = "sek"  # noqa: S105
    old_ts = "1000000000"  # well in the past
    sig = _sign(payload, "m", old_ts, secret)
    assert (
        verify_webhook(
            payload,
            webhook_id="m",
            webhook_timestamp=old_ts,
            webhook_signature=f"v1,{sig}",
            secret=secret,
        )
        is False
    )


def test_verify_webhook_failure() -> None:
    import time

    assert (
        verify_webhook(
            b"x",
            webhook_id="m",
            webhook_timestamp=str(int(time.time())),
            webhook_signature="v1,deadbeef",
            secret="sek",  # noqa: S106
        )
        is False
    )


# --- ParallelTaskRunTool ---


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_task_run_tool_executes(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.task_run.execute.return_value = _result(
        {
            "run": {"run_id": "tr-1", "status": "completed"},
            "output": {"text": "hello"},
            "basis": [{"field": "answer", "citations": []}],
        },
    )
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        tool = ParallelTaskRunTool(processor="lite")
        result = tool.invoke({"input": "Hi"})
    sync_client.task_run.execute.assert_called_once()
    kwargs = sync_client.task_run.execute.call_args.kwargs
    assert kwargs["input"] == "Hi"
    assert kwargs["processor"] == "lite"
    assert result["output"] == {"text": "hello"}
    assert result["basis"][0]["field"] == "answer"


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_task_run_tool_with_mcp_servers_uses_create_path(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    """BYOMCP forces the create+result path so mcp_servers can be passed."""
    sync_client = Mock()
    sync_client.beta.task_run.create.return_value = SimpleNamespace(run_id="tr-2")
    sync_client.task_run.result.return_value = _result(
        {"run": {"run_id": "tr-2"}, "output": "ok"},
    )
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        tool = ParallelTaskRunTool(
            processor="base",
            mcp_servers=[
                McpServer(
                    name="my_mcp",
                    url="https://example.com/mcp",
                    headers={"Authorization": "Bearer abc"},
                ),
            ],
        )
        tool.invoke({"input": "Hi"})

    # Should NOT have called task_run.execute (no mcp_servers there).
    sync_client.task_run.execute.assert_not_called()
    sync_client.beta.task_run.create.assert_called_once()
    kwargs = sync_client.beta.task_run.create.call_args.kwargs
    assert kwargs["mcp_servers"] == [
        {
            "type": "url",
            "name": "my_mcp",
            "url": "https://example.com/mcp",
            "headers": {"Authorization": "Bearer abc"},
        },
    ]
    assert "mcp-server-2025-07-17" in kwargs["betas"]


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_task_run_tool_wraps_errors(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.task_run.execute.side_effect = Exception("API down")
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        tool = ParallelTaskRunTool(processor="lite")
        with pytest.raises(ValueError, match="Error calling Parallel Task API"):
            tool.invoke({"input": "Hi"})


# --- ParallelDeepResearch ---


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_deep_research_runnable(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.task_run.execute.return_value = _result({"output": "report"})
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        research = ParallelDeepResearch(processor="core")
        result = research.invoke("research question")
    kwargs = sync_client.task_run.execute.call_args.kwargs
    assert kwargs["processor"] == "core"
    assert kwargs["input"] == "research question"
    assert result["output"] == "report"


# --- ParallelTaskGroup ---


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_task_group_run(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    sync_client = Mock()
    sync_client.beta.task_group.create.return_value = SimpleNamespace(
        task_group_id="tg-1",
    )
    sync_client.beta.task_group.add_runs.return_value = SimpleNamespace(
        run_ids=["r1", "r2", "r3"],
    )
    sync_client.task_run.result.side_effect = [
        _result({"output": f"out-{i}"}) for i in range(3)
    ]
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        group = ParallelTaskGroup(processor="lite")
        results = group.run(["a", "b", "c"])
    assert [r["output"] for r in results] == ["out-0", "out-1", "out-2"]
    # add_runs got 3 inputs each tagged with the configured processor.
    add_kwargs = sync_client.beta.task_group.add_runs.call_args.kwargs
    assert len(add_kwargs["inputs"]) == 3
    assert all(i["processor"] == "lite" for i in add_kwargs["inputs"])
