"""Unit tests for the Task API surfaces."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, Field

from langchain_parallel import (
    McpServer,
    ParallelDeepResearch,
    ParallelEnrichment,
    ParallelTaskGroup,
    ParallelTaskRunTool,
    build_task_spec,
    parse_basis,
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


# --- parse_basis ---


def test_parse_basis_top_level() -> None:
    """Basis at the top level (TaskRunTool typed-output shape)."""
    result = {
        "run": {"run_id": "tr-1", "interaction_id": "iact_1"},
        "output": {"founder": "Elon Musk", "year": 2002},
        "basis": [
            {
                "field": "founder",
                "confidence": "high",
                "citations": [{"url": "https://example.com/a"}],
            },
            {
                "field": "year",
                "confidence": "low",
                "citations": [{"url": "https://example.com/b"}],
            },
        ],
    }
    parsed = parse_basis(result)
    assert parsed["citations_by_field"]["founder"][0]["url"] == "https://example.com/a"
    assert parsed["low_confidence_fields"] == ["year"]
    assert parsed["interaction_id"] == "iact_1"


def test_parse_basis_nested_under_output() -> None:
    """Basis nested under `output` (DeepResearch / structured-output shape)."""
    result = {
        "interaction_id": "iact_top",
        "output": {
            "content": {"capital": "Paris"},
            "basis": [
                {"field": "capital", "confidence": "MEDIUM", "citations": []},
            ],
        },
    }
    parsed = parse_basis(result)
    assert parsed["citations_by_field"] == {"capital": []}
    assert parsed["low_confidence_fields"] == []
    assert parsed["interaction_id"] == "iact_top"


def test_parse_basis_handles_missing_basis() -> None:
    """No basis present → empty results, no errors."""
    parsed = parse_basis({"output": "free text", "run": {"run_id": "tr-x"}})
    assert parsed == {
        "citations_by_field": {},
        "low_confidence_fields": [],
        "interaction_id": None,
    }


def test_parse_basis_skips_malformed_entries() -> None:
    """Non-dict entries and entries without a `field` are ignored."""
    result = {
        "basis": [
            "not a dict",
            {"confidence": "low"},  # missing field
            {"field": "", "confidence": "low"},  # blank field
            {"field": "ok", "confidence": "low", "citations": [{"url": "u"}]},
        ],
    }
    parsed = parse_basis(result)
    assert parsed["citations_by_field"] == {"ok": [{"url": "u"}]}
    assert parsed["low_confidence_fields"] == ["ok"]


# --- Default processors ---


def test_default_processors_are_fast_variants() -> None:
    """All four Task surfaces default to a `-fast` processor variant."""

    class _Out(BaseModel):
        x: int

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        assert ParallelTaskRunTool().processor == "lite-fast"
        assert ParallelDeepResearch().processor == "pro-fast"
        assert ParallelTaskGroup().processor == "lite-fast"
        assert ParallelEnrichment(output_schema=_Out).processor == "core-fast"


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


# --- build_task_spec ---


class _CompanyIn(BaseModel):
    company: str = Field(description="Company name")


class _CompanyOut(BaseModel):
    headquarters: str
    founding_year: int


def test_build_task_spec_pydantic() -> None:
    spec = build_task_spec(input_schema=_CompanyIn, output_schema=_CompanyOut)
    assert spec["output_schema"]["type"] == "json"
    assert "headquarters" in spec["output_schema"]["json_schema"]["properties"]
    assert spec["input_schema"]["type"] == "json"
    assert "company" in spec["input_schema"]["json_schema"]["properties"]


def test_build_task_spec_mixed_shapes() -> None:
    """Strings become text schemas; raw dicts are wrapped as json schemas."""
    spec = build_task_spec(
        input_schema="A natural-language description of the input",
        output_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
    )
    assert spec["input_schema"]["type"] == "text"
    assert spec["input_schema"]["description"].startswith("A natural-language")
    assert spec["output_schema"]["type"] == "json"
    assert spec["output_schema"]["json_schema"]["properties"]["x"]["type"] == "integer"


# --- ParallelEnrichment ---


@patch("langchain_parallel.tasks.get_parallel_client")
@patch("langchain_parallel.tasks.get_async_parallel_client")
def test_enrichment_invoke(
    mock_async: Mock,
    mock_sync: Mock,
) -> None:
    """Coerces pydantic inputs, passes default_task_spec to add_runs."""
    sync_client = Mock()
    sync_client.beta.task_group.create.return_value = SimpleNamespace(
        task_group_id="tg-en-1",
    )
    sync_client.beta.task_group.add_runs.return_value = SimpleNamespace(
        run_ids=["r1", "r2"],
    )
    sync_client.task_run.result.side_effect = [
        _result(
            {
                "output": {
                    "content": {
                        "headquarters": "San Francisco, CA",
                        "founding_year": 2021,
                    },
                },
            },
        ),
        _result(
            {
                "output": {
                    "content": {
                        "headquarters": "San Francisco, CA",
                        "founding_year": 2015,
                    },
                },
            },
        ),
    ]
    mock_sync.return_value = sync_client
    mock_async.return_value = Mock()

    with patch("langchain_parallel.tasks.get_api_key", return_value="k"):
        enricher = ParallelEnrichment(
            input_schema=_CompanyIn,
            output_schema=_CompanyOut,
            processor="core",
        )
        results = enricher.invoke(
            [
                _CompanyIn(company="Anthropic"),
                {"company": "OpenAI"},
            ],
        )

    # `default_task_spec` was passed to add_runs.
    add_kwargs = sync_client.beta.task_group.add_runs.call_args.kwargs
    assert "default_task_spec" in add_kwargs
    spec = add_kwargs["default_task_spec"]
    assert spec["input_schema"]["type"] == "json"
    assert spec["output_schema"]["type"] == "json"
    # Pydantic input got coerced to a dict.
    inputs = add_kwargs["inputs"]
    assert inputs[0]["input"] == {"company": "Anthropic"}
    assert inputs[1]["input"] == {"company": "OpenAI"}
    # Each run uses the configured processor.
    assert all(i["processor"] == "core" for i in inputs)
    # Results come back in the same order.
    assert results[0]["output"]["content"]["founding_year"] == 2021
    assert results[1]["output"]["content"]["founding_year"] == 2015
