"""Unit tests for ParallelMonitor (httpx wrapper)."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from langchain_parallel import MonitorWebhook, ParallelMonitor


@patch("langchain_parallel.monitors.get_api_key", return_value="k")
def test_create_monitor(_mock_key: object) -> None:
    """create() POSTs to /v1alpha/monitors with the right body and headers."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        captured["x-api-key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            json={"monitor_id": "mon-1", "status": "active"},
        )

    monitor = ParallelMonitor()
    with patch.object(
        ParallelMonitor,
        "_request",
        lambda self, method, path, *, json=None: (
            captured.update(
                method=method,
                path=path,
                body=json,
                api_key=self._resolved_key,
            ),
            {"monitor_id": "mon-1", "status": "active"},
        )[1],
    ):
        result = monitor.create(
            query="SEC filings",
            frequency="1h",
            webhook=MonitorWebhook(url="https://x", secret="s"),  # noqa: S106
            metadata={"team": "research"},
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1alpha/monitors"
    assert captured["api_key"] == "k"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["query"] == "SEC filings"
    assert body["frequency"] == "1h"
    assert body["webhook"] == {"url": "https://x", "secret": "s"}
    assert body["metadata"] == {"team": "research"}
    assert result["monitor_id"] == "mon-1"


@patch("langchain_parallel.monitors.get_api_key", return_value="k")
def test_http_error_wrapped(_mock_key: object) -> None:
    """A 4xx from the API surfaces as ValueError with the response code."""
    monitor = ParallelMonitor()

    def fake_request(*_args: object, **_kwargs: object) -> dict[str, object]:
        # Simulate the body of `_request`'s raise-for-status path.
        request = httpx.Request("GET", "https://api.parallel.ai/v1alpha/monitors/x")
        response = httpx.Response(404, json={"error": "not found"}, request=request)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            msg = (
                f"Parallel Monitor API error: {e.response.status_code} "
                f"{e.response.text[:300]}"
            )
            raise ValueError(msg) from e
        return {}

    with (
        patch.object(ParallelMonitor, "_request", fake_request),
        pytest.raises(ValueError, match="Parallel Monitor API error: 404"),
    ):
        monitor.retrieve("missing")
