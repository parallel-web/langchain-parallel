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
            frequency="6h",
            webhook=MonitorWebhook(
                url="https://x",
                event_types=["monitor.event.detected"],
            ),
            metadata={"team": "research"},
            include_backfill=True,
        )

    assert captured["method"] == "POST"
    assert captured["path"] == "/v1alpha/monitors"
    assert captured["api_key"] == "k"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["query"] == "SEC filings"
    assert body["frequency"] == "6h"
    assert body["webhook"] == {
        "url": "https://x",
        "event_types": ["monitor.event.detected"],
    }
    assert body["metadata"] == {"team": "research"}
    assert body["include_backfill"] is True
    assert result["monitor_id"] == "mon-1"


def test_invalid_frequency_raises() -> None:
    """Frequency must be `<n><unit>` from 1h to 30d."""
    import pytest

    from langchain_parallel.monitors import _validate_frequency

    with pytest.raises(ValueError, match="Invalid frequency"):
        _validate_frequency("15m")  # sub-hour not supported
    with pytest.raises(ValueError, match="outside the supported range"):
        _validate_frequency("60d")  # over 30d
    # Valid values pass through.
    assert _validate_frequency("1h") == "1h"
    assert _validate_frequency("3d") == "3d"
    assert _validate_frequency("2w") == "2w"


@patch("langchain_parallel.monitors.get_api_key", return_value="k")
def test_list_events_path_and_query(_mock_key: object) -> None:
    """list_events hits /events with lookback_period (not /event_groups)."""
    captured: dict[str, object] = {}

    monitor = ParallelMonitor()
    with patch.object(
        ParallelMonitor,
        "_request",
        lambda self, method, path, *, json=None: (
            captured.update(method=method, path=path),
            {"events": []},
        )[1],
    ):
        monitor.list_events("mon-1", lookback_period="7d")

    assert captured["method"] == "GET"
    assert captured["path"] == "/v1alpha/monitors/mon-1/events?lookback_period=7d"


@patch("langchain_parallel.monitors.get_api_key", return_value="k")
def test_simulate_event_with_type(_mock_key: object) -> None:
    captured: dict[str, object] = {}

    monitor = ParallelMonitor()
    with patch.object(
        ParallelMonitor,
        "_request",
        lambda self, method, path, *, json=None: (
            captured.update(method=method, path=path),
            {},
        )[1],
    ):
        monitor.simulate_event("mon-1", event_type="monitor.execution.completed")

    assert captured["path"] == (
        "/v1alpha/monitors/mon-1/simulate_event?event_type=monitor.execution.completed"
    )


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
