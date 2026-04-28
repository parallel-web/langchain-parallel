"""LangChain integration for Parallel's Monitor API (alpha).

The Monitor API runs scheduled web queries and emits webhook events when
results change. As of `parallel-web` 0.5.1 the SDK does not expose this
surface, so this module talks to ``/v1alpha/monitors`` directly via httpx.

The Monitor API is currently in **alpha**; endpoints and shapes may change
without notice. Track https://docs.parallel.ai/monitor-api/monitor-quickstart
for updates.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key

_MONITORS_PATH = "/v1alpha/monitors"

MonitorEventType = Literal[
    "monitor.event.detected",
    "monitor.execution.completed",
    "monitor.execution.failed",
]

# `<n><unit>` where unit ∈ {h, d, w} and the resulting duration is between 1h
# and 30d. Validated as a regex; numeric range checked separately.
_FREQUENCY_RE = re.compile(r"^\d+[hdw]$")


def _validate_frequency(value: str) -> str:
    """Frequency is `<n><unit>` from 1h to 30d (units: h/d/w)."""
    if not _FREQUENCY_RE.match(value):
        msg = (
            f"Invalid frequency '{value}'. Expected '<n><unit>' where unit is "
            f"h, d, or w (e.g. '1h', '6h', '3d', '1w', '2w')."
        )
        raise ValueError(msg)
    n, unit = int(value[:-1]), value[-1]
    hours = {"h": n, "d": n * 24, "w": n * 24 * 7}[unit]
    if not (1 <= hours <= 30 * 24):
        msg = f"frequency '{value}' is outside the supported range (1h-30d)."
        raise ValueError(msg)
    return value


class MonitorWebhook(BaseModel):
    """Webhook config for a Parallel monitor.

    Per the create-monitor API, the webhook object is ``{url, event_types}``.
    The signing secret is configured at the org webhook-endpoint level (in
    the Parallel dashboard), not per-monitor — see
    :func:`langchain_parallel.tasks.verify_webhook` for verification.
    """

    url: str = Field(description="HTTPS URL the monitor will POST events to.")
    event_types: Optional[list[MonitorEventType]] = Field(
        default=None,
        description=(
            "Optional subset of event types to forward. Defaults to all "
            "(detected / completed / failed)."
        ),
    )

    def to_sdk(self) -> dict[str, Any]:
        out: dict[str, Any] = {"url": self.url}
        if self.event_types is not None:
            out["event_types"] = list(self.event_types)
        return out


class ParallelMonitor(BaseModel):
    """Manage scheduled web monitors via the Parallel Monitor API (alpha).

    Each method is a thin wrapper around an HTTP call. Returned dicts are
    the API response bodies as-is.

    Setup:
        ```bash
        export PARALLEL_API_KEY="your-api-key"
        ```

    Example:
        ```python
        from langchain_parallel import ParallelMonitor, MonitorWebhook

        m = ParallelMonitor()

        monitor = m.create(
            query="Track new SEC filings related to Anthropic",
            frequency="6h",
            webhook=MonitorWebhook(
                url="https://example.com/parallel-webhook",
                event_types=["monitor.event.detected"],
            ),
            metadata={"team": "research"},
        )
        print(monitor["monitor_id"])

        # `/events` returns a flat list flattened out of event groups.
        events = m.list_events(monitor["monitor_id"], lookback_period="7d")
        for ev in events.get("events", []):
            print(ev["type"], ev.get("event_date"))
        ```
    """

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://api.parallel.ai")
    timeout: float = Field(default=60.0)

    _resolved_key: Optional[str] = None

    @model_validator(mode="after")
    def _resolve_key(self) -> ParallelMonitor:
        self._resolved_key = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        return self

    def _headers(self) -> dict[str, str]:
        if self._resolved_key is None:
            msg = "API key not initialized."
            raise RuntimeError(msg)
        return {
            "x-api-key": self._resolved_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _build_create_body(
        *,
        query: str,
        frequency: str,
        webhook: Optional[MonitorWebhook],
        metadata: Optional[dict[str, Any]],
        output_schema: Optional[dict[str, Any]],
        source_policy: Optional[dict[str, Any]],
        include_backfill: Optional[bool],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "frequency": _validate_frequency(frequency),
        }
        if webhook is not None:
            body["webhook"] = webhook.to_sdk()
        if metadata is not None:
            body["metadata"] = metadata
        if output_schema is not None:
            body["output_schema"] = output_schema
        if source_policy is not None:
            body["source_policy"] = source_policy
        if include_backfill is not None:
            body["include_backfill"] = include_backfill
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json,
                )
                response.raise_for_status()
                return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            msg = (
                f"Parallel Monitor API error: {e.response.status_code} "
                f"{e.response.text[:300]}"
            )
            raise ValueError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Parallel Monitor API request failed: {e!s}"
            raise ValueError(msg) from e

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method,
                    url,
                    headers=self._headers(),
                    json=json,
                )
                response.raise_for_status()
                return response.json() if response.content else {}
        except httpx.HTTPStatusError as e:
            msg = (
                f"Parallel Monitor API error: {e.response.status_code} "
                f"{e.response.text[:300]}"
            )
            raise ValueError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Parallel Monitor API request failed: {e!s}"
            raise ValueError(msg) from e

    # ---- CRUD ----

    def create(
        self,
        *,
        query: str,
        frequency: str,
        webhook: Optional[MonitorWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        source_policy: Optional[dict[str, Any]] = None,
        include_backfill: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Create a new web monitor.

        Args:
            query: What to monitor for material changes.
            frequency: ``<n><unit>`` where unit ∈ {h, d, w}, between 1h and 30d
                (e.g. ``"1h"``, ``"6h"``, ``"3d"``, ``"2w"``).
            webhook: Optional :class:`MonitorWebhook` to receive events.
            metadata: Free-form string-valued metadata persisted on the run.
            output_schema: Optional JSON schema for structured monitor events
                (see https://docs.parallel.ai/monitor-api/monitor-structured-outputs).
            source_policy: Domain include/exclude lists and freshness floor.
            include_backfill: If True, the first execution returns historical
                events matching the query.
        """
        return self._request(
            "POST",
            _MONITORS_PATH,
            json=self._build_create_body(
                query=query,
                frequency=frequency,
                webhook=webhook,
                metadata=metadata,
                output_schema=output_schema,
                source_policy=source_policy,
                include_backfill=include_backfill,
            ),
        )

    def retrieve(self, monitor_id: str) -> dict[str, Any]:
        """Retrieve a monitor's current configuration and status."""
        return self._request("GET", f"{_MONITORS_PATH}/{monitor_id}")

    def list(
        self,
        *,
        limit: Optional[int] = None,
    ) -> Any:
        """List active monitors."""
        path = _MONITORS_PATH
        if limit is not None:
            path = f"{path}?limit={limit}"
        return self._request("GET", path)

    def delete(self, monitor_id: str) -> dict[str, Any]:
        """Delete a monitor."""
        return self._request("DELETE", f"{_MONITORS_PATH}/{monitor_id}")

    # ---- Events ----

    def list_events(
        self,
        monitor_id: str,
        *,
        lookback_period: Optional[str] = None,
    ) -> dict[str, Any]:
        """List recent events for a monitor.

        The response is a flat ``{"events": [...]}`` list with entries of
        type ``event``, ``completion``, or ``error``. Event groups are
        flattened into individual events per the Parallel API contract.

        Args:
            monitor_id: The monitor id.
            lookback_period: How far back to fetch (e.g. ``"10d"``, ``"1w"``).
                Defaults to the API default of ``10d``.
        """
        path = f"{_MONITORS_PATH}/{monitor_id}/events"
        if lookback_period is not None:
            path = f"{path}?lookback_period={lookback_period}"
        return self._request("GET", path)

    def get_event_group(
        self,
        monitor_id: str,
        event_group_id: str,
    ) -> dict[str, Any]:
        """Retrieve a single event group."""
        return self._request(
            "GET",
            f"{_MONITORS_PATH}/{monitor_id}/event_groups/{event_group_id}",
        )

    def simulate_event(
        self,
        monitor_id: str,
        *,
        event_type: Optional[MonitorEventType] = None,
    ) -> dict[str, Any]:
        """Trigger a synthetic event to test webhook wiring.

        Args:
            monitor_id: The monitor id.
            event_type: Which event to simulate; defaults to
                ``monitor.event.detected``.
        """
        path = f"{_MONITORS_PATH}/{monitor_id}/simulate_event"
        if event_type is not None:
            path = f"{path}?event_type={event_type}"
        return self._request("POST", path)

    # ---- Async variants ----

    async def acreate(
        self,
        *,
        query: str,
        frequency: str,
        webhook: Optional[MonitorWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
        output_schema: Optional[dict[str, Any]] = None,
        source_policy: Optional[dict[str, Any]] = None,
        include_backfill: Optional[bool] = None,
    ) -> dict[str, Any]:
        return await self._arequest(
            "POST",
            _MONITORS_PATH,
            json=self._build_create_body(
                query=query,
                frequency=frequency,
                webhook=webhook,
                metadata=metadata,
                output_schema=output_schema,
                source_policy=source_policy,
                include_backfill=include_backfill,
            ),
        )

    async def aretrieve(self, monitor_id: str) -> dict[str, Any]:
        return await self._arequest("GET", f"{_MONITORS_PATH}/{monitor_id}")

    async def alist(self, *, limit: Optional[int] = None) -> Any:
        path = _MONITORS_PATH
        if limit is not None:
            path = f"{path}?limit={limit}"
        return await self._arequest("GET", path)

    async def adelete(self, monitor_id: str) -> dict[str, Any]:
        return await self._arequest("DELETE", f"{_MONITORS_PATH}/{monitor_id}")

    async def alist_events(
        self,
        monitor_id: str,
        *,
        lookback_period: Optional[str] = None,
    ) -> dict[str, Any]:
        path = f"{_MONITORS_PATH}/{monitor_id}/events"
        if lookback_period is not None:
            path = f"{path}?lookback_period={lookback_period}"
        return await self._arequest("GET", path)

    async def aget_event_group(
        self,
        monitor_id: str,
        event_group_id: str,
    ) -> dict[str, Any]:
        return await self._arequest(
            "GET",
            f"{_MONITORS_PATH}/{monitor_id}/event_groups/{event_group_id}",
        )

    async def asimulate_event(
        self,
        monitor_id: str,
        *,
        event_type: Optional[MonitorEventType] = None,
    ) -> dict[str, Any]:
        path = f"{_MONITORS_PATH}/{monitor_id}/simulate_event"
        if event_type is not None:
            path = f"{path}?event_type={event_type}"
        return await self._arequest("POST", path)
