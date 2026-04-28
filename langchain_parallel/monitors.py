"""LangChain integration for Parallel's Monitor API (alpha).

The Monitor API runs scheduled web queries and emits webhook events when
results change. As of `parallel-web` 0.5.1 the SDK does not expose this
surface, so this module talks to ``/v1alpha/monitors`` directly via httpx.

The Monitor API is currently in **alpha**; endpoints and shapes may change
without notice. Track https://docs.parallel.ai/monitor-api/monitor-quickstart
for updates.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key

_MONITORS_PATH = "/v1alpha/monitors"


class MonitorWebhook(BaseModel):
    """Webhook config for a Parallel monitor."""

    url: str = Field(description="HTTPS URL the monitor will POST events to.")
    secret: Optional[str] = Field(
        default=None,
        description=(
            "Shared secret used to HMAC-sign event payloads. Verify with "
            "`langchain_parallel.tasks.verify_webhook`."
        ),
    )

    def to_sdk(self) -> dict[str, Any]:
        out: dict[str, Any] = {"url": self.url}
        if self.secret is not None:
            out["secret"] = self.secret
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
            query="Track new SEC filings from anthropic-related entities",
            frequency="1h",
            webhook=MonitorWebhook(
                url="https://example.com/parallel-webhook",
                secret="..."
            ),
            metadata={"team": "research"},
        )
        print(monitor["monitor_id"])

        for ev in m.list_events(monitor["monitor_id"])["event_groups"]:
            print(ev)
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

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
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
        frequency: Literal["1h", "1d", "1w"],
        webhook: Optional[MonitorWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create a new web monitor."""
        body: dict[str, Any] = {"query": query, "frequency": frequency}
        if webhook is not None:
            body["webhook"] = webhook.to_sdk()
        if metadata is not None:
            body["metadata"] = metadata
        return self._request("POST", _MONITORS_PATH, json=body)

    def retrieve(self, monitor_id: str) -> dict[str, Any]:
        """Retrieve a monitor's current configuration and status."""
        return self._request("GET", f"{_MONITORS_PATH}/{monitor_id}")

    def list(
        self,
        *,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """List active monitors."""
        path = _MONITORS_PATH
        if limit is not None:
            path = f"{path}?limit={limit}"
        return self._request("GET", path)

    def update(
        self,
        monitor_id: str,
        *,
        query: Optional[str] = None,
        frequency: Optional[Literal["1h", "1d", "1w"]] = None,
        webhook: Optional[MonitorWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Patch a monitor in place."""
        body: dict[str, Any] = {}
        if query is not None:
            body["query"] = query
        if frequency is not None:
            body["frequency"] = frequency
        if webhook is not None:
            body["webhook"] = webhook.to_sdk()
        if metadata is not None:
            body["metadata"] = metadata
        return self._request(
            "PATCH",
            f"{_MONITORS_PATH}/{monitor_id}",
            json=body,
        )

    def delete(self, monitor_id: str) -> dict[str, Any]:
        """Delete a monitor."""
        return self._request("DELETE", f"{_MONITORS_PATH}/{monitor_id}")

    # ---- Events ----

    def list_events(
        self,
        monitor_id: str,
        *,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        """List recent event groups for a monitor (up to 300)."""
        path = f"{_MONITORS_PATH}/{monitor_id}/event_groups"
        if limit is not None:
            path = f"{path}?limit={limit}"
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

    def simulate_event(self, monitor_id: str) -> dict[str, Any]:
        """Trigger a synthetic event to test webhook wiring."""
        return self._request(
            "POST",
            f"{_MONITORS_PATH}/{monitor_id}/simulate_event",
        )

    # ---- Async variants ----

    async def acreate(
        self,
        *,
        query: str,
        frequency: Literal["1h", "1d", "1w"],
        webhook: Optional[MonitorWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "frequency": frequency}
        if webhook is not None:
            body["webhook"] = webhook.to_sdk()
        if metadata is not None:
            body["metadata"] = metadata
        return await self._arequest("POST", _MONITORS_PATH, json=body)

    async def aretrieve(self, monitor_id: str) -> dict[str, Any]:
        return await self._arequest("GET", f"{_MONITORS_PATH}/{monitor_id}")

    async def alist(self, *, limit: Optional[int] = None) -> dict[str, Any]:
        path = _MONITORS_PATH
        if limit is not None:
            path = f"{path}?limit={limit}"
        return await self._arequest("GET", path)

    async def adelete(self, monitor_id: str) -> dict[str, Any]:
        return await self._arequest("DELETE", f"{_MONITORS_PATH}/{monitor_id}")
