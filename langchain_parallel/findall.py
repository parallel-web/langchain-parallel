"""LangChain integration for Parallel's FindAll API.

FindAll discovers entities from the web that match a natural-language
objective plus a set of boolean match conditions. Returns a ranked list of
candidates with citations.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from parallel import AsyncParallel, Parallel
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_parallel_client, get_parallel_client

# Preview generator caps match_limit at 10; we still allow up to 1000 in the
# input schema and validate against the chosen generator at call time.
_PREVIEW_MAX_LIMIT = 10
_DEFAULT_POLL_TIMEOUT = 600.0
_POLL_INITIAL = 2.0
_POLL_MAX = 10.0

FindAllEventType = Literal[
    "findall.candidate.generated",
    "findall.candidate.matched",
    "findall.candidate.unmatched",
    "findall.candidate.enriched",
    "findall.run.completed",
    "findall.run.cancelled",
    "findall.run.failed",
]


class FindAllWebhook(BaseModel):
    """Webhook config for a FindAll run."""

    url: str = Field(description="HTTPS URL the run will POST events to.")
    event_types: Optional[list[FindAllEventType]] = Field(
        default=None,
        description="Optional subset of event types to forward. Defaults to all.",
    )

    def to_sdk(self) -> dict[str, Any]:
        out: dict[str, Any] = {"url": self.url}
        if self.event_types is not None:
            out["event_types"] = list(self.event_types)
        return out


class FindAllMatchCondition(BaseModel):
    """One boolean condition the API uses to filter candidates.

    The pair (name, description) names a True/False question the API
    answers per candidate; only candidates with all conditions True are
    returned.
    """

    name: str = Field(description="Short identifier (slug-like).")
    description: str = Field(
        description=(
            "Question the API answers per candidate, e.g. "
            "'Is this company headquartered in the US?'"
        ),
    )

    def to_sdk(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description}


class FindAllExcludeEntry(BaseModel):
    """One entry in the FindAll exclude_list."""

    name: str
    url: str

    def to_sdk(self) -> dict[str, Any]:
        return {"name": self.name, "url": self.url}


class ParallelFindAllInput(BaseModel):
    """Input schema for :class:`ParallelFindAllTool`."""

    objective: str = Field(
        description=(
            "Natural-language description of what to find. e.g. 'AI startups "
            "founded in 2024 that have raised more than $10M'."
        ),
    )
    entity_type: str = Field(
        description=(
            "Short noun describing the entity class, e.g. 'company', "
            "'researcher', 'product'."
        ),
    )
    match_conditions: list[FindAllMatchCondition] = Field(
        min_length=1,
        description="Boolean conditions every candidate must satisfy.",
    )
    match_limit: int = Field(
        ge=5,
        le=1000,
        description=(
            "Maximum number of matching candidates to return. Must be in "
            "[5, 1000]; the `preview` generator further caps this at 10."
        ),
    )
    exclude_list: Optional[list[FindAllExcludeEntry]] = Field(
        default=None,
        description=(
            "Entities to skip (use to refresh runs without re-discovering). "
            "URL matching is sensitive to canonicalization; use the same "
            "URL form across runs."
        ),
    )
    webhook: Optional[FindAllWebhook] = Field(
        default=None,
        description="Optional webhook to receive run/candidate events.",
    )
    metadata: Optional[dict[str, Any]] = Field(default=None)
    timeout: Optional[float] = Field(
        default=None,
        description=(
            "Polling timeout (seconds). FindAll runs are typically multi-minute."
        ),
    )


class ParallelFindAllTool(BaseTool):
    """Run a Parallel FindAll discovery and return the matched candidates.

    FindAll discovers entities from the web that satisfy a natural-language
    objective plus a set of boolean match conditions. Useful for lead
    generation, market mapping, and entity enumeration.

    Setup:
        ```bash
        export PARALLEL_API_KEY="your-api-key"
        ```

    Key init args:
        generator: Literal["preview", "base", "core", "pro"]
            Discovery generator. Higher tiers find more candidates but
            cost more and take longer. ``"preview"`` is a small free sample.

    Invocation:
        ```python
        from langchain_parallel import (
            ParallelFindAllTool,
            FindAllMatchCondition,
        )

        tool = ParallelFindAllTool(generator="base")
        result = tool.invoke({
            "objective": "AI agent startups founded after 2023",
            "entity_type": "company",
            "match_conditions": [
                FindAllMatchCondition(
                    name="founded_after_2023",
                    description="Was this company founded after January 1 2023?",
                ),
                FindAllMatchCondition(
                    name="builds_ai_agents",
                    description="Does this company build AI agents as a core product?",
                ),
            ],
            "match_limit": 25,
        })
        for candidate in result["candidates"]:
            print(candidate["name"], candidate["url"])
        ```

    Returns:
        Dict with ``candidates`` (list), ``run`` (status info), and
        ``last_event_id``. Each candidate carries:

        - ``candidate_id``, ``name``, ``url``, ``description``
        - ``match_status``: ``"generated" | "matched" | "unmatched"``
        - ``output``: ``{<name>: {"type": "match_condition" | "enrichment",
          "value": <bool|...>, "is_matched": <bool>}}``
        - ``basis``: list of citations / reasoning per output field

        Note: the ``preview`` generator returns at most 10 candidates and
        does not support enrichments, cancellation, or extension.
    """

    name: str = "parallel_findall"
    description: str = (
        "Discover entities from the web that match a natural-language objective "
        "plus a set of boolean match conditions. Useful for lead generation, "
        "market mapping, and entity enumeration. Returns a ranked candidate list."
    )
    args_schema: type[BaseModel] = ParallelFindAllInput

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://api.parallel.ai")

    generator: Literal["preview", "base", "core", "pro"] = Field(default="base")
    """Discovery generator tier."""

    _client: Optional[Parallel] = None
    _async_client: Optional[AsyncParallel] = None

    @model_validator(mode="after")
    def _initialize_clients(self) -> ParallelFindAllTool:
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self

    def _build_create_kwargs(
        self,
        *,
        objective: str,
        entity_type: str,
        match_conditions: list[FindAllMatchCondition],
        match_limit: int,
        exclude_list: Optional[list[FindAllExcludeEntry]],
        webhook: Optional[FindAllWebhook],
        metadata: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.generator == "preview" and match_limit > _PREVIEW_MAX_LIMIT:
            msg = (
                f"The 'preview' generator caps match_limit at "
                f"{_PREVIEW_MAX_LIMIT}; got {match_limit}. Use 'base', "
                f"'core', or 'pro' for larger runs."
            )
            raise ValueError(msg)
        kwargs: dict[str, Any] = {
            "objective": objective,
            "entity_type": entity_type,
            "match_conditions": [m.to_sdk() for m in match_conditions],
            "match_limit": match_limit,
            "generator": self.generator,
        }
        if exclude_list:
            kwargs["exclude_list"] = [e.to_sdk() for e in exclude_list]
        if webhook is not None:
            kwargs["webhook"] = webhook.to_sdk()
        if metadata is not None:
            kwargs["metadata"] = metadata
        return kwargs

    def _format_result(self, result: Any) -> dict[str, Any]:
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    _TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed"})

    def _wait_for_completion(self, findall_id: str, timeout: float) -> str:
        """Poll ``retrieve()`` until the run terminates (or we time out).

        Returns the terminal status string (``"completed"``, ``"cancelled"``,
        or ``"failed"``).

        On timeout, attempts a best-effort ``cancel()`` and re-raises
        ``TimeoutError``.
        """
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        deadline = time.monotonic() + timeout
        wait = _POLL_INITIAL
        while True:
            info = self._client.beta.findall.retrieve(findall_id)
            status = info.status.status
            if status in self._TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                # Best-effort cancel so we don't leak the run server-side.
                with contextlib.suppress(Exception):
                    self._client.beta.findall.cancel(findall_id)
                msg = (
                    f"FindAll run {findall_id} did not complete within "
                    f"{timeout}s (last status: {status})."
                )
                raise TimeoutError(msg)
            time.sleep(wait)
            wait = min(wait * 1.5, _POLL_MAX)

    async def _await_completion(self, findall_id: str, timeout: float) -> str:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        deadline = time.monotonic() + timeout
        wait = _POLL_INITIAL
        while True:
            info = await self._async_client.beta.findall.retrieve(findall_id)
            status = info.status.status
            if status in self._TERMINAL_STATUSES:
                return status
            if time.monotonic() >= deadline:
                with contextlib.suppress(Exception):
                    await self._async_client.beta.findall.cancel(findall_id)
                msg = (
                    f"FindAll run {findall_id} did not complete within "
                    f"{timeout}s (last status: {status})."
                )
                raise TimeoutError(msg)
            await asyncio.sleep(wait)
            wait = min(wait * 1.5, _POLL_MAX)

    def cancel(self, findall_id: str) -> dict[str, Any]:
        """Cancel a running FindAll run."""
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        result: Any = self._client.beta.findall.cancel(findall_id)
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    async def acancel(self, findall_id: str) -> dict[str, Any]:
        """Async cancel a running FindAll run."""
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        result: Any = await self._async_client.beta.findall.cancel(findall_id)
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    def _run(
        self,
        objective: str,
        entity_type: str,
        match_conditions: list[FindAllMatchCondition],
        match_limit: int,
        exclude_list: Optional[list[FindAllExcludeEntry]] = None,
        webhook: Optional[FindAllWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        kwargs = self._build_create_kwargs(
            objective=objective,
            entity_type=entity_type,
            match_conditions=match_conditions,
            match_limit=match_limit,
            exclude_list=exclude_list,
            webhook=webhook,
            metadata=metadata,
        )
        poll_timeout = timeout if timeout is not None else _DEFAULT_POLL_TIMEOUT
        try:
            run = self._client.beta.findall.create(**kwargs)
            status = self._wait_for_completion(run.findall_id, poll_timeout)
            if status == "failed":
                msg = f"FindAll run {run.findall_id} terminated with status=failed."
                raise ValueError(msg)
            result = self._client.beta.findall.result(run.findall_id)
        except (TimeoutError, ValueError):
            raise
        except Exception as e:
            msg = f"Error calling Parallel FindAll API: {e!s}"
            raise ValueError(msg) from e
        return self._format_result(result)

    async def _arun(
        self,
        objective: str,
        entity_type: str,
        match_conditions: list[FindAllMatchCondition],
        match_limit: int,
        exclude_list: Optional[list[FindAllExcludeEntry]] = None,
        webhook: Optional[FindAllWebhook] = None,
        metadata: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        kwargs = self._build_create_kwargs(
            objective=objective,
            entity_type=entity_type,
            match_conditions=match_conditions,
            match_limit=match_limit,
            exclude_list=exclude_list,
            webhook=webhook,
            metadata=metadata,
        )
        poll_timeout = timeout if timeout is not None else _DEFAULT_POLL_TIMEOUT
        try:
            run = await self._async_client.beta.findall.create(**kwargs)
            status = await self._await_completion(run.findall_id, poll_timeout)
            if status == "failed":
                msg = f"FindAll run {run.findall_id} terminated with status=failed."
                raise ValueError(msg)
            result = await self._async_client.beta.findall.result(run.findall_id)
        except (TimeoutError, ValueError):
            raise
        except Exception as e:
            msg = f"Error calling Parallel FindAll API: {e!s}"
            raise ValueError(msg) from e
        return self._format_result(result)
