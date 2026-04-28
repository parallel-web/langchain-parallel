"""LangChain integration for Parallel's Task API.

Three primary surfaces:

- :class:`ParallelTaskRunTool` — agent-callable tool that runs a single
  Parallel Task synchronously (via ``client.task_run.execute``) and
  returns the structured result with citations.
- :class:`ParallelDeepResearch` — high-level :class:`~langchain_core.runnables.Runnable`
  wrapper for deep-research tasks. Defaults to the ``core`` processor and
  always returns the full ``basis`` (citations + reasoning + confidence).
- :class:`ParallelTaskGroup` — batch executor backed by the Task Group
  API. Useful for bulk enrichment.

A :func:`verify_webhook` helper validates HMAC signatures on incoming
Parallel webhooks.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Literal, Optional, Union

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from parallel import AsyncParallel, Parallel
from pydantic import BaseModel, Field, SecretStr, model_validator

from ._client import get_api_key, get_async_parallel_client, get_parallel_client

_TIER_PROCESSORS = (
    "lite",
    "base",
    "core",
    "core2x",
    "pro",
    "ultra",
    "ultra2x",
    "ultra4x",
    "ultra8x",
)
ALL_PROCESSORS = tuple(
    name + suffix for name in _TIER_PROCESSORS for suffix in ("", "-fast")
)
ProcessorLiteral = Literal[
    "lite",
    "base",
    "core",
    "core2x",
    "pro",
    "ultra",
    "ultra2x",
    "ultra4x",
    "ultra8x",
    "lite-fast",
    "base-fast",
    "core-fast",
    "core2x-fast",
    "pro-fast",
    "ultra-fast",
    "ultra2x-fast",
    "ultra4x-fast",
    "ultra8x-fast",
]

_MCP_BETA_HEADER = "mcp-server-2025-07-17"


class McpServer(BaseModel):
    """One BYOMCP server description for a Task Run.

    Mirrors :class:`parallel.types.McpServerParam` so callers don't have to
    import from the SDK directly.
    """

    name: str = Field(description="Unique name for the MCP server.")
    url: str = Field(description="HTTPS URL of a Streamable HTTP MCP endpoint.")
    headers: Optional[dict[str, str]] = Field(
        default=None,
        description="Headers (e.g. Authorization) sent on every MCP call.",
    )
    allowed_tools: Optional[list[str]] = Field(
        default=None,
        description=(
            "If set, restrict the run to tools whose names appear in this list."
        ),
    )

    def to_sdk(self) -> dict[str, Any]:
        """Render to the SDK kwargs shape."""
        out: dict[str, Any] = {"type": "url", "name": self.name, "url": self.url}
        if self.headers is not None:
            out["headers"] = self.headers
        if self.allowed_tools is not None:
            out["allowed_tools"] = list(self.allowed_tools)
        return out


_DEFAULT_WEBHOOK_TOLERANCE_SECONDS = 5 * 60


def verify_webhook(
    payload: bytes,
    *,
    webhook_id: str,
    webhook_timestamp: str,
    webhook_signature: str,
    secret: str,
    tolerance_seconds: int = _DEFAULT_WEBHOOK_TOLERANCE_SECONDS,
) -> bool:
    """Verify a Parallel webhook signature (Standard Webhooks scheme).

    Parallel signs webhook payloads using HMAC-SHA256 over
    ``"<webhook-id>.<webhook-timestamp>.<body>"``, base64-encoded
    with padding. The signature is delivered as the ``webhook-signature``
    header (possibly with multiple space-delimited ``v1,<sig>`` entries).
    See https://docs.parallel.ai/resources/webhook-setup.

    The verification flow is:

    1. Reject if the timestamp drifts more than ``tolerance_seconds`` from now
       (replay protection).
    2. Compute the expected signature.
    3. Compare against every ``v1,<sig>`` entry in the header.

    Args:
        payload: Raw request body bytes (do not decode-then-encode — must be
            byte-identical to what Parallel signed).
        webhook_id: Value of the ``webhook-id`` header.
        webhook_timestamp: Value of the ``webhook-timestamp`` header (Unix
            seconds as a string).
        webhook_signature: Value of the ``webhook-signature`` header.
        secret: Your webhook signing secret (from the Parallel dashboard).
        tolerance_seconds: Reject signatures whose timestamp differs from
            ``time.time()`` by more than this many seconds. Defaults to 5
            minutes (the Standard Webhooks recommendation).

    Returns:
        True if the signature is valid and within tolerance; False otherwise.
    """
    import base64
    import time

    try:
        ts = int(webhook_timestamp)
    except (TypeError, ValueError):
        return False
    if abs(time.time() - ts) > tolerance_seconds:
        return False

    signed = f"{webhook_id}.{webhook_timestamp}.{payload.decode('utf-8')}"
    digest = hmac.new(
        secret.encode("utf-8"),
        signed.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(digest).decode("ascii")

    # The header may carry multiple signatures, space-delimited, each prefixed
    # with a version (e.g. "v1,<sig> v1,<sig>"). Match if any entry matches.
    for entry in webhook_signature.split():
        if "," not in entry:
            continue
        version, sig = entry.split(",", 1)
        if version != "v1":
            continue
        if hmac.compare_digest(expected, sig):
            return True
    return False


class _TaskClientMixin(BaseModel):
    """Shared API-key + client plumbing for the Task surfaces."""

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://api.parallel.ai")

    _client: Optional[Parallel] = None
    _async_client: Optional[AsyncParallel] = None

    @model_validator(mode="after")
    def _initialize_clients(self) -> _TaskClientMixin:
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self


class ParallelTaskRunInput(BaseModel):
    """Input schema for :class:`ParallelTaskRunTool`."""

    input: Union[str, dict[str, Any]] = Field(
        description=(
            "The task input. Either a freeform string (matches a default text "
            "input schema) or a dict matching the tool's configured "
            "`input_schema` if a TaskSpec was set."
        ),
    )
    metadata: Optional[dict[str, Union[str, float, bool]]] = Field(
        default=None,
        description="Free-form metadata persisted on the run.",
    )
    previous_interaction_id: Optional[str] = Field(
        default=None,
        description=(
            "Chain context from a prior run. Pass the `interaction_id` from "
            "an earlier `ParallelTaskRunTool` / `ParallelDeepResearch` result "
            "to continue a multi-turn research thread. See "
            "https://docs.parallel.ai/task-api/guides/interactions."
        ),
    )


class ParallelTaskRunTool(BaseTool):
    """Run a single Parallel Task synchronously and return the structured result.

    This is the agent-friendly path: an LLM calls the tool with `input`, the
    tool blocks until the Parallel Task Run completes, and returns a dict
    containing the output, citations (`basis`), and run metadata.

    For long-running deep-research tasks, prefer :class:`ParallelDeepResearch`,
    which is a :class:`~langchain_core.runnables.Runnable` and returns the same
    result shape.

    Setup:
        ```bash
        export PARALLEL_API_KEY="your-api-key"
        ```

    Key init args:
        processor: Literal[...]
            Which Parallel processor to run. ``"lite"`` is fastest;
            ``"core"``/``"pro"`` for deep research; ``"ultra"`` family for
            the highest-quality long-running tasks.
        output_schema: Optional[type[BaseModel] | dict | str]
            If a pydantic class, the SDK parses the response into an instance
            of the class. If a dict, it's used as the JSON schema. If a
            string, it's used as the natural-language output description
            (text output mode).
        mcp_servers: Optional[list[McpServer]]
            BYOMCP servers exposed to the run.
        api_key: Optional[SecretStr]

    Invocation:
        ```python
        from langchain_parallel import ParallelTaskRunTool

        tool = ParallelTaskRunTool(processor="lite")
        result = tool.invoke({"input": "Who founded SpaceX, in one sentence?"})
        # The structured output is at result["output"]["content"]; per-field
        # citations are at result["output"]["basis"]; the run id is at
        # result["run"]["run_id"].
        print(result["output"]["content"])
        print(result["output"]["basis"])
        ```
    """

    name: str = "parallel_task_run"
    description: str = (
        "Run a single Parallel Task synchronously. Inputs are either freeform "
        "text or a dict matching the configured input_schema. Returns the "
        "structured output, per-field citations (basis), and run id."
    )
    args_schema: type[BaseModel] = ParallelTaskRunInput

    api_key: Optional[SecretStr] = Field(default=None)
    base_url: str = Field(default="https://api.parallel.ai")

    processor: ProcessorLiteral = Field(default="lite")
    """Which processor to run (lite, base, core, pro, ultra family)."""

    task_output_schema: Optional[Union[type[BaseModel], dict[str, Any], str]] = None
    """Output spec: pydantic class, JSON-schema dict, or text description.

    (Named `task_output_schema` to avoid shadowing `BaseTool.output_schema`.)
    """

    task_spec: Optional[dict[str, Any]] = None
    """Optional full TaskSpec dict ``{input_schema, output_schema}``.

    When set, takes precedence over ``task_output_schema`` and unlocks
    structured ``input_schema`` validation. Required for the Task Group
    API's structured-batch pattern. See
    https://docs.parallel.ai/task-api/guides/specify-a-task.
    """

    mcp_servers: Optional[list[McpServer]] = None
    """BYOMCP servers to make available to the run."""

    _client: Optional[Parallel] = None
    _async_client: Optional[AsyncParallel] = None

    @model_validator(mode="after")
    def _initialize_clients(self) -> ParallelTaskRunTool:
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None,
        )
        self._client = get_parallel_client(api_key_str, self.base_url)
        self._async_client = get_async_parallel_client(api_key_str, self.base_url)
        return self

    def _build_execute_kwargs(
        self,
        task_input: Union[str, dict[str, Any]],
        metadata: Optional[dict[str, Union[str, float, bool]]],
        timeout: Optional[float],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "input": task_input,
            "processor": self.processor,
        }
        if self.task_output_schema is not None:
            # The SDK accepts the pydantic class directly via the `output=` kwarg
            # and parses for you; for dicts/strings it treats them as schemas.
            kwargs["output"] = self.task_output_schema
        if metadata is not None:
            kwargs["metadata"] = metadata
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _build_create_kwargs(
        self,
        task_input: Union[str, dict[str, Any]],
        metadata: Optional[dict[str, Union[str, float, bool]]],
        previous_interaction_id: Optional[str],
        timeout: Optional[float],
    ) -> dict[str, Any]:
        """Build kwargs for ``client.beta.task_run.create`` (the create+poll path).

        Used when the user passes ``mcp_servers``, ``task_spec``, or any
        other field that ``execute()`` doesn't accept.
        """
        kwargs: dict[str, Any] = {
            "input": task_input,
            "processor": self.processor,
        }
        if self.task_spec is not None:
            kwargs["task_spec"] = self.task_spec
        if metadata is not None:
            kwargs["metadata"] = metadata
        if previous_interaction_id is not None:
            kwargs["previous_interaction_id"] = previous_interaction_id
        if self.mcp_servers:
            kwargs["mcp_servers"] = [m.to_sdk() for m in self.mcp_servers]
            kwargs["betas"] = [_MCP_BETA_HEADER]
        if timeout is not None:
            kwargs["timeout"] = timeout
        return kwargs

    def _format_result(self, result: Any) -> dict[str, Any]:
        """Convert the SDK's TaskRunResult / ParsedTaskRunResult to a dict.

        Surfaces the run id, output, basis (citations + reasoning + confidence),
        any usage info the API returned, and the top-level ``interaction_id``
        for multi-turn context chaining.
        """
        payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        # ParsedTaskRunResult exposes `.parsed` as the typed pydantic instance;
        # if our caller asked for a pydantic schema, surface the parsed object.
        if hasattr(result, "parsed") and result.parsed is not None:
            parsed = result.parsed
            payload["parsed"] = (
                parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
            )
        # Promote interaction_id from the nested `run` dict so callers can
        # chain multi-turn research without diving into the structure.
        run_block = payload.get("run") or {}
        if isinstance(run_block, dict) and run_block.get("interaction_id"):
            payload["interaction_id"] = run_block["interaction_id"]
        return payload

    def _needs_create_path(self) -> bool:
        """`execute()` doesn't accept mcp_servers or task_spec; route via create."""
        return bool(self.mcp_servers) or self.task_spec is not None

    def _run(
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002 -- SDK uses `input` too
        metadata: Optional[dict[str, Union[str, float, bool]]] = None,
        previous_interaction_id: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        if run_manager:
            run_manager.on_text(
                f"Starting Parallel task ({self.processor})...\n",
                color="blue",
            )

        try:
            if self._needs_create_path():
                run = self._client.beta.task_run.create(
                    **self._build_create_kwargs(
                        input,
                        metadata,
                        previous_interaction_id,
                        timeout,
                    ),
                )
                result = self._client.task_run.result(run.run_id, timeout=timeout)
            else:
                kwargs = self._build_execute_kwargs(input, metadata, timeout)
                if previous_interaction_id is not None:
                    # `execute()` doesn't take previous_interaction_id; route via
                    # create() so multi-turn chaining works on the simple path too.
                    create_kwargs = self._build_create_kwargs(
                        input,
                        metadata,
                        previous_interaction_id,
                        timeout,
                    )
                    if self.task_output_schema is not None:
                        create_kwargs["task_spec"] = create_kwargs.get(
                            "task_spec",
                        ) or {"output_schema": self.task_output_schema}
                    run = self._client.beta.task_run.create(**create_kwargs)
                    result = self._client.task_run.result(run.run_id, timeout=timeout)
                else:
                    result = self._client.task_run.execute(**kwargs)
        except Exception as e:
            msg = f"Error calling Parallel Task API: {e!s}"
            raise ValueError(msg) from e

        if run_manager:
            run_manager.on_text("Task completed\n", color="green")
        return self._format_result(result)

    async def _arun(
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002
        metadata: Optional[dict[str, Union[str, float, bool]]] = None,
        previous_interaction_id: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict[str, Any]:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        if run_manager:
            await run_manager.on_text(
                f"Starting Parallel task ({self.processor})...\n",
                color="blue",
            )

        try:
            if self._needs_create_path() or previous_interaction_id is not None:
                create_kwargs = self._build_create_kwargs(
                    input,
                    metadata,
                    previous_interaction_id,
                    timeout,
                )
                if (
                    not self._needs_create_path()
                    and self.task_output_schema is not None
                ):
                    create_kwargs["task_spec"] = create_kwargs.get(
                        "task_spec",
                    ) or {"output_schema": self.task_output_schema}
                run = await self._async_client.beta.task_run.create(**create_kwargs)
                result = await self._async_client.task_run.result(
                    run.run_id,
                    timeout=timeout,
                )
            else:
                result = await self._async_client.task_run.execute(
                    **self._build_execute_kwargs(input, metadata, timeout),
                )
        except Exception as e:
            msg = f"Error calling Parallel Task API: {e!s}"
            raise ValueError(msg) from e

        if run_manager:
            await run_manager.on_text("Task completed\n", color="green")
        return self._format_result(result)


class ParallelDeepResearch(Runnable[Union[str, dict[str, Any]], dict[str, Any]]):
    """High-level Runnable for Parallel deep-research tasks.

    Defaults to the ``core`` processor and always returns the full `basis`
    (citations + reasoning + confidence). Lower friction than wiring up
    :class:`ParallelTaskRunTool` manually when all you want is "do deep
    research on this question."

    Example:
        ```python
        research = ParallelDeepResearch(processor="core")
        result = research.invoke("Latest developments in renewable energy")
        print(result["output"]["content"])
        for fact in result["output"].get("basis", []):
            print(fact["field"], "->", fact["citations"])
        ```
    """

    def __init__(
        self,
        *,
        processor: ProcessorLiteral = "core",
        output_schema: Optional[Union[type[BaseModel], dict[str, Any], str]] = None,
        api_key: Optional[Union[str, SecretStr]] = None,
        base_url: str = "https://api.parallel.ai",
        mcp_servers: Optional[list[McpServer]] = None,
    ) -> None:
        self.processor = processor
        self._output_schema = output_schema
        self.mcp_servers = mcp_servers
        self._tool = ParallelTaskRunTool(
            processor=processor,
            task_output_schema=output_schema,
            api_key=(
                api_key
                if isinstance(api_key, SecretStr) or api_key is None
                else SecretStr(api_key)
            ),
            base_url=base_url,
            mcp_servers=mcp_servers,
        )

    def invoke(  # type: ignore[override]
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002
        config: Any = None,
        *,
        previous_interaction_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool_input: dict[str, Any] = {"input": input}
        if previous_interaction_id is not None:
            tool_input["previous_interaction_id"] = previous_interaction_id
        return self._tool.invoke(tool_input, config=config, **kwargs)

    async def ainvoke(  # type: ignore[override]
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002
        config: Any = None,
        *,
        previous_interaction_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool_input: dict[str, Any] = {"input": input}
        if previous_interaction_id is not None:
            tool_input["previous_interaction_id"] = previous_interaction_id
        return await self._tool.ainvoke(tool_input, config=config, **kwargs)


class ParallelTaskGroup(_TaskClientMixin):
    """Batch task runner backed by the Task Group API.

    Use when you have a list of inputs and want them all processed in
    parallel. Returns a list of result dicts in the same order as the
    input list.

    Example:
        ```python
        group = ParallelTaskGroup(processor="lite")
        results = group.run(
            inputs=[
                "Founder of Anthropic?",
                "Founder of OpenAI?",
                "Founder of Google DeepMind?",
            ]
        )
        for inp, out in zip(inputs, results):
            print(inp, "->", out["output"])
        ```
    """

    processor: ProcessorLiteral = Field(default="lite")
    """Default processor for runs added to this group."""

    metadata: Optional[dict[str, Union[str, float, bool]]] = Field(default=None)

    def _kick_off_runs(
        self,
        inputs: list[Union[str, dict[str, Any]]],
    ) -> list[str]:
        """Create a group, add runs, and return the list of run_ids."""
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        group = self._client.beta.task_group.create(metadata=self.metadata)
        run_inputs: list[Any] = [
            {"input": inp, "processor": self.processor} for inp in inputs
        ]
        response = self._client.beta.task_group.add_runs(
            group.task_group_id,
            inputs=run_inputs,
            refresh_status=True,
        )
        return list(response.run_ids or [])

    async def _akick_off_runs(
        self,
        inputs: list[Union[str, dict[str, Any]]],
    ) -> list[str]:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        group = await self._async_client.beta.task_group.create(
            metadata=self.metadata,
        )
        run_inputs: list[Any] = [
            {"input": inp, "processor": self.processor} for inp in inputs
        ]
        response = await self._async_client.beta.task_group.add_runs(
            group.task_group_id,
            inputs=run_inputs,
            refresh_status=True,
        )
        return list(response.run_ids or [])

    def run(
        self,
        inputs: list[Union[str, dict[str, Any]]],
        *,
        timeout: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Submit a batch of inputs and block until all results are ready."""
        if self._client is None:
            msg = "Parallel client not initialized."
            raise RuntimeError(msg)
        try:
            run_ids = self._kick_off_runs(inputs)
            results: list[dict[str, Any]] = []
            for run_id in run_ids:
                result = self._client.task_run.result(run_id, timeout=timeout)
                results.append(
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else dict(result),
                )
            return results
        except Exception as e:
            msg = f"Error calling Parallel Task Group API: {e!s}"
            raise ValueError(msg) from e

    async def arun(
        self,
        inputs: list[Union[str, dict[str, Any]]],
        *,
        timeout: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if self._async_client is None:
            msg = "Async Parallel client not initialized."
            raise RuntimeError(msg)
        try:
            run_ids = await self._akick_off_runs(inputs)
            results: list[dict[str, Any]] = []
            for run_id in run_ids:
                result = await self._async_client.task_run.result(
                    run_id,
                    timeout=timeout,
                )
                results.append(
                    result.model_dump()
                    if hasattr(result, "model_dump")
                    else dict(result),
                )
            return results
        except Exception as e:
            msg = f"Error calling Parallel Task Group API: {e!s}"
            raise ValueError(msg) from e
