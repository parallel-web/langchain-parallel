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

ALL_PROCESSORS = (
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
]


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


def verify_webhook(
    payload: bytes,
    signature: str,
    secret: str,
) -> bool:
    """Verify the HMAC-SHA256 signature on an incoming Parallel webhook.

    Parallel signs webhook payloads with a shared secret you configure when
    creating the webhook (`webhook.secret`). The signature is sent in the
    ``parallel-signature`` header; this function returns True iff the signature
    matches.

    Args:
        payload: Raw request body (bytes).
        signature: Value of the ``parallel-signature`` header.
        secret: The webhook signing secret you configured.

    Returns:
        True if valid, False otherwise.
    """
    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


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
        print(result["output"])
        print(result["basis"])  # citations on lite/base/core/etc.
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

    def _format_result(self, result: Any) -> dict[str, Any]:
        """Convert the SDK's TaskRunResult / ParsedTaskRunResult to a dict.

        Surfaces the run id, output, basis (citations), and any usage info
        the API returned.
        """
        payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        # ParsedTaskRunResult exposes `.parsed` as the typed pydantic instance;
        # if our caller asked for a pydantic schema, surface the parsed object.
        if hasattr(result, "parsed") and result.parsed is not None:
            parsed = result.parsed
            payload["parsed"] = (
                parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
            )
        return payload

    def _run(
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002 -- SDK uses `input` too
        metadata: Optional[dict[str, Union[str, float, bool]]] = None,
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

        if self.mcp_servers:
            # `execute` doesn't take mcp_servers; fall through to the create+result
            # path so we can pass them.
            create_kwargs = self._build_execute_kwargs(input, metadata, timeout)
            create_kwargs["mcp_servers"] = [m.to_sdk() for m in self.mcp_servers]
            create_kwargs["betas"] = ["mcp-server-2025-07-17"]
            try:
                run = self._client.beta.task_run.create(**create_kwargs)
                result = self._client.task_run.result(run.run_id, timeout=timeout)
            except Exception as e:
                msg = f"Error calling Parallel Task API: {e!s}"
                raise ValueError(msg) from e
        else:
            try:
                result = self._client.task_run.execute(
                    **self._build_execute_kwargs(input, metadata, timeout),
                )
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

        if self.mcp_servers:
            create_kwargs = self._build_execute_kwargs(input, metadata, timeout)
            create_kwargs["mcp_servers"] = [m.to_sdk() for m in self.mcp_servers]
            create_kwargs["betas"] = ["mcp-server-2025-07-17"]
            try:
                run = await self._async_client.beta.task_run.create(**create_kwargs)
                result = await self._async_client.task_run.result(
                    run.run_id,
                    timeout=timeout,
                )
            except Exception as e:
                msg = f"Error calling Parallel Task API: {e!s}"
                raise ValueError(msg) from e
        else:
            try:
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
        print(result["output"])
        for fact in result.get("basis", []):
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._tool.invoke({"input": input}, config=config, **kwargs)

    async def ainvoke(  # type: ignore[override]
        self,
        input: Union[str, dict[str, Any]],  # noqa: A002
        config: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._tool.ainvoke({"input": input}, config=config, **kwargs)


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
