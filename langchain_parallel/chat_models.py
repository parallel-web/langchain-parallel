"""Parallel Web chat model integration.

This module provides the ChatParallelWeb class for interacting with Parallel's
Chat API through an OpenAI-compatible interface.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, Optional, Union, cast

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_json_schema
from langchain_core.utils.pydantic import is_basemodel_subclass
from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from ._client import get_api_key, get_async_openai_client, get_openai_client

# Models that support response_format JSON schema. The `speed` model ignores it.
_STRUCTURED_OUTPUT_MODELS: frozenset[str] = frozenset({"lite", "base", "core"})


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to OpenAI message format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    msg = f"Unsupported message type: {type(message)}"
    raise ValueError(msg)


def _prepare_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Prepare messages for API call by merging consecutive messages and converting to dict format."""  # noqa: E501
    merged_messages = _merge_consecutive_messages(messages)
    return [_convert_message_to_dict(msg) for msg in merged_messages]


def _create_response_metadata(response: Any, choice: Any) -> dict[str, Any]:
    """Create response metadata from API response.

    Uses LangChain 1.x standard keys (`model_name`, `finish_reason`,
    `system_fingerprint`). Surfaces Parallel-specific fields (`basis`,
    `interaction_id`) when present.
    """
    metadata: dict[str, Any] = {
        "model_name": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "created": getattr(response, "created", None),
    }
    system_fingerprint = getattr(response, "system_fingerprint", None)
    if system_fingerprint is not None:
        metadata["system_fingerprint"] = system_fingerprint
    basis = getattr(response, "basis", None)
    if basis:
        metadata["basis"] = (
            [b.model_dump() if hasattr(b, "model_dump") else b for b in basis]
            if isinstance(basis, list)
            else basis
        )
    interaction_id = getattr(response, "interaction_id", None)
    if interaction_id is not None:
        metadata["interaction_id"] = interaction_id
    return metadata


def _create_ai_message(content: str, response_metadata: dict[str, Any]) -> AIMessage:
    """Create AIMessage with standard format."""
    return AIMessage(
        content=content,
        response_metadata=response_metadata,
        usage_metadata=None,  # Parallel doesn't return usage metadata
    )


def _create_stream_response_metadata(chunk: Any, choice: Any) -> dict[str, Any]:
    """Create response metadata for streaming chunks."""
    response_metadata: dict[str, Any] = {}
    if hasattr(choice, "finish_reason") and choice.finish_reason is not None:
        response_metadata["finish_reason"] = str(choice.finish_reason)
    if hasattr(chunk, "model"):
        response_metadata["model_name"] = chunk.model
    if getattr(chunk, "system_fingerprint", None) is not None:
        response_metadata["system_fingerprint"] = chunk.system_fingerprint
    if getattr(chunk, "interaction_id", None) is not None:
        response_metadata["interaction_id"] = chunk.interaction_id
    basis = getattr(chunk, "basis", None)
    if basis:
        response_metadata["basis"] = (
            [b.model_dump() if hasattr(b, "model_dump") else b for b in basis]
            if isinstance(basis, list)
            else basis
        )
    return response_metadata


def _merge_consecutive_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Merge consecutive messages of the same type to satisfy API requirements.

    Parallel requires messages to alternate between user and assistant roles.
    This function merges consecutive messages of the same type.
    """
    if not messages:
        return messages

    merged: list[BaseMessage] = []
    current_content = []
    current_type = None

    for message in messages:
        message_type = type(message)

        if message_type == current_type:
            # Same type as previous, accumulate content
            current_content.append(str(message.content))
        else:
            # Different type, save previous and start new
            if current_type is not None and current_content:
                # Create merged message of the previous type
                merged_content = "\n\n".join(current_content)
                if current_type == SystemMessage:
                    merged.append(SystemMessage(content=merged_content))
                elif current_type == HumanMessage:
                    merged.append(HumanMessage(content=merged_content))
                elif current_type == AIMessage:
                    merged.append(AIMessage(content=merged_content))

            # Start new message
            current_type = message_type
            current_content = [str(message.content)]

    # Don't forget the last message
    if current_type is not None and current_content:
        merged_content = "\n\n".join(current_content)
        if current_type == SystemMessage:
            merged.append(SystemMessage(content=merged_content))
        elif current_type == HumanMessage:
            merged.append(HumanMessage(content=merged_content))
        elif current_type == AIMessage:
            merged.append(AIMessage(content=merged_content))

    return merged


class ChatParallelWeb(BaseChatModel):
    """Parallel Web chat model integration.

    This integration connects to Parallel's Chat API, which provides
    real-time web research capabilities through an OpenAI-compatible interface.

    Setup:
        Install `langchain-parallel` and set environment variable
        `PARALLEL_API_KEY`.

        ```bash
        pip install -U langchain-parallel
        export PARALLEL_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model: str
            Name of Parallel Web model to use. Defaults to "speed".
        temperature: Optional[float]
            Sampling temperature (ignored by Parallel).
        max_tokens: Optional[int]
            Max number of tokens to generate (ignored by Parallel).

    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Parallel API key. If not passed in will be read from env var
            PARALLEL_API_KEY.
        base_url: str
            Base URL for Parallel API. Defaults to "https://api.parallel.ai".

    Instantiate:
        ```python
        from langchain_parallel import ChatParallelWeb

        llm = ChatParallelWeb(
            model="speed",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful assistant with access to real-time web "
                "information."
            ),
            ("human", "What are the latest developments in AI?"),
        ]
        llm.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in llm.stream(messages):
            print(chunk.content, end="")
        ```

    Async:
        ```python
        await llm.ainvoke(messages)

        # stream:
        async for chunk in llm.astream(messages):
            print(chunk.content, end="")

        # batch:
        await llm.abatch([messages])
        ```

    Token usage:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.usage_metadata
        ```

    Response metadata:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.response_metadata
        ```

    """

    model: str = Field(default="speed")
    """The name of the model to use.

    One of:

    - ``"speed"`` (default): low-latency conversational answers, no citations.
    - ``"lite"`` / ``"base"`` / ``"core"``: research models with web access
      that return source citations on ``response_metadata['basis']`` and
      support ``response_format`` JSON schemas via
      :meth:`with_structured_output`.
    """

    api_key: Optional[SecretStr] = Field(default=None)
    """Parallel API key. If not provided, will be read from
    PARALLEL_API_KEY env var."""

    base_url: str = Field(default="https://api.parallel.ai")
    """Base URL for Parallel API."""

    temperature: Optional[float] = Field(default=None)
    """Sampling temperature (ignored by Parallel)."""

    max_tokens: Optional[int] = Field(default=None)
    """Max number of tokens to generate (ignored by Parallel)."""

    timeout: Optional[float] = Field(default=None)
    """Timeout for requests."""

    max_retries: int = Field(default=2)
    """Max number of retries."""

    # OpenAI-compatible parameters that are ignored by Parallel
    response_format: Optional[dict[str, Any]] = Field(default=None)
    """Response format (ignored by Parallel)."""

    tools: Optional[list[dict[str, Any]]] = Field(default=None)
    """Tools for function calling (ignored by Parallel)."""

    tool_choice: Optional[str] = Field(default=None)
    """Tool choice parameter (ignored by Parallel)."""

    stream_options: Optional[dict[str, Any]] = Field(default=None)
    """Stream options (ignored by Parallel)."""

    top_p: Optional[float] = Field(default=None)
    """Top-p sampling parameter (ignored by Parallel)."""

    frequency_penalty: Optional[float] = Field(default=None)
    """Frequency penalty (ignored by Parallel)."""

    presence_penalty: Optional[float] = Field(default=None)
    """Presence penalty (ignored by Parallel)."""

    logit_bias: Optional[dict[str, float]] = Field(default=None)
    """Logit bias (ignored by Parallel)."""

    seed: Optional[int] = Field(default=None)
    """Random seed (ignored by Parallel)."""

    user: Optional[str] = Field(default=None)
    """User identifier (ignored by Parallel)."""

    _client: Optional[openai.OpenAI] = None
    _async_client: Optional[openai.AsyncOpenAI] = None

    @model_validator(mode="before")
    @classmethod
    def _accept_model_name_alias(cls, values: Any) -> Any:
        """Accept ``model_name="..."`` as a back-compat alias for ``model="..."``.

        Pre-0.3.0 the field was declared as ``Field(alias="model_name")``,
        meaning users had to pass ``model_name=`` and ``model=`` was silently
        ignored. The alias was removed in 0.3.0 to fix that footgun; this
        validator preserves callers that still use ``model_name=``.
        """
        if (
            isinstance(values, dict)
            and "model_name" in values
            and "model" not in values
        ):
            values = {**values, "model": values.pop("model_name")}
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key exists and initialize clients."""
        # Get API key from parameter or environment - this will raise if not found
        api_key_str = get_api_key(
            self.api_key.get_secret_value() if self.api_key else None
        )

        # Set the api_key field if it was loaded from environment
        if not self.api_key:
            self.api_key = SecretStr(api_key_str)

        # Initialize both sync and async OpenAI clients configured for Parallel
        self._client = get_openai_client(api_key_str, self.base_url)
        self._async_client = get_async_openai_client(api_key_str, self.base_url)
        return self

    @property
    def client(self) -> openai.OpenAI:
        """Get the sync OpenAI client, initializing if needed."""
        if self._client is None:
            msg = (
                "Client not initialized. Please ensure the model is properly validated."
            )
            raise ValueError(msg)
        return self._client

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        """Get the async OpenAI client, initializing if needed."""
        if self._async_client is None:
            msg = (
                "Async client not initialized. "
                "Please ensure the model is properly validated."
            )
            raise ValueError(msg)
        return self._async_client

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-parallel-web"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": self.response_format,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
        }

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Return secrets for LangChain serialization."""
        return {"api_key": "PARALLEL_API_KEY"}

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """Return attributes for LangChain serialization."""
        attributes: dict[str, Any] = {"model_name": self.model}
        if self.base_url:
            attributes["base_url"] = self.base_url
        return attributes

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object."""
        return ["langchain_parallel", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @contextlib.contextmanager
    def _handle_errors(self) -> Iterator[None]:
        """Handle errors from Parallel API."""
        try:
            yield
        except AuthenticationError as e:
            msg = (
                f"Authentication failed with Parallel API. "
                f"Please check your API key: {e!s}"
            )
            raise ValueError(msg)
        except RateLimitError as e:
            msg = f"Rate limit exceeded for Parallel API. Please try again later: {e!s}"
            raise ValueError(msg)
        except Exception as e:
            msg = f"Error calling Parallel API: {e!s}"
            raise ValueError(msg)

    def _process_non_stream_response(self, response: Any) -> ChatResult:
        """Process a non-streaming response into a ChatResult."""
        choice = response.choices[0]
        content = choice.message.content or ""
        response_metadata = _create_response_metadata(response, choice)
        response_metadata["model_name"] = response_metadata["model_name"] or self.model

        message = _create_ai_message(content, response_metadata)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _process_stream_chunk(
        self, chunk: Any, run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> Optional[ChatGenerationChunk]:
        """Process a streaming chunk into a ChatGenerationChunk."""
        if not (hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0):
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = ""
        if hasattr(delta, "content") and delta.content is not None:
            content = delta.content

        response_metadata = _create_stream_response_metadata(chunk, choice)

        chunk_message = AIMessageChunk(
            content=content,
            response_metadata=response_metadata,
            usage_metadata=None,  # Parallel doesn't return usage metadata
        )

        if run_manager and content:
            run_manager.on_llm_new_token(content)

        return ChatGenerationChunk(message=chunk_message)

    async def _process_async_stream_chunk(
        self, chunk: Any, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None
    ) -> Optional[ChatGenerationChunk]:
        """Process an async streaming chunk into a ChatGenerationChunk."""
        if not (chunk.choices and len(chunk.choices) > 0):
            return None

        choice = chunk.choices[0]
        delta = choice.delta

        content = ""
        if hasattr(delta, "content") and delta.content is not None:
            content = delta.content

        response_metadata = _create_stream_response_metadata(chunk, choice)

        chunk_message = AIMessageChunk(
            content=content,
            response_metadata=response_metadata,
            usage_metadata=None,  # Parallel doesn't return usage metadata
        )

        if run_manager and content:
            await run_manager.on_llm_new_token(content)

        return ChatGenerationChunk(message=chunk_message)

    def _build_create_kwargs(
        self,
        messages: list[dict[str, Any]],
        stop: Optional[list[str]],
        *,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build kwargs for the OpenAI ``chat.completions.create`` call.

        Per-call ``extra`` (typically populated by ``with_structured_output``)
        wins over instance-level fields.
        """
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": cast(Any, messages),
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
        }
        if self.response_format is not None:
            create_kwargs["response_format"] = self.response_format
        # Per-call overrides from the runnable kwargs. Drop None values.
        create_kwargs.update({k: v for k, v in extra.items() if v is not None})
        return create_kwargs

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using Parallel's chat API."""
        openai_messages = _prepare_messages(messages)

        with self._handle_errors():
            response = self.client.chat.completions.create(
                **self._build_create_kwargs(
                    openai_messages,
                    stop,
                    stream=False,
                    extra=kwargs,
                ),
            )

            return self._process_non_stream_response(response)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from Parallel's chat API."""
        openai_messages = _prepare_messages(messages)

        with self._handle_errors():
            stream = self.client.chat.completions.create(
                **self._build_create_kwargs(
                    openai_messages,
                    stop,
                    stream=True,
                    extra=kwargs,
                ),
            )

            for chunk in stream:
                chunk_result = self._process_stream_chunk(chunk, run_manager)
                if chunk_result is not None:
                    yield chunk_result

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate a response using Parallel's chat API."""
        openai_messages = _prepare_messages(messages)

        with self._handle_errors():
            response = await self.async_client.chat.completions.create(
                **self._build_create_kwargs(
                    openai_messages,
                    stop,
                    stream=False,
                    extra=kwargs,
                ),
            )

            return self._process_non_stream_response(response)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async stream responses from Parallel's chat API."""
        openai_messages = _prepare_messages(messages)

        with self._handle_errors():
            stream = await self.async_client.chat.completions.create(
                **self._build_create_kwargs(
                    openai_messages,
                    stop,
                    stream=True,
                    extra=kwargs,
                ),
            )

            async for chunk in stream:
                chunk_result = await self._process_async_stream_chunk(
                    chunk, run_manager
                )
                if chunk_result is not None:
                    yield chunk_result

    def with_structured_output(
        self,
        schema: Optional[Union[dict[str, Any], type[BaseModel]]] = None,
        *,
        method: Literal["json_schema", "function_calling", "json_mode"] = "json_schema",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict[str, Any], BaseModel]]:
        """Return a Runnable that produces structured output.

        Parallel's research models (``lite``, ``base``, ``core``) accept the
        OpenAI ``response_format`` parameter with a JSON schema. The ``speed``
        model silently ignores it; this method raises if you try to use it
        on a non-supporting model so the failure is loud.

        Args:
            schema: A pydantic v2 model class or a JSON schema dict.
            method: ``"json_schema"`` (default) for strict-typed output, or
                ``"json_mode"`` to ask the model for any valid JSON object.
                ``"function_calling"`` is accepted for cross-provider
                compatibility and is routed to ``"json_schema"`` since
                Parallel's chat API does not support tool calling.
            include_raw: If True, return ``{"raw": AIMessage, "parsed": ...,
                "parsing_error": ...}`` instead of just the parsed value.
            strict: Forwarded to the API's ``response_format`` JSON schema.
                Defaults to True for pydantic schemas, None for raw dicts.
            **kwargs: Reserved for forward compatibility; unused.
        """
        if kwargs:
            msg = f"Received unsupported kwargs: {sorted(kwargs)}"
            raise ValueError(msg)
        if self.model not in _STRUCTURED_OUTPUT_MODELS:
            msg = (
                f"Structured output requires one of the research models "
                f"({sorted(_STRUCTURED_OUTPUT_MODELS)}); the '{self.model}' "
                f"model silently ignores response_format. Re-instantiate with "
                f"`ChatParallelWeb(model='lite' | 'base' | 'core')`."
            )
            raise ValueError(msg)
        if method == "function_calling":
            # Parallel chat doesn't support tool calling; route to json_schema
            # since the user-visible result is equivalent.
            method = "json_schema"
        if method not in {"json_schema", "json_mode"}:
            msg = (
                f"Unsupported method '{method}'. Use 'json_schema', "
                f"'function_calling' (routed to json_schema), or 'json_mode'."
            )
            raise ValueError(msg)

        if method == "json_mode":
            # `json_mode` only enables JSON output without a schema constraint;
            # if a schema is also passed, accept it for cross-provider compat
            # but only use it for the parser, not for the API call.
            response_format: dict[str, Any] = {"type": "json_object"}
            schema_is_pydantic = (
                schema is not None
                and isinstance(schema, type)
                and is_basemodel_subclass(schema)
            )
            output_parser: Runnable = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if schema_is_pydantic
                else JsonOutputParser()
            )
        else:
            if schema is None:
                msg = "method='json_schema' requires a schema."
                raise ValueError(msg)
            is_pydantic = isinstance(schema, type) and is_basemodel_subclass(schema)
            strict_value: Optional[bool]
            if is_pydantic:
                json_schema = convert_to_json_schema(schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                strict_value = True if strict is None else strict
            else:
                json_schema = dict(schema)  # type: ignore[arg-type]
                output_parser = JsonOutputParser()
                strict_value = strict
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "output"),
                    "schema": json_schema,
                },
            }
            if strict_value is not None:
                response_format["json_schema"]["strict"] = strict_value

        bound = self.bind(response_format=response_format)
        if include_raw:

            def _parse_with_capture(raw: AIMessage) -> dict[str, Any]:
                try:
                    return {
                        "raw": raw,
                        "parsed": output_parser.invoke(raw),
                        "parsing_error": None,
                    }
                except Exception as e:
                    return {"raw": raw, "parsed": None, "parsing_error": e}

            return bound | _parse_with_capture
        return bound | output_parser


#: Forward-compat alias for :class:`ChatParallelWeb`.
#:
#: Prefer ChatParallel in new code; ChatParallelWeb will continue to
#: work indefinitely as an alias for this class.
ChatParallel = ChatParallelWeb
