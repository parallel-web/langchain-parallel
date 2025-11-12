"""Parallel Web chat model integration.

This module provides the ChatParallelWeb class for interacting with Parallel's
Chat API through an OpenAI-compatible interface.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, cast

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from openai import AuthenticationError, RateLimitError
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

from ._client import get_api_key, get_async_openai_client, get_openai_client


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
    """Create response metadata from API response."""
    return {
        "model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "created": getattr(response, "created", None),
    }


def _create_ai_message(content: str, response_metadata: dict[str, Any]) -> AIMessage:
    """Create AIMessage with standard format."""
    return AIMessage(
        content=content,
        response_metadata=response_metadata,
        usage_metadata=None,  # Parallel doesn't return usage metadata
    )


def _create_stream_response_metadata(chunk: Any, choice: Any) -> dict[str, Any]:
    """Create response metadata for streaming chunks."""
    response_metadata = {}
    if hasattr(choice, "finish_reason") and choice.finish_reason is not None:
        response_metadata["finish_reason"] = str(choice.finish_reason)
    if hasattr(chunk, "model"):
        response_metadata["model"] = chunk.model
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
        Install ``langchain-parallel`` and set environment variable
        ``PARALLEL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-parallel
            export PARALLEL_API_KEY="your-api-key"

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
        .. code-block:: python

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

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful assistant with access to real-time web "
                    "information."
                ),
                ("human", "What are the latest developments in AI?"),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="")

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            async for chunk in llm.astream(messages):
                print(chunk.content, end="")

            # batch:
            await llm.abatch([messages])

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

    Response metadata:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

    """

    model: str = Field(default="speed", alias="model_name")
    """The name of the model to use. Defaults to 'speed' for Parallel."""

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
        attributes: dict[str, Any] = {}
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
        response_metadata["model"] = response_metadata["model"] or self.model

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
                model=self.model,
                messages=cast(Any, openai_messages),
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
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
                model=self.model,
                messages=cast(Any, openai_messages),
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
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
                model=self.model,
                messages=cast(Any, openai_messages),
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
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
                model=self.model,
                messages=cast(Any, openai_messages),
                stream=True,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
            )

            async for chunk in stream:
                chunk_result = await self._process_async_stream_chunk(
                    chunk, run_manager
                )
                if chunk_result is not None:
                    yield chunk_result
