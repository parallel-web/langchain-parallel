"""Internal client utilities for Parallel integration.

Private module. Helpers in here are not part of the public API and may
change without notice; import from :mod:`langchain_parallel` instead.
"""

from __future__ import annotations

import os
from typing import Optional

import openai
from parallel import AsyncParallel, Parallel

__all__: list[str] = []


def get_api_key(api_key: Optional[str] = None) -> str:
    """Retrieve the Parallel API key from argument or environment variables.

    Args:
        api_key: Optional API key string.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not found.
    """
    if api_key:
        return api_key

    env_key = os.environ.get("PARALLEL_API_KEY")
    if env_key:
        return env_key

    msg = (
        "Parallel API key not found. Please pass it as an argument or set the "
        "PARALLEL_API_KEY environment variable."
    )
    raise ValueError(msg)


def get_openai_client(api_key: str, base_url: str) -> openai.OpenAI:
    """Returns a configured sync OpenAI client for the Chat API."""
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def get_async_openai_client(api_key: str, base_url: str) -> openai.AsyncOpenAI:
    """Returns a configured async OpenAI client for the Chat API."""
    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


def get_parallel_client(api_key: str, base_url: str) -> Parallel:
    """Returns a configured sync Parallel SDK client."""
    return Parallel(api_key=api_key, base_url=base_url)


def get_async_parallel_client(api_key: str, base_url: str) -> AsyncParallel:
    """Returns a configured async Parallel SDK client."""
    return AsyncParallel(api_key=api_key, base_url=base_url)
