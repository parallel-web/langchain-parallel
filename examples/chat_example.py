"""Example usage of Parallel Chat integration."""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_parallel import ChatParallelWeb

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def basic_example() -> None:
    """Basic synchronous chat example."""
    print("=== Basic Chat Example ===")

    # Initialize the chat model
    chat = ChatParallelWeb(
        model_name="speed",  # Parallel's chat model
        temperature=0.7,  # Optional: temperature (ignored by Parallel)
        max_tokens=None,  # Optional: max tokens (ignored by Parallel)
    )

    # Create messages
    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. Provide concise, "
                "accurate answers based on current information."
            )
        ),
        HumanMessage(
            content="What are the latest developments in renewable energy technology?"
        ),
    ]

    # Get response
    try:
        response = chat.invoke(messages)
        print(f"Response: {response.content[:200]}...")

        if hasattr(response, "usage_metadata") and getattr(
            response, "usage_metadata", None
        ):
            print(f"Usage: {response.usage_metadata}")

    except ValueError as e:
        if "API key not found" in str(e):
            print("Error: API key not found. Please set PARALLEL_API_KEY")
        else:
            print(f"Error: {e}")


def streaming_example() -> None:
    """Streaming example for real-time responses."""
    print("\n=== Streaming Chat Example ===")

    chat = ChatParallelWeb()

    messages = [
        SystemMessage(content="You are a creative writing assistant."),
        HumanMessage(content="Write a short poem about the ocean."),
    ]

    print("Streaming response:")
    try:
        full_response = ""
        for chunk in chat.stream(messages):
            if chunk.content:
                content_str = str(chunk.content) if chunk.content else ""
                print(content_str, end="", flush=True)
                full_response += content_str

        print(f"\nTotal response length: {len(full_response)} characters")

    except ValueError as e:
        if "API key not found" in str(e):
            print("Error: API key not found. Please set PARALLEL_API_KEY")
        else:
            print(f"Error: {e}")


async def async_example() -> None:
    """Asynchronous example."""
    print("\n=== Async Chat Example ===")

    chat = ChatParallelWeb()

    messages = [
        SystemMessage(content="You are a technology expert."),
        HumanMessage(content="Explain quantum computing in simple terms."),
    ]

    try:
        # Async invoke
        response = await chat.ainvoke(messages)
        print(f"Async response: {response.content[:200]}...")

        # Async streaming
        print("\nAsync streaming:")
        async for chunk in chat.astream(
            [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Count from 1 to 5."),
            ]
        ):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        print("\nAsync streaming completed")

    except ValueError as e:
        if "API key not found" in str(e):
            print("Error: API key not found. Please set PARALLEL_API_KEY")
        else:
            print(f"Error: {e}")


def conversation_example() -> None:
    """Example of maintaining conversation context."""
    print("\n=== Conversation Example ===")

    chat = ChatParallelWeb()

    # Start with system message
    messages: list[BaseMessage] = [
        SystemMessage(
            content="You are a helpful assistant that remembers conversation context."
        )
    ]

    # Simulate a conversation
    user_inputs = [
        "What's the capital of France?",
        "What's the population of that city?",
        "What language do they speak there?",
    ]

    try:
        for i, user_input in enumerate(user_inputs, 1):
            print(f"\nTurn {i}")
            print(f"User: {user_input}")

            # Add user message
            messages.append(HumanMessage(content=user_input))

            # Get response
            response = chat.invoke(messages)
            print(f"AI: {response.content[:150]}...")

            # Add assistant response to conversation history
            messages.append(response)
            print(f"Conversation history: {len(messages)} messages")

    except ValueError as e:
        if "API key not found" in str(e):
            print("Error: API key not found. Please set PARALLEL_API_KEY")
        else:
            print(f"Error: {e}")


def main() -> None:
    """Run all examples."""
    print("=== Parallel Chat Examples ===")

    # Check if API key is set
    if not os.getenv("PARALLEL_API_KEY"):
        print("Error: PARALLEL_API_KEY environment variable not set")
        print("Please set your API key: export PARALLEL_API_KEY='your-api-key'")
        return

    print("API key found in environment")

    # Run examples
    try:
        basic_example()
        streaming_example()
        asyncio.run(async_example())
        conversation_example()

        print("\n=== All examples completed successfully ===")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
