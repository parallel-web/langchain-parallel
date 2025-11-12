"""Examples of Parallel Extract Tool integration."""

from __future__ import annotations

import asyncio
import os

from langchain_parallel import ParallelExtractTool

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def basic_extract_examples() -> None:
    """Basic extract tool examples."""
    print("=== Basic Extract Tool Examples ===")

    # Initialize the extract tool
    tool = ParallelExtractTool()

    # Example 1: Extract from a single URL
    print("\nExample 1: Extract from a single URL")
    result = tool.invoke(
        {"urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]}
    )

    print(f"Extracted {len(result)} result")
    print(f"Title: {result[0]['title']}")
    print(f"URL: {result[0]['url']}")
    print(f"Content length: {len(result[0]['content'])} characters")
    print(f"Content preview: {result[0]['content'][:200]}...")


def batch_extract_examples() -> None:
    """Batch extraction examples."""
    print("\n=== Batch Extract Examples ===")

    tool = ParallelExtractTool()

    # Example 2: Extract from multiple URLs
    print("\nExample 2: Extract from multiple URLs")
    result = tool.invoke(
        {
            "urls": [
                "https://en.wikipedia.org/wiki/Machine_learning",
                "https://en.wikipedia.org/wiki/Deep_learning",
                "https://en.wikipedia.org/wiki/Natural_language_processing",
            ]
        }
    )

    print(f"Extracted {len(result)} results")
    for i, item in enumerate(result, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   URL: {item['url']}")
        print(f"   Content length: {len(item['content'])} characters")


def focused_extraction_examples() -> None:
    """Examples with search objective and queries."""
    print("\n=== Focused Extraction Examples ===")

    tool = ParallelExtractTool()

    # Example 3: Extract with search objective
    print("\nExample 3: Extract with search objective")
    result = tool.invoke(
        {
            "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
            "search_objective": (
                "What are the main applications and ethical concerns of AI?"
            ),
            "excerpts": True,
            "full_content": False,
        }
    )

    print(f"Title: {result[0]['title']}")
    print(f"Number of relevant excerpts: {len(result[0].get('excerpts', []))}")
    print(f"Content preview: {result[0]['content'][:300]}...")

    # Example 4: Extract with search queries
    print("\nExample 4: Extract with search queries")
    result = tool.invoke(
        {
            "urls": ["https://en.wikipedia.org/wiki/Machine_learning"],
            "search_queries": [
                "neural networks",
                "training algorithms",
                "applications",
            ],
            "excerpts": True,
        }
    )

    print(f"Title: {result[0]['title']}")
    print(f"Number of relevant excerpts: {len(result[0].get('excerpts', []))}")
    print("Relevant excerpts focused on the queries:")
    for i, excerpt in enumerate(result[0].get("excerpts", [])[:3], 1):
        print(f"\n  Excerpt {i}: {excerpt[:150]}...")


def content_length_control_examples() -> None:
    """Examples with content length control."""
    print("\n=== Content Length Control Examples ===")

    # Example 5: Limit content length
    print("\nExample 5: Extract with content length limit")
    tool = ParallelExtractTool(max_chars_per_extract=2000)

    result = tool.invoke({"urls": ["https://en.wikipedia.org/wiki/Quantum_computing"]})

    print(f"Title: {result[0]['title']}")
    print(f"Content length: {len(result[0]['content'])} characters (limited to ~2000)")
    print(f"Content preview: {result[0]['content'][:200]}...")


def error_handling_examples() -> None:
    """Examples with error handling."""
    print("\n=== Error Handling Examples ===")

    tool = ParallelExtractTool()

    # Example 4: Handle mixed valid/invalid URLs
    print("\nExample 4: Extract with error handling")
    result = tool.invoke(
        {
            "urls": [
                "https://en.wikipedia.org/wiki/Python_(programming_language)",
                "https://this-domain-does-not-exist-12345.com/",
            ]
        }
    )

    print(f"Extracted {len(result)} results")
    for item in result:
        if "error_type" in item:
            print(f"\n❌ Failed: {item['url']}")
            print(f"   Error: {item['content']}")
        else:
            print(f"\n✓ Success: {item['url']}")
            print(f"   Content length: {len(item['content'])} characters")


async def async_extract_examples() -> None:
    """Async extraction examples."""
    print("\n=== Async Extract Examples ===")

    tool = ParallelExtractTool()

    # Example 5: Async extraction
    print("\nExample 5: Async extraction")
    result = await tool.ainvoke(
        {
            "urls": [
                "https://en.wikipedia.org/wiki/JavaScript",
                "https://en.wikipedia.org/wiki/TypeScript",
            ]
        }
    )

    print(f"Extracted {len(result)} results asynchronously")
    for i, item in enumerate(result, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Content length: {len(item['content'])} characters")


def agent_integration_example() -> None:
    """Example of using extract tool with an agent."""
    print("\n=== Agent Integration Example ===")

    from langchain_parallel import ChatParallelWeb

    # Initialize tools
    extract_tool = ParallelExtractTool(max_chars_per_extract=3000)
    chat = ChatParallelWeb()

    # Extract content
    print("\nExtracting content from URLs...")
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    ]

    result = extract_tool.invoke({"urls": urls})

    # Prepare context for the chat model
    context = "\n\n".join(
        [
            f"Source: {doc['title']} ({doc['url']})\n{doc['content'][:1000]}..."
            for doc in result
        ]
    )

    # Generate summary using extracted content
    print("\nGenerating summary with extracted context...")
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes content."),
        HumanMessage(
            content=(
                f"Summarize the key differences between AGI and AI "
                f"based on this content:\n\n{context}"
            )
        ),
    ]

    response = chat.invoke(messages)
    print(f"\nSummary: {response.content}")


def main() -> None:
    """Main function demonstrating Parallel Extract Tool usage."""
    print("=== Parallel Extract Tool Examples ===")

    # Check if API key is set
    if not os.getenv("PARALLEL_API_KEY"):
        print("Error: PARALLEL_API_KEY environment variable not set")
        print("Please set your API key: export PARALLEL_API_KEY='your-api-key'")
        return

    print("API key found in environment")
    print("Starting extract tool examples...")

    # Run examples
    try:
        # Basic examples
        basic_extract_examples()

        # Batch extraction
        batch_extract_examples()

        # Focused extraction
        focused_extraction_examples()

        # Content length control
        content_length_control_examples()

        # Error handling
        error_handling_examples()

        # Async extraction
        asyncio.run(async_extract_examples())

        # Agent integration
        agent_integration_example()

        print("\n=== All examples completed successfully ===")
        print("\nKey features demonstrated:")
        print("  - Single URL extraction")
        print("  - Batch extraction from multiple URLs")
        print("  - Focused extraction with search objective")
        print("  - Focused extraction with search queries")
        print("  - Content length control")
        print("  - Error handling for failed extractions")
        print("  - Async extraction")
        print("  - Integration with chat models")

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("\nTroubleshooting tips:")
        print("  - Ensure your API key is valid")
        print("  - Check your internet connection")
        print("  - Verify the Parallel service is accessible")
        raise


if __name__ == "__main__":
    main()
