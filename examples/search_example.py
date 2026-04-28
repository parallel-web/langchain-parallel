"""Examples of Parallel Search integration."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_parallel import ParallelSearchTool, SourcePolicy

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def basic_search_examples() -> None:
    """Basic search tool examples."""
    print("=== Basic Search Examples ===")

    search_tool = ParallelSearchTool()

    # Example 1: Objective + search_queries (the recommended GA shape).
    print("\nExample 1: Objective + search_queries")
    result = search_tool.invoke(
        {
            "search_queries": [
                "latest AI developments 2026",
                "AI research breakthroughs",
            ],
            "objective": "What are the latest developments in artificial intelligence?",
            "max_results": 5,
        },
    )
    print(f"Found {len(result.get('results', []))} results")
    display_results(result, max_results=2)
    display_metadata(result)

    # Example 2: Multiple search queries (no objective).
    print("\nExample 2: Multiple search queries")
    result2 = search_tool.invoke(
        {
            "search_queries": [
                "AI developments 2026",
                "latest artificial intelligence news",
                "machine learning breakthroughs",
            ],
            "max_results": 8,
            "include_metadata": True,
        },
    )
    print(f"Found {len(result2.get('results', []))} results")
    display_results(result2, max_results=3)
    display_metadata(result2)


def search_examples() -> None:
    """Search features examples."""
    print("\n=== Search Examples ===")

    search_tool = ParallelSearchTool()

    # Example 3: Academic search with domain filtering and fetch policy.
    print("\nExample 3: Academic search with domain filtering and fetch policy")
    result3 = search_tool.invoke(
        {
            "search_queries": [
                "climate change research findings",
                "global warming peer reviewed studies",
            ],
            "objective": "Latest climate change research and findings",
            "source_policy": SourcePolicy(
                include_domains=["nature.com", "science.org", "arxiv.org"],
                exclude_domains=["reddit.com", "twitter.com", "facebook.com"],
            ),
            "max_results": 5,
            "excerpts": {"max_chars_per_result": 2000},
            "mode": "advanced",  # Higher quality with more retrieval and compression.
            "fetch_policy": {
                "max_age_seconds": 86400,  # Cache content for 1 day.
                "timeout_seconds": 60,
            },
            "include_metadata": True,
        },
    )
    print("Academic sources search completed")
    display_results(result3, max_results=2, show_excerpts=True)
    display_metadata(result3)

    # Example 4: Multiple-topic news search with the basic (low-latency) mode.
    print("\nExample 4: Multiple topic news search (basic mode)")
    result4 = search_tool.invoke(
        {
            "search_queries": [
                "tech industry layoffs 2026",
                "startup funding trends",
                "AI company acquisitions",
            ],
            "max_results": 6,
            "mode": "basic",  # Low-latency mode; pair with 2-3 high-quality queries.
            "include_metadata": True,
        },
    )
    print("Multiple query search completed")
    display_results(result4, max_results=3)
    display_metadata(result4)


async def async_search_examples() -> None:
    """Async search examples."""
    print("\n=== Async Search Examples ===")

    search_tool = ParallelSearchTool()

    # Example 5: Async search.
    print("\nExample 5: Async search execution")
    result5 = await search_tool.ainvoke(
        {
            "search_queries": ["quantum computing breakthroughs"],
            "objective": "Latest developments in quantum computing",
            "max_results": 4,
            "include_metadata": True,
        },
    )
    print("Async search completed")
    display_results(result5, max_results=2)
    display_metadata(result5)

    # Example 6: Parallel async searches.
    print("\nExample 6: Parallel async searches")
    tasks = [
        search_tool.ainvoke(
            {
                "search_queries": ["artificial intelligence news"],
                "max_results": 3,
            },
        ),
        search_tool.ainvoke(
            {
                "search_queries": ["machine learning research"],
                "max_results": 3,
            },
        ),
        search_tool.ainvoke(
            {
                "search_queries": ["robotics developments"],
                "max_results": 3,
            },
        ),
    ]
    results = await asyncio.gather(*tasks)

    for i, result in enumerate(results, 1):
        print(f"\nParallel search {i} results: {len(result.get('results', []))} found")
        display_results(result, max_results=1)


def display_results(
    result: dict[str, Any],
    *,
    max_results: int = 5,
    show_excerpts: bool = False,
) -> None:
    """Display search results in a formatted way."""
    if "results" not in result:
        print("No results found in response")
        print(f"Response keys: {list(result.keys())}")
        return

    for i, res in enumerate(result["results"][:max_results], 1):
        print(f"\nResult {i}:")
        print(f"  URL: {res.get('url', 'N/A')}")
        print(f"  Title: {res.get('title', 'N/A')}")
        excerpts = res.get("excerpts", [])
        if excerpts:
            print(f"  Excerpts: {len(excerpts)} found")
            if show_excerpts:
                for j, excerpt in enumerate(excerpts[:2], 1):
                    print(f"    {j}. {excerpt[:200]}...")
            else:
                print(f"    First: {excerpts[0][:100]}...")


def display_metadata(result: dict[str, Any]) -> None:
    """Display search metadata if available."""
    if "search_metadata" not in result:
        return
    metadata = result["search_metadata"]
    print("\n  Search Metadata:")
    print(f"    Endpoint: {metadata.get('endpoint', 'N/A')}")
    print(f"    Duration: {metadata.get('search_duration_seconds', 'N/A')}s")
    print(f"    Results: {metadata.get('actual_results_returned', 'N/A')}")


def practical_use_cases() -> None:
    """Practical use case examples."""
    print("\n=== Practical Use Cases ===")

    search_tool = ParallelSearchTool()

    # Use case 1: Research assistance.
    print("\nUse Case 1: Research Assistant")
    research_result = search_tool.invoke(
        {
            "search_queries": [
                "renewable energy adoption 2026",
                "solar wind energy growth",
            ],
            "objective": "Analysis of renewable energy adoption trends",
            "source_policy": SourcePolicy(
                include_domains=["iea.org", "irena.org", "energy.gov", "nature.com"],
                exclude_domains=["blog.com", "personal-site.com"],
            ),
            "max_results": 10,
            "excerpts": {"max_chars_per_result": 2500},
            "include_metadata": True,
        },
    )
    print("Research completed - energy analysis")
    print(f"Found {len(research_result.get('results', []))} authoritative sources")
    display_metadata(research_result)

    # Use case 2: News monitoring.
    print("\nUse Case 2: News Monitoring Dashboard")
    news_result = search_tool.invoke(
        {
            "search_queries": [
                "tech industry news today",
                "AI company funding",
                "cybersecurity breaches 2026",
                "cloud computing trends",
            ],
            "max_results": 15,
            "include_metadata": True,
        },
    )
    print("News monitoring completed")
    print(f"Found {len(news_result.get('results', []))} relevant news items")
    display_metadata(news_result)

    # Use case 3: Competitive analysis.
    print("\nUse Case 3: Competitive Analysis")
    competitor_result = search_tool.invoke(
        {
            "search_queries": [
                "tech company product launches",
                "big tech strategic moves",
            ],
            "objective": (
                "Latest product launches and strategic moves by major tech companies"
            ),
            "source_policy": SourcePolicy(
                include_domains=[
                    "techcrunch.com",
                    "theverge.com",
                    "wired.com",
                    "ars-technica.com",
                ],
                exclude_domains=["reddit.com", "twitter.com"],
            ),
            "max_results": 12,
            "include_metadata": True,
        },
    )
    print("Competitive analysis completed")
    display_results(competitor_result, max_results=2)
    display_metadata(competitor_result)


async def main() -> None:
    """Main function demonstrating Parallel Search Tool usage."""
    print("=== Parallel Search Examples ===")

    if not os.getenv("PARALLEL_API_KEY"):
        print("Error: PARALLEL_API_KEY environment variable not set")
        print("Please set your API key: export PARALLEL_API_KEY='your-api-key'")
        return

    print("API key found in environment")
    print("Starting search examples...")

    try:
        basic_search_examples()
        search_examples()
        await async_search_examples()
        practical_use_cases()

        print("\n=== All examples completed successfully ===")
        print("\nKey features demonstrated:")
        print("  - search_queries + objective (GA /v1 endpoint)")
        print("  - Multi-query search")
        print("  - Domain filtering with SourcePolicy")
        print("  - FetchPolicy for cache control")
        print("  - Search modes: basic (low-latency) and advanced (high-quality)")
        print("  - Async + parallel execution")
        print("  - Metadata collection")

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("\nTroubleshooting tips:")
        print("  - Ensure your API key is valid")
        print("  - Check your internet connection")
        print("  - Verify the Parallel service is accessible")
        raise


def run_sync_examples() -> None:
    """Run only synchronous examples for testing."""
    print("=== Running Synchronous Examples Only ===")

    if not os.getenv("PARALLEL_API_KEY"):
        print("Error: PARALLEL_API_KEY environment variable not set")
        return

    try:
        basic_search_examples()
        search_examples()
        practical_use_cases()
        print("\n=== Sync examples completed successfully ===")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
