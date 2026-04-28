"""ParallelSearchRetriever example: drop-in BaseRetriever for RAG."""

from __future__ import annotations

import asyncio

from langchain_parallel import ParallelSearchRetriever

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def basic_retrieval() -> None:
    print("=== ParallelSearchRetriever: basic ===")
    retriever = ParallelSearchRetriever(
        max_results=3,
        excerpts={"max_chars_per_result": 800},
    )
    docs = retriever.invoke("breakthroughs in fusion energy 2025")
    for d in docs:
        print("-", d.metadata.get("title"), "/", d.metadata.get("url"))


def configured_retrieval() -> None:
    print("\n=== ParallelSearchRetriever: domain-filtered ===")
    retriever = ParallelSearchRetriever(
        max_results=5,
        excerpts={"max_chars_per_result": 1500},
        mode="basic",
        source_policy={
            "include_domains": ["nature.com", "science.org", "iter.org"],
        },
    )
    docs = retriever.invoke(
        "What's the latest peer-reviewed result on net-energy-gain fusion?",
    )
    print(f"got {len(docs)} docs")
    for d in docs[:3]:
        print("-", d.metadata.get("url"))


async def async_retrieval() -> None:
    print("\n=== ParallelSearchRetriever: async ===")
    retriever = ParallelSearchRetriever(max_results=3)
    docs = await retriever.ainvoke("Latest GLP-1 trial results 2025")
    print(f"got {len(docs)} docs")


def main() -> None:
    basic_retrieval()
    configured_retrieval()
    asyncio.run(async_retrieval())


if __name__ == "__main__":
    main()
