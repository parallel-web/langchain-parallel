"""ParallelFindAllTool example: entity discovery with match conditions."""

from __future__ import annotations

import asyncio

from langchain_parallel import FindAllMatchCondition, ParallelFindAllTool

# Set your API key: export PARALLEL_API_KEY="your-api-key"


async def preview_discovery() -> None:
    print("=== ParallelFindAllTool: preview generator ===")
    tool = ParallelFindAllTool()
    result = await tool.ainvoke(
        {
            "objective": "Pure-play public LLM API providers",
            "entity_type": "company",
            "match_conditions": [
                FindAllMatchCondition(
                    name="public_us",
                    description="Company is publicly traded on a US exchange",
                ).model_dump(),
                FindAllMatchCondition(
                    name="llm_api_revenue",
                    description="Primary revenue is selling LLM inference via API",
                ).model_dump(),
            ],
            "generator": "preview",
            "match_limit": 5,
        },
    )
    for c in result.get("candidates", []):
        print("-", c.get("name"), "/", c.get("url"))


def main() -> None:
    asyncio.run(preview_discovery())


if __name__ == "__main__":
    main()
