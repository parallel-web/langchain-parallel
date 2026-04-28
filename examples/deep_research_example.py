"""Deep-research example using ParallelDeepResearch."""

from __future__ import annotations

from pydantic import BaseModel

from langchain_parallel import ParallelDeepResearch

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def basic_deep_research() -> None:
    print("=== ParallelDeepResearch: untyped ===")
    # Default processor is "pro-fast" — the -fast variant of pro,
    # 2-5x faster than "pro" at similar accuracy.
    research = ParallelDeepResearch()
    result = research.invoke("In one sentence, what is the capital of France?")
    output = result["output"]
    print("output:", output["content"] if isinstance(output, dict) else output)


def typed_deep_research() -> None:
    print("\n=== ParallelDeepResearch: typed output ===")

    class CityFact(BaseModel):
        capital: str
        population_millions: float

    research = ParallelDeepResearch(output_schema=CityFact)
    result = research.invoke("Capital and population (in millions) of France?")
    print("output:", result["output"])


def main() -> None:
    basic_deep_research()
    typed_deep_research()


if __name__ == "__main__":
    main()
