"""Structured-batch enrichment example using ParallelEnrichment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_parallel import ParallelEnrichment

# Set your API key: export PARALLEL_API_KEY="your-api-key"


class CompanyInput(BaseModel):
    company: str = Field(description="Company name")


class CompanyOutput(BaseModel):
    headquarters: str
    founding_year: int


def enrich_company_facts() -> None:
    print("=== ParallelEnrichment: typed batch ===")
    # Default processor is "core-fast" — pass "core" or "pro" for higher accuracy.
    enricher = ParallelEnrichment(
        input_schema=CompanyInput,
        output_schema=CompanyOutput,
    )
    results = enricher.invoke(
        [
            CompanyInput(company="Anthropic"),
            {"company": "OpenAI"},
        ],
    )
    for r in results:
        print(r["output"]["content"])


def main() -> None:
    enrich_company_facts()


if __name__ == "__main__":
    main()
