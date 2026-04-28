"""Single-task example using ParallelTaskRunTool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_parallel import ParallelTaskRunTool

# Set your API key: export PARALLEL_API_KEY="your-api-key"


def untyped_run() -> None:
    print("=== ParallelTaskRunTool: untyped run ===")
    tool = ParallelTaskRunTool()  # default processor="lite-fast"
    result = tool.invoke({"input": "Who founded SpaceX, in one sentence?"})
    output = result["output"]
    print("output:", output["content"] if isinstance(output, dict) else output)
    print("run_id:", result["run"]["run_id"])


def typed_run_with_citations() -> None:
    print("\n=== ParallelTaskRunTool: typed run with citations ===")

    class FounderFact(BaseModel):
        founder: str = Field(description="Full name of the founder")
        year: int = Field(description="Year the company was founded")

    tool = ParallelTaskRunTool(task_output_schema=FounderFact)
    result = tool.invoke({"input": "Who founded SpaceX and in what year?"})
    print("output:", result["output"])
    for fact in result.get("basis", []):
        print(f"- {fact.get('field')}: confidence={fact.get('confidence')}")


def main() -> None:
    untyped_run()
    typed_run_with_citations()


if __name__ == "__main__":
    main()
