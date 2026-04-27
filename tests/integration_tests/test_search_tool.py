from __future__ import annotations

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_parallel.search_tool import ParallelWebSearchTool


class TestParallelWebSearchToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[ParallelWebSearchTool]:
        return ParallelWebSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        # API key will be read from environment variable PARALLEL_API_KEY
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {
            "search_queries": [
                "latest AI developments",
                "AI breakthroughs 2026",
            ],
            "objective": "Latest developments in AI",
            "max_results": 3,
        }
        # Note: passing only `objective` (no search_queries) also works in
        # 0.3.x but routes to /v1beta with a DeprecationWarning. Prefer the
        # GA shape above; the fallback will be removed in 0.4.0.
