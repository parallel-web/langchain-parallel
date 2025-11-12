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
            "objective": "What are the latest developments in AI?",
            "max_results": 3,
        }
