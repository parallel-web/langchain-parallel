from importlib import metadata

from langchain_parallel._types import (
    ExcerptSettings,
    FetchPolicy,
    FullContentSettings,
    SourcePolicy,
)
from langchain_parallel.chat_models import ChatParallel, ChatParallelWeb
from langchain_parallel.extract_tool import ParallelExtractTool
from langchain_parallel.findall import (
    FindAllExcludeEntry,
    FindAllMatchCondition,
    ParallelFindAllTool,
)
from langchain_parallel.mcp import parallel_mcp_toolkit
from langchain_parallel.monitors import MonitorWebhook, ParallelMonitor
from langchain_parallel.retrievers import ParallelSearchRetriever
from langchain_parallel.search_tool import ParallelSearchTool, ParallelWebSearchTool
from langchain_parallel.tasks import (
    McpServer,
    ParallelDeepResearch,
    ParallelTaskGroup,
    ParallelTaskRunTool,
    verify_webhook,
)

try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatParallel",
    "ChatParallelWeb",
    "ExcerptSettings",
    "FetchPolicy",
    "FindAllExcludeEntry",
    "FindAllMatchCondition",
    "FullContentSettings",
    "McpServer",
    "MonitorWebhook",
    "ParallelDeepResearch",
    "ParallelExtractTool",
    "ParallelFindAllTool",
    "ParallelMonitor",
    "ParallelSearchRetriever",
    "ParallelSearchTool",
    "ParallelTaskGroup",
    "ParallelTaskRunTool",
    "ParallelWebSearchTool",
    "SourcePolicy",
    "__version__",
    "parallel_mcp_toolkit",
    "verify_webhook",
]
