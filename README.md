# LangChain Parallel Web Integration

This package provides LangChain integrations for [Parallel](https://docs.parallel.ai/), covering Parallel's full developer-facing API surface.

## Features

| Surface | Class | Backed by |
|---|---|---|
| Chat completions with citations + structured output | [`ChatParallel`](#chat-models) | `/chat/completions` (lite/base/core) |
| Web search → Documents (RAG) | [`ParallelSearchRetriever`](#retriever-rag) | `/v1/search` |
| Web search tool (agents) | [`ParallelSearchTool`](#search-api) | `/v1/search` |
| Web content extraction | [`ParallelExtractTool`](#extract-api) | `/v1/extract` |
| Single Task Run + citations | [`ParallelTaskRunTool`](#task-api) | `/v1/tasks/runs` |
| Deep-research Runnable | [`ParallelDeepResearch`](#task-api) | `/v1/tasks/runs` |
| Bulk task batching | [`ParallelTaskGroup`](#task-api) | `/v1beta/tasks/groups` |
| Entity discovery | [`ParallelFindAllTool`](#findall-api) | `/v1beta/findall` |
| Scheduled web monitors | [`ParallelMonitor`](#monitor-api-alpha) | `/v1alpha/monitors` |
| Hosted MCP servers as LangChain tools | [`parallel_mcp_toolkit()`](#mcp-toolkit) | `search.parallel.ai` + `task-mcp.parallel.ai` |
| Webhook signature verification | [`verify_webhook()`](#webhook-signature-verification) | HMAC-SHA256 |

> Old names (`ChatParallelWeb`, `ParallelWebSearchTool`) continue to work as aliases for `ChatParallel` and `ParallelSearchTool`.

## Installation

```bash
pip install langchain-parallel
```

## Setup

1. Get your API key from [Parallel](https://parallel.ai/)
2. Set your API key as an environment variable:

```bash
export PARALLEL_API_KEY="your-api-key-here"
```

## Chat Models

### ChatParallelWeb

The `ChatParallelWeb` class provides access to Parallel's Chat API, which combines language models with real-time web research capabilities.

#### Picking a model

| Model | Latency | Citations (`response_metadata["basis"]`) | Structured output |
|-------|---------|------------------------------------------|-------------------|
| `speed` (default) | ~3s | none | not supported |
| `lite` | seconds | yes | `with_structured_output()` |
| `base` | seconds–minutes | yes | `with_structured_output()` |
| `core` | minutes | yes (most thorough) | `with_structured_output()` |

#### Basic Usage

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_parallel.chat_models import ChatParallelWeb

chat = ChatParallelWeb(model="speed")

messages = [
    SystemMessage(content="You are a helpful assistant with access to real-time web information."),
    HumanMessage(content="What are the latest developments in artificial intelligence?"),
]

response = chat.invoke(messages)
print(response.content)
# Citations on the research models (lite/base/core):
print(response.response_metadata.get("basis"))
```

#### Structured output (research models)

```python
from pydantic import BaseModel, Field
from langchain_parallel import ChatParallelWeb

class Founder(BaseModel):
    name: str = Field(description="Full name of the founder")
    company: str = Field(description="Company they founded")

structured = ChatParallelWeb(model="lite").with_structured_output(Founder)
result = structured.invoke([("human", "Who founded SpaceX?")])
print(result)
# Founder(name='Elon Musk', company='SpaceX')
```

#### Streaming Responses

```python
# Stream responses for real-time output
for chunk in chat.stream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

#### Async Operations

```python
import asyncio

async def main():
    # Async invoke
    response = await chat.ainvoke(messages)
    print(response.content)
    
    # Async streaming
    async for chunk in chat.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

#### Conversation Context

```python
# Maintain conversation history
messages = [
    SystemMessage(content="You are a helpful assistant.")
]

# First turn
messages.append(HumanMessage(content="What is machine learning?"))
response = chat.invoke(messages)
messages.append(response)  # Add assistant response

# Second turn with context
messages.append(HumanMessage(content="How does it work?"))
response = chat.invoke(messages)
print(response.content)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"speed"` | Parallel model name |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses `PARALLEL_API_KEY` env var if not provided) |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL |
| `temperature` | `Optional[float]` | `None` | Sampling temperature (ignored by Parallel) |
| `max_tokens` | `Optional[int]` | `None` | Max tokens (ignored by Parallel) |
| `timeout` | `Optional[float]` | `None` | Request timeout |
| `max_retries` | `int` | `2` | Max retry attempts |


## Real-Time Web Research

Parallel's Chat API provides real-time access to web information, making it perfect for:

- **Current Events**: Get up-to-date information about recent events
- **Market Data**: Access current stock prices, market trends
- **Research**: Find the latest research papers, developments
- **Weather**: Get current weather conditions
- **News**: Access breaking news and recent articles

```python
# Example: Current events
messages = [
    SystemMessage(content="You are a research assistant with access to real-time web data."),
    HumanMessage(content="What happened in the stock market today?")
]

response = chat.invoke(messages)
print(response.content)  # Gets real-time market information
```

## Integration with LangChain

### Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant with access to real-time web information."),
    ("human", "{question}")
])

chain = prompt | chat | StrOutputParser()

# Use the chain
result = chain.invoke({"question": "What are the latest AI breakthroughs?"})
print(result)
```

### Agents

Parallel's Chat API does not support tool calling, so `ChatParallelWeb` cannot be the LLM that drives an agent. Use it as a research assistant inside a chain (above), or use Parallel's tools (`ParallelWebSearchTool`, `ParallelExtractTool`) with a tool-calling chat model (Anthropic, OpenAI, etc.) — see the **Tool Usage in Agents** section below.

## Search API

The Search API provides direct access to Parallel's web search capabilities, returning structured, compressed excerpts optimized for LLM consumption.

### ParallelWebSearchTool

The search tool provides direct access to Parallel's Search API:

```python
from langchain_parallel import ParallelWebSearchTool

search_tool = ParallelWebSearchTool()

result = search_tool.invoke({
    "search_queries": ["renewable energy 2026", "solar power developments"],
    "max_results": 5,
})

print(result["search_id"], len(result["results"]))
for r in result["results"]:
    print(r["title"], "-", r["url"])
```





### Search API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | `Optional[str]` | `None` | Natural-language description of research goal (≤5000 chars). |
| `search_queries` | `Optional[List[str]]` | `None` | 1-5 keyword queries (3-6 words each, ≤200 chars). Required by the GA `/v1` endpoint; if omitted, the call routes to the deprecated `/v1beta` endpoint with a `DeprecationWarning` (slated for removal in 0.4.0). Pair with an optional `objective` for best results. |
| `max_results` | `int` | `10` | Maximum results to return (1–40). |
| `excerpts` | `Optional[ExcerptSettings]` | `None` | Per-result excerpt-size cap. |
| `max_chars_total` | `Optional[int]` | `None` | Cap on total excerpt characters across all results. |
| `mode` | `Optional[Literal["basic", "advanced"]]` | `None` (API default `advanced`) | `basic` is lower-latency; `advanced` is higher quality with more retrieval and compression. Legacy values `fast`, `one-shot` (→ `basic`) and `agentic` (→ `advanced`) are accepted with a `DeprecationWarning`. |
| `source_policy` | `Optional[SourcePolicy]` | `None` | Domain include/exclude lists and freshness floor (`after_date`). |
| `fetch_policy` | `Optional[FetchPolicy]` | `None` | Cache vs live-fetch policy (e.g. `FetchPolicy(max_age_seconds=86400, timeout_seconds=60)`). |
| `location` | `Optional[str]` | `None` | ISO 3166-1 alpha-2 country code (e.g. `"us"`, `"gb"`). |
| `client_model` | `Optional[str]` | `None` | Identifier of the calling LLM, used for model-specific result optimizations. |
| `session_id` | `Optional[str]` | `None` | Shared id grouping related Search/Extract calls in one task. |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses `PARALLEL_API_KEY` env var if not provided). |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL. |

### Search with Specific Queries

You can provide specific search queries instead of an objective:

```python
# Search with specific queries
result = search_tool.invoke({
    "search_queries": [
        "renewable energy 2024",
        "solar power developments",
        "wind energy statistics"
    ],
    "max_results": 8
})
```

### Tool Usage in Agents

Use the search tool with a tool-calling chat model (e.g. Anthropic Claude or OpenAI) and `create_agent`. Note that Parallel's own Chat API does not currently support tool calling, so use a different model class for the agent's LLM and use Parallel as a tool.

```python
from langchain.agents import create_agent
from langchain_parallel import ParallelWebSearchTool, ParallelExtractTool

agent = create_agent(
    "anthropic:claude-haiku-4-5",
    tools=[ParallelWebSearchTool(), ParallelExtractTool()],
    system_prompt=(
        "You are a research assistant. Use parallel_web_search to find "
        "current information and parallel_extract to read specific pages."
    ),
)

result = agent.invoke({"messages": [("human", "Latest AI breakthroughs?")]})
print(result["messages"][-1].content)
```

See `docs/demo_agent.ipynb` for a full walkthrough.

## Extract API

The Extract API provides clean content extraction from web pages, returning structured markdown-formatted content optimized for LLM consumption.

### ParallelExtractTool

The extract tool extracts clean, structured content from web pages:

```python
from langchain_parallel import ParallelExtractTool

# Initialize the extract tool
extract_tool = ParallelExtractTool()

# Extract from a single URL
result = extract_tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
})

print(result)
# [
#     {
#         "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
#         "title": "Artificial intelligence - Wikipedia",
#         "content": "# Artificial intelligence\n\nMain content in markdown...",
#         "publish_date": "2024-01-15"  # Optional
#     }
# ]
```

### Extract with Search Objective and Advanced Options

Focus extraction on specific topics using search objectives, with control over excerpts and fetch policy:

```python
# Extract content focused on a specific objective with excerpt settings
result = extract_tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    "search_objective": "What are the main applications and ethical concerns of AI?",
    "excerpts": {"max_chars_per_result": 2000},
    "full_content": False,
    "fetch_policy": {
        "max_age_seconds": 86400,
        "timeout_seconds": 60,
        "disable_cache_fallback": False
    }
})

# Returns relevant excerpts focused on the objective
print(result[0]["excerpts"])  # List of relevant text excerpts
```

### Extract with Search Queries

Extract content relevant to specific search queries:

```python
# Extract content focused on specific queries
result = extract_tool.invoke({
    "urls": [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning"
    ],
    "search_queries": ["neural networks", "training algorithms", "applications"],
    "excerpts": True
})

for item in result:
    print(f"Title: {item['title']}")
    print(f"Relevant excerpts: {len(item['excerpts'])}")
    print()
```

### Content Length Control

```python
# Control full content length per extraction
result = extract_tool.invoke({
    "urls": ["https://en.wikipedia.org/wiki/Quantum_computing"],
    "full_content": {"max_chars_per_result": 3000}
})

print(f"Content length: {len(result[0]['content'])} characters")
```

### Extract API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `urls` | `List[str]` | Required | List of URLs to extract content from (up to 20 per request). |
| `search_objective` | `Optional[str]` | `None` | Natural language objective to focus extraction (≤5000 chars). |
| `search_queries` | `Optional[List[str]]` | `None` | Specific keyword queries to focus extraction. |
| `excerpts` | `Union[bool, ExcerptSettings]` | `True` | In v1 GA, excerpts are always returned; the bool is kept for backward compatibility, and `ExcerptSettings(max_chars_per_result=…)` controls per-result size. |
| `full_content` | `Union[bool, FullContentSettings]` | `False` | Include full page content in addition to excerpts. |
| `max_chars_total` | `Optional[int]` | `None` | Cap on total excerpt characters across all results. Does not affect `full_content`. |
| `fetch_policy` | `Optional[FetchPolicy]` | `None` | Cache vs live content policy. |
| `client_model` | `Optional[str]` | `None` | Identifier of the calling LLM, used for model-specific result optimizations. |
| `session_id` | `Optional[str]` | `None` | Shared id grouping related Search/Extract calls in one task. |
| `max_chars_per_extract` | `Optional[int]` | `None` | Tool-level default cap on `full_content` size; only applied when `full_content=True`. |
| `api_key` | `Optional[SecretStr]` | `None` | API key (uses `PARALLEL_API_KEY` env var if not provided). |
| `base_url` | `str` | `"https://api.parallel.ai"` | API base URL. |

### Error Handling

The extract tool gracefully handles failed extractions:

```python
# Mix of valid and invalid URLs
result = extract_tool.invoke({
    "urls": [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://this-domain-does-not-exist-12345.com/"
    ]
})

for item in result:
    if "error_type" in item:
        print(f"Failed: {item['url']} - {item['content']}")
    else:
        print(f"Success: {item['url']} - {len(item['content'])} chars")
```

### Async Extract

```python
import asyncio

async def extract_async():
    result = await extract_tool.ainvoke({
        "urls": ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
    })
    return result

# Run async extraction
result = asyncio.run(extract_async())
```

## Retriever (RAG)

`ParallelSearchRetriever` is a `BaseRetriever` that returns Parallel Search results as `Document`s. Drops in to any LangChain RAG pipeline.

```python
from langchain_parallel import ParallelSearchRetriever, SourcePolicy

retriever = ParallelSearchRetriever(
    max_results=5,
    mode="advanced",
    source_policy=SourcePolicy(include_domains=["nature.com", "arxiv.org"]),
    objective="Focus on peer-reviewed material",  # forwarded on every call
)

docs = retriever.invoke("recent advances in protein folding")
for doc in docs:
    print(doc.metadata["title"], "-", doc.metadata["url"])
    print(doc.page_content[:200])
```

`Document.metadata` carries `url`, `title`, `publish_date`, `search_id`, the original `excerpts` list, and the `query` that produced the document.

## Task API

The Task API exposes Parallel's research processors (`lite`, `base`, `core`, `pro`, `ultra`) and the `basis` citation graph. Three surfaces:

- `ParallelTaskRunTool` — agent-callable tool for a single Task Run.
- `ParallelDeepResearch` — `Runnable` wrapper that defaults to `core` and is the lower-friction path for deep-research questions.
- `ParallelTaskGroup` — batch executor for fan-out/fan-in workloads.

### Single Task with citations

```python
from langchain_parallel import ParallelTaskRunTool

tool = ParallelTaskRunTool(processor="lite")
result = tool.invoke({"input": "Who founded SpaceX, in one sentence?"})
print(result["output"])
print(result["basis"])  # per-field citations + reasoning + confidence
```

### Deep research (Runnable)

```python
from langchain_parallel import ParallelDeepResearch

research = ParallelDeepResearch(processor="core")
result = research.invoke("Latest developments in renewable energy storage")
print(result["output"])
for fact in result.get("basis", []):
    print(fact["field"], "->", fact["citations"])
```

### Structured output (pydantic)

```python
from pydantic import BaseModel, Field
from langchain_parallel import ParallelTaskRunTool

class CompanyFacts(BaseModel):
    name: str
    founded: int = Field(description="Year the company was founded")
    headquarters: str

tool = ParallelTaskRunTool(
    processor="base",
    task_output_schema=CompanyFacts,
)
result = tool.invoke({"input": "Tell me about Anthropic"})
print(result["parsed"])  # CompanyFacts instance, fields populated
```

### Batch (Task Group)

```python
from langchain_parallel import ParallelTaskGroup

group = ParallelTaskGroup(processor="lite")
results = group.run([
    "Founder of Anthropic?",
    "Founder of OpenAI?",
    "Founder of Google DeepMind?",
])
for r in results:
    print(r["output"])
```

### BYOMCP (bring-your-own MCP servers)

```python
from langchain_parallel import McpServer, ParallelTaskRunTool

tool = ParallelTaskRunTool(
    processor="base",
    mcp_servers=[
        McpServer(
            name="my_internal_data",
            url="https://mcp.example.com/internal",
            headers={"Authorization": "Bearer ..."},
        ),
    ],
)
```

## FindAll API

Discover entities from the web that satisfy a natural-language objective plus boolean match conditions.

```python
from langchain_parallel import (
    ParallelFindAllTool,
    FindAllMatchCondition,
)

tool = ParallelFindAllTool(generator="base")
result = tool.invoke({
    "objective": "AI agent startups founded after 2023",
    "entity_type": "company",
    "match_conditions": [
        FindAllMatchCondition(
            name="founded_after_2023",
            description="Was this company founded after January 1 2023?",
        ),
        FindAllMatchCondition(
            name="builds_ai_agents",
            description="Does this company build AI agents as a core product?",
        ),
    ],
    "match_limit": 25,
})
for candidate in result["candidates"]:
    print(candidate["name"], "-", candidate["url"])
```

Generators: `preview` (small free sample), `base`, `core`, `pro` (highest quality, longest-running).

## Monitor API (alpha)

Schedule recurring web queries that emit webhook events on change. The Monitor API is **alpha**; shapes may change without notice. The current SDK doesn't expose this surface, so `ParallelMonitor` talks to `/v1alpha/monitors` directly.

```python
from langchain_parallel import ParallelMonitor, MonitorWebhook

monitors = ParallelMonitor()

m = monitors.create(
    query="Track new SEC filings related to Anthropic",
    frequency="1h",
    webhook=MonitorWebhook(
        url="https://example.com/parallel-webhook",
        secret="...",  # used to HMAC-sign payloads
    ),
)

events = monitors.list_events(m["monitor_id"])
print(len(events["event_groups"]))
```

## Webhook signature verification

Validates HMAC-SHA256 signatures on incoming Task Run / FindAll / Monitor webhooks.

```python
from langchain_parallel import verify_webhook

@app.post("/parallel-webhook")
async def webhook(request):
    body = await request.body()
    signature = request.headers["parallel-signature"]
    if not verify_webhook(body, signature, secret="..."):
        return Response(status_code=401)
    # ... process the event
```

## MCP toolkit

Wrap Parallel's hosted MCP servers (Search MCP + Task MCP) as LangChain `BaseTool`s. Useful when you want to mix Parallel tools with other MCP servers in the same agent runtime, or when you've standardized on MCP for cross-language reasons. For Python-only use cases, the native tools above are simpler and don't require the extra dependency.

```bash
pip install "langchain-parallel[mcp]"  # pulls in langchain-mcp-adapters
```

```python
from langchain_parallel import parallel_mcp_toolkit

tools = await parallel_mcp_toolkit()  # returns list[BaseTool]
# Includes: web_search, web_fetch (Search MCP);
#           createDeepResearch, createTaskGroup, getStatus, getResultMarkdown (Task MCP)
```

## Error Handling

```python
try:
    response = chat.invoke(messages)
    print(response.content)
except ValueError as e:
    if "API key not found" in str(e):
        print("Please set your PARALLEL_API_KEY environment variable")
    else:
        print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Examples

See the `examples/` and `docs/` directories for complete working examples:

- `examples/chat_example.py` - Chat model usage examples
- `docs/search_tool.ipynb` - Search tool examples and tutorials
- `docs/extract_tool.ipynb` - Extract tool examples and tutorials
- Basic synchronous usage
- Streaming responses
- Async operations
- Conversation management
- Tool usage in agents

## API Compatibility

This integration provides access to two Parallel APIs:

### Chat API Compatibility
The Chat API uses Parallel's OpenAI-compatible interface:

- **Supported**: Messages, streaming, response_format (JSON schema)
- **Ignored**: temperature, max_tokens, top_p, stop, most OpenAI-specific parameters
- **Not Supported**: Function calling, multimodal inputs (images/audio), tool usage

### Search API Features
The Search API provides direct web search capabilities:

- **Supported**: Objective-based search, query-based search, two processor tiers
- **Output**: Structured results with URLs, titles, and relevant excerpts
- **Integration**: Works with LangChain tools, retrievers, and agents

### Extract API Features
The Extract API provides clean content extraction from web pages:

- **Supported**: Batch URL extraction, content length control, markdown formatting
- **Output**: Clean, structured content with metadata (title, publish date, etc.)
- **Integration**: Works with LangChain tools and agents
- **Error Handling**: Gracefully handles failed extractions with detailed error info

## Performance & Rate Limits

### Chat API
- **Default Rate Limit**: 300 requests per minute
- **Performance**: 3 second p50 TTFT (time to first token) with streaming
- **Use Cases**: Interactive chat, real-time responses

### Search API
- **Default Rate Limit**: Contact Parallel for rate limit information
- **Performance**: Varies based on query complexity and result count
- **Use Cases**: Real-time web information, research, content discovery

### Production Usage
Contact [Parallel](https://parallel.ai/) for:
- Higher rate limits
- Enterprise features
- Custom configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

### Documentation
- [Parallel Documentation](https://docs.parallel.ai/)
- [Chat API Reference](https://docs.parallel.ai/chat-api)
- [Search API Reference](https://docs.parallel.ai/search/search-quickstart)
- [LangChain Documentation](https://python.langchain.com/)

### Getting Help
- [GitHub Issues](https://github.com/parallel-web/langchain-parallel/issues)
- [Parallel Support](mailto:support@parallel.ai)

## Changelog

See [`CHANGELOG.md`](./CHANGELOG.md) for the full version history.
