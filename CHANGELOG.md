# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-28

This is a feature release covering Phase 2 of the modernization roadmap. Adds five new public surfaces (retriever, Task API, FindAll, Monitor, MCP toolkit) and removes the deprecation paths that 0.3.0 introduced.

### Added

- **`ParallelSearchRetriever`** — `BaseRetriever` returning `list[Document]` with rich `metadata` (`url`, `title`, `publish_date`, `search_id`, `excerpts`, `query`). Drops in to any RAG pipeline. Sync + async.
- **Task API** (`langchain_parallel.tasks`):
  - `ParallelTaskRunTool` — agent-callable tool that runs a single Parallel Task synchronously via `client.task_run.execute(...)`. Surfaces the structured output, `basis` citations, and run id. Supports the full processor menu (`lite`, `base`, `core`, `core2x`, `pro`, `ultra`, `ultra2x/4x/8x`).
  - `ParallelDeepResearch` — `Runnable[str|dict, dict]` wrapper defaulting to the `core` processor; lower-friction shape for "do deep research on this question."
  - `ParallelTaskGroup` — batch runner that creates a Task Group, fans out runs, and collects all results. Useful for bulk enrichment.
  - **BYOMCP support** — pass `mcp_servers=[McpServer(...)]` to `ParallelTaskRunTool` / `ParallelDeepResearch` to expose your own Streamable-HTTP MCP endpoints to the run.
  - **Webhook signature verification** — `verify_webhook(payload, signature, secret)` HMAC-SHA256 helper for incoming `task_run.status` webhooks.
- **`ParallelFindAllTool`** (`langchain_parallel.findall`) — entity discovery via `client.beta.findall.create` + `result`. Returns ranked candidates that satisfy a natural-language objective and a set of boolean match conditions. Generators: `preview`, `base`, `core`, `pro`. Sync + async.
- **`ParallelMonitor`** (`langchain_parallel.monitors`) — thin httpx wrapper around `/v1alpha/monitors`. Create / retrieve / update / delete monitors; list event groups; simulate events. The Parallel SDK (0.5.1) does not yet expose this surface, so this module talks to the API directly. The Monitor API is **alpha** and shapes may change without notice.
- **`parallel_mcp_toolkit()`** (`langchain_parallel.mcp`) — factory that returns Parallel's hosted Search MCP and Task MCP tools as LangChain `BaseTool`s, via the optional `langchain-mcp-adapters` dependency. Install with `pip install "langchain-parallel[mcp]"`. Useful when you want to mix Parallel tools with other MCP servers in the same agent runtime.

### Removed

- **`mode="one-shot"` / `"agentic"` / `"fast"`** — the legacy `mode` strings deprecated in 0.3.0 are removed. Use `mode="basic"` (formerly `one-shot`/`fast`) or `mode="advanced"` (formerly `agentic`).
- **Search `objective`-only call (no `search_queries`)** — the `/v1beta` fallback path deprecated in 0.3.0 is removed. `search_queries` is now a required field on `ParallelWebSearchInput`. Pass at least one keyword query alongside any `objective`.
- **`Extract.excerpts=False`** — the no-op DeprecationWarning path is removed. The field is now `Optional[ExcerptSettings]` (was `Union[bool, ExcerptSettings]`); pass `ExcerptSettings(max_chars_per_result=…)` to control per-result size, or omit the field for the API default.
- **`search_metadata["endpoint"]`** key — no longer emitted (the v1beta fallback that introduced it is gone).

### Changed

- `pyproject.toml`: added optional extra `[mcp]` pulling in `langchain-mcp-adapters` for the MCP toolkit.

### Migration from 0.3.x

```python
# 0.3.x (DeprecationWarning, still worked)        # 0.4.x
tool.invoke({"objective": "..."})                 # tool.invoke({"search_queries": ["..."], "objective": "..."})
tool.invoke({"mode": "one-shot", ...})            # tool.invoke({"mode": "basic", ...})
tool.invoke({"mode": "agentic", ...})             # tool.invoke({"mode": "advanced", ...})
extract_tool.invoke({..., "excerpts": False})     # extract_tool.invoke({...})  # excerpts always returned in v1
```

The new surfaces are all additive — existing 0.3.x code that was warning-free continues to work without changes.

## [0.3.0] - 2026-04-27

This release migrates Search and Extract to Parallel's v1 GA endpoints, surfaces citations + structured output on the chat model, and bumps the SDK to `0.5.1`.

### Added

- **Canonical naming**: new aliases `ChatParallel` and `ParallelSearchTool` are the recommended names going forward; the previous `ChatParallelWeb` and `ParallelWebSearchTool` continue to work indefinitely as aliases (same class objects).
- **Search/Extract GA endpoints**: `ParallelSearchTool` and `ParallelExtractTool` now call `client.search` / `client.extract` (the `/v1` GA paths). New parameters surfaced from the GA contract on both tools: `max_chars_total`, `client_model`, `session_id`, `location` (Search). The `advanced_settings` envelope is built automatically from the existing flat fields.
- **`ChatParallel.with_structured_output()`**: returns a `Runnable` producing a typed object (pydantic model or dict) via Parallel's `response_format` JSON-schema on the research models (`lite`, `base`, `core`). `method="json_schema"` (default), `method="json_mode"`, and `method="function_calling"` (routed to `json_schema` for cross-provider compatibility) are supported. Raises a clear `ValueError` on `model="speed"` since that model silently ignores structured-output requests. `include_raw=True` returns `{"raw", "parsed", "parsing_error"}` and properly captures parser failures.
- **Citations on chat responses**: for the research models, `AIMessage.response_metadata["basis"]` carries the API's per-field citations / reasoning / confidence list. `response_metadata["interaction_id"]` is surfaced for multi-turn context chaining; `system_fingerprint` is forwarded when present.
- **`SourcePolicy` pydantic model** in `langchain_parallel._types` mirroring the API's `include_domains` / `exclude_domains` / `after_date`. Both `SourcePolicy(...)` and a raw dict are accepted on `ParallelSearchTool`.

### Deprecated

- **Search without `search_queries`**: calls passing only `objective` route to the deprecated `/v1beta` endpoint with a `DeprecationWarning`. The fallback will be **removed in 0.4.0**; the Parallel API itself sunsets `/v1beta` no earlier than June 2026. Pair `objective` with `search_queries=[...]` (1-5 keyword strings, 3-6 words each) to use the GA `/v1` endpoint.
- **Legacy `mode` values**: `"fast"`, `"one-shot"`, and `"agentic"` continue to call the API correctly with a `DeprecationWarning` mapping them to the GA values (`"fast"` / `"one-shot"` → `"basic"`, `"agentic"` → `"advanced"`). The GA values `"basic"` and `"advanced"` are now the canonical set.
- **`Extract.excerpts=False`**: the GA Extract API always returns excerpts and has no flag to disable them; passing `False` is accepted with a `DeprecationWarning` and ignored. Use `ExcerptSettings(max_chars_per_result=…)` to control per-result size.

### Changed

- **`response_metadata["model_name"]`**: chat completions now emit the LangChain 1.x standard key `model_name` (was `model`). Tracing systems and `langchain-tests`' standard suite check for this name.
- **`parallel-web` SDK bumped** from `^0.3.3` to `^0.5.1`. Brings in the v1 GA Search/Extract types, `AdvancedSearchSettingsParam` / `AdvancedExtractSettingsParam`, and the FindAll / Task Group / Monitor surfaces (not yet exposed by this integration — see `IMPROVEMENT_PLAN.md` Phase 2).
- **Slimmed `_client.py`**: the four hand-rolled `ParallelSearchClient` / `AsyncParallelSearchClient` / `ParallelExtractClient` / `AsyncParallelExtractClient` wrapper classes have been removed in favor of using `parallel.Parallel` / `parallel.AsyncParallel` directly. Internal change; no public surface impact.
- **`_run`/`_arun` deduped**: extracted `_finalize_response`, `_start_text`, and `_completion_text` helpers on both tools so the sync and async bodies are now ~25 lines each instead of ~50.
- `ParallelExtractTool.full_content` precedence is now explicit: an explicit `FullContentSettings` (or dict) on the call always wins over the tool-level `max_chars_per_extract`; the latter only applies when `full_content=True` is passed as a plain bool.

### Fixed

- `ChatParallel(model="lite")` now actually selects the `lite` model. Pre-0.3.0 the `Field(alias="model_name")` on the `model` field silently swallowed the `model=` kwarg and forced callers into the default `"speed"`. Both `ChatParallel(model="lite")` and `ChatParallel(model_name="lite")` work in 0.3.0 — the latter via a `model_validator` that maps `model_name=` to `model=` for back-compat. `lc_attributes` still serializes the field as `model_name` for tracing parity.
- `py.typed` is now bundled into the wheel via the `[tool.poetry] include` directive, so downstream `mypy` runs see the package's type information.
- `with_structured_output(include_raw=True)` correctly populates `parsing_error` on parse failure (previously always `None`).

### Migration

For most users, **no code changes are required**. The recommended-but-optional updates to silence deprecation warnings:

- **Search**: add `search_queries=[…]` (1-5 keyword strings, 3-6 words each) to use the GA `/v1` endpoint.
  ```python
  # 0.2.x (still works in 0.3.x with a DeprecationWarning; will break in 0.4.0)
  tool.invoke({"objective": "What are the latest AI breakthroughs?"})

  # 0.3.x preferred (GA /v1 endpoint)
  tool.invoke({
      "search_queries": ["latest AI breakthroughs", "AI advances 2026"],
      "objective": "What are the latest AI breakthroughs?",
  })
  ```
- **Search mode**: rename `mode="one-shot"`/`"fast"` → `mode="basic"` and `mode="agentic"` → `mode="advanced"`.
- **Chat**: prefer `ChatParallel(model="lite")` (or `"base"` / `"core"`) over `model_name="..."`. Read citations from `response.response_metadata["basis"]` and structured outputs via `chat.with_structured_output(MyPydanticModel)`. The old class name `ChatParallelWeb` continues to work.

## [0.2.0] - 2025-12-01

### Changed

- **BREAKING**: Minimum Python version raised from 3.9 to 3.10 (aligns with langchain-core 1.x requirements)
- **BREAKING**: Updated `langchain-core` dependency from `^0.3.76` to `>=1.1.0,<2.0.0` for LangChain 1.0 compatibility
- Updated `langchain-tests` from `^0.3.8` to `^1.0.0`
- Updated `httpx` from `^0.27.0` to `>=0.28.1,<1.0.0` (required by langchain-tests 1.x)

### Notes

- This release aligns with the official LangChain 1.0 ecosystem
- Dependency constraints now match official integrations like `langchain-openai` and `langchain-anthropic`
- No code changes required - all langchain-core APIs used remain backward compatible

## [0.1.0] - Initial Release

### Added

- `ChatParallelWeb` - Chat model integration for Parallel's Chat API
- `ParallelWebSearchTool` - Search tool with web research capabilities
- `ParallelExtractTool` - Content extraction from web pages
