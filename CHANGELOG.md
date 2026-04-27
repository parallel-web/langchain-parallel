# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-27

### Added

- **Search/Extract GA endpoints**: `ParallelWebSearchTool` and `ParallelExtractTool` now call `client.search` and `client.extract` (the `/v1` GA paths) by default, replacing the deprecated `client.beta.*` calls. New parameters surfaced from the GA contract: `max_chars_total`, `client_model`, `session_id`, `location` on both tools.
- **Citations on chat responses**: when `model` is `lite`, `base`, or `core`, `AIMessage.response_metadata["basis"]` now carries the API's per-field citations / reasoning / confidence. `interaction_id` is also surfaced for context chaining across calls.
- **`ChatParallelWeb.with_structured_output()`**: returns a `Runnable` that produces a typed object (pydantic model or dict) using Parallel's `response_format` JSON-schema feature on the research models. `method="json_schema"` (default), `method="json_mode"`, and `method="function_calling"` (routed to `json_schema` for cross-provider compatibility) are supported. Raises a clear error when called on `model="speed"` since that model silently ignores structured-output requests.
- **`SourcePolicy` pydantic model** in `langchain_parallel._types` mirroring the API's `include_domains` / `exclude_domains` / `after_date` shape.
- **`tool.response_format = "content_and_artifact"`** on both Search and Extract tools — agents see a compact summary string while consumers reading from the `ToolMessage` get the full structured payload via `.artifact`.

### Changed

- **BREAKING — tool return shape**: `ParallelWebSearchTool` and `ParallelExtractTool` now return `(content_str, artifact)` per the LangChain `content_and_artifact` convention. Direct `tool.invoke({...})` returns just the content string; the tool-call form (`{"args": {...}, "id": ..., "name": ..., "type": "tool_call"}`) returns a `ToolMessage` whose `.artifact` carries the full Parallel response. To keep the old direct-dict access, use `_, artifact = tool._run(...)` or unpack the `ToolMessage`.
- **BREAKING — `mode` strings**: legacy values `"fast"`, `"one-shot"`, and `"agentic"` continue to work but emit a `DeprecationWarning` and are mapped to `basic` / `basic` / `advanced` respectively. The GA values `"basic"` and `"advanced"` are the new canonical set.
- **`ChatParallelWeb.model` alias removed (with back-compat shim)**: the `model_name` alias on the `model` field has been removed because it silently swallowed `ChatParallelWeb(model="lite")` and forced users into the default `"speed"`. Both `ChatParallelWeb(model="lite")` and `ChatParallelWeb(model_name="lite")` now work — the latter via a `model_validator` that maps `model_name=` to `model=`. `lc_attributes` still serializes as `model_name` for tracing parity.
- **Search behavior**: when `search_queries` is omitted, the tool falls back to the deprecated `/v1beta/search` endpoint with a `DeprecationWarning`. The GA endpoint requires `search_queries` (1–5 keyword strings); pass them explicitly to silence the warning.
- **`response_metadata["model_name"]`**: chat completions now emit `model_name` (the LangChain 1.x standard key) instead of `model`. Standard tests check for `model_name`.
- **`parallel-web` SDK bumped** from `^0.3.3` to `^0.5.1`. Brings in v1 GA Search/Extract types, `AdvancedSearchSettingsParam` / `AdvancedExtractSettingsParam`, and FindAll / Task Group surface (not yet exposed by this integration; see the IMPROVEMENT_PLAN.md roadmap for Phase 2).
- **Slimmed `_client.py`**: the four hand-rolled `ParallelSearchClient` / `AsyncParallelSearchClient` / `ParallelExtractClient` / `AsyncParallelExtractClient` wrappers have been removed. Tools now instantiate the `parallel.Parallel` / `parallel.AsyncParallel` SDK clients directly. Internal change; no public surface impact.
- `ParallelExtractTool.full_content` precedence is now explicit: a `FullContentSettings` (or dict) on the call always wins over the tool-level `max_chars_per_extract`; the latter only applies when `full_content=True` is passed as a plain bool.

### Fixed

- `ChatParallelWeb(model="lite")` now actually selects the `lite` model. Previously the `alias="model_name"` on the field meant the `model=` kwarg was silently ignored and the default `"speed"` was used.
- `py.typed` is now bundled into the wheel via the `[tool.poetry] include` directive, so downstream `mypy` runs see the package's type information.

### Migration

- **Tools**: existing code that does `result = tool.invoke({...})` and treats `result` as a dict/list should switch to either `_, result = tool._run(...)` or use the tool-call envelope:
  ```python
  msg = tool.invoke({"args": {...}, "id": "1", "name": tool.name, "type": "tool_call"})
  result = msg.artifact
  ```
- **Search**: callers using only `objective` (no `search_queries`) keep working but should add `search_queries=["...","..."]` to silence the deprecation warning and use the GA endpoint.
- **Search modes**: rename `mode="one-shot"` → `mode="basic"` (or `"advanced"` for higher quality), `mode="agentic"` → `mode="advanced"`, `mode="fast"` → `mode="basic"`.
- **Chat**: code that did `ChatParallelWeb(model_name="...")` continues to work via `model_name` mapping in `lc_attributes`. New code should prefer `ChatParallelWeb(model="lite")`. Read citations from `response.response_metadata["basis"]`.

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
