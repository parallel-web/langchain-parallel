# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-27

This release migrates Search and Extract to Parallel's v1 GA endpoints, surfaces citations + structured output on the chat model, and bumps the SDK to `0.5.1`.

### Added

- **Canonical naming**: new aliases `ChatParallel` and `ParallelSearchTool` are the recommended names going forward; the previous `ChatParallelWeb` and `ParallelWebSearchTool` continue to work indefinitely as aliases (same class objects).
- **Search/Extract GA endpoints**: `ParallelSearchTool` and `ParallelExtractTool` now call `client.search` / `client.extract` (the `/v1` GA paths). New parameters surfaced from the GA contract on both tools: `max_chars_total`, `client_model`, `session_id`, `location` (Search). The `advanced_settings` envelope is built automatically from the existing flat fields.
- **`ChatParallel.with_structured_output()`**: returns a `Runnable` producing a typed object (pydantic model or dict) via Parallel's `response_format` JSON-schema on the research models (`lite`, `base`, `core`). `method="json_schema"` (default), `method="json_mode"`, and `method="function_calling"` (routed to `json_schema` for cross-provider compatibility) are supported. Raises a clear `ValueError` on `model="speed"` since that model silently ignores structured-output requests. `include_raw=True` returns `{"raw", "parsed", "parsing_error"}` and properly captures parser failures.
- **Citations on chat responses**: for the research models, `AIMessage.response_metadata["basis"]` carries the API's per-field citations / reasoning / confidence list. `response_metadata["interaction_id"]` is surfaced for multi-turn context chaining; `system_fingerprint` is forwarded when present.
- **`SourcePolicy` pydantic model** in `langchain_parallel._types` mirroring the API's `include_domains` / `exclude_domains` / `after_date`. Both `SourcePolicy(...)` and a raw dict are accepted on `ParallelSearchTool`.

### Changed

- **BREAKING — `search_queries` is required for Search**. Previously the tool accepted `objective` alone (silently calling the deprecated `/v1beta` endpoint). With v1 GA, calls without `search_queries` raise `ValueError` with a migration hint pointing at https://docs.parallel.ai/search/search-migration-guide. Existing call sites that pair `objective` with `search_queries` are unaffected.
- **`mode` strings**: legacy values `"fast"`, `"one-shot"`, and `"agentic"` continue to call the API correctly with a `DeprecationWarning` mapping them to the GA values (`"fast"` / `"one-shot"` → `"basic"`, `"agentic"` → `"advanced"`). The GA values `"basic"` and `"advanced"` are now the canonical set.
- **Extract `excerpts: bool` is now a no-op**: the GA Extract API always returns excerpts, so passing `excerpts=True` (the default) is unchanged on the wire and `excerpts=False` is accepted with a `DeprecationWarning`. Use `ExcerptSettings(max_chars_per_result=…)` to control per-result size.
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

- **Search** (only required change for most users): add `search_queries=[…]` (1-5 keyword strings, 3-6 words each). Pair with the existing `objective=…` for best results.
  ```python
  # Before (0.2.x — silently used /v1beta)
  tool.invoke({"objective": "What are the latest AI breakthroughs?"})

  # After (0.3.x — uses /v1 GA)
  tool.invoke({
      "search_queries": ["latest AI breakthroughs", "AI advances 2026"],
      "objective": "What are the latest AI breakthroughs?",
  })
  ```
- **Search mode**: rename `mode="one-shot"`/`"fast"` → `mode="basic"` and `mode="agentic"` → `mode="advanced"` to silence the legacy-value deprecation warning.
- **Chat**: prefer `ChatParallel(model="lite")` (or `"base"` / `"core"`) over `model_name="..."`. Read citations off `response.response_metadata["basis"]` and structured outputs via `chat.with_structured_output(MyPydanticModel)`. The old class name `ChatParallelWeb` continues to work.

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
