# langchain-parallel — Improvement Plan

_Drafted 2026-04-27. Targets the next minor (`0.3.0`) and a follow-on `0.4.0`._

## TL;DR

`langchain-parallel` 0.2.0 ships a solid baseline (Chat, Search, Extract) but is now behind on three fronts:

1. **Parallel's API has moved on.** Search and Extract went GA at `/v1/...` (April 2026) with a new `mode`/`advanced_settings` shape that supersedes `processor`/flat params. The SDK we depend on (`parallel-web ^0.3.3`) is two minor versions behind the current `0.5.1`. Three entirely new product surfaces — **Task Run / Task Group**, **FindAll**, **Monitor** — and Parallel's hosted **Search MCP** + **Task MCP** servers, plus **BYOMCP** inside Tasks, are not exposed at all.
2. **LangChain 1.x conventions have hardened.** Standard tests now check `response_metadata.model_name` (we emit `model`), expect `bind_tools` / `with_structured_output` for OpenAI-compatible chat backends, and we don't ship `py.typed` in the wheel. We're not registered with `init_chat_model("parallel:...")`.
3. **High-leverage idioms are missing.** No `BaseRetriever` returning `Document`s for RAG (`langchain-exa` ships one — it's ~30 lines and unlocks the standard retriever surface). Tools return raw dicts instead of using `response_format="content_and_artifact"` to deliver an LLM-friendly string + the raw JSON artifact. Citations / `FieldBasis` (Parallel's flagship feature for grounded outputs) are dropped on the floor.

This plan is split into three phases. **Phase 1** is a 0.3.0 release that's mostly additive plus a few targeted fixes. **Phase 2** is the bigger 0.4.0 expansion (Task API, FindAll, Monitor, retrievers, init_chat_model). **Phase 3** is polish (docs, examples, CI, ecosystem).

---

## Current state (1-page audit)

**What works:**
- Three primary classes — `ChatParallelWeb`, `ParallelWebSearchTool`, `ParallelExtractTool` — with full sync + async, streaming for chat, and pydantic input schemas.
- Sync + async client wrappers around `parallel-web` SDK and `openai` SDK.
- Standard-test scaffolding (`langchain-tests ^1.0.0`) with capability flags wired up.
- `lc_secrets` / `lc_namespace` / `is_lc_serializable` set on `ChatParallelWeb`.
- Reasonable tests, unit + integration; CI matrix on Python 3.10–3.13.

**Concrete gaps and bugs:**

| # | Location | Issue |
|---|----------|-------|
| 1 | `chat_models.py:55` | `_create_response_metadata` emits key `"model"`. LangChain 1.x standard expects `"model_name"`. |
| 2 | `chat_models.py:128–273` | Exposes `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `logit_bias`, `user`, `tool_choice`, `stream_options`, `response_format`, `tools` — all silently ignored. Misleading; also makes standard tests over-claim. |
| 3 | `chat_models.py:128` | No `bind_tools` / `with_structured_output`. Parallel Chat now supports `response_format` JSON schema for `lite/base/core` — should be exposed. |
| 4 | `chat_models.py` | `model` defaults to `"speed"` only. Parallel ships `lite`, `base`, `core` with citations on `response.basis` — not surfaced. Citations lost. |
| 5 | `_client.py` | Uses `client.beta.search` / `client.beta.extract`. SDK 0.5.x exposes stable `client.search` / `client.extract` with the v1 GA contract. We're on a deprecated path. |
| 6 | `_client.py:48–158` | Hand-rolled wrappers re-implement what the SDK already does. Just call the SDK directly from the tool. Three fewer classes to maintain. |
| 7 | `search_tool.py:19–75` | `ParallelWebSearchInput` does not surface the GA params: `max_chars_total`, `client_model`, `session_id`, `location`, `after_date` (in `source_policy`). `mode` strings aren't validated against `Literal["basic","advanced"]`. |
| 8 | `extract_tool.py:18–68` | Similar gaps — no `location`, no GA `advanced_settings` envelope, no support for the new `usage[]` array in the response. |
| 9 | `*Tool` | `response_format = "content"` (default). For search/extract this loses the structure: agents see a JSON-stringified dict. Should be `"content_and_artifact"` returning `(formatted_str, raw_dict)`. |
| 10 | `*Tool` | No `BaseRetriever` exists. `langchain-exa` ships `ExaSearchRetriever`; we should ship `ParallelSearchRetriever` returning `list[Document]` with rich `metadata`. |
| 11 | `pyproject.toml` | `parallel-web = "^0.3.3"` — pins us to a deprecated minor. Bump to `^0.5.1`. Also missing `include = ["langchain_parallel/py.typed"]` in `[tool.poetry]`, so type info doesn't reach downstream installs. |
| 12 | `__init__.py` | No public exports for the new surfaces (Tasks, FindAll, Monitor) — those don't exist yet. |
| 13 | repo | Not registered in `langchain.chat_models.base._BUILTIN_PROVIDERS` — `init_chat_model("parallel:speed")` raises `ValueError`. |
| 14 | repo | No `embeddings` / `retrievers` / `tasks` modules. No `ParallelTaskRun` / `ParallelTaskGroup` / `ParallelFindAll` / `ParallelMonitor`. |
| 15 | docs | The `docs/*.ipynb` and `examples/*.py` predate `create_agent` (LangChain 1.x); only `demo_agent.ipynb` is current. README still references `create_openai_functions_agent`. |
| 16 | tests | No tests for the new surfaces (because they don't exist). Standard tests are passing only because every advanced capability is set to `False`. |

---

## Phase 1 — `0.3.0` (mostly additive, a few targeted fixes)

Goal: remove the bit-rot, light up Parallel's GA APIs, and fix the LangChain 1.x conformance issues. No new product surfaces yet — just make what we have correct and current.

### 1.1 Bump to `parallel-web ^0.5.1` and switch to GA endpoints

- `pyproject.toml`: `parallel-web = "^0.5.1"`.
- Replace `client.beta.search(...)` with `client.search(...)` (and async equivalents). Same for `extract`.
- Map our params onto the GA shape:
  - Search: `objective` + `search_queries` stay; `max_results`, `excerpts`, `source_policy`, `fetch_policy`, `mode` move under `advanced_settings`. Add `max_chars_total`, `client_model`, `session_id`.
  - Extract: same envelope with `advanced_settings`. Surface `usage[]`.
- Pass the SDK's typed param objects (`AdvancedSearchSettings`, `AdvancedExtractSettings`, `SourcePolicy`, `FetchPolicy`, `ExcerptSettings`) through directly when the user provides them. Accept dicts too (auto-convert).
- Map `processor` → `mode` for back-compat: if a caller passes `processor="pro"`, translate to `mode="advanced"` and emit a `DeprecationWarning`.

### 1.2 Drop the hand-rolled `_client.py` wrappers

- Keep only `get_api_key`, `get_openai_client`, `get_async_openai_client`. Delete `ParallelSearchClient` / `AsyncParallelSearchClient` / `ParallelExtractClient` / `AsyncParallelExtractClient`. Tools call `parallel.Parallel(...)` and `parallel.AsyncParallel(...)` directly. ~150 fewer lines, one fewer indirection layer, the SDK's typed errors propagate cleanly.

### 1.3 Chat model fixes

- `chat_models.py`: change response metadata key `"model"` → `"model_name"` (matches `langchain-openai`/`-anthropic` and `langchain-tests`).
- Drop the dead-weight ignored params (`top_p`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `seed`, `user`, `tools`, `tool_choice`, `stream_options`, `response_format`). Keep only what's actually meaningful: `model`, `api_key`, `base_url`, `timeout`, `max_retries`.
  - For `temperature` / `max_tokens` / `stop`: keep but add a one-line `@model_validator` warning if set, with `stacklevel=2`. They're in the OpenAI surface so users will reach for them.
- Implement `bind_tools` and `with_structured_output` for the models that support them (Parallel's `lite/base/core` accept `response_format` with JSON schema). Gate on `model`: raise a clear error on `speed` ("structured output / tool calling requires the `lite`, `base`, or `core` models"). Standard tests' `has_structured_output` flag becomes `True` for those models.
- Surface citations: when the response carries `basis`/`citations`, attach to `AIMessage.response_metadata["basis"]` and to `AIMessageChunk` on the final streaming chunk. Document via the standard "Token usage" section in the docstring, but call it "Research basis" instead.
- Add a class-level docstring section for the model menu and what each one is for (latency, citations, JSON output) — pulled from `https://docs.parallel.ai/chat-api/chat-quickstart`.

### 1.4 Tool fixes

- Both tools: switch to `response_format = "content_and_artifact"`. Return a tuple of `(human_readable_str, raw_response_dict)`. Agents see compact, copy-paste-grade text; downstream code still has the full structured data via `ToolMessage.artifact`.
- `ParallelWebSearchTool`: add `location`, `max_chars_total`, `session_id`, `client_model`. Validate `mode` as `Literal["basic","advanced"]` (with the `processor` shim above).
- `ParallelExtractTool`: same advanced-settings overhaul. Stop the silent override where `max_chars_per_extract` clobbers a user-passed `full_content` boolean (`extract_tool.py:189`) — make precedence explicit ("if `full_content` is a settings object, that wins").
- Wire `RunnableConfig` through and respect `config.callbacks` cleanly (the `run_manager.on_text` color-coded notifications are nice but assume a console — make them quieter and only emit them if a callback exists).

### 1.5 Standard-tests / packaging hygiene

- Add `include = ["langchain_parallel/py.typed"]` to `pyproject.toml` `[tool.poetry]`. Verify `parallel.types.beta.*` types don't poison downstream `mypy` runs (the `[[tool.mypy.overrides]]` block already handles this internally; we want it to flow through).
- Override `standard_chat_model_params` in unit/integration tests to omit params Parallel ignores.
- Flip `has_structured_output = True` for `lite`/`base`/`core` model variants and add a parametrized integration test class.
- Tools tests: add `tool_invoke_params_example` for `ParallelExtractTool` (currently only Search has one).

### 1.6 Deliverables for 0.3.0

```
A langchain_parallel/_client.py            (slimmed)
M langchain_parallel/chat_models.py        (model_name fix; drop ignored params; bind_tools; with_structured_output; basis surfacing)
M langchain_parallel/search_tool.py        (GA params; content_and_artifact; processor->mode shim)
M langchain_parallel/extract_tool.py       (GA params; content_and_artifact; precedence fix)
M langchain_parallel/_types.py             (re-export SDK types as the canonical surface; keep our own as backward-compat)
M pyproject.toml                           (parallel-web ^0.5.1; py.typed include)
M README.md                                (model menu; create_agent; cite docs)
M CHANGELOG.md
+ tests for new behavior
```

Breaking changes minimal — `processor` still works via deprecation shim; tool return types change shape (was `dict`, now `(str, dict)`), so this is a minor-version bump. Document in `CHANGELOG.md` with migration snippets.

---

## Phase 2 — `0.4.0` (new product surfaces)

Goal: cover the rest of the Parallel API and ship the LangChain idioms users will actually reach for.

### 2.1 `ParallelSearchRetriever` — `BaseRetriever` returning `Document`s

```
langchain_parallel/retrievers.py
  class ParallelSearchRetriever(BaseRetriever):
      api_key: SecretStr | None
      base_url: str = "https://api.parallel.ai"
      mode: Literal["basic", "advanced"] = "advanced"
      max_results: int = 10
      max_chars_per_result: int | None = None
      source_policy: SourcePolicy | None = None
      fetch_policy: FetchPolicy | None = None
      location: str | None = None

      def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]: ...
      async def _aget_relevant_documents(self, query: str, *, run_manager) -> list[Document]: ...
```

`Document.page_content` = excerpts joined; `metadata` = `{"url", "title", "publish_date", "search_id", "score", ...}`. ~50 lines including tests. This is the single highest-leverage addition for RAG users.

### 2.2 Task API: `ParallelTaskRun` + `ParallelTaskRunner`

Two complementary surfaces:

- `ParallelTaskRunTool(BaseTool)` — for letting a chat agent kick off a deep-research task and pull the result. Args: `input`, `processor` (Literal of all processors), `task_spec` (input/output schemas), `mcp_servers`, `source_policy`, `metadata`. Synchronous `_run` blocks via `client.task_run.execute(...)`; streaming variant via `_run_with_events` (SSE) emits progress through `run_manager.on_text`.
- `ParallelDeepResearch` — a `Runnable[str | dict, dict]` higher-level wrapper that always uses `pro`/`ultra` and returns `{output, basis, citations}`. Lower friction for "I want a research report" use cases.

Implementation notes:
- Use `client.task_run.execute(..., poll=True, timeout=...)` for blocking calls, or `client.task_run.events(run_id)` (SSE) for live updates.
- Surface `basis` (citations, reasoning, confidence per output field) on every result. This is the killer feature.
- Return type uses `response_format="content_and_artifact"` to keep the LLM-facing content compact while preserving the structured result.

Add `BYOMCP` support: a `mcp_servers: list[McpServerParam]` arg on the task tool, with a thin pydantic model wrapping the SDK's `McpServerParam`. Pass `betas=["mcp-server-2025-07-17"]`.

### 2.3 Task Groups: `ParallelTaskGroup`

A `Runnable[list[Union[str, dict]], list[dict]]` (batch of inputs → batch of results) backed by `client.beta.task_group.{create,add_runs,events,get_runs}`. Useful for enrichment pipelines (1k–10k inputs in one shot). Stream events via `_atransform`. Webhooks supported via a `webhook_url` field; HMAC verification helper exposed as `langchain_parallel.tasks.verify_webhook(payload, signature, secret)`.

### 2.4 FindAll: `ParallelFindAllTool` + `ParallelFindAllRunner`

`ParallelFindAllTool(BaseTool)` for entity discovery in agents — `objective`, `entity_type`, `match_conditions`, `match_limit`, `generator` (Literal of preview/base/core/pro). Returns the candidate list as artifact, an LLM-friendly summary as content. Streaming via SSE for live discovery. New since mid-2025; entirely missing today.

### 2.5 Monitor: `ParallelMonitor`

A `Runnable` that creates/updates monitors, plus a `verify_webhook` helper. Lower priority — most users will configure monitors out-of-band — but worth a thin wrapper for parity.

### 2.6 MCP toolkit: `parallel_mcp_toolkit()`

A factory returning the Parallel-hosted MCP servers as `langchain` MCP tools (using the `langchain-mcp-adapters` package): wrap `https://search.parallel.ai/mcp-oauth` and `https://task-mcp.parallel.ai/mcp` so users get `web_search`, `web_fetch`, `createDeepResearch`, `createTaskGroup`, `getStatus`, `getResultMarkdown` as drop-in `BaseTool`s. ~20 lines, unlocks immediate use of the hosted servers.

### 2.7 `init_chat_model("parallel:...")` registration

- Open a PR against `langchain-ai/langchain` adding `"parallel": ("langchain_parallel", "ChatParallelWeb", _call)` to `_BUILTIN_PROVIDERS` in `libs/langchain_v1/langchain/chat_models/base.py`.
- Add a `langchain[parallel]` extra in the same PR.
- Add a model-name prefix rule (`speed` / `lite` / `base` / `core` → `parallel`) to `_attempt_infer_model_provider` so `init_chat_model("speed")` works without an explicit provider.

### 2.8 Deliverables for 0.4.0

```
+ langchain_parallel/retrievers.py
+ langchain_parallel/tasks.py            (ParallelTaskRunTool, ParallelDeepResearch, ParallelTaskGroup, verify_webhook)
+ langchain_parallel/findall.py          (ParallelFindAllTool, ParallelFindAllRunner)
+ langchain_parallel/monitors.py
+ langchain_parallel/mcp.py              (parallel_mcp_toolkit factory)
M langchain_parallel/__init__.py         (export new surfaces)
+ tests for each new module (unit + integration)
+ docs/{retriever,task_run,findall,monitor,mcp}.ipynb
+ external PR to langchain-ai/langchain
```

---

## Phase 3 — polish & ecosystem

### 3.1 Docs and examples

- Rewrite README around `create_agent` + `init_chat_model("parallel:speed")`. Drop `create_openai_functions_agent` references.
- New section: "Picking the right surface" (matrix: Chat vs Search vs Extract vs Task vs FindAll vs Monitor).
- New section: "Citations and basis" — every example shows how to access `response.basis` / `result.basis`.
- One notebook per surface; consolidate `docs/chat.ipynb`, `docs/search_tool.ipynb`, `docs/extract_tool.ipynb` against the new APIs and ensure they all match the `demo_agent.ipynb` pattern.
- Add a recipes doc: `RAG with Parallel` (Retriever), `Deep research agent` (Task), `Bulk enrichment` (Task Group), `Lead generation` (FindAll), `Change tracking` (Monitor). These are the obvious cross-API workflows.
- Mirror the Parallel docs page at `https://docs.parallel.ai/integrations/langchain.md` — keep README and that page synchronized.

### 3.2 Tests / CI

- Add a real VCR cassette path for chat (`enable_vcr_tests = True` once recordings stabilize) — speeds up CI and removes API-key flakiness for community contributors.
- Standard-tests: add `ToolsUnitTests` / `ToolsIntegrationTests` for `ParallelExtractTool` (currently missing — only Search has the wrapper).
- Run integration tests on a nightly cron, not every PR (rate limits + cost). PR-time CI runs unit tests only.
- Add `pre-commit` config matching `ruff` rules already in `pyproject.toml`.
- Codespell list (`pyproject.toml` already has the group) — turn on in CI.

### 3.3 Observability and ergonomics

- Trace tags: every API call should set `tags=["parallel", "search"|"extract"|"chat"|"task"|"findall"|"monitor"]` and `metadata={"parallel_search_id": ...}` on the run-manager so LangSmith traces are filterable.
- Add `Parallel-User-Agent` headers identifying `langchain-parallel/<version>` so Parallel can attribute usage.
- Surface SDK warnings (the new `warnings[]` array in v1 responses) by emitting a Python `warnings.warn(..., stacklevel=3)` per warning — invisible in production, very visible in dev.
- Better errors: catch `parallel.AuthenticationError`, `parallel.RateLimitError`, `parallel.APIStatusError` and translate to `ValueError`/`ToolException` with actionable messages, not the current `Exception` swallow.

### 3.4 Repo housekeeping

- `examples/` and `docs/` files written at v0.1 are stale — many import-paths and patterns are out of date. Rewrite or delete.
- `TESTING.md` references `tests/integration_tests/test_tools.py` and `tests/unit_tests/test_search.py` — those files don't exist (it's `test_search_tool.py`). Fix.
- `Makefile`'s `lint_diff` target uses `--relative=libs/partners/parallel-web` — that's a leftover from the LangChain monorepo style and incorrect for this standalone repo.
- Add a security policy / `SECURITY.md` (where to report API-key leakage etc.).

---

## Phased timeline (rough)

| Phase | Scope | Est. effort | Release |
|-------|-------|-------------|---------|
| 1 | bump SDK; GA endpoints; chat fixes; content_and_artifact; std-test conformance | 2–3 days | `0.3.0` |
| 2 | retriever; task run; task group; findall; monitor; mcp toolkit; init_chat_model PR | 1–2 weeks | `0.4.0` |
| 3 | docs/examples rewrite; observability; CI hardening | 3–5 days | rolling into `0.4.x` |

## Open questions

1. **Naming.** The package is `langchain-parallel` and the chat class is `ChatParallelWeb`. Other Parallel APIs aren't really "web" — should we drop the suffix and standardize on `ChatParallel`, `ParallelSearchTool`, `ParallelExtractTool`, `ParallelSearchRetriever`, etc.? If yes, ship the new names and keep `ChatParallelWeb` / `ParallelWebSearchTool` as aliases for one minor.
2. **Should `with_structured_output` on Chat use `task_spec` under the hood?** For long/structured outputs, dispatching to a Task Run with `output_schema=<json_schema>` and a `core` processor is much higher quality than `chat_completions` `response_format`. Worth a `method="task_run"` variant.
3. **Should we publish a top-level `from langchain_parallel import create_research_agent(...)` convenience** that pre-configures `create_agent` with our search+extract+task tools and a sensible system prompt? Lowers onboarding to one import.
4. **Embeddings.** Parallel doesn't expose embeddings today. If they ship an embeddings endpoint, `ParallelEmbeddings(Embeddings)` is the obvious add and gets registered with `init_embeddings("parallel:...")` similarly.
5. **Async-first vs sync-first.** Phase-2 surfaces (Tasks, FindAll) are inherently long-running. Should we expose them only as `Runnable` async + an `astream_events` SSE bridge, and not bother with sync? Async-only would let us shed a lot of duplication.

## Success criteria

- All standard tests for `ChatModelUnitTests`, `ChatModelIntegrationTests`, `ToolsUnitTests`, `ToolsIntegrationTests` pass with the right capability flags set.
- `init_chat_model("parallel:speed")` works after the upstream PR lands.
- README and the `docs.parallel.ai/integrations/langchain.md` page agree, line-for-line, on the public API.
- A user can do RAG in <10 lines (`ParallelSearchRetriever`) and a deep-research agent in <30 lines (`create_agent` + `ParallelTaskRunTool`).
- Citations / `basis` are accessible on every output that has them.

## Non-goals

- Replacing Parallel's first-party SDK. We wrap; we don't reimplement.
- Custom event-loop or threading magic in chat streaming. The OpenAI SDK already handles it.
- Browser-side / TS port. Out of scope.
- Caching layer. LangChain has `set_llm_cache(...)`; users opt in.
