# Testing Guide for LangChain Parallel Web Integration

This document explains how to run the LangChain standard tests for both ChatParallelWeb and ParallelWebSearchTool integrations.

## Overview

Both integrations follow the official [LangChain standard testing framework](https://python.langchain.com/docs/contributing/testing/):

- **ChatParallelWeb**: Chat model integration with OpenAI-compatible interface
- **ParallelWebSearchTool**: Search tool integration for web research capabilities

## Architecture

### Chat Model (ChatParallelWeb)
- **OpenAI Compatibility**: Uses OpenAI-compatible interface with Parallel endpoint
- **Message Handling**: Automatic message merging for API compliance
- **LangChain Integration**: Full metadata support (`lc_secrets`, `lc_attributes`, etc.)

### Search Tool (ParallelWebSearchTool)
- **Direct API Access**: Direct integration with Parallel Search API
- **Input Validation**: Comprehensive parameter validation
- **Metadata Collection**: Search timing, result counts, processor information
- **Async Support**: Full async/await support with proper event loop handling

## Test Types

### 1. Unit Tests (`tests/unit_tests/`)
- **Purpose**: Test components in isolation without external API calls
- **Network**: Disabled (no external connections)
- **API Key**: Uses mock/test values
- **Files**:
  - `test_chat_models.py` - Chat model unit tests
  - `test_search.py` - Search tool unit tests

### 2. Integration Tests (`tests/integration_tests/`)
- **Purpose**: Test components with real API calls
- **Network**: Enabled (requires internet connection)
- **API Key**: Requires valid `PARALLEL_API_KEY` environment variable
- **Files**:
  - `test_chat_models.py` - Chat model integration tests
  - `test_tools.py` - Search tool integration tests

## Running Tests

### Unit Tests Only (Recommended for Development)
```bash
# Run unit tests with network disabled
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/unit_tests/ -v
```

### Integration Tests (Requires API Key)
```bash
# Set your API key first
export PARALLEL_API_KEY="your-actual-api-key-here"

# Run integration tests
poetry run pytest --asyncio-mode=auto tests/integration_tests/ -v
```

### All Tests
```bash
# Run all tests (integration tests will fail without API key)
poetry run pytest --disable-socket --allow-unix-socket --asyncio-mode=auto tests/ -v
```

## Test Configuration

### Chat Model Capabilities (ChatParallelWeb)
| Capability | Supported | Notes |
|------------|-----------|-------|
| **Basic Chat** | ✅ | Full sync/async support |
| **Streaming** | ✅ | Real-time response streaming |
| **System Messages** | ✅ | OpenAI-compatible interface |
| **Usage Metadata** | ❌ | API doesn't return usage data |
| **JSON Mode** | ❌ | Not supported by API |
| **Tool Calling** | ❌ | Not supported by API |
| **Structured Output** | ❌ | Not supported by API |
| **Image Inputs** | ❌ | Not supported by API |
| **Audio/Video Inputs** | ❌ | Not supported by API |

### Search Tool Capabilities (ParallelWebSearchTool)
| Capability | Supported | Notes |
|------------|-----------|-------|
| **Objective-based Search** | ✅ | Natural language search goals |
| **Query-based Search** | ✅ | Traditional keyword searches |
| **Domain Filtering** | ✅ | Include/exclude specific domains |
| **Processor Selection** | ✅ | Base (fast) and Pro (quality) modes |
| **Async Support** | ✅ | Full async/await functionality |
| **Metadata Collection** | ✅ | Search timing and result statistics |
| **Input Validation** | ✅ | Parameter validation and error handling |

### Current Test Status

#### Chat Model Tests
- **Unit Tests**: 8+ tests covering initialization, serialization, environment setup
- **Integration Tests**: 40+ tests covering chat functionality, streaming, message handling
- **Skipped Tests**: Usage metadata, tool calling, structured output (not supported by API)

#### Search Tool Tests
- **Unit Tests**: 10+ tests covering validation, error handling, metadata collection
- **Integration Tests**: 5+ tests covering real API calls and response parsing
- **All Tests**: Currently passing with proper API key configuration
