"""Test chat model integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel, SecretStr

from langchain_parallel.chat_models import ChatParallelWeb

_TEST_KEY = SecretStr("test")


class TestChatParallelWebUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatParallelWeb]:
        return ChatParallelWeb

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "speed",
            "api_key": "test-api-key",
        }

    @property
    def standard_chat_model_params(self) -> dict:
        """Parallel ignores most OpenAI sampling params; keep tests honest."""
        return {}

    # Configure capabilities based on Parallel's Chat API features
    @property
    def has_tool_calling(self) -> bool:
        """Parallel Chat API tool calling support - currently not implemented."""
        return False

    @property
    def has_tool_choice(self) -> bool:
        """Parallel ignores tool choice parameter."""
        return False

    @property
    def has_structured_output(self) -> bool:
        """Parallel Chat API structured output support - currently not implemented.

        Currently not implemented in Parallel Chat API.
        """
        return False

    @property
    def supports_json_mode(self) -> bool:
        """Parallel ignores JSON mode parameter."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Parallel Chat API does not currently return usage metadata."""
        return False

    @property
    def supports_anthropic_computer_use(self) -> bool:
        """Parallel Chat API does not support Anthropic computer use."""
        return False

    @property
    def supports_image_inputs(self) -> bool:
        """Parallel Chat API image input support - not confirmed."""
        return False

    @property
    def supports_image_urls(self) -> bool:
        """Parallel does not support image URLs."""
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        """Parallel does not support PDF inputs."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Parallel Chat API does not support audio inputs."""
        return False

    @property
    def supports_video_inputs(self) -> bool:
        """Parallel Chat API does not support video inputs."""
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """Parallel does not support image tool messages."""
        return False

    @property
    def structured_output_kwargs(self) -> dict:
        """Additional kwargs for with_structured_output.

        Parallel research models (`lite`, `base`, `core`) accept
        ``response_format`` JSON schemas; ``function_calling`` is not
        supported. The base class doesn't enable structured output
        (see :attr:`has_structured_output`); subclasses that flip the
        flag should default to ``method='json_schema'``.
        """
        return {"method": "json_schema"}

    @property
    def supported_usage_metadata_details(self) -> dict:
        """Parallel supports basic usage metadata."""
        return {
            "invoke": [],
            "stream": [],
        }

    @property
    def enable_vcr_tests(self) -> bool:
        """Disable VCR tests for now."""
        return False

    @property
    def supports_system_messages(self) -> bool:
        """Parallel Chat API supports system messages via OpenAI interface.

        Supports system messages through OpenAI-compatible API.
        """
        return True

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters for testing initialization from environment variables."""
        return (
            {
                "PARALLEL_API_KEY": "test-env-api-key",
            },
            {
                "model": "speed",
            },
            {
                "api_key": "test-env-api-key",
            },
        )


class TestChatParallelWebUnitLite(TestChatParallelWebUnit):
    """Unit tests parametrized for the `lite` research model.

    `lite` (and `base`/`core`) accept ``response_format`` JSON schema, so the
    structured-output capability flag is True for those models.
    """

    @property
    def chat_model_params(self) -> dict:
        return {"model": "lite", "api_key": "test-api-key"}

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def structured_output_kwargs(self) -> dict:
        # Parallel research models use json_schema, not function_calling.
        return {"method": "json_schema"}


class _Founder(BaseModel):
    name: str
    company: str


class TestChatParallelWebDirect:
    """Direct unit tests for behaviors the standard suite doesn't cover."""

    def test_model_kwarg_actually_sets_model(self) -> None:
        """`ChatParallelWeb(model='lite')` selects 'lite' (regression test)."""
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        assert chat.model == "lite"

    def test_model_name_alias_back_compat(self) -> None:
        """`ChatParallelWeb(model_name='lite')` still works via the validator shim."""
        # Pre-0.3.0 callers used `model_name=`; the validator maps it back
        # to `model=`. `model_name` isn't a real field, so silence the type
        # checker on this back-compat call.
        chat = ChatParallelWeb(model_name="lite", api_key=_TEST_KEY)  # type: ignore[call-arg]
        assert chat.model == "lite"

    def test_lc_attributes_exposes_model_name(self) -> None:
        """`lc_attributes` surfaces the model under the LangChain-standard key."""
        chat = ChatParallelWeb(model="core", api_key=_TEST_KEY)
        assert chat.lc_attributes["model_name"] == "core"

    def test_response_metadata_surfaces_basis_and_interaction_id(self) -> None:
        """Basis / interaction_id / system_fingerprint round-trip on AIMessage."""
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="Elon Musk founded SpaceX."),
        )
        response = SimpleNamespace(
            choices=[choice],
            model="lite",
            created=1700000000,
            system_fingerprint="fp-1",
            interaction_id="int-1",
            basis=[
                SimpleNamespace(
                    model_dump=lambda: {"field": "answer", "citations": []},
                ),
            ],
        )
        result = chat._process_non_stream_response(response)
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.response_metadata["model_name"] == "lite"
        assert msg.response_metadata["finish_reason"] == "stop"
        assert msg.response_metadata["system_fingerprint"] == "fp-1"
        assert msg.response_metadata["interaction_id"] == "int-1"
        assert msg.response_metadata["basis"] == [
            {"field": "answer", "citations": []},
        ]

    def test_with_structured_output_rejects_speed(self) -> None:
        """Speed silently ignores response_format; raise to make this loud."""
        chat = ChatParallelWeb(model="speed", api_key=_TEST_KEY)
        with pytest.raises(ValueError, match="research models"):
            chat.with_structured_output(_Founder)

    def test_with_structured_output_binds_response_format(self) -> None:
        """Binding a pydantic schema produces a json_schema response_format."""
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        runnable = chat.with_structured_output(_Founder)
        assert isinstance(runnable, RunnableSequence)
        bound = runnable.first
        rf = bound.kwargs["response_format"]  # type: ignore[attr-defined]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "_Founder"
        assert rf["json_schema"]["strict"] is True
        assert "name" in rf["json_schema"]["schema"]["properties"]

    def test_with_structured_output_function_calling_routes_to_json_schema(
        self,
    ) -> None:
        """method='function_calling' is routed to json_schema for compat.

        Parallel chat doesn't actually support tool calling; we accept
        ``function_calling`` for cross-provider compatibility and produce a
        json_schema response_format under the hood.
        """
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        runnable = chat.with_structured_output(_Founder, method="function_calling")
        assert isinstance(runnable, RunnableSequence)
        bound = runnable.first
        assert bound.kwargs["response_format"]["type"] == "json_schema"  # type: ignore[attr-defined]

    def test_with_structured_output_json_mode(self) -> None:
        """method='json_mode' produces a json_object response_format."""
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        runnable = chat.with_structured_output(method="json_mode")
        assert isinstance(runnable, RunnableSequence)
        bound = runnable.first
        assert bound.kwargs["response_format"] == {"type": "json_object"}  # type: ignore[attr-defined]

    def test_with_structured_output_include_raw_failure_capture(self) -> None:
        """include_raw=True populates parsing_error on parse failure."""
        from langchain_core.runnables import RunnableLambda

        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        runnable = chat.with_structured_output(_Founder, include_raw=True)
        # The capture lambda is the last step; pull it out and exercise directly
        # so we don't need a live API call.
        assert isinstance(runnable, RunnableSequence)
        capture = next(
            step.func for step in runnable.steps if isinstance(step, RunnableLambda)
        )
        result = capture(AIMessage(content="not json"))
        assert isinstance(result["raw"], AIMessage)
        assert result["parsed"] is None
        assert result["parsing_error"] is not None

    def test_with_structured_output_include_raw_success(self) -> None:
        """include_raw=True wraps the parsed pydantic object."""
        from langchain_core.runnables import RunnableLambda

        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        runnable = chat.with_structured_output(_Founder, include_raw=True)
        assert isinstance(runnable, RunnableSequence)
        capture = next(
            step.func for step in runnable.steps if isinstance(step, RunnableLambda)
        )
        result = capture(
            AIMessage(content='{"name": "Elon Musk", "company": "SpaceX"}'),
        )
        assert isinstance(result["parsed"], _Founder)
        assert result["parsed"].name == "Elon Musk"
        assert result["parsing_error"] is None

    def test_chat_parallel_is_alias_of_chat_parallel_web(self) -> None:
        """``ChatParallel`` is the new canonical name; old name still works."""
        from langchain_parallel import ChatParallel, ChatParallelWeb

        assert ChatParallel is ChatParallelWeb
        assert ChatParallel(model="lite", api_key=_TEST_KEY).model == "lite"

    def test_response_metadata_stream_chunk_includes_basis(self) -> None:
        """Streaming chunks expose basis on the last chunk."""
        chat = ChatParallelWeb(model="lite", api_key=_TEST_KEY)
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content="."),
                ),
            ],
            model="lite",
            interaction_id="int-2",
            basis=[
                SimpleNamespace(
                    model_dump=lambda: {"field": "answer", "citations": []},
                ),
            ],
            system_fingerprint=None,
        )
        out = chat._process_stream_chunk(chunk, run_manager=Mock())
        assert out is not None
        meta = out.message.response_metadata
        assert meta["model_name"] == "lite"
        assert meta["interaction_id"] == "int-2"
        assert meta["basis"] == [{"field": "answer", "citations": []}]
