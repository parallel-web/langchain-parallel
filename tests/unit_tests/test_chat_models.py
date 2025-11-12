"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_parallel.chat_models import ChatParallelWeb


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
        """Additional kwargs for with_structured_output."""
        return {"method": "function_calling"}

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
