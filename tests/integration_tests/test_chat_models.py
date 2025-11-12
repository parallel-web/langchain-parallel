"""Test ChatParallelWeb chat model."""

from __future__ import annotations

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_parallel.chat_models import ChatParallelWeb


class TestChatParallelWebIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatParallelWeb]:
        return ChatParallelWeb

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "speed",
            # API key will be read from environment variable PARALLEL_API_KEY
        }

    # Configure capabilities based on Parallel's Chat API features
    @property
    def has_tool_calling(self) -> bool:
        """Parallel Chat API tool calling support - currently not implemented."""
        return False

    @property
    def has_structured_output(self) -> bool:
        """Parallel Chat API structured output support - currently not implemented.

        Currently not implemented in Parallel Chat API.
        """
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
    def supports_video_inputs(self) -> bool:
        """Parallel Chat API does not support video inputs."""
        return False

    @property
    def supports_audio_inputs(self) -> bool:
        """Parallel Chat API does not support audio inputs."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Parallel Chat API does not currently return usage metadata."""
        return False

    @property
    def supports_system_messages(self) -> bool:
        """Parallel Chat API supports system messages via OpenAI interface.

        Supports system messages through OpenAI-compatible API.
        """
        return True

    @property
    def has_tool_choice(self) -> bool:
        """Parallel ignores tool choice parameter."""
        return False

    @property
    def supports_json_mode(self) -> bool:
        """Parallel Chat API JSON mode support - currently not implemented."""
        return False

    @property
    def supports_image_urls(self) -> bool:
        """Parallel Chat API does not support image URLs."""
        return False

    @property
    def supports_pdf_inputs(self) -> bool:
        """Parallel does not support PDF inputs."""
        return False

    @property
    def supports_image_tool_message(self) -> bool:
        """Parallel Chat API does not support image tool messages."""
        return False

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
