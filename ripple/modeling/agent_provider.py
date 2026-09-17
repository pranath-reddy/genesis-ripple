"""Lazy provider construction for the optional live onboarding agent.

This module deliberately does not load ``.env`` files and never accepts an API
key as a function argument. OpenAI reads ``OPENAI_API_KEY`` only at its live
boundary. Bedrock uses the named profile selected by the credential-free RIPPLe
Bedrock settings model.
"""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

LiveAgentProvider = Literal["openai", "bedrock"]


class LiveAgentConfigurationError(RuntimeError):
    """A safe, credential-free explanation of an invalid live configuration."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class LiveAgentConfiguration(BaseModel):
    """Non-secret identity for one live provider selection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    provider: Literal["openai"] = "openai"
    api: Literal["responses"] = "responses"
    model_id: str = Field(min_length=1, max_length=128)


def load_live_agent_configuration() -> LiveAgentConfiguration:
    """Validate live-only environment state without returning the credential."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None or not api_key.strip():
        raise LiveAgentConfigurationError(
            code="openai_api_key_missing",
            message=(
                "OPENAI_API_KEY is not available in this process. Export it locally "
                "before invoking the live onboarding command."
            ),
        )
    # Do not retain the credential in a configuration object or exception.
    del api_key

    model_id = os.environ.get("RIPPLE_AGENT_MODEL")
    if model_id is None or not model_id.strip():
        raise LiveAgentConfigurationError(
            code="agent_model_missing",
            message=(
                "RIPPLE_AGENT_MODEL is not available in this process. Export an "
                "explicit OpenAI model ID before invoking the live onboarding command."
            ),
        )
    if model_id != model_id.strip() or _MODEL_ID_PATTERN.fullmatch(model_id) is None:
        raise LiveAgentConfigurationError(
            code="agent_model_invalid",
            message="RIPPLE_AGENT_MODEL is not a safe, normalized model identifier.",
        )
    return LiveAgentConfiguration(model_id=model_id)


def build_openai_responses_model_from_environment() -> tuple[
    Any, LiveAgentConfiguration
]:
    """Build a PydanticAI OpenAI Responses model only at the live-call boundary."""

    configuration = load_live_agent_configuration()
    try:
        from pydantic_ai.models.openai import OpenAIResponsesModel
    except ModuleNotFoundError:
        raise LiveAgentConfigurationError(
            code="pydantic_ai_unavailable",
            message=(
                "The optional PydanticAI OpenAI dependency is not installed in this "
                "interpreter. Use the dedicated agent environment."
            ),
        ) from None
    return OpenAIResponsesModel(configuration.model_id), configuration


def build_live_agent_model_from_environment(
    provider: LiveAgentProvider,
) -> tuple[Any, Any]:
    """Build one explicitly selected live provider without exposing credentials.

    ``openai`` preserves the existing Responses configuration. ``bedrock`` reuses
    the scientist subsystem's typed profile/region/model settings and Converse
    adapter. Neither branch loads a ``.env`` file.
    """

    if provider == "openai":
        return build_openai_responses_model_from_environment()
    if provider != "bedrock":
        raise LiveAgentConfigurationError(
            code="live_agent_provider_invalid",
            message="The live onboarding provider must be openai or bedrock.",
        )

    try:
        from ..scientist.providers import (
            BedrockProviderConfigurationError,
            build_bedrock_converse_model,
            expand_bedrock_agent_settings,
            load_bedrock_agent_settings,
        )
    except ImportError:
        raise LiveAgentConfigurationError(
            code="bedrock_provider_unavailable",
            message="The optional RIPPLe Bedrock provider is unavailable.",
        ) from None

    try:
        settings = expand_bedrock_agent_settings(
            load_bedrock_agent_settings(),
            max_output_tokens=4_096,
            input_token_limit=32_768,
            output_token_limit=4_096,
            request_limit=8,
        )
        return build_bedrock_converse_model(settings), settings.runtime_identity()
    except BedrockProviderConfigurationError as exc:
        raise LiveAgentConfigurationError(
            code=exc.code,
            message=exc.safe_message,
        ) from None


__all__ = [
    "LiveAgentConfiguration",
    "LiveAgentConfigurationError",
    "LiveAgentProvider",
    "build_live_agent_model_from_environment",
    "build_openai_responses_model_from_environment",
    "load_live_agent_configuration",
]
