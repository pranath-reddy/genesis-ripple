"""Safe, lazy PydanticAI integration for AWS Bedrock Converse.

Only non-secret provider identity is represented here. AWS credentials are not
fields, function parameters, logs, or serialized artifacts. The AWS SDK resolves
the named local profile when a live model is explicitly constructed. This module
does not load ``.env`` files and importing it performs no AWS operation.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d+$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")

_PROFILE_ENV = "RIPPLE_BEDROCK_PROFILE"
_REGION_ENV = "RIPPLE_BEDROCK_REGION"
_MODEL_ENV = "RIPPLE_BEDROCK_MODEL_ID"


class BedrockProviderConfigurationError(RuntimeError):
    """Credential-free error raised before a Bedrock request can be sent."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


class BedrockRuntimeIdentity(BaseModel):
    """Serializable, non-secret identity of a Bedrock runtime selection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    provider: Literal["aws-bedrock-converse"] = "aws-bedrock-converse"
    profile_name: str
    region_name: str
    model_id: str


class BedrockAgentSettings(BaseModel):
    """Validated settings for the Bedrock agent boundary.

    These settings deliberately contain no credential material. To select a
    different identity, pass this model explicitly or set the three dedicated
    ``RIPPLE_BEDROCK_*`` environment variables read by
    :func:`load_bedrock_agent_settings`.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    profile_name: str = "ripple-bedrock"
    region_name: str = "ap-south-1"
    model_id: str = "in.openai.gpt-5.6-luna"
    max_output_tokens: int = Field(default=256, ge=32, le=4_096)
    retries: int = Field(default=2, ge=0, le=2)
    request_limit: int = Field(default=7, ge=1, le=8)
    input_token_limit: int = Field(default=12_000, ge=128, le=32_768)
    output_token_limit: int = Field(default=2_000, ge=32, le=4_096)
    total_token_limit: int = Field(default=14_000, ge=160, le=36_864)
    connect_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    read_timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)

    @model_validator(mode="after")
    def validate_identifiers_and_limits(self) -> "BedrockAgentSettings":
        if _PROFILE_PATTERN.fullmatch(self.profile_name) is None:
            raise ValueError("profile_name is not a normalized AWS profile name")
        if _REGION_PATTERN.fullmatch(self.region_name) is None:
            raise ValueError("region_name is not a normalized AWS region name")
        if _MODEL_ID_PATTERN.fullmatch(self.model_id) is None:
            raise ValueError("model_id is not a normalized Bedrock model identifier")
        if self.max_output_tokens > self.output_token_limit:
            raise ValueError("max_output_tokens cannot exceed output_token_limit")
        if self.input_token_limit + self.output_token_limit > self.total_token_limit:
            raise ValueError(
                "total_token_limit must cover input_token_limit plus output_token_limit"
            )
        return self

    def runtime_identity(self) -> BedrockRuntimeIdentity:
        """Return the only provider metadata permitted in artifacts or logs."""

        return BedrockRuntimeIdentity(
            profile_name=self.profile_name,
            region_name=self.region_name,
            model_id=self.model_id,
        )


class BedrockTypedSmokeOutput(BaseModel):
    """Tiny typed result used to verify the PydanticAI/Bedrock boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )

    status: Literal["ready"]
    provider: Literal["aws-bedrock-converse"]
    task: Literal["typed-output-smoke"]


@dataclass(frozen=True)
class BedrockTypedSmokeRunRecord:
    """Typed smoke output plus provider-reported usage and call latency."""

    output: BedrockTypedSmokeOutput
    requests: int
    tool_calls: int
    input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    elapsed_seconds: float


def expand_bedrock_agent_settings(
    settings: BedrockAgentSettings,
    *,
    max_output_tokens: int,
    input_token_limit: int,
    output_token_limit: int,
    request_limit: int,
) -> BedrockAgentSettings:
    """Keep provider identity while meeting one typed agent's minimum limits."""

    payload = settings.model_dump(mode="python")
    payload.update(
        {
            "max_output_tokens": max(settings.max_output_tokens, max_output_tokens),
            "input_token_limit": max(settings.input_token_limit, input_token_limit),
            "output_token_limit": max(
                settings.output_token_limit,
                output_token_limit,
                max_output_tokens,
            ),
            "request_limit": max(settings.request_limit, request_limit),
        }
    )
    payload["total_token_limit"] = max(
        settings.total_token_limit,
        payload["input_token_limit"] + payload["output_token_limit"],
    )
    return BedrockAgentSettings.model_validate(payload, strict=True)


def load_bedrock_agent_settings() -> BedrockAgentSettings:
    """Load non-secret provider selection from the process environment.

    Missing variables use explicit project defaults. A variable that is present
    but empty, padded with whitespace, or malformed fails closed. Credential
    variables are intentionally neither inspected nor copied.
    """

    values: dict[str, str] = {}
    for environment_name, field_name in (
        (_PROFILE_ENV, "profile_name"),
        (_REGION_ENV, "region_name"),
        (_MODEL_ENV, "model_id"),
    ):
        raw_value = os.environ.get(environment_name)
        if raw_value is None:
            continue
        if not raw_value or raw_value != raw_value.strip():
            raise BedrockProviderConfigurationError(
                code="bedrock_environment_value_invalid",
                message=(
                    f"{environment_name} must be a non-empty normalized value when set."
                ),
            )
        values[field_name] = raw_value

    try:
        return BedrockAgentSettings.model_validate(values)
    except ValueError:
        raise BedrockProviderConfigurationError(
            code="bedrock_settings_invalid",
            message=(
                "The non-secret RIPPLe Bedrock profile, region, or model settings "
                "failed validation."
            ),
        ) from None


def build_bedrock_converse_model(
    settings: BedrockAgentSettings | None = None,
) -> Any:
    """Construct a PydanticAI Bedrock model without sending a request.

    The SDK receives only a profile name and region. Callers cannot pass raw AWS
    keys through this interface.
    """

    resolved = settings or load_bedrock_agent_settings()
    try:
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider
    except ImportError:
        raise BedrockProviderConfigurationError(
            code="bedrock_dependencies_unavailable",
            message=(
                "The optional PydanticAI Bedrock dependencies are unavailable. "
                "Install requirements-scientist-agent.txt in the Python 3.12 "
                "agent environment."
            ),
        ) from None

    try:
        provider = BedrockProvider(
            profile_name=resolved.profile_name,
            region_name=resolved.region_name,
            aws_connect_timeout=resolved.connect_timeout_seconds,
            aws_read_timeout=resolved.read_timeout_seconds,
        )
        model_profile = provider.model_profile(resolved.model_id)
        if model_profile is None and resolved.model_id.startswith("in.openai."):
            # PydanticAI 2.1 does not recognize Bedrock's India inference
            # prefix.  Luna's Converse endpoint has been verified to reject
            # the ``reasoning_effort`` field which that release assigns to
            # other OpenAI-on-Bedrock profiles, so declare only the common
            # Converse capabilities here.  Do not invent model-specific
            # request fields.
            model_profile = BedrockModelProfile(
                bedrock_thinking_variant=None,
                supports_thinking=False,
                thinking_always_enabled=False,
            )
        return BedrockConverseModel(
            resolved.model_id,
            provider=provider,
            profile=model_profile,
            settings={
                "max_tokens": resolved.max_output_tokens,
            },
        )
    except Exception:
        # SDK/profile exceptions are deliberately replaced with a fixed message:
        # neither exception text nor resolved credential state crosses this API.
        raise BedrockProviderConfigurationError(
            code="bedrock_runtime_initialization_failed",
            message=(
                "AWS Bedrock runtime initialization failed for the configured "
                "non-secret profile identity. No model request was sent."
            ),
        ) from None


def build_bedrock_typed_smoke_agent(
    settings: BedrockAgentSettings | None = None,
) -> Any:
    """Build, but do not run, a bounded PydanticAI typed-output smoke agent."""

    resolved = settings or load_bedrock_agent_settings()
    model = build_bedrock_converse_model(resolved)
    try:
        from pydantic_ai import Agent, PromptedOutput
    except ImportError:
        raise BedrockProviderConfigurationError(
            code="pydantic_ai_unavailable",
            message="PydanticAI is unavailable in the Python 3.12 agent environment.",
        ) from None

    # Luna is not currently declared by PydanticAI as supporting Bedrock native
    # JSON Schema output. PromptedOutput still validates the returned JSON with
    # Pydantic before it can leave this boundary.
    return Agent(
        model=model,
        output_type=PromptedOutput(
            BedrockTypedSmokeOutput,
            name="ripple_bedrock_typed_smoke",
            description="Return the three fixed smoke-test identity fields.",
        ),
        system_prompt=(
            "You are a connectivity smoke check. Return only the requested typed "
            "object. Do not call tools, infer scientific facts, or add commentary."
        ),
        retries=resolved.retries,
        name="ripple_bedrock_typed_smoke",
    )


def run_bedrock_typed_output_smoke(
    settings: BedrockAgentSettings | None = None,
) -> BedrockTypedSmokeOutput:
    """Perform one explicit, fixed-prompt live smoke request.

    This function is the network boundary and is never called during import or
    provider construction. Its prompt contains no user data, source code, images,
    paths, credentials, or scientific artifacts.
    """

    resolved = settings or load_bedrock_agent_settings()
    agent = build_bedrock_typed_smoke_agent(resolved)
    try:
        from pydantic_ai.usage import UsageLimits
    except ImportError:
        raise BedrockProviderConfigurationError(
            code="pydantic_ai_unavailable",
            message="PydanticAI is unavailable in the Python 3.12 agent environment.",
        ) from None

    result = agent.run_sync(
        (
            "Return status='ready', provider='aws-bedrock-converse', and "
            "task='typed-output-smoke'."
        ),
        usage_limits=UsageLimits(
            request_limit=resolved.request_limit,
            tool_calls_limit=0,
            input_tokens_limit=resolved.input_token_limit,
            output_tokens_limit=resolved.output_token_limit,
            total_tokens_limit=resolved.total_token_limit,
            count_tokens_before_request=False,
        ),
    )
    return result.output


def run_bedrock_typed_output_smoke_with_usage(
    settings: BedrockAgentSettings | None = None,
) -> BedrockTypedSmokeRunRecord:
    """Perform the fixed smoke request and retain its aggregate usage evidence."""

    resolved = settings or load_bedrock_agent_settings()
    agent = build_bedrock_typed_smoke_agent(resolved)
    try:
        from pydantic_ai.usage import UsageLimits
    except ImportError:
        raise BedrockProviderConfigurationError(
            code="pydantic_ai_unavailable",
            message="PydanticAI is unavailable in the Python 3.12 agent environment.",
        ) from None

    started = time.monotonic()
    result = agent.run_sync(
        (
            "Return status='ready', provider='aws-bedrock-converse', and "
            "task='typed-output-smoke'."
        ),
        usage_limits=UsageLimits(
            request_limit=resolved.request_limit,
            tool_calls_limit=0,
            input_tokens_limit=resolved.input_token_limit,
            output_tokens_limit=resolved.output_token_limit,
            total_tokens_limit=resolved.total_token_limit,
            count_tokens_before_request=False,
        ),
    )
    elapsed_seconds = time.monotonic() - started
    usage = result.usage
    return BedrockTypedSmokeRunRecord(
        output=result.output,
        requests=usage.requests,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        elapsed_seconds=elapsed_seconds,
    )


__all__ = [
    "BedrockAgentSettings",
    "BedrockProviderConfigurationError",
    "BedrockRuntimeIdentity",
    "BedrockTypedSmokeOutput",
    "BedrockTypedSmokeRunRecord",
    "build_bedrock_converse_model",
    "build_bedrock_typed_smoke_agent",
    "load_bedrock_agent_settings",
    "run_bedrock_typed_output_smoke",
    "run_bedrock_typed_output_smoke_with_usage",
]
