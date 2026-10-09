"""Optional live-model providers for the RIPPLe scientist control plane.

Importing this package does not initialize an AWS session or contact a network.
Provider-specific SDK imports are delayed until a caller explicitly constructs a
live model.
"""

from .bedrock import (
    BedrockAgentSettings,
    BedrockProviderConfigurationError,
    BedrockRuntimeIdentity,
    BedrockTypedSmokeOutput,
    build_bedrock_converse_model,
    build_bedrock_typed_smoke_agent,
    expand_bedrock_agent_settings,
    load_bedrock_agent_settings,
    run_bedrock_typed_output_smoke,
)

__all__ = [
    "BedrockAgentSettings",
    "BedrockProviderConfigurationError",
    "BedrockRuntimeIdentity",
    "BedrockTypedSmokeOutput",
    "build_bedrock_converse_model",
    "build_bedrock_typed_smoke_agent",
    "expand_bedrock_agent_settings",
    "load_bedrock_agent_settings",
    "run_bedrock_typed_output_smoke",
]
