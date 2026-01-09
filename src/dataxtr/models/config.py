"""Model configuration and registry."""

from dataclasses import dataclass
from enum import Enum


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"


class ModelTier(str, Enum):
    """Model capability tiers for task routing."""

    FAST = "fast"  # Cheap, fast models for simple tasks
    STANDARD = "standard"  # Balanced models
    POWERFUL = "powerful"  # Best reasoning capability
    VISION = "vision"  # Vision-capable models


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: ModelProvider
    model_id: str
    tier: ModelTier
    supports_vision: bool
    supports_tools: bool
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float


# Model registry with all supported models
MODEL_REGISTRY: dict[str, ModelConfig] = {
    # Anthropic Models
    "claude-3-haiku": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-haiku-20240307",
        tier=ModelTier.FAST,
        supports_vision=True,
        supports_tools=True,
        max_tokens=4096,
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
    ),
    "claude-3-sonnet": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        tier=ModelTier.STANDARD,
        supports_vision=True,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "claude-3-opus": ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-3-opus-20240229",
        tier=ModelTier.POWERFUL,
        supports_vision=True,
        supports_tools=True,
        max_tokens=4096,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    # OpenAI Models
    "gpt-3.5-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-3.5-turbo",
        tier=ModelTier.FAST,
        supports_vision=False,
        supports_tools=True,
        max_tokens=4096,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
    ),
    "gpt-4-turbo": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4-turbo",
        tier=ModelTier.STANDARD,
        supports_vision=True,
        supports_tools=True,
        max_tokens=4096,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
    "gpt-4o": ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        tier=ModelTier.POWERFUL,
        supports_vision=True,
        supports_tools=True,
        max_tokens=4096,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    # Google Models
    "gemini-flash": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-flash",
        tier=ModelTier.FAST,
        supports_vision=True,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
    "gemini-pro": ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-1.5-pro",
        tier=ModelTier.POWERFUL,
        supports_vision=True,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
    ),
    # Groq Models (fast inference)
    "groq-llama-8b": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.1-8b-instant",
        tier=ModelTier.FAST,
        supports_vision=False,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
    ),
    "groq-llama-70b": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.3-70b-versatile",
        tier=ModelTier.STANDARD,
        supports_vision=False,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
    ),
    "groq-llama-vision": ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.2-90b-vision-preview",
        tier=ModelTier.STANDARD,
        supports_vision=True,
        supports_tools=True,
        max_tokens=8192,
        cost_per_1k_input=0.0009,
        cost_per_1k_output=0.0009,
    ),
}


def get_models_by_tier(tier: ModelTier) -> list[ModelConfig]:
    """Get all models of a specific tier."""
    return [m for m in MODEL_REGISTRY.values() if m.tier == tier]


def get_models_by_provider(provider: ModelProvider) -> list[ModelConfig]:
    """Get all models from a specific provider."""
    return [m for m in MODEL_REGISTRY.values() if m.provider == provider]
