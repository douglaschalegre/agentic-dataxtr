"""Model routing for task-based model selection."""

from typing import Optional

from langchain_core.language_models import BaseChatModel

from dataxtr.models.config import (
    MODEL_REGISTRY,
    ModelConfig,
    ModelProvider,
    ModelTier,
)
from dataxtr.schemas.fields import ExtractionComplexity


class ModelRouter:
    """Routes to appropriate model based on task requirements."""

    def __init__(
        self,
        preferred_provider: Optional[ModelProvider] = None,
        fallback_enabled: bool = True,
        cost_optimization: bool = True,
    ):
        """Initialize the model router.

        Args:
            preferred_provider: Prefer models from this provider when possible
            fallback_enabled: Allow fallback to other providers if preferred unavailable
            cost_optimization: Sort by cost when multiple options available
        """
        self.preferred_provider = preferred_provider
        self.fallback_enabled = fallback_enabled
        self.cost_optimization = cost_optimization

    def select_model(
        self,
        complexity: ExtractionComplexity,
        requires_vision: bool = False,
        requires_tools: bool = True,
        upgrade: bool = False,
    ) -> ModelConfig:
        """Select appropriate model based on requirements.

        Args:
            complexity: Task complexity level
            requires_vision: Whether vision capability is needed
            requires_tools: Whether tool use is needed
            upgrade: Whether to upgrade to next tier

        Returns:
            Selected model configuration
        """
        # Map complexity to tier
        tier_map = {
            ExtractionComplexity.SIMPLE: ModelTier.FAST,
            ExtractionComplexity.COMPLEX: ModelTier.STANDARD,
            ExtractionComplexity.VISUAL: ModelTier.STANDARD,
        }

        target_tier = tier_map[complexity]

        # Upgrade tier if requested
        if upgrade:
            tier_upgrade = {
                ModelTier.FAST: ModelTier.STANDARD,
                ModelTier.STANDARD: ModelTier.POWERFUL,
                ModelTier.POWERFUL: ModelTier.POWERFUL,
            }
            target_tier = tier_upgrade[target_tier]

        # Visual tasks need vision-capable models
        if complexity == ExtractionComplexity.VISUAL:
            requires_vision = True

        # Filter compatible models
        candidates = [
            config
            for config in MODEL_REGISTRY.values()
            if config.tier == target_tier
            and (not requires_vision or config.supports_vision)
            and (not requires_tools or config.supports_tools)
        ]

        # Apply provider preference
        if self.preferred_provider and candidates:
            provider_candidates = [c for c in candidates if c.provider == self.preferred_provider]
            if provider_candidates or not self.fallback_enabled:
                candidates = provider_candidates

        # Cost optimization - sort by total cost
        if self.cost_optimization and candidates:
            candidates.sort(key=lambda x: x.cost_per_1k_input + x.cost_per_1k_output)

        if not candidates:
            raise ValueError(
                f"No suitable model found for tier={target_tier}, "
                f"vision={requires_vision}, tools={requires_tools}"
            )

        return candidates[0]

    def get_chat_model(self, config: ModelConfig) -> BaseChatModel:
        """Instantiate the appropriate chat model.

        Args:
            config: Model configuration

        Returns:
            Instantiated chat model
        """
        if config.provider == ModelProvider.ANTHROPIC:
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=config.model_id,
                max_tokens=config.max_tokens,
                temperature=0,  # Deterministic for extraction
            )

        elif config.provider == ModelProvider.OPENAI:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_id,
                max_tokens=config.max_tokens,
                temperature=0,
            )

        elif config.provider == ModelProvider.GOOGLE:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.model_id,
                max_output_tokens=config.max_tokens,
                temperature=0,
            )

        elif config.provider == ModelProvider.GROQ:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=config.model_id,
                max_tokens=config.max_tokens,
                temperature=0,
            )

        raise ValueError(f"Unknown provider: {config.provider}")

    def upgrade_model(self, current_config: ModelConfig) -> ModelConfig:
        """Get a more powerful model than the current one.

        Args:
            current_config: Current model configuration

        Returns:
            Upgraded model configuration
        """
        tier_upgrade = {
            ModelTier.FAST: ModelTier.STANDARD,
            ModelTier.STANDARD: ModelTier.POWERFUL,
            ModelTier.POWERFUL: ModelTier.POWERFUL,
            ModelTier.VISION: ModelTier.POWERFUL,
        }

        new_tier = tier_upgrade[current_config.tier]

        candidates = [
            config
            for config in MODEL_REGISTRY.values()
            if config.tier == new_tier
            and config.supports_vision >= current_config.supports_vision
            and config.supports_tools >= current_config.supports_tools
        ]

        # Prefer same provider if possible
        same_provider = [c for c in candidates if c.provider == current_config.provider]
        if same_provider:
            return same_provider[0]

        if candidates:
            return candidates[0]

        # Already at max, return same
        return current_config

    def get_model_for_task(
        self,
        complexity: ExtractionComplexity,
        model_hint: Optional[str] = None,
    ) -> tuple[BaseChatModel, ModelConfig]:
        """Get a model instance for a specific task.

        Args:
            complexity: Task complexity
            model_hint: Optional hint like 'upgrade' or specific model name

        Returns:
            Tuple of (model instance, config)
        """
        upgrade = model_hint == "upgrade"

        # Check if hint is a specific model name
        if model_hint and model_hint in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model_hint]
        else:
            config = self.select_model(complexity, upgrade=upgrade)

        model = self.get_chat_model(config)
        return model, config
