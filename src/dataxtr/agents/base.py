"""Base agent class for the extraction system."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool


class BaseAgent(ABC):
    """Base class for all agents in the extraction system."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the agent.

        Args:
            model: The chat model to use
            tools: Optional list of tools for the agent
            system_prompt: Optional system prompt for the agent
        """
        self.model = model
        self.tools = tools or []
        self.system_prompt = system_prompt

        # Bind tools to model if provided
        if self.tools:
            self.model = self.model.bind_tools(self.tools)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the agent's main task.

        Args:
            **kwargs: Task-specific arguments

        Returns:
            Task-specific result
        """
        pass

    def _build_prompt(self, human_template: str = "{input}") -> ChatPromptTemplate:
        """Build the prompt template for this agent.

        Args:
            human_template: Template for the human message

        Returns:
            ChatPromptTemplate instance
        """
        messages = []
        if self.system_prompt:
            messages.append(("system", self.system_prompt))
        messages.append(("human", human_template))
        return ChatPromptTemplate.from_messages(messages)
