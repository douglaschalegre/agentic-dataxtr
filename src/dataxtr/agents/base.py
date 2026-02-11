"""Base agent class for the extraction system."""

import json
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
        structured_output_method: Optional[str] = None,
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
        self.structured_output_method = structured_output_method

        # Bind tools to model if provided
        if self.tools:
            self.model = self.model.bind_tools(self.tools)

    def _with_structured_output(self, schema: Any) -> Any:
        """Get structured output wrapper, with provider-specific fallback."""
        if self.structured_output_method:
            try:
                return self.model.with_structured_output(
                    schema,
                    method=self.structured_output_method,
                    include_raw=True,
                )
            except TypeError:
                return self.model.with_structured_output(schema)
        return self.model.with_structured_output(schema)

    def _normalize_structured_result(self, result: Any) -> Any:
        """Normalize `include_raw` structured output payloads."""
        if not (isinstance(result, dict) and "parsed" in result):
            return result

        parsed = result.get("parsed")
        if parsed is not None:
            return parsed

        raw = result.get("raw")
        raw_content = getattr(raw, "content", raw)

        if isinstance(raw_content, str):
            return json.loads(raw_content)

        if isinstance(raw_content, list):
            text_parts: list[str] = []
            for item in raw_content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                else:
                    text_parts.append(str(item))
            return json.loads("".join(text_parts))

        raise ValueError(f"Failed to parse structured output: {result.get('parsing_error')}")

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
