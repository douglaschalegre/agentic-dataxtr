"""Extraction agent for data extraction from documents."""

import json
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from dataxtr.agents.base import BaseAgent
from dataxtr.schemas.fields import FieldGroup
from dataxtr.schemas.results import ExtractionResult, GroupExtractionResult
from dataxtr.tools.document_tools import DOCUMENT_TOOLS


class FieldExtractionOutput(BaseModel):
    """Structured output for field extraction."""

    extractions: list[ExtractionResult] = Field(description="Extracted values for each field")


class ExtractionAgent(BaseAgent):
    """Agent that extracts field values from documents."""

    SYSTEM_PROMPT = """You are a data extraction expert. Your task is to extract
specific field values from document content.

For each field in the group:
1. Use the provided tools to search and read relevant document sections
2. Extract the exact value that matches the field description
3. Provide a confidence score (0.0-1.0) based on:
   - 1.0: Exact match found, clearly visible
   - 0.8-0.9: High confidence, minor ambiguity
   - 0.6-0.7: Moderate confidence, some interpretation needed
   - 0.4-0.5: Low confidence, multiple possible values
   - 0.1-0.3: Very low confidence, guessing
4. Record the source location (page number, section)
5. Include the raw text snippet where the value was found

If a required field cannot be found, still include it with:
- extracted_value: null
- confidence: 0.0
- An explanation in raw_text

Use tools efficiently - search first, then read specific sections."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
        model_name: str = "unknown",
    ):
        """Initialize the extraction agent.

        Args:
            model: Chat model for extraction
            tools: Document interaction tools
            model_name: Name of the model being used
        """
        super().__init__(
            model=model,
            tools=tools or DOCUMENT_TOOLS,
            system_prompt=self.SYSTEM_PROMPT,
        )
        self.model_name = model_name

    async def execute(
        self,
        field_group: FieldGroup,
        document_content: dict[str, Any],
        document_metadata: dict[str, Any],
    ) -> GroupExtractionResult:
        """Extract fields for a specific group.

        Args:
            field_group: The field group to extract
            document_content: Parsed document content
            document_metadata: Document metadata

        Returns:
            Extraction results for the group
        """
        return await self.extract(field_group, document_content, document_metadata)

    async def extract(
        self,
        field_group: FieldGroup,
        document_content: dict[str, Any],
        document_metadata: dict[str, Any],
    ) -> GroupExtractionResult:
        """Extract fields for a specific group using agentic tool use.

        Args:
            field_group: The field group to extract
            document_content: Parsed document content
            document_metadata: Document metadata

        Returns:
            Extraction results for the group
        """
        start_time = time.time()

        # Build the extraction prompt
        fields_desc = "\n".join(
            f"- {f.name}: {f.description} (type: {f.field_type.value}, required: {f.required})"
            for f in field_group.fields
        )

        extraction_prompt = f"""Extract the following fields from the document:

Group: {field_group.group_name}
Context hint: {field_group.context_hint}

Fields to extract:
{fields_desc}

Document overview:
- Pages: {document_metadata.get("page_count", "unknown")}
- Has tables: {document_metadata.get("has_tables", False)}
- Has images: {document_metadata.get("has_images", False)}

First use search_document or get_document_structure to locate relevant content,
then use read_document_section to extract the values.

After gathering information, provide the extracted values."""

        # Run the agentic loop with tool use
        messages = [HumanMessage(content=extraction_prompt)]
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            response = await self.model.ainvoke(messages)
            messages.append(response)

            # Check if there are tool calls
            if not response.tool_calls:
                break

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Find and execute the tool
                tool_result = await self._execute_tool(tool_name, tool_args)

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        # Parse the final response into structured output
        results = await self._parse_extraction_response(response, field_group, document_content)

        extraction_time = int((time.time() - start_time) * 1000)

        return GroupExtractionResult(
            group_id=field_group.group_id,
            results=results,
            model_used=self.model_name,
            extraction_time_ms=extraction_time,
        )

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool execution result
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.ainvoke(tool_args)

        return f"Tool {tool_name} not found"

    async def _parse_extraction_response(
        self,
        response: AIMessage,
        field_group: FieldGroup,
        document_content: dict[str, Any],
    ) -> list[ExtractionResult]:
        """Parse the agent's response into extraction results.

        Args:
            response: Final AI response
            field_group: The field group being extracted
            document_content: Document content for context

        Returns:
            List of extraction results
        """
        # Use a structured output call to parse the response
        structured_model = self.model.with_structured_output(FieldExtractionOutput)

        parse_prompt = f"""Based on your extraction work, provide the final structured results.

Fields to report:
{json.dumps([f.model_dump() for f in field_group.fields], indent=2)}

Your extraction findings:
{response.content}

For each field, provide:
- field_name: exact field name from the list
- extracted_value: the extracted value (or null if not found)
- confidence: your confidence score 0.0-1.0
- source_location: where you found it (e.g., "Page 1, Header section")
- extraction_method: "text", "ocr", "table", or "vision"
- raw_text: the original text snippet containing the value"""

        result = await structured_model.ainvoke(parse_prompt)
        return result.extractions
