"""Field preparation agent for grouping and strategy assignment."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from dataxtr.agents.base import BaseAgent
from dataxtr.schemas.fields import (
    ExtractionComplexity,
    FieldDefinition,
    FieldGroup,
)


class FieldGroupingOutput(BaseModel):
    """Structured output for field grouping."""

    # Some providers return only field names (strings) here; the agent normalizes later.
    groups: list[dict[str, Any]] = Field(description="List of field groups")
    reasoning: str = Field(description="Explanation of grouping decisions")


class FieldPrepAgent(BaseAgent):
    """Agent that analyzes schema and groups fields for extraction."""

    SYSTEM_PROMPT = """You are a document analysis expert. Your task is to analyze
a list of fields to extract and group them logically based on:

1. Semantic context (personal info, financial data, dates, addresses)
2. Document location (fields likely found together)
3. Extraction complexity (simple text vs tables vs handwritten/visual)

For each group, determine the extraction strategy:
- "simple": Plain text fields that can be extracted with basic LLM (names, dates, simple values)
- "complex": Fields requiring reasoning, calculations, or interpretation
- "visual": Fields requiring OCR, image analysis, or handwritten recognition

Output field groups with:
- A unique group_id (e.g., "personal_info", "financial_data")
- A descriptive group_name
- The fields belonging to that group
- Recommended extraction_strategy (simple/complex/visual)
- A context_hint about where in the document to look
- Any dependencies on other groups (if extraction order matters)

Be efficient - group related fields together to minimize API calls."""

    def __init__(self, model: BaseChatModel):
        """Initialize the field preparation agent.

        Args:
            model: Chat model for analysis
        """
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT)

    async def execute(
        self,
        schema_fields: list[FieldDefinition],
        document_metadata: dict[str, Any],
        document_content: dict[str, Any],
    ) -> list[FieldGroup]:
        """Analyze fields and create logical groupings.

        Args:
            schema_fields: List of fields to extract
            document_metadata: Metadata about the document
            document_content: Parsed document content

        Returns:
            List of field groups with extraction strategies
        """
        return await self.analyze_and_group(schema_fields, document_metadata, document_content)

    async def analyze_and_group(
        self,
        schema_fields: list[FieldDefinition],
        document_metadata: dict[str, Any],
        document_content: dict[str, Any],
    ) -> list[FieldGroup]:
        """Analyze fields and create logical groupings.

        Args:
            schema_fields: List of fields to extract
            document_metadata: Metadata about the document
            document_content: Parsed document content

        Returns:
            List of field groups with extraction strategies
        """
        prompt = self._build_prompt()

        # Use structured output for reliable parsing
        structured_model = self.model.with_structured_output(FieldGroupingOutput)

        # Build input context
        input_context = {
            "fields": [f.model_dump() for f in schema_fields],
            "document_info": {
                "page_count": document_metadata.get("page_count"),
                "has_tables": document_metadata.get("has_tables"),
                "has_images": document_metadata.get("has_images"),
                "sections": document_content.get("sections", [])[:10],  # Limit
            },
        }

        result = await structured_model.ainvoke(
            prompt.format(input=json.dumps(input_context, indent=2))
        )

        # Gemini via Antigravity may return only field names (strings) instead of full
        # FieldDefinition objects. Fall back to mapping those names back onto the
        # provided schema.
        if isinstance(result, FieldGroupingOutput):
            output = result
        elif isinstance(result, dict):
            output = FieldGroupingOutput.model_validate(result)
        else:
            output = FieldGroupingOutput.model_validate(result)

        schema_by_name = {f.name: f for f in schema_fields}
        normalized_groups: list[FieldGroup] = []
        for g in output.groups:
            group_id = str(g.get("group_id", "default"))
            group_name = str(g.get("group_name", group_id))
            extraction_strategy = g.get("extraction_strategy", ExtractionComplexity.COMPLEX)
            context_hint = str(g.get("context_hint", ""))

            raw_fields = g.get("fields", [])
            fields: list[FieldDefinition] = []
            for item in raw_fields:
                if isinstance(item, FieldDefinition):
                    fields.append(item)
                elif isinstance(item, str):
                    if item in schema_by_name:
                        fields.append(schema_by_name[item])
                elif isinstance(item, dict) and "name" in item:
                    name = str(item.get("name"))
                    if name in schema_by_name:
                        fields.append(schema_by_name[name])
                    else:
                        fields.append(FieldDefinition.model_validate(item))
                else:
                    fields.append(FieldDefinition.model_validate(item))

            deps_any: Any = g.get("dependencies", [])
            deps: list[str]
            if deps_any is None:
                deps = []
            elif isinstance(deps_any, str):
                deps = [deps_any]
            elif isinstance(deps_any, list):
                deps = [str(d) for d in deps_any]
            else:
                deps = [str(deps_any)]

            normalized_groups.append(
                FieldGroup(
                    group_id=group_id,
                    group_name=group_name,
                    fields=fields,
                    extraction_strategy=extraction_strategy,
                    context_hint=context_hint,
                    dependencies=deps,
                )
            )

        return normalized_groups

    def _create_default_grouping(self, schema_fields: list[FieldDefinition]) -> list[FieldGroup]:
        """Create a default grouping if LLM fails.

        Args:
            schema_fields: List of fields to group

        Returns:
            Single group containing all fields
        """
        return [
            FieldGroup(
                group_id="default",
                group_name="All Fields",
                fields=schema_fields,
                extraction_strategy=ExtractionComplexity.COMPLEX,
                context_hint="Search entire document",
            )
        ]
