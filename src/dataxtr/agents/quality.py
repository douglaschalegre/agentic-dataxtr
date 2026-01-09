"""Quality validation agent using LLM-as-judge pattern."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel

from dataxtr.agents.base import BaseAgent
from dataxtr.schemas.fields import FieldDefinition, FieldGroup
from dataxtr.schemas.quality import QualityReport
from dataxtr.schemas.results import GroupExtractionResult


class QualityAgent(BaseAgent):
    """Agent that validates extractions using LLM-as-judge pattern."""

    SYSTEM_PROMPT = """You are a quality assurance expert for data extraction.
Your task is to validate extracted data against the schema and assess quality.

For each extraction, evaluate:

1. **Completeness** - Are all required fields present with non-null values?

2. **Format validity** - Does the extracted value match the expected type?
   - TEXT: Any string value
   - NUMBER: Valid numeric value
   - DATE: Valid date format
   - CURRENCY: Numeric with optional currency symbol
   - BOOLEAN: true/false or yes/no
   - TABLE: Structured data with rows
   - LIST: Array of values

3. **Confidence calibration** - Is the confidence score appropriate?
   - Low confidence (<0.5) on required fields is a warning
   - Very low confidence (<0.3) on required fields is an error

4. **Consistency** - Are related fields consistent with each other?
   - Dates should be in logical order
   - Totals should match line items
   - Names should be consistent across fields

5. **Plausibility** - Do values make sense in context?
   - Dates not in the future for historical documents
   - Amounts within reasonable ranges
   - No obvious OCR errors

Provide your assessment as a QualityReport with:
- passed: Overall pass/fail (fail if any errors, pass with warnings ok)
- overall_confidence: Weighted average of field confidences
- issues: List of specific problems found
- recommendation: What to do next
  - "accept": Quality is good, use results
  - "retry_same_model": Minor issues, retry might help
  - "retry_different_model": Needs more capable model
  - "manual_review": Cannot be resolved automatically
- reasoning: Your analysis summary"""

    def __init__(self, model: BaseChatModel):
        """Initialize the quality agent.

        Args:
            model: Chat model for validation
        """
        super().__init__(model=model, system_prompt=self.SYSTEM_PROMPT)

    async def execute(
        self,
        extraction_result: GroupExtractionResult,
        field_group: FieldGroup,
        schema_fields: list[FieldDefinition],
    ) -> QualityReport:
        """Validate extraction results.

        Args:
            extraction_result: Results to validate
            field_group: The field group definition
            schema_fields: Full schema for context

        Returns:
            Quality validation report
        """
        return await self.validate(extraction_result, field_group, schema_fields)

    async def validate(
        self,
        extraction_result: GroupExtractionResult,
        field_group: FieldGroup,
        schema_fields: list[FieldDefinition],
    ) -> QualityReport:
        """Validate extraction results and generate quality report.

        Args:
            extraction_result: Results to validate
            field_group: The field group definition
            schema_fields: Full schema for context

        Returns:
            Quality validation report
        """
        # Build validation context
        validation_context = {
            "group_id": extraction_result.group_id,
            "extracted_data": [r.model_dump() for r in extraction_result.results],
            "expected_fields": [f.model_dump() for f in field_group.fields],
            "validation_rules": {
                f.name: f.validation_rules for f in schema_fields if f.validation_rules
            },
            "model_used": extraction_result.model_used,
            "retry_count": extraction_result.retry_count,
            "extraction_time_ms": extraction_result.extraction_time_ms,
        }

        prompt = self._build_prompt()

        # Use structured output for reliable parsing
        structured_model = self.model.with_structured_output(QualityReport)

        result = await structured_model.ainvoke(
            prompt.format(input=json.dumps(validation_context, indent=2))
        )

        # Ensure group_id is set
        result.group_id = extraction_result.group_id

        return result

    def _quick_validation(
        self,
        extraction_result: GroupExtractionResult,
        field_group: FieldGroup,
    ) -> dict[str, Any]:
        """Perform quick programmatic validation before LLM review.

        Args:
            extraction_result: Results to validate
            field_group: The field group definition

        Returns:
            Dict of validation findings
        """
        findings = {
            "missing_required": [],
            "low_confidence": [],
            "null_values": [],
        }

        extracted_names = {r.field_name for r in extraction_result.results}
        expected_names = {f.name for f in field_group.fields}

        # Check for missing fields
        for field in field_group.fields:
            if field.name not in extracted_names and field.required:
                findings["missing_required"].append(field.name)

        # Check extraction quality
        for result in extraction_result.results:
            if result.extracted_value is None:
                field_def = next(
                    (f for f in field_group.fields if f.name == result.field_name),
                    None,
                )
                if field_def and field_def.required:
                    findings["null_values"].append(result.field_name)

            if result.confidence < 0.5:
                findings["low_confidence"].append(
                    {"field": result.field_name, "confidence": result.confidence}
                )

        return findings
