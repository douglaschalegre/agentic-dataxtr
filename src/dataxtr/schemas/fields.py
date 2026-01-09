"""Field definition and grouping schemas."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Types of fields that can be extracted."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    BOOLEAN = "boolean"
    TABLE = "table"
    LIST = "list"


class ExtractionComplexity(str, Enum):
    """Complexity level determining model selection."""

    SIMPLE = "simple"  # Fast/cheap model (Haiku, GPT-3.5-turbo)
    COMPLEX = "complex"  # Powerful model (Sonnet, GPT-4)
    VISUAL = "visual"  # Vision model (Claude Vision, GPT-4V)


class FieldDefinition(BaseModel):
    """Schema definition for a field to extract."""

    name: str = Field(description="Unique field identifier")
    description: str = Field(description="Human description of what to extract")
    field_type: FieldType = Field(default=FieldType.TEXT)
    required: bool = Field(default=True)
    validation_rules: Optional[dict] = Field(default=None, description="Custom validation rules")
    examples: Optional[list[str]] = Field(default=None, description="Example values for this field")


class FieldGroup(BaseModel):
    """A logical grouping of related fields."""

    group_id: str = Field(description="Unique group identifier")
    group_name: str = Field(description="Human-readable group name")
    fields: list[FieldDefinition] = Field(description="Fields in this group")
    extraction_strategy: ExtractionComplexity = Field(
        default=ExtractionComplexity.SIMPLE,
        description="Complexity level for model selection",
    )
    context_hint: str = Field(
        default="", description="Where in the document to look for these fields"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Other group_ids that must be extracted first",
    )
