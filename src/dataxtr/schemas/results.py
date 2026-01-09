"""Extraction result schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Result from extracting a single field."""

    field_name: str = Field(description="Name of the extracted field")
    extracted_value: Any = Field(description="The extracted value")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    source_location: Optional[str] = Field(
        default=None, description="Page/section reference where value was found"
    )
    extraction_method: str = Field(
        default="text", description="Method used: text, ocr, table, vision"
    )
    raw_text: Optional[str] = Field(
        default=None, description="Original text snippet containing the value"
    )


class GroupExtractionResult(BaseModel):
    """Results from extracting an entire field group."""

    group_id: str = Field(description="ID of the field group")
    results: list[ExtractionResult] = Field(
        default_factory=list, description="Extraction results for each field"
    )
    model_used: str = Field(description="Model identifier used for extraction")
    extraction_time_ms: int = Field(
        default=0, description="Time taken for extraction in milliseconds"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
