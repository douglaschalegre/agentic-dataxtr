"""Quality validation schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class QualityIssue(BaseModel):
    """Issue identified during quality validation."""

    field_name: str = Field(description="Name of the field with the issue")
    issue_type: Literal[
        "missing", "invalid_format", "low_confidence", "inconsistent", "out_of_range"
    ] = Field(description="Type of quality issue")
    severity: Literal["error", "warning"] = Field(
        description="Severity level of the issue"
    )
    message: str = Field(description="Human-readable description of the issue")
    suggested_fix: Optional[str] = Field(
        default=None, description="Suggested action to fix the issue"
    )


class QualityReport(BaseModel):
    """Quality assessment report from the quality agent."""

    group_id: str = Field(description="ID of the field group being validated")
    passed: bool = Field(description="Whether the extraction passed quality checks")
    overall_confidence: float = Field(
        ge=0.0, le=1.0, description="Aggregate confidence score"
    )
    issues: list[QualityIssue] = Field(
        default_factory=list, description="List of identified issues"
    )
    recommendation: Literal[
        "accept", "retry_same_model", "retry_different_model", "manual_review"
    ] = Field(description="Recommended action based on quality assessment")
    reasoning: str = Field(
        description="Explanation for the quality assessment and recommendation"
    )
