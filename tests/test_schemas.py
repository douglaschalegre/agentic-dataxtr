"""Tests for schema definitions."""

import pytest

from dataxtr.schemas.fields import (
    ExtractionComplexity,
    FieldDefinition,
    FieldGroup,
    FieldType,
)
from dataxtr.schemas.quality import QualityIssue, QualityReport
from dataxtr.schemas.results import ExtractionResult, GroupExtractionResult


class TestFieldDefinition:
    """Tests for FieldDefinition schema."""

    def test_create_basic_field(self):
        """Test creating a basic field definition."""
        field = FieldDefinition(
            name="test_field",
            description="A test field",
        )

        assert field.name == "test_field"
        assert field.description == "A test field"
        assert field.field_type == FieldType.TEXT
        assert field.required is True

    def test_create_field_with_all_options(self):
        """Test creating a field with all options."""
        field = FieldDefinition(
            name="amount",
            description="Total amount",
            field_type=FieldType.CURRENCY,
            required=False,
            validation_rules={"min": 0},
            examples=["$100.00", "$250.50"],
        )

        assert field.field_type == FieldType.CURRENCY
        assert field.required is False
        assert field.validation_rules == {"min": 0}
        assert len(field.examples) == 2


class TestFieldGroup:
    """Tests for FieldGroup schema."""

    def test_create_field_group(self, sample_invoice_schema):
        """Test creating a field group."""
        group = FieldGroup(
            group_id="invoice_basics",
            group_name="Invoice Basics",
            fields=sample_invoice_schema,
            extraction_strategy=ExtractionComplexity.SIMPLE,
            context_hint="First page header",
        )

        assert group.group_id == "invoice_basics"
        assert len(group.fields) == 2
        assert group.extraction_strategy == ExtractionComplexity.SIMPLE


class TestExtractionResult:
    """Tests for extraction result schemas."""

    def test_create_extraction_result(self):
        """Test creating an extraction result."""
        result = ExtractionResult(
            field_name="invoice_number",
            extracted_value="INV-001",
            confidence=0.95,
            source_location="Page 1, Header",
            extraction_method="text",
            raw_text="Invoice Number: INV-001",
        )

        assert result.field_name == "invoice_number"
        assert result.confidence == 0.95

    def test_confidence_bounds(self):
        """Test that confidence is bounded 0-1."""
        with pytest.raises(ValueError):
            ExtractionResult(
                field_name="test",
                extracted_value="value",
                confidence=1.5,  # Invalid
            )


class TestQualityReport:
    """Tests for quality report schema."""

    def test_create_quality_report(self):
        """Test creating a quality report."""
        report = QualityReport(
            group_id="test_group",
            passed=True,
            overall_confidence=0.9,
            issues=[],
            recommendation="accept",
            reasoning="All fields extracted with high confidence",
        )

        assert report.passed is True
        assert report.recommendation == "accept"

    def test_quality_report_with_issues(self):
        """Test creating a report with issues."""
        issue = QualityIssue(
            field_name="amount",
            issue_type="low_confidence",
            severity="warning",
            message="Confidence below threshold",
            suggested_fix="Retry with OCR",
        )

        report = QualityReport(
            group_id="test_group",
            passed=True,  # Warnings don't fail
            overall_confidence=0.6,
            issues=[issue],
            recommendation="retry_same_model",
            reasoning="Low confidence on amount field",
        )

        assert len(report.issues) == 1
        assert report.issues[0].severity == "warning"
