"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_invoice_schema():
    """Sample invoice extraction schema for testing."""
    from dataxtr.schemas.fields import FieldDefinition, FieldType

    return [
        FieldDefinition(
            name="invoice_number",
            description="The invoice number",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="total_amount",
            description="Total invoice amount",
            field_type=FieldType.CURRENCY,
            required=True,
        ),
    ]


@pytest.fixture
def sample_field_group():
    """Sample field group for testing."""
    from dataxtr.schemas.fields import (
        ExtractionComplexity,
        FieldDefinition,
        FieldGroup,
        FieldType,
    )

    return FieldGroup(
        group_id="test_group",
        group_name="Test Group",
        fields=[
            FieldDefinition(
                name="test_field",
                description="A test field",
                field_type=FieldType.TEXT,
                required=True,
            ),
        ],
        extraction_strategy=ExtractionComplexity.SIMPLE,
        context_hint="Test context",
    )


@pytest.fixture
def sample_extraction_result():
    """Sample extraction result for testing."""
    from dataxtr.schemas.results import ExtractionResult, GroupExtractionResult

    return GroupExtractionResult(
        group_id="test_group",
        results=[
            ExtractionResult(
                field_name="test_field",
                extracted_value="test_value",
                confidence=0.95,
                source_location="Page 1",
                extraction_method="text",
            ),
        ],
        model_used="test-model",
        extraction_time_ms=100,
    )
