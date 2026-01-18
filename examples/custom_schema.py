"""Example showing how to define custom extraction schemas."""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dataxtr.graph.builder import build_simple_extraction_graph
from dataxtr.graph.state import create_initial_state
from dataxtr.schemas.fields import FieldDefinition, FieldType

load_dotenv()


def create_resume_schema() -> list[FieldDefinition]:
    """Create a schema for extracting resume/CV data."""
    return [
        FieldDefinition(
            name="full_name",
            description="The candidate's full name",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="email",
            description="Email address",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
        ),
        FieldDefinition(
            name="phone",
            description="Phone number",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="location",
            description="City and country of residence",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="summary",
            description="Professional summary or objective statement",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="skills",
            description="List of technical and soft skills",
            field_type=FieldType.LIST,
            required=True,
        ),
        FieldDefinition(
            name="work_experience",
            description="Work history with company names, titles, dates, and descriptions",
            field_type=FieldType.TABLE,
            required=True,
        ),
        FieldDefinition(
            name="education",
            description="Educational background with institutions, degrees, and dates",
            field_type=FieldType.TABLE,
            required=True,
        ),
        FieldDefinition(
            name="certifications",
            description="Professional certifications and licenses",
            field_type=FieldType.LIST,
            required=False,
        ),
    ]


def create_receipt_schema() -> list[FieldDefinition]:
    """Create a schema for extracting receipt data."""
    return [
        FieldDefinition(
            name="merchant_name",
            description="Name of the store or merchant",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="merchant_address",
            description="Store address",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="transaction_date",
            description="Date of the transaction",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="transaction_time",
            description="Time of the transaction",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="items",
            description="List of purchased items with prices",
            field_type=FieldType.TABLE,
            required=True,
        ),
        FieldDefinition(
            name="subtotal",
            description="Subtotal before tax",
            field_type=FieldType.CURRENCY,
            required=False,
        ),
        FieldDefinition(
            name="tax",
            description="Tax amount",
            field_type=FieldType.CURRENCY,
            required=False,
        ),
        FieldDefinition(
            name="total",
            description="Total amount paid",
            field_type=FieldType.CURRENCY,
            required=True,
        ),
        FieldDefinition(
            name="payment_method",
            description="How payment was made (cash, credit card, etc.)",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="last_four_digits",
            description="Last 4 digits of card if paid by card",
            field_type=FieldType.TEXT,
            required=False,
        ),
    ]


def create_contract_schema() -> list[FieldDefinition]:
    """Create a schema for extracting contract data."""
    return [
        FieldDefinition(
            name="contract_title",
            description="Title or type of the contract",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="effective_date",
            description="Date when the contract becomes effective",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="expiration_date",
            description="Date when the contract expires",
            field_type=FieldType.DATE,
            required=False,
        ),
        FieldDefinition(
            name="party_a_name",
            description="Name of the first party",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="party_a_address",
            description="Address of the first party",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="party_b_name",
            description="Name of the second party",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="party_b_address",
            description="Address of the second party",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="contract_value",
            description="Total value or amount of the contract",
            field_type=FieldType.CURRENCY,
            required=False,
        ),
        FieldDefinition(
            name="payment_terms",
            description="Payment schedule and terms",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="governing_law",
            description="Jurisdiction or governing law",
            field_type=FieldType.TEXT,
            required=False,
        ),
    ]


async def extract_with_schema(
    file_path: str,
    schema: list[FieldDefinition],
    doc_type: str = "pdf",
) -> dict:
    """Extract data using a custom schema.

    Args:
        file_path: Path to the document
        schema: List of field definitions
        doc_type: Document type (pdf, image, docx, xlsx)

    Returns:
        Extracted data
    """
    initial_state = create_initial_state(
        document_path=file_path,
        document_type=doc_type,
        schema_fields=schema,
        max_iterations=3,
    )

    # Use simple graph for faster processing
    graph = build_simple_extraction_graph()
    result = await graph.ainvoke(initial_state)

    return result["final_results"]


async def main():
    """Demonstrate custom schema usage."""
    import sys

    print("Available schemas:")
    print("  1. Resume/CV")
    print("  2. Receipt")
    print("  3. Contract")
    print()

    if len(sys.argv) < 3:
        print("Usage: python custom_schema.py <schema_number> <file_path>")
        print("\nExample:")
        print("  python custom_schema.py 1 resume.pdf")
        print("  python custom_schema.py 2 receipt.jpg")
        return

    schema_choice = sys.argv[1]
    file_path = sys.argv[2]

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return

    # Select schema
    schemas = {
        "1": ("Resume", create_resume_schema()),
        "2": ("Receipt", create_receipt_schema()),
        "3": ("Contract", create_contract_schema()),
    }

    if schema_choice not in schemas:
        print(f"Invalid schema choice: {schema_choice}")
        return

    schema_name, schema = schemas[schema_choice]

    # Determine document type from extension
    ext = Path(file_path).suffix.lower()
    doc_type_map = {
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".png": "image",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".csv": "csv",
    }
    doc_type = doc_type_map.get(ext, "pdf")

    print(f"Extracting {schema_name} data from: {file_path}")
    print(f"Document type: {doc_type}")
    print("-" * 50)

    results = await extract_with_schema(file_path, schema, doc_type)

    print("\nExtraction Results:")
    for field_name, data in results.items():
        print(f"\n{field_name}: {data.get('value', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
