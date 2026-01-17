"""Test PDF table extraction with chunking using the provided financial spreadsheet.

This test demonstrates Docling-based chunking for extracting structured data
from a financial planning spreadsheet.
"""

import asyncio
import os
from pathlib import Path

from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state
from dataxtr.schemas.fields import FieldDefinition, FieldType


async def test_financial_extraction():
    """Extract financial data from spreadsheet with table extraction."""

    # The image appears to be a screenshot, let's assume it's saved
    image_path = "financial_spreadsheet.png"

    # Check if we have the image
    if not Path(image_path).exists():
        print(f"⚠️  Image file not found: {image_path}")
        print("Please save the screenshot as 'financial_spreadsheet.png' in the current directory.")
        return

    print("=" * 70)
    print("TESTING FINANCIAL DATA EXTRACTION WITH CHUNKING")
    print("=" * 70)

    # Define comprehensive schema for the financial spreadsheet
    schema = [
        # Top-level summary
        FieldDefinition(
            name="total_house_cost",
            description="Total custo da casa (Total house cost)",
            field_type=FieldType.CURRENCY,
            required=True,
            examples=["R$6,653.04"],
        ),

        # Salary table
        FieldDefinition(
            name="salary_breakdown",
            description="Table of salaries with percentages and contributions. Contains columns: Salários, %, Contribuição",
            field_type=FieldType.TABLE,
            required=True,
            examples=[
                "Table with salary amounts, percentages, and contribution values",
                "Multiple rows showing different salary sources",
            ],
        ),

        # Detailed expenses table
        FieldDefinition(
            name="detailed_expenses",
            description="Detalhado table with all monthly expenses. Contains expense names, amounts, and type (Fixo/Variável)",
            field_type=FieldType.TABLE,
            required=True,
            examples=[
                "Table with expenses like Financiamento AP, Condomínio, Internet, etc.",
                "Each row has expense name, amount in R$, and Fixed or Variable classification",
            ],
        ),

        # Total expenses
        FieldDefinition(
            name="total_expenses",
            description="Total of all expenses (last row of detailed table)",
            field_type=FieldType.CURRENCY,
            required=True,
            examples=["R$6,653.04"],
        ),

        # Reserve projection table
        FieldDefinition(
            name="reserve_projection",
            description="Projeção da reserva guardando o máximo - table showing monthly savings projection from 9/2024 to 2/2025",
            field_type=FieldType.TABLE,
            required=True,
            examples=[
                "Table with months and projected reserve amounts",
                "Rows like 9/2024: R$10,781.00, 10/2024: R$11,896.36, etc.",
            ],
        ),
    ]

    # Test 1: WITH CHUNKING (should work better for tables)
    print("\n🟢 TEST 1: WITH CHUNKING ENABLED")
    print("-" * 70)

    state_with_chunking = create_initial_state(
        document_path=image_path,
        document_type="image",  # Using image type for screenshot
        schema_fields=schema,
        max_iterations=3,
        use_chunking=True,  # ✅ Enable chunking
    )

    graph = build_extraction_graph()

    print("🔄 Starting extraction with chunking...")
    try:
        result = await graph.ainvoke(state_with_chunking)

        print("\n📊 EXTRACTION RESULTS:")
        print("=" * 70)

        if result["workflow_status"] == "completed":
            print("✅ Status: COMPLETED\n")

            final_results = result.get("final_results", {})

            # Display extracted data
            for field_name, value in final_results.items():
                print(f"\n📌 {field_name}:")
                print("-" * 50)

                if isinstance(value, dict):
                    if "rows" in value:
                        # Table data
                        print(f"   Type: TABLE")
                        if "headers" in value:
                            print(f"   Headers: {value['headers']}")
                        rows = value.get("rows", [])
                        print(f"   Rows: {len(rows)}")

                        # Show first few rows
                        for i, row in enumerate(rows[:5]):
                            print(f"   Row {i+1}: {row}")
                        if len(rows) > 5:
                            print(f"   ... and {len(rows) - 5} more rows")
                    else:
                        print(f"   {value}")
                else:
                    print(f"   Value: {value}")

            # Show extraction metadata
            print("\n" + "=" * 70)
            print("📈 EXTRACTION METADATA:")
            print("-" * 70)

            metadata = result.get("document_metadata", {})
            print(f"Document type: {result.get('document_type', 'N/A')}")
            print(f"Chunking enabled: {metadata.get('chunking_enabled', False)}")
            print(f"Has tables: {metadata.get('has_tables', False)}")
            print(f"Has images: {metadata.get('has_images', False)}")

            # Show chunk information
            content = result.get("document_content", {})
            if "chunk_summary" in content:
                summary = content["chunk_summary"]
                print(f"\n🧩 Chunk Summary:")
                print(f"   Total chunks: {summary.get('total', 0)}")
                print(f"   Table chunks: {summary.get('tables', 0)}")
                print(f"   Image chunks: {summary.get('images', 0)}")
                print(f"   Text chunks: {summary.get('text', 0)}")

            # Show quality reports
            quality_reports = result.get("quality_reports", [])
            if quality_reports:
                print(f"\n✨ Quality Reports: {len(quality_reports)}")
                for i, qr in enumerate(quality_reports[:3]):
                    print(f"   Report {i+1}:")
                    print(f"      Group: {qr.group_id}")
                    print(f"      Passed: {qr.passed}")
                    print(f"      Recommendation: {qr.recommendation}")

        else:
            print(f"❌ Status: {result['workflow_status']}")
            errors = result.get("errors", [])
            if errors:
                print("\n🚫 Errors:")
                for error in errors:
                    print(f"   - {error}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


async def test_without_chunking():
    """Test extraction WITHOUT chunking for comparison."""

    image_path = "financial_spreadsheet.png"

    if not Path(image_path).exists():
        print(f"⚠️  Image file not found: {image_path}")
        return

    print("\n" + "=" * 70)
    print("🔵 TEST 2: WITHOUT CHUNKING (for comparison)")
    print("-" * 70)

    # Simple schema for comparison
    schema = [
        FieldDefinition(
            name="detailed_expenses",
            description="Table of detailed monthly expenses",
            field_type=FieldType.TABLE,
            required=True,
        ),
    ]

    state_no_chunking = create_initial_state(
        document_path=image_path,
        document_type="image",
        schema_fields=schema,
        use_chunking=False,  # ❌ Traditional extraction
    )

    graph = build_extraction_graph()

    print("🔄 Starting extraction without chunking...")
    try:
        result = await graph.ainvoke(state_no_chunking)

        print(f"\nStatus: {result['workflow_status']}")

        if result.get("final_results"):
            expenses = result["final_results"].get("detailed_expenses")
            if expenses and isinstance(expenses, dict):
                print(f"Extracted table with {len(expenses.get('rows', []))} rows")

        print("=" * 70)

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Make sure you have API key configured
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: No API keys found!")
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        print("Or use DEFAULT_LLM_PROVIDER=ollama if you have Ollama running.")
        print()

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     FINANCIAL DATA EXTRACTION TEST - DOCLING CHUNKING           ║
    ╚══════════════════════════════════════════════════════════════════╝

    This test will extract structured financial data from a spreadsheet
    using Docling's semantic chunking for better table recognition.

    Expected extractions:
    - Total house cost (R$6,653.04)
    - Salary breakdown table (3 rows)
    - Detailed expenses table (~13 items)
    - Total expenses (R$6,653.04)
    - Reserve projection table (6 months)
    """)

    # Run the test
    asyncio.run(test_financial_extraction())

    # Uncomment to compare with non-chunking approach
    # asyncio.run(test_without_chunking())
