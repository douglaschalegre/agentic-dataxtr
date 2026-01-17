"""Example demonstrating Docling-based chunking for better PDF table extraction.

This example shows how enabling chunking improves table extraction from PDFs
by using Docling to semantically chunk the document into text, tables, and images.
"""

import asyncio
import os
from pathlib import Path

from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state
from dataxtr.schemas.fields import FieldDefinition, FieldType


async def main():
    """Extract data from a PDF with tables using chunking."""
    # Sample PDF path - replace with your PDF containing tables
    pdf_path = "sample_invoice.pdf"

    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        print("Please provide a PDF file with tables to extract from.")
        return

    # Define extraction schema for invoice with table data
    schema = [
        FieldDefinition(
            name="invoice_number",
            description="Invoice number or ID",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="invoice_date",
            description="Date of the invoice",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="vendor_name",
            description="Name of the vendor or company",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="line_items",
            description="Table of line items with description, quantity, and price",
            field_type=FieldType.TABLE,
            required=True,
            examples=[
                "Table with columns: Item, Description, Quantity, Unit Price, Total",
                "Product listing with quantities and prices",
            ],
        ),
        FieldDefinition(
            name="total_amount",
            description="Total amount of the invoice",
            field_type=FieldType.CURRENCY,
            required=True,
        ),
    ]

    # Create initial state WITH CHUNKING ENABLED
    # This enables Docling-based semantic chunking for better table extraction
    print("🔄 Creating extraction workflow with chunking enabled...")
    initial_state = create_initial_state(
        document_path=pdf_path,
        document_type="pdf",
        schema_fields=schema,
        max_iterations=3,
        use_chunking=True,  # ⭐ Enable chunking for better table extraction
    )

    # Build and run the extraction graph
    print("🚀 Starting extraction process...")
    graph = build_extraction_graph()

    try:
        final_state = await graph.ainvoke(initial_state)

        # Display results
        print("\n" + "=" * 60)
        print("📊 EXTRACTION RESULTS")
        print("=" * 60)

        if final_state["workflow_status"] == "completed":
            results = final_state.get("final_results", {})

            print(f"\n✅ Extraction completed successfully!\n")

            for field_name, value in results.items():
                print(f"📌 {field_name}:")
                if isinstance(value, dict) and "rows" in value:
                    # Table data
                    print(f"   Type: Table ({len(value.get('rows', []))} rows)")
                    if "headers" in value:
                        print(f"   Headers: {value['headers']}")
                    if value.get("rows"):
                        print(f"   First row: {value['rows'][0]}")
                else:
                    print(f"   {value}")
                print()

            # Show chunk information if available
            if "document_metadata" in final_state:
                metadata = final_state["document_metadata"]
                if metadata.get("chunking_enabled"):
                    print("\n🧩 CHUNKING INFORMATION:")
                    content = final_state.get("document_content", {})
                    if "chunk_summary" in content:
                        summary = content["chunk_summary"]
                        print(f"   Total chunks: {summary.get('total', 0)}")
                        print(f"   Table chunks: {summary.get('tables', 0)}")
                        print(f"   Image chunks: {summary.get('images', 0)}")
                        print(f"   Text chunks: {summary.get('text', 0)}")

        else:
            print(f"\n❌ Extraction failed!")
            errors = final_state.get("errors", [])
            for error in errors:
                print(f"   Error: {error}")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback

        traceback.print_exc()


async def compare_with_without_chunking():
    """Compare extraction results with and without chunking.

    This demonstrates the difference between traditional extraction
    and chunking-based extraction.
    """
    pdf_path = "sample_invoice.pdf"

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        return

    # Simple schema for comparison
    schema = [
        FieldDefinition(
            name="line_items_table",
            description="Table of line items",
            field_type=FieldType.TABLE,
            required=True,
        )
    ]

    print("=" * 60)
    print("COMPARISON: Traditional vs Chunking-Based Extraction")
    print("=" * 60)

    # WITHOUT CHUNKING
    print("\n🔵 Test 1: WITHOUT chunking (traditional)")
    state_no_chunking = create_initial_state(
        document_path=pdf_path,
        document_type="pdf",
        schema_fields=schema,
        use_chunking=False,  # ❌ Traditional extraction
    )

    graph = build_extraction_graph()
    result_no_chunking = await graph.ainvoke(state_no_chunking)

    # WITH CHUNKING
    print("\n🟢 Test 2: WITH chunking (Docling-based)")
    state_with_chunking = create_initial_state(
        document_path=pdf_path,
        document_type="pdf",
        schema_fields=schema,
        use_chunking=True,  # ✅ Chunking enabled
    )

    result_with_chunking = await graph.ainvoke(state_with_chunking)

    # Compare results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)

    print("\n❌ Without chunking:")
    print(f"   Status: {result_no_chunking['workflow_status']}")
    if result_no_chunking.get("final_results"):
        table_data = result_no_chunking["final_results"].get("line_items_table")
        if table_data and isinstance(table_data, dict):
            print(f"   Rows extracted: {len(table_data.get('rows', []))}")

    print("\n✅ With chunking:")
    print(f"   Status: {result_with_chunking['workflow_status']}")
    if result_with_chunking.get("final_results"):
        table_data = result_with_chunking["final_results"].get("line_items_table")
        if table_data and isinstance(table_data, dict):
            print(f"   Rows extracted: {len(table_data.get('rows', []))}")

    # Show chunk info
    if result_with_chunking.get("document_content", {}).get("chunk_summary"):
        summary = result_with_chunking["document_content"]["chunk_summary"]
        print(f"\n   Chunks created:")
        print(f"      - Tables: {summary.get('tables', 0)}")
        print(f"      - Text: {summary.get('text', 0)}")
        print(f"      - Images: {summary.get('images', 0)}")


if __name__ == "__main__":
    # Set up environment (optional)
    # os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    print("=" * 60)
    print("PDF TABLE EXTRACTION WITH CHUNKING")
    print("Powered by Docling + LangGraph")
    print("=" * 60)

    # Run basic example
    asyncio.run(main())

    # Uncomment to run comparison
    # asyncio.run(compare_with_without_chunking())
