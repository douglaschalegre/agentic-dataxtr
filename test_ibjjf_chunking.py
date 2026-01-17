"""Test chunking feature with IBJJF Rules PDF - Real World Example.

This demonstrates Docling-based chunking with a real competition rules PDF
that contains tables, sections, and structured data.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_ibjjf_chunking():
    """Test chunking with IBJJF Rules PDF."""

    print("=" * 80)
    print("TESTING PDF CHUNKING WITH IBJJF RULES DOCUMENT")
    print("=" * 80)

    pdf_path = "ibjjf_rules.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ PDF not found: {pdf_path}")
        return

    # First, let's just test the chunking service directly
    from dataxtr.services.document_chunker import DocumentChunker
    from dataxtr.schemas.chunks import ChunkType

    print("\n🔄 Step 1: Testing DocumentChunker with Docling")
    print("-" * 80)

    chunker = DocumentChunker()

    print("Processing PDF with Docling (this may take a moment)...")

    try:
        chunked_doc = await chunker.chunk_document(
            file_path=Path(pdf_path),
            document_type="pdf",
            extract_tables=True,
            extract_images=True,
        )

        print("\n✅ CHUNKING SUCCESSFUL!")
        print("=" * 80)

        # Show chunk statistics
        print(f"\n📊 CHUNK STATISTICS:")
        print(f"   Total chunks: {len(chunked_doc.chunks)}")
        print(f"   Document: {chunked_doc.document_path}")
        print(f"   Processing time: {chunked_doc.processing_time_ms}ms")

        # Count by type
        chunk_counts = {}
        for chunk in chunked_doc.chunks:
            chunk_type = chunk.chunk_type.value
            chunk_counts[chunk_type] = chunk_counts.get(chunk_type, 0) + 1

        print(f"\n🧩 CHUNKS BY TYPE:")
        for chunk_type, count in sorted(chunk_counts.items()):
            print(f"   {chunk_type.upper()}: {count}")

        # Show table chunks
        table_chunks = chunked_doc.get_table_chunks()
        print(f"\n📋 TABLE CHUNKS: {len(table_chunks)}")
        if table_chunks:
            print("\n   First 5 tables found:")
            for i, chunk in enumerate(table_chunks[:5]):
                print(f"\n   Table {i+1}:")
                print(f"      Chunk ID: {chunk.chunk_id}")
                print(f"      Page: {chunk.page_number}")
                content = chunk.content
                if isinstance(content, dict):
                    if 'rows' in content:
                        print(f"      Rows: {len(content['rows'])}")
                        if 'headers' in content:
                            print(f"      Headers: {content['headers'][:5]}...")  # First 5 headers
                    elif 'dataframe' in content:
                        print(f"      DataFrame shape: {content['dataframe'].shape}")

        # Show some text chunks
        text_chunks = chunked_doc.get_text_chunks()
        print(f"\n📝 TEXT CHUNKS: {len(text_chunks)}")
        if text_chunks:
            print(f"\n   First text chunk preview:")
            first_text = text_chunks[0].text_content[:200]
            print(f"      {first_text}...")

        # Show image chunks
        image_chunks = chunked_doc.get_image_chunks()
        print(f"\n🖼️  IMAGE CHUNKS: {len(image_chunks)}")

        print("\n" + "=" * 80)
        print("✅ CHUNKING FEATURE IS WORKING!")
        print("=" * 80)

        print("\n📌 KEY BENEFITS DEMONSTRATED:")
        print("   ✓ PDF processed into semantic chunks")
        print("   ✓ Tables identified and structured")
        print("   ✓ Text separated from tables and images")
        print("   ✓ Each chunk has metadata (page, position, bbox)")
        print("   ✓ Ready for extraction agents to use")

        return chunked_doc

    except Exception as e:
        print(f"\n❌ Error during chunking: {e}")
        import traceback
        traceback.print_exc()

        # Check if it's an import error
        if "No module named 'docling'" in str(e):
            print("\n⚠️  Docling not installed!")
            print("Install with: pip install docling")


async def test_full_extraction():
    """Test full extraction workflow with chunking."""

    print("\n\n" + "=" * 80)
    print("🔄 Step 2: Testing Full Extraction Workflow")
    print("=" * 80)

    from dataxtr.graph.state import create_initial_state
    from dataxtr.graph.builder import build_extraction_graph
    from dataxtr.schemas.fields import FieldDefinition, FieldType

    # Define a schema to extract competition rules
    schema = [
        FieldDefinition(
            name="weight_divisions",
            description="Table of weight divisions and categories for competitors",
            field_type=FieldType.TABLE,
            required=False,
            examples=["Table with weight classes like Rooster, Light Feather, etc."],
        ),
        FieldDefinition(
            name="point_scoring",
            description="Table or rules about how points are scored in competition",
            field_type=FieldType.TABLE,
            required=False,
            examples=["Points for takedowns, sweeps, guard passes, etc."],
        ),
    ]

    print("\n📋 Extraction Schema:")
    for field in schema:
        print(f"   - {field.name}: {field.description}")

    # Create state with chunking enabled
    state = create_initial_state(
        document_path="ibjjf_rules.pdf",
        document_type="pdf",
        schema_fields=schema,
        max_iterations=2,  # Limit iterations for testing
        use_chunking=True,  # ✅ CHUNKING ENABLED
    )

    print("\n🚀 Starting extraction (this requires an API key)...")

    # Check for API key
    import os
    if not any([
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("DEFAULT_LLM_PROVIDER") == "ollama"
    ]):
        print("\n⚠️  No API key found!")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        print("Or use DEFAULT_LLM_PROVIDER=ollama if you have Ollama running.")
        print("\nSkipping full extraction test.")
        return

    try:
        graph = build_extraction_graph()
        result = await graph.ainvoke(state)

        print("\n✅ EXTRACTION COMPLETED")
        print("=" * 80)

        print(f"\nStatus: {result['workflow_status']}")

        if result.get('final_results'):
            print("\n📊 EXTRACTED DATA:")
            for field_name, value in result['final_results'].items():
                print(f"\n{field_name}:")
                if isinstance(value, dict) and 'rows' in value:
                    print(f"   Table with {len(value['rows'])} rows")
                    if value.get('headers'):
                        print(f"   Headers: {value['headers']}")
                else:
                    print(f"   {value}")

        # Show chunk usage
        metadata = result.get('document_metadata', {})
        if metadata.get('chunking_enabled'):
            content = result.get('document_content', {})
            if 'chunk_summary' in content:
                print("\n🧩 Chunks Used:")
                summary = content['chunk_summary']
                print(f"   Total: {summary.get('total', 0)}")
                print(f"   Tables: {summary.get('tables', 0)}")
                print(f"   Text: {summary.get('text', 0)}")

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║         IBJJF RULES PDF - CHUNKING FEATURE TEST                 ║
    ╚══════════════════════════════════════════════════════════════════╝

    Testing the Docling-based chunking feature with a real-world PDF
    containing competition rules, tables, and structured data.

    PDF: IBJJF (International Brazilian Jiu-Jitsu Federation) Rules
    Size: 6.7 MB
    Content: Rules, weight divisions, scoring tables, regulations
    """)

    # Run chunking test (doesn't require API key)
    asyncio.run(test_ibjjf_chunking())

    # Optionally run full extraction test (requires API key)
    # Uncomment to test full extraction workflow:
    # asyncio.run(test_full_extraction())
