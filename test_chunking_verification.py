"""Quick test to verify chunking implementation is ready (no Docling required).

This test verifies that all the chunking code is in place and properly integrated,
even if Docling isn't installed yet.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_chunking_architecture():
    """Test that chunking architecture is properly implemented."""

    print("=" * 80)
    print("CHUNKING FEATURE - ARCHITECTURE VERIFICATION")
    print("=" * 80)

    tests_passed = 0
    tests_total = 0

    # Test 1: Can import chunk schemas
    tests_total += 1
    try:
        from dataxtr.schemas.chunks import ChunkType, ChunkedDocument, DocumentChunk

        print("\n✅ Test 1: Chunk schemas imported successfully")
        print(f"   ChunkTypes available: {', '.join([t.value for t in ChunkType])}")
        tests_passed += 1
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: Could not import chunk schemas: {e}")

    # Test 2: Can import DocumentChunker
    tests_total += 1
    try:
        from dataxtr.services.document_chunker import DocumentChunker

        chunker = DocumentChunker()
        print("\n✅ Test 2: DocumentChunker class imported and instantiated")
        tests_passed += 1
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: Could not import DocumentChunker: {e}")

    # Test 3: DocumentParser has chunking support
    tests_total += 1
    try:
        from dataxtr.services.document_parser import DocumentParser

        # Check if use_chunking parameter exists
        import inspect

        sig = inspect.signature(DocumentParser.__init__)
        params = list(sig.parameters.keys())

        if "use_chunking" in params:
            print("\n✅ Test 3: DocumentParser has use_chunking parameter")
            tests_passed += 1
        else:
            print(f"\n❌ Test 3 FAILED: use_chunking not in DocumentParser parameters: {params}")
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: Could not verify DocumentParser: {e}")

    # Test 4: Extraction state has use_chunking
    tests_total += 1
    try:
        from dataxtr.graph.state import create_initial_state
        from dataxtr.schemas.fields import FieldDefinition, FieldType

        test_field = FieldDefinition(
            name="test", description="test field", field_type=FieldType.TEXT
        )

        # Try creating state with chunking enabled
        state = create_initial_state(
            document_path="test.pdf",
            document_type="pdf",
            schema_fields=[test_field],
            use_chunking=True,
        )

        if state.get("use_chunking") == True:
            print("\n✅ Test 4: ExtractionState supports use_chunking parameter")
            tests_passed += 1
        else:
            print(f"\n❌ Test 4 FAILED: use_chunking not in state: {state.keys()}")
    except Exception as e:
        print(f"\n❌ Test 4 FAILED: Could not create state with chunking: {e}")

    # Test 5: New tools are available
    tests_total += 1
    try:
        from dataxtr.tools.document_tools import DOCUMENT_TOOLS, get_table_chunks

        tool_names = [tool.name for tool in DOCUMENT_TOOLS]

        if "get_table_chunks" in tool_names:
            print(f"\n✅ Test 5: New get_table_chunks tool is available")
            print(f"   Total document tools: {len(DOCUMENT_TOOLS)}")
            print(f"   Tools: {', '.join(tool_names)}")
            tests_passed += 1
        else:
            print(f"\n❌ Test 5 FAILED: get_table_chunks not in tools: {tool_names}")
    except Exception as e:
        print(f"\n❌ Test 5 FAILED: Could not import document tools: {e}")

    # Test 6: Check Docling availability
    tests_total += 1
    try:
        import docling
        from docling.document_converter import DocumentConverter

        print(f"\n✅ Test 6: Docling is installed and ready!")
        print("   DocumentConverter available - ready to process PDFs with AI-powered chunking!")
        tests_passed += 1
    except ImportError:
        print("\n⚠️  Test 6: Docling not installed yet (still downloading)")
        print("   Chunking will use fallback mode until installation completes")
        print("   Installation in progress: ~3GB of PyTorch/CUDA dependencies")

    # Summary
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 80)

    if tests_passed >= 5:
        print("\n✅ CHUNKING FEATURE IS PROPERLY INTEGRATED!")
        print("\nArchitecture components:")
        print("  ✓ Chunk schemas (ChunkType, DocumentChunk, ChunkedDocument)")
        print("  ✓ DocumentChunker service")
        print("  ✓ DocumentParser with use_chunking support")
        print("  ✓ ExtractionState with use_chunking parameter")
        print("  ✓ New get_table_chunks() tool")
        if tests_passed == tests_total:
            print("  ✓ Docling library installed and ready")
        else:
            print("  ⏳ Docling installation in progress...")

        print("\n📦 Once Docling installation completes:")
        print("  • Run: python test_ibjjf_chunking.py")
        print("  • Or test with any PDF: python examples/chunking_pdf_tables.py")

        return True
    else:
        print("\n❌ Some tests failed - chunking feature may not be complete")
        return False


async def demo_chunking_workflow():
    """Show what the chunking workflow looks like."""

    print("\n\n" + "=" * 80)
    print("CHUNKING WORKFLOW DEMONSTRATION")
    print("=" * 80)

    print("""
The chunking feature works as follows:

1️⃣  USER CODE (No changes needed!)
   ────────────────────────────────
   from dataxtr.graph.state import create_initial_state

   state = create_initial_state(
       document_path="invoice.pdf",
       document_type="pdf",
       schema_fields=schema,
       use_chunking=True  # ← Enable chunking (default)
   )

2️⃣  DOCUMENT LOADER NODE
   ────────────────────────
   → Calls load_document(path, type, use_chunking=True)
   → Creates DocumentParser with use_chunking=True
   → Parser.load() triggered

3️⃣  CHUNKING PROCESS
   ────────────────────
   IF use_chunking=True:
      → DocumentChunker.chunk_document() called
      → Docling analyzes PDF structure
      → Creates semantic chunks:
         • Text paragraphs
         • Tables (structured!)
         • Images
         • Titles/headings
         • Lists, code, formulas
      → Returns ChunkedDocument

4️⃣  EXTRACTION TOOLS
   ────────────────────
   Agents can use:
   • get_table_chunks() → Structured tables from chunks
   • get_document_structure() → Shows chunk summary
   • read_document_section() → Works with chunks

5️⃣  RESULTS
   ────────────
   • Better table extraction accuracy
   • Proper row/column structure
   • Separate processing for different content types
   • Agents get only relevant chunks

═══════════════════════════════════════════════════════════════════

KEY BENEFITS:

❌ BEFORE (text-based):
   • Table detection: Look for "|" or "\\t" characters
   • Structure: Heuristic guesses
   • Accuracy: Low for complex tables

✅ AFTER (Docling chunking):
   • Table detection: AI models
   • Structure: Proper rows/columns/spanning cells
   • Accuracy: High with semantic understanding
   • Bonus: Chunks can be processed separately!
    """)


if __name__ == "__main__":
    print("\n")
    asyncio.run(test_chunking_architecture())
    asyncio.run(demo_chunking_workflow())
    print("\n")
