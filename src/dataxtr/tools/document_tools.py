"""Document interaction tools for LangChain agents."""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from dataxtr.services.document_parser import DocumentParser
from dataxtr.services.ocr_service import OCRService
from dataxtr.services.table_extractor import TableExtractor


# Tool input schemas
class ReadSectionInput(BaseModel):
    """Input for reading a document section."""

    page_numbers: list[int] = Field(description="Page numbers to read (1-indexed)")
    section_hint: Optional[str] = Field(
        default=None, description="Section name or header to look for"
    )


class OCRRegionInput(BaseModel):
    """Input for OCR on specific region."""

    page_number: int = Field(description="Page number (1-indexed)")
    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None,
        description="Bounding box as (x1, y1, x2, y2) in normalized coords 0-1",
    )


class TableExtractionInput(BaseModel):
    """Input for extracting tables."""

    page_number: int = Field(description="Page number containing the table")
    table_index: int = Field(
        default=0, description="Index of table on page if multiple"
    )


class SearchDocumentInput(BaseModel):
    """Input for searching document content."""

    query: str = Field(description="Text or pattern to search for")
    context_chars: int = Field(
        default=200, description="Characters of context around match"
    )


@tool(args_schema=ReadSectionInput)
async def read_document_section(
    page_numbers: list[int], section_hint: Optional[str] = None
) -> str:
    """Read text content from specific pages or sections of the document.

    Use this to get raw text from known locations in the document.

    Args:
        page_numbers: List of page numbers to read (1-indexed)
        section_hint: Optional section name to look for

    Returns:
        Extracted text content from the specified pages/sections
    """
    parser = DocumentParser.get_current()

    if section_hint:
        return await parser.find_section(section_hint, page_numbers)
    else:
        return await parser.read_pages(page_numbers)


@tool(args_schema=OCRRegionInput)
async def ocr_region(
    page_number: int, bbox: Optional[tuple[float, float, float, float]] = None
) -> str:
    """Perform OCR on a specific page or region of the document.

    Use this for scanned documents or when text extraction fails.

    Args:
        page_number: Page number to OCR (1-indexed)
        bbox: Optional bounding box for specific region (normalized 0-1 coords)

    Returns:
        OCR-extracted text
    """
    parser = DocumentParser.get_current()
    ocr = OCRService()

    image = await parser.get_page_image(page_number)
    return await ocr.extract_text(image, bbox)


@tool(args_schema=TableExtractionInput)
async def extract_table(page_number: int, table_index: int = 0) -> dict:
    """Extract structured table data from a document page.

    Returns table as dictionary with headers and rows.

    Args:
        page_number: Page containing the table (1-indexed)
        table_index: Which table on the page (0-indexed)

    Returns:
        Dictionary with 'headers' and 'rows' keys
    """
    parser = DocumentParser.get_current()
    extractor = TableExtractor()

    page_content = await parser.read_pages([page_number])
    tables = await extractor.find_tables(page_content)

    if table_index >= len(tables):
        return {"error": f"Only {len(tables)} tables found on page {page_number}"}

    table = tables[table_index]
    return table.to_dict()


@tool(args_schema=SearchDocumentInput)
async def search_document(query: str, context_chars: int = 200) -> list[dict]:
    """Search the document for specific text or patterns.

    Returns matching locations with surrounding context.

    Args:
        query: Search term or pattern
        context_chars: Characters of context around each match

    Returns:
        List of matches with page number, text, and context
    """
    parser = DocumentParser.get_current()

    matches = await parser.search(query)

    results = []
    for match in matches[:10]:  # Limit results
        context = await parser.get_context(match.page, match.position, context_chars)
        results.append(
            {
                "page": match.page,
                "text": match.text,
                "context": context,
            }
        )

    return results


@tool
async def get_document_structure() -> dict:
    """Get the overall structure of the document.

    Returns table of contents, section headers, and page count.

    Returns:
        Document structure with sections and metadata
    """
    parser = DocumentParser.get_current()

    return {
        "page_count": parser.page_count,
        "sections": await parser.get_sections(),
        "has_tables": parser.has_tables,
        "has_images": parser.has_images,
        "document_type": parser.document_type,
    }


# List of all tools for agent binding
DOCUMENT_TOOLS = [
    read_document_section,
    ocr_region,
    extract_table,
    search_document,
    get_document_structure,
]
