"""Document chunk schemas for semantic processing."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Types of document chunks."""

    TEXT = "text"  # Regular text content
    TABLE = "table"  # Structured table data
    IMAGE = "image"  # Image content
    TITLE = "title"  # Section titles/headings
    LIST = "list"  # List items
    CODE = "code"  # Code blocks
    FORMULA = "formula"  # Mathematical formulas


class DocumentChunk(BaseModel):
    """A semantic chunk of document content."""

    chunk_id: str = Field(description="Unique chunk identifier")
    chunk_type: ChunkType = Field(description="Type of content in this chunk")
    content: Any = Field(description="The actual content (text, table data, image base64, etc.)")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    position: Optional[int] = Field(
        default=None, description="Position in document (for ordering)"
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        default=None, description="Bounding box coordinates (x0, y0, x1, y1)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (confidence, headers for tables, etc.)",
    )
    parent_section: Optional[str] = Field(
        default=None, description="Parent section/heading if applicable"
    )

    @property
    def text_content(self) -> str:
        """Get text representation of content."""
        if self.chunk_type == ChunkType.TEXT:
            return str(self.content)
        elif self.chunk_type == ChunkType.TABLE:
            # Convert table to text representation
            if isinstance(self.content, dict):
                return str(self.content)
            return str(self.content)
        elif self.chunk_type == ChunkType.TITLE:
            return str(self.content)
        elif self.chunk_type == ChunkType.LIST:
            if isinstance(self.content, list):
                return "\n".join(str(item) for item in self.content)
            return str(self.content)
        else:
            return str(self.content)


class ChunkedDocument(BaseModel):
    """A document split into semantic chunks."""

    document_path: str = Field(description="Original document path")
    document_type: str = Field(description="Document type (pdf, docx, etc.)")
    chunks: list[DocumentChunk] = Field(description="Ordered list of document chunks")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document-level metadata"
    )
    total_pages: Optional[int] = Field(default=None, description="Total page count")
    processing_time_ms: int = Field(
        default=0, description="Time taken to process document"
    )

    def get_chunks_by_type(self, chunk_type: ChunkType) -> list[DocumentChunk]:
        """Get all chunks of a specific type."""
        return [chunk for chunk in self.chunks if chunk.chunk_type == chunk_type]

    def get_chunks_by_page(self, page_number: int) -> list[DocumentChunk]:
        """Get all chunks from a specific page."""
        return [
            chunk for chunk in self.chunks if chunk.page_number == page_number
        ]

    def get_text_chunks(self) -> list[DocumentChunk]:
        """Get all text chunks."""
        return self.get_chunks_by_type(ChunkType.TEXT)

    def get_table_chunks(self) -> list[DocumentChunk]:
        """Get all table chunks."""
        return self.get_chunks_by_type(ChunkType.TABLE)

    def get_image_chunks(self) -> list[DocumentChunk]:
        """Get all image chunks."""
        return self.get_chunks_by_type(ChunkType.IMAGE)
