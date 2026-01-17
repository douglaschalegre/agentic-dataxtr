"""Document chunking service using Docling for semantic processing."""

import time
from pathlib import Path
from typing import Optional

import structlog

from dataxtr.schemas.chunks import ChunkType, ChunkedDocument, DocumentChunk

logger = structlog.get_logger(__name__)


class DocumentChunker:
    """Chunks documents into semantic units using Docling."""

    def __init__(self):
        """Initialize the document chunker."""
        self._converter = None

    def _get_converter(self):
        """Lazy-load Docling converter to avoid import overhead."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter

                self._converter = DocumentConverter()
                logger.info("Docling converter initialized successfully")
            except ImportError as e:
                logger.error("Failed to import Docling", error=str(e))
                raise ImportError(
                    "Docling is not installed. Install it with: pip install docling"
                ) from e
        return self._converter

    async def chunk_document(
        self,
        file_path: Path,
        document_type: str,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> ChunkedDocument:
        """
        Chunk a document into semantic units.

        Args:
            file_path: Path to the document
            document_type: Type of document (pdf, docx, etc.)
            extract_tables: Whether to extract tables as separate chunks
            extract_images: Whether to extract images as separate chunks

        Returns:
            ChunkedDocument with semantic chunks
        """
        start_time = time.time()

        logger.info(
            "Starting document chunking",
            file_path=str(file_path),
            document_type=document_type,
        )

        # Currently only support PDF chunking with Docling
        if document_type == "pdf":
            chunks = await self._chunk_pdf(
                file_path,
                extract_tables=extract_tables,
                extract_images=extract_images,
            )
        else:
            # Fallback to basic chunking for other types
            logger.warning(
                "Docling chunking not supported for this type, using basic chunking",
                document_type=document_type,
            )
            chunks = await self._basic_chunk(file_path, document_type)

        processing_time_ms = int((time.time() - start_time) * 1000)

        chunked_doc = ChunkedDocument(
            document_path=str(file_path),
            document_type=document_type,
            chunks=chunks,
            metadata={
                "chunk_count": len(chunks),
                "table_chunks": sum(
                    1 for c in chunks if c.chunk_type == ChunkType.TABLE
                ),
                "image_chunks": sum(
                    1 for c in chunks if c.chunk_type == ChunkType.IMAGE
                ),
                "text_chunks": sum(
                    1 for c in chunks if c.chunk_type == ChunkType.TEXT
                ),
            },
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "Document chunking completed",
            chunk_count=len(chunks),
            processing_time_ms=processing_time_ms,
        )

        return chunked_doc

    async def _chunk_pdf(
        self,
        file_path: Path,
        extract_tables: bool = True,
        extract_images: bool = True,
    ) -> list[DocumentChunk]:
        """Chunk a PDF using Docling."""
        try:
            converter = self._get_converter()

            # Convert document using Docling
            result = converter.convert(str(file_path))

            chunks: list[DocumentChunk] = []
            chunk_counter = 0
            current_section = None

            # Process the document structure
            # Docling returns a structured document with elements
            for element in result.document.iterate_items():
                chunk_counter += 1

                # Determine chunk type based on element type
                element_type = element.self_ref.split("/")[0] if hasattr(element, "self_ref") else "text"

                # Map Docling element types to our ChunkType
                chunk_type = self._map_element_type(element_type)

                # Get page number if available
                page_number = None
                if hasattr(element, "prov") and element.prov:
                    page_number = element.prov[0].page_no if element.prov else None

                # Get bounding box if available
                bbox = None
                if hasattr(element, "prov") and element.prov:
                    prov = element.prov[0]
                    if hasattr(prov, "bbox"):
                        bbox_obj = prov.bbox
                        bbox = (bbox_obj.l, bbox_obj.t, bbox_obj.r, bbox_obj.b)

                # Extract content based on type
                content = None
                metadata = {}

                if chunk_type == ChunkType.TABLE and extract_tables:
                    # Extract table data
                    content = self._extract_table_from_element(element)
                    metadata["table_cells"] = len(content.get("rows", [])) if isinstance(content, dict) else 0

                elif chunk_type == ChunkType.IMAGE and extract_images:
                    # Extract image data
                    content = self._extract_image_from_element(element)
                    metadata["image_format"] = "base64"

                elif chunk_type == ChunkType.TEXT:
                    # Extract text content
                    content = element.text if hasattr(element, "text") else str(element)

                elif chunk_type == ChunkType.TITLE:
                    # Section title
                    content = element.text if hasattr(element, "text") else str(element)
                    current_section = content

                else:
                    content = element.text if hasattr(element, "text") else str(element)

                # Skip empty chunks
                if not content:
                    continue

                chunk = DocumentChunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    chunk_type=chunk_type,
                    content=content,
                    page_number=page_number,
                    position=chunk_counter,
                    bbox=bbox,
                    metadata=metadata,
                    parent_section=current_section,
                )

                chunks.append(chunk)

            logger.info(
                "PDF chunking completed",
                total_chunks=len(chunks),
                table_chunks=sum(1 for c in chunks if c.chunk_type == ChunkType.TABLE),
            )

            return chunks

        except Exception as e:
            logger.error("Failed to chunk PDF with Docling", error=str(e))
            # Fallback to basic chunking
            return await self._basic_chunk(file_path, "pdf")

    def _map_element_type(self, element_type: str) -> ChunkType:
        """Map Docling element type to ChunkType."""
        type_mapping = {
            "table": ChunkType.TABLE,
            "picture": ChunkType.IMAGE,
            "figure": ChunkType.IMAGE,
            "title": ChunkType.TITLE,
            "heading": ChunkType.TITLE,
            "section_header": ChunkType.TITLE,
            "list": ChunkType.LIST,
            "list_item": ChunkType.LIST,
            "code": ChunkType.CODE,
            "formula": ChunkType.FORMULA,
            "text": ChunkType.TEXT,
            "paragraph": ChunkType.TEXT,
        }

        element_lower = element_type.lower()
        return type_mapping.get(element_lower, ChunkType.TEXT)

    def _extract_table_from_element(self, element) -> dict:
        """Extract table data from Docling element."""
        try:
            # Docling provides table data in structured format
            if hasattr(element, "export_to_dataframe"):
                df = element.export_to_dataframe()
                return {
                    "headers": df.columns.tolist(),
                    "rows": df.values.tolist(),
                    "dataframe": df,  # Keep original DataFrame
                }
            elif hasattr(element, "data"):
                # Fallback to raw data
                return {"data": element.data}
            else:
                return {"text": str(element)}
        except Exception as e:
            logger.warning("Failed to extract table data", error=str(e))
            return {"text": str(element)}

    def _extract_image_from_element(self, element) -> Optional[str]:
        """Extract image as base64 from Docling element."""
        try:
            if hasattr(element, "image"):
                import base64
                from io import BytesIO

                img = element.image
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                return img_base64
            return None
        except Exception as e:
            logger.warning("Failed to extract image", error=str(e))
            return None

    async def _basic_chunk(
        self, file_path: Path, document_type: str
    ) -> list[DocumentChunk]:
        """
        Fallback basic chunking when Docling is not available or not supported.

        This creates a single chunk with the full document content.
        """
        logger.info("Using basic fallback chunking", document_type=document_type)

        # Read file content based on type
        if document_type == "pdf":
            import fitz
            doc = fitz.open(str(file_path))
            content = "\n\n".join(page.get_text() for page in doc)
            page_count = len(doc)
            doc.close()
        else:
            # For other types, just read as text
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            page_count = 1

        return [
            DocumentChunk(
                chunk_id="chunk_1",
                chunk_type=ChunkType.TEXT,
                content=content,
                page_number=None,
                position=1,
                metadata={"fallback": True, "total_pages": page_count},
            )
        ]
