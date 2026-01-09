"""Table extraction service."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image


@dataclass
class ExtractedTable:
    """Represents an extracted table."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    confidence: float = 1.0
    page_number: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "headers": self.headers,
            "rows": self.rows,
            "confidence": self.confidence,
        }

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""

        lines = []

        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")
        elif self.rows:
            # Use first row as header if no headers
            lines.append("| " + " | ".join(self.rows[0]) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.rows[0])) + " |")
            self.rows = self.rows[1:]

        # Data rows
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


class TableExtractor:
    """Service for extracting tables from documents."""

    def __init__(self, use_camelot: bool = True):
        """Initialize table extractor.

        Args:
            use_camelot: Whether to use camelot-py for PDF tables
        """
        self.use_camelot = use_camelot

    async def extract_from_pdf(
        self,
        pdf_path: Path,
        page_numbers: Optional[list[int]] = None,
    ) -> list[ExtractedTable]:
        """Extract tables from a PDF file.

        Args:
            pdf_path: Path to PDF file
            page_numbers: Optional list of pages to extract from (1-indexed)

        Returns:
            List of extracted tables
        """
        tables = []

        if self.use_camelot:
            try:
                import camelot

                # Convert page numbers to camelot format
                pages = "all"
                if page_numbers:
                    pages = ",".join(str(p) for p in page_numbers)

                # Try lattice (bordered tables) first
                camelot_tables = camelot.read_pdf(
                    str(pdf_path), pages=pages, flavor="lattice"
                )

                if not camelot_tables:
                    # Fallback to stream (borderless tables)
                    camelot_tables = camelot.read_pdf(
                        str(pdf_path), pages=pages, flavor="stream"
                    )

                for ct in camelot_tables:
                    df = ct.df
                    tables.append(
                        ExtractedTable(
                            headers=df.iloc[0].tolist() if len(df) > 0 else [],
                            rows=df.iloc[1:].values.tolist() if len(df) > 1 else [],
                            confidence=ct.accuracy / 100,
                            page_number=ct.page,
                        )
                    )

            except Exception:
                # Fallback to text-based extraction
                tables = await self._extract_tables_from_text(pdf_path, page_numbers)

        return tables

    async def _extract_tables_from_text(
        self, pdf_path: Path, page_numbers: Optional[list[int]] = None
    ) -> list[ExtractedTable]:
        """Extract tables using text pattern matching."""
        import fitz

        tables = []
        doc = fitz.open(str(pdf_path))

        pages_to_process = range(len(doc))
        if page_numbers:
            pages_to_process = [p - 1 for p in page_numbers if 0 < p <= len(doc)]

        for page_num in pages_to_process:
            page = doc[page_num]
            text = page.get_text()

            # Find table-like patterns
            found_tables = self._find_tables_in_text(text)
            for t in found_tables:
                t.page_number = page_num + 1
                tables.append(t)

        doc.close()
        return tables

    def _find_tables_in_text(self, text: str) -> list[ExtractedTable]:
        """Find tables in text using heuristics."""
        tables = []

        # Pattern 1: Tab-separated values
        lines = text.split("\n")
        current_table_lines = []

        for line in lines:
            if "\t" in line or line.count("  ") >= 2:
                current_table_lines.append(line)
            elif current_table_lines:
                if len(current_table_lines) >= 2:
                    table = self._parse_table_lines(current_table_lines)
                    if table:
                        tables.append(table)
                current_table_lines = []

        # Handle last table
        if len(current_table_lines) >= 2:
            table = self._parse_table_lines(current_table_lines)
            if table:
                tables.append(table)

        return tables

    def _parse_table_lines(self, lines: list[str]) -> Optional[ExtractedTable]:
        """Parse lines into a table structure."""
        if not lines:
            return None

        rows = []
        for line in lines:
            # Split by tabs or multiple spaces
            if "\t" in line:
                cells = [c.strip() for c in line.split("\t")]
            else:
                cells = [c.strip() for c in re.split(r"\s{2,}", line)]

            cells = [c for c in cells if c]  # Remove empty cells
            if cells:
                rows.append(cells)

        if len(rows) < 2:
            return None

        return ExtractedTable(
            headers=rows[0],
            rows=rows[1:],
            confidence=0.7,  # Lower confidence for text-based extraction
        )

    async def extract_from_image(self, image: Image.Image) -> list[ExtractedTable]:
        """Extract tables from an image using OCR.

        Args:
            image: PIL Image containing tables

        Returns:
            List of extracted tables
        """
        # Use OCR to get text, then parse
        import pytesseract

        text = pytesseract.image_to_string(image)
        return self._find_tables_in_text(text)

    async def find_tables(self, page_content: str) -> list[ExtractedTable]:
        """Find tables in already-extracted page content.

        Args:
            page_content: Text content of a page

        Returns:
            List of found tables
        """
        return self._find_tables_in_text(page_content)
