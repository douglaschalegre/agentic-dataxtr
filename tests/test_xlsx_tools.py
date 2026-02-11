"""Tests for XLSX parsing and table tool behavior."""

from pathlib import Path

from openpyxl import Workbook

from dataxtr.services.document_parser import DocumentParser


async def test_xlsx_parser_populates_pages_and_tables(tmp_path: Path):
    """XLSX loader should expose sheet text and structured tables."""
    file_path = tmp_path / "sample.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["name", "value"])
    ws.append(["A", 10])
    ws.append(["B", 20])
    wb.save(file_path)

    parser = DocumentParser(file_path=file_path, document_type="xlsx")
    content, metadata = await parser.load()

    assert metadata["page_count"] == 1
    assert content["pages"]
    assert "Sheet: Sheet1" in content["pages"][0]

    tables = await parser.get_tables(1)
    assert len(tables) == 1
    assert tables[0]["headers"] == ["name", "value"]
    assert tables[0]["rows"] == [["A", "10"], ["B", "20"]]
