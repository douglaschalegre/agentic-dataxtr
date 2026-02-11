"""Excel extraction example demonstrating spreadsheet data extraction."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state
from dataxtr.schemas.fields import FieldDefinition, FieldType

# Load environment variables
load_dotenv()


async def extract_excel(xlsx_path: str) -> dict:
    """Extract data from an Excel spreadsheet.

    Args:
        xlsx_path: Path to the Excel file

    Returns:
        Full state with extracted data
    """
    # Define the extraction schema for Excel data
    # TODO: Customize these fields based on your specific Excel structure
    schema = [
        FieldDefinition(
            name="Cabeça - Y",
            description="Posição Cabeça - Y (m) do M5-M3-P5",
            field_type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="Posição Objetivo",
            description="Nome do maior Posição Objetivo - Y (m)",
            field_type=FieldType.NUMBER,
            required=True,
        ),
        FieldDefinition(
            name="Nomes dos Poços",
            description="Nomes de todos os poços com função WAG",
            field_type=FieldType.LIST,
            required=True,
        ),
    ]

    # Create initial state
    initial_state = create_initial_state(
        document_path=xlsx_path,
        document_type="xlsx",
        schema_fields=schema,
        max_iterations=3,
        preferred_provider="openai_codex",
        preferred_model="openai-codex-gpt-5.3-codex",
    )

    # Build and run the graph
    graph = build_extraction_graph()
    state = await graph.ainvoke(initial_state)

    return state


async def main():
    """Run the Excel extraction example."""
    import sys

    xlsx_path = sys.argv[1]

    if not Path(xlsx_path).exists():
        print(f"Error: File not found: {xlsx_path}")
        print("\nUsage: python excel_extraction.py <path_to_xlsx>")
        print("\nExample:")
        print('  python excel_extraction.py "examples/Malha Caso 15.xlsx"')
        return

    print(f"Extracting data from: {xlsx_path}")
    print("-" * 50)

    state = await extract_excel(xlsx_path)
    results = state.get("final_results", {}) if isinstance(state, dict) else {}

    print("\nDiagnostics:")
    print("=" * 50)
    print(f"Workflow status: {state.get('workflow_status', 'unknown')}")

    errors = state.get("errors", [])
    if errors:
        print(f"Errors: {errors}")

    field_groups = state.get("field_groups", [])
    if field_groups:
        print("\nField groups:")
        for g in field_groups:
            print(
                f"- {getattr(g, 'group_id', '?')} | {getattr(g, 'group_name', '')} | "
                f"strategy={getattr(g, 'extraction_strategy', '')} | hint={getattr(g, 'context_hint', '')}"
            )

    extraction_results = state.get("extraction_results", [])
    if extraction_results:
        print("\nExtraction results (raw):")
        for r in extraction_results:
            print(
                f"- group {getattr(r, 'group_id', '?')} model={getattr(r, 'model_used', '')} "
                f"retry={getattr(r, 'retry_count', 0)} count={len(getattr(r, 'results', []))}"
            )
            for res in getattr(r, "results", []):
                value = getattr(res, "extracted_value", None)
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(
                    f"    field={getattr(res, 'field_name', '')} value={value} "
                    f"conf={getattr(res, 'confidence', None)}"
                )

    quality_reports = state.get("quality_reports", [])
    if quality_reports:
        print("\nQuality reports:")
        for qr in quality_reports:
            print(
                f"- group {getattr(qr, 'group_id', '?')} passed={getattr(qr, 'passed', None)} "
                f"recommendation={getattr(qr, 'recommendation', '')}"
            )

    if not results:
        print("\nNo results returned. Check errors and adjust the schema.")
        return

    print("\nExtraction Results:")
    print("=" * 50)

    for field_name, data in results.items():
        value = data.get("value", "N/A")
        confidence = data.get("confidence", 0)
        source = data.get("source", "unknown")
        method = data.get("method", "")

        print(f"\n{field_name}:")

        # Handle table data specially
        if isinstance(value, (list, dict)):
            print(f"  Type: {type(value).__name__}")
            print(f"  Preview: {json.dumps(value, indent=2, default=str)[:200]}...")
        else:
            print(f"  Value: {value}")

        print(f"  Confidence: {confidence:.1%}")
        print(f"  Source: {source}")
        if method:
            print(f"  Method: {method}")

    # Also output as JSON
    print("\n" + "=" * 50)
    print("JSON Output:")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
