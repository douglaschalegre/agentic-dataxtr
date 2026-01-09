"""Basic extraction example demonstrating the dataxtr workflow."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state
from dataxtr.schemas.fields import FieldDefinition, FieldType

# Load environment variables
load_dotenv()


async def extract_invoice(pdf_path: str) -> dict:
    """Extract data from an invoice PDF.

    Args:
        pdf_path: Path to the invoice PDF file

    Returns:
        Full state with extracted data and diagnostics plus intermediate debugging info
    """
    # Define the extraction schema tailored to the bundled insurance PDF
    schema = [
        FieldDefinition(
            name="razao_social",
            description="Razão Social da seguradora",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="cnpj",
            description="CNPJ da seguradora",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"pattern": r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}"},
        ),
        FieldDefinition(
            name="numero_negocio",
            description="Número do negócio ou proposta",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="proponente",
            description="Nome completo do proponente/segurado",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="cpf",
            description="CPF do proponente",
            field_type=FieldType.TEXT,
            required=True,
            validation_rules={"pattern": r"\d{3}\.\d{3}\.\d{3}-\d{2}"},
        ),
        FieldDefinition(
            name="endereco",
            description="Endereço completo do proponente (logradouro, número, complemento, bairro, cidade, estado, CEP)",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="vigencia_inicio",
            description="Data de início da vigência",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="vigencia_fim",
            description="Data de fim da vigência",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="tipo_cobertura",
            description="Tipo de cobertura (ex: Prédio + Conteúdo)",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="classe_bonus",
            description="Classe de bônus informada",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="seguradora",
            description="Seguradora e código",
            field_type=FieldType.TEXT,
            required=True,
        ),
        FieldDefinition(
            name="vencimento",
            description="Data de vencimento da proposta ou apólice",
            field_type=FieldType.DATE,
            required=True,
        ),
        FieldDefinition(
            name="apolice_anterior",
            description="Número da apólice anterior (se houver)",
            field_type=FieldType.TEXT,
            required=False,
        ),
        FieldDefinition(
            name="coberturas",
            description="Tabela de coberturas com limite e prêmio líquido",
            field_type=FieldType.TABLE,
            required=True,
        ),
        FieldDefinition(
            name="premio_liquido_total",
            description="Prêmio líquido total",
            field_type=FieldType.CURRENCY,
            required=True,
        ),
    ]

    # Create initial state
    initial_state = create_initial_state(
        document_path=pdf_path,
        document_type="pdf",
        schema_fields=schema,
        max_iterations=3,
    )

    # Build and run the graph
    graph = build_extraction_graph()
    state = await graph.ainvoke(initial_state)

    return state


async def main():
    """Run the extraction example."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python basic_extraction.py <path_to_pdf>")
        print("\nExample:")
        print("  python basic_extraction.py invoice.pdf")
        return

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        return

    print(f"Extracting data from: {pdf_path}")
    print("-" * 50)

    state = await extract_invoice(pdf_path)
    results = state.get("final_results", {}) if isinstance(state, dict) else {}

    print("\nDiagnostics:")
    print("=" * 50)
    print(f"Workflow status: {state.get('workflow_status', 'unknown')}")
    print(f"Errors: {state.get('errors', [])}")

    field_groups = state.get("field_groups", [])
    if field_groups:
        print("Field groups:")
        for g in field_groups:
            print(
                f"- {getattr(g, 'group_id', '?')} | {getattr(g, 'group_name', '')} | "
                f"strategy={getattr(g, 'extraction_strategy', '')} | hint={getattr(g, 'context_hint', '')}"
            )

    extraction_results = state.get("extraction_results", [])
    if extraction_results:
        print("Extraction results (raw):")
        for r in extraction_results:
            print(
                f"- group {getattr(r, 'group_id', '?')} model={getattr(r, 'model_used', '')} "
                f"retry={getattr(r, 'retry_count', 0)} count={len(getattr(r, 'results', []) )}"
            )
            for res in getattr(r, 'results', []):
                print(
                    f"    field={getattr(res, 'field_name', '')} value={getattr(res, 'extracted_value', None)} "
                    f"conf={getattr(res, 'confidence', None)} src={getattr(res, 'source_location', '')}"
                )

    quality_reports = state.get("quality_reports", [])
    if quality_reports:
        print("Quality reports:")
        for qr in quality_reports:
            print(
                f"- group {getattr(qr, 'group_id', '?')} passed={getattr(qr, 'passed', None)} "
                f"recommendation={getattr(qr, 'recommendation', '')}"
            )

    if not results:
        print("\nNo results returned. Try adjusting the schema or checking logs.")
        return

    print("\nExtraction Results:")
    print("=" * 50)

    for field_name, data in results.items():
        value = data.get("value", "N/A")
        confidence = data.get("confidence", 0)
        source = data.get("source", "unknown")
        method = data.get("method", "")

        print(f"\n{field_name}:")
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
