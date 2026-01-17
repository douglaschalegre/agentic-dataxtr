# dataxtr

Agentic data extraction system using LangGraph.

## Features

- **Supervisor-orchestrated workflow** with 3 specialized agents
- **Multi-provider LLM support** (Claude, OpenAI, Gemini, Groq, Ollama)
- **Multiple document formats** (PDF, DOCX, XLSX, images)
- **Smart document chunking** with Docling for better table extraction
- **Smart model routing** based on extraction complexity
- **Quality validation** with LLM-as-judge pattern
- **Local models** via Ollama (FREE, no API keys needed)

## Installation

```bash
uv sync --dev
```

## Quick Start

### Option 1: Using Cloud Providers (Groq/Claude/OpenAI)

```bash
# Configure API keys
cp .env.example .env
# Edit .env and add your API keys

# Run extraction
uv run python examples/basic_extraction.py invoice.pdf
```

### Option 2: Using Ollama (Free, Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull gemma3:4b

# Configure (optional - uses localhost:11434 by default)
cp .env.example .env

# Run extraction
DEFAULT_LLM_PROVIDER=ollama uv run python examples/basic_extraction.py invoice.pdf
```

## Document Chunking for Better Table Extraction

By default, dataxtr uses **Docling** to semantically chunk PDF documents into text, tables, and images. This significantly improves table extraction accuracy compared to traditional text-based parsing.

### How It Works

1. **Semantic Chunking**: Docling analyzes document structure and creates separate chunks for:
   - Text paragraphs
   - Tables (with proper structure recognition)
   - Images
   - Titles/headings
   - Lists and code blocks

2. **Better Table Recognition**: Instead of using heuristics like detecting `|` or `\t` characters, Docling uses AI models to:
   - Identify table boundaries
   - Recognize table structure (rows, columns, spanning cells)
   - Extract data as structured DataFrames

3. **Chunk-Based Extraction**: The extraction agent can:
   - Access tables directly as structured data
   - Process only relevant chunks instead of full pages
   - Extract different data types (tables, text, images) separately

### Example

```python
from dataxtr.graph.state import create_initial_state
from dataxtr.graph.builder import build_extraction_graph
from dataxtr.schemas.fields import FieldDefinition, FieldType

# Define schema with table field
schema = [
    FieldDefinition(
        name="line_items",
        description="Table of invoice line items",
        field_type=FieldType.TABLE,
        required=True,
    )
]

# Create state with chunking enabled (default)
initial_state = create_initial_state(
    document_path="invoice.pdf",
    document_type="pdf",
    schema_fields=schema,
    use_chunking=True,  # ✅ Enables Docling chunking (default)
)

# Run extraction
graph = build_extraction_graph()
result = await graph.ainvoke(initial_state)
```

### Disabling Chunking

If you want to use traditional extraction without chunking:

```python
initial_state = create_initial_state(
    # ...
    use_chunking=False,  # ❌ Disable chunking
)
```

### When to Use Chunking

**Use chunking (recommended):**
- PDFs with tables (invoices, reports, statements)
- Documents with complex layouts
- Multi-page documents with mixed content types

**Disable chunking when:**
- Processing simple text-only documents
- Working with already-structured formats (CSV, XLSX)
- Docling dependencies are not available

See `examples/chunking_pdf_tables.py` for a complete example.

## Architecture

```
Document → Loader → Field Prep Agent → Supervisor ←→ Extraction Agent(s)
                                           ↓              ↓
                                      Aggregator ← Quality Agent (retry loop)
```

## License

MIT
