# Agent Reference Guide - dataxtr

This document contains everything an AI agent needs to know to work effectively with this codebase.

---

## Project Overview

**dataxtr** is an agentic data extraction system built with LangGraph that uses a supervisor pattern to orchestrate three specialized agents for extracting structured data from documents.

**Tech Stack:**
- Python 3.11+
- LangGraph for workflow orchestration
- Multi-provider LLM support (Claude, OpenAI, Gemini, Groq, Ollama)
- PyMuPDF, python-docx, openpyxl for document parsing
- Tesseract for OCR
- Camelot for table extraction

**Package Manager:** `uv` (not pip/poetry/pipenv)

---

## Architecture

### Workflow Graph
```
START
  → Document Loader (parse document)
  → Field Prep Agent (group fields by context)
  → Supervisor (orchestrates extraction loop)
      ↓
      → Extraction Agent(s) (parallel extraction with tools)
      → Quality Agent (LLM-as-judge validation)
      ↓ (retry if needed with upgraded model)
      ← back to Supervisor
  → Aggregator (combine results)
  → END
```

### Three Agent Pattern

1. **Field Preparation Agent** (`src/dataxtr/agents/field_prep.py`)
   - Analyzes schema fields and groups them by semantic context
   - Assigns extraction strategy: `simple` | `complex` | `visual`
   - Provides document location hints
   - Uses structured output (Pydantic models)

2. **Extraction Agent** (`src/dataxtr/agents/extraction.py`)
   - Agentic tool use loop (max 10 iterations)
   - Uses document tools: `read_document_section`, `search_document`, `extract_table`, `ocr_region`
   - Model selection based on complexity (fast/standard/powerful)
   - Returns `GroupExtractionResult` with confidence scores

3. **Quality Agent** (`src/dataxtr/agents/quality.py`)
   - LLM-as-judge pattern for validation
   - Checks: completeness, format validity, confidence calibration, consistency
   - Returns `QualityReport` with recommendation: `accept` | `retry_same_model` | `retry_different_model` | `manual_review`
   - Triggers supervisor retry loop with model upgrade

### Supervisor Logic (`src/dataxtr/graph/nodes.py`)
- Manages retry queue with max iterations (default: 3)
- Upgrades model on retry (e.g., Haiku → Sonnet → Opus)
- Sequential processing of field groups (not parallel by default)
- Handles extraction failures gracefully

---

## Project Structure

```
src/dataxtr/
├── schemas/              # Pydantic models (data contracts)
│   ├── fields.py        # FieldDefinition, FieldGroup, FieldType, ExtractionComplexity
│   ├── results.py       # ExtractionResult, GroupExtractionResult
│   └── quality.py       # QualityIssue, QualityReport
├── graph/               # LangGraph workflow
│   ├── state.py         # ExtractionState TypedDict with custom reducers
│   ├── nodes.py         # Node implementations (document_loader, field_prep, etc.)
│   └── builder.py       # build_extraction_graph() and build_simple_extraction_graph()
├── agents/              # Agent implementations
│   ├── base.py          # BaseAgent abstract class
│   ├── field_prep.py    # Field grouping agent
│   ├── extraction.py    # Data extraction agent with tool use
│   └── quality.py       # Validation agent
├── models/              # LLM configuration and routing
│   ├── config.py        # ModelProvider, ModelTier, MODEL_REGISTRY
│   └── router.py        # ModelRouter for task-based selection
├── services/            # Document processing services
│   ├── document_parser.py  # Unified parser for PDF/DOCX/XLSX/images
│   ├── ocr_service.py      # Tesseract OCR wrapper
│   └── table_extractor.py  # Camelot-based table extraction
└── tools/               # LangChain tools for document interaction
    └── document_tools.py   # DOCUMENT_TOOLS list

examples/                # Usage examples
tests/                   # Pytest tests
```

---

## Core Concepts

### 1. State Management (LangGraph)

**ExtractionState** (`graph/state.py`) is a TypedDict with custom reducers:
- `merge_dicts`: Merges document_content and document_metadata
- `merge_extraction_results`: Upserts results by group_id
- `merge_quality_reports`: Upserts reports by group_id
- `append_errors`: Accumulates error messages

**Important:** Custom reducers handle parallel node execution correctly.

### 2. Model Routing

**Three-tier system** (`models/config.py`):
- **FAST**: Haiku, GPT-3.5, Gemini Flash, Groq Llama 8B (simple text extraction)
- **STANDARD**: Sonnet, GPT-4 Turbo, Groq Llama 70B (complex reasoning)
- **POWERFUL**: Opus, GPT-4o, Gemini Pro (maximum capability)

**Complexity mapping** (`models/router.py`):
```python
ExtractionComplexity.SIMPLE → ModelTier.FAST
ExtractionComplexity.COMPLEX → ModelTier.STANDARD
ExtractionComplexity.VISUAL → ModelTier.STANDARD (with vision=True)
```

**Cost optimization enabled by default**: Sorts models by `cost_per_1k_input + cost_per_1k_output`

### 3. Document Tools

**Context-aware tools** (`tools/document_tools.py`):
- `DocumentParser.get_current()` retrieves parser from ContextVar
- Tools are async and return strings/dicts
- Available to extraction agent via `DOCUMENT_TOOLS` list

**Tool list:**
- `read_document_section(page_numbers, section_hint)` - Read specific pages
- `ocr_region(page_number, bbox)` - OCR with optional bounding box
- `extract_table(page_number, table_index)` - Structured table data
- `search_document(query, context_chars)` - Search with context
- `get_document_structure()` - Sections and metadata

### 4. Structured Output

**All agents use `.with_structured_output()`**:
```python
structured_model = self.model.with_structured_output(FieldGroupingOutput)
result = await structured_model.ainvoke(prompt.format(input=...))
```

This ensures reliable parsing without JSON extraction errors.

---

## How to Add Features

### Adding a New LLM Provider

1. **Add to enum** (`models/config.py`):
```python
class ModelProvider(str, Enum):
    NEW_PROVIDER = "new_provider"
```

2. **Add models to registry**:
```python
"provider-model-name": ModelConfig(
    provider=ModelProvider.NEW_PROVIDER,
    model_id="actual-model-id",
    tier=ModelTier.FAST,  # or STANDARD/POWERFUL
    supports_vision=False,
    supports_tools=True,
    max_tokens=4096,
    cost_per_1k_input=0.001,
    cost_per_1k_output=0.002,
)
```

3. **Add to router** (`models/router.py`):
```python
elif config.provider == ModelProvider.NEW_PROVIDER:
    from langchain_new_provider import ChatNewProvider
    return ChatNewProvider(model=config.model_id, max_tokens=config.max_tokens, temperature=0)
```

4. **Add to .env.example**:
```bash
NEW_PROVIDER_API_KEY=your-key-here
```

5. **Add dependency**:
```bash
uv add langchain-new-provider
```

### Adding a New Document Type

1. **Add to type union** (`graph/state.py`):
```python
document_type: Literal["pdf", "image", "docx", "xlsx", "csv", "new_type"]
```

2. **Add parser method** (`services/document_parser.py`):
```python
async def _load_new_type(self) -> tuple[dict, dict]:
    # Parse the document
    return content, metadata
```

3. **Add to load dispatcher**:
```python
async def load(self) -> tuple[dict, dict]:
    if self.document_type == "new_type":
        return await self._load_new_type()
```

### Adding a New Tool

1. **Define input schema**:
```python
class NewToolInput(BaseModel):
    param: str = Field(description="Parameter description")
```

2. **Implement tool**:
```python
@tool(args_schema=NewToolInput)
async def new_tool(param: str) -> str:
    """Tool description for LLM.

    Args:
        param: Parameter description

    Returns:
        Result description
    """
    parser = DocumentParser.get_current()
    # Implementation
    return result
```

3. **Add to tool list** (`tools/document_tools.py`):
```python
DOCUMENT_TOOLS = [
    read_document_section,
    ocr_region,
    extract_table,
    search_document,
    get_document_structure,
    new_tool,  # Add here
]
```

### Adding Custom Validation Rules

In `QualityAgent._quick_validation()` or enhance the LLM prompt with specific rules:

```python
# In quality agent system prompt
"5. Custom validations:
   - Invoice numbers must match pattern INV-\d{4}
   - Dates must not be in the future
   - Amounts must be positive"
```

---

## Development Workflow

### Setup
```bash
uv sync --dev              # Install dependencies
cp .env.example .env       # Configure API keys
```

### Running
```bash
# Basic extraction
uv run python examples/basic_extraction.py invoice.pdf

# Custom schema
uv run python examples/custom_schema.py 1 resume.pdf

# Tests
uv run pytest
uv run pytest -v          # Verbose
uv run pytest tests/test_schemas.py  # Specific file
```

### Adding Dependencies
```bash
uv add package-name        # Production dependency
uv add --dev package-name  # Development dependency
```

### Code Quality
```bash
uv run ruff check .        # Linting
uv run ruff format .       # Formatting
uv run mypy src/dataxtr    # Type checking
```

---

## Configuration

### Environment Variables (.env)

**Required for each provider you use:**
```bash
GROQ_API_KEY=gsk_...           # Groq (cheap, fast)
ANTHROPIC_API_KEY=sk-ant-...   # Claude
OPENAI_API_KEY=sk-...          # OpenAI
GOOGLE_API_KEY=...             # Gemini
OLLAMA_BASE_URL=http://localhost:11434  # Ollama (local, free - no API key needed)
```

**Optional:**
```bash
DEFAULT_LLM_PROVIDER=ollama    # Preferred provider (ollama for free local models)
LOG_LEVEL=DEBUG                # Logging verbosity
```

### Model Selection

**Automatic (default):**
- Cost-optimized: picks cheapest model for tier
- Example: `FAST` → Ollama gemma3:4b (FREE, local) or Groq Llama 8B ($0.00013/1k tokens)

**Manual override:**
```python
router = ModelRouter(
    preferred_provider=ModelProvider.OLLAMA,  # Prefer Ollama (local, free)
    fallback_enabled=False,                    # Don't fallback to other providers
    cost_optimization=True,                    # Still optimize within Ollama models
)
```

**Direct model selection:**
```python
config = MODEL_REGISTRY["groq-llama-70b"]
model = router.get_chat_model(config)
```

---

## Common Patterns

### Creating Custom Extraction Schemas

```python
from dataxtr.schemas.fields import FieldDefinition, FieldType

schema = [
    FieldDefinition(
        name="field_name",
        description="Clear description for LLM",
        field_type=FieldType.TEXT,  # TEXT, NUMBER, DATE, CURRENCY, BOOLEAN, TABLE, LIST
        required=True,
        validation_rules={"pattern": r"^\d{4}$"},  # Optional
        examples=["example1", "example2"],          # Optional
    ),
]
```

### Running the Graph

```python
from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state

# Create state
state = create_initial_state(
    document_path="/path/to/doc.pdf",
    document_type="pdf",
    schema_fields=schema,
    max_iterations=3,  # Retry limit
)

# Build and run
graph = build_extraction_graph()
result = await graph.ainvoke(state)

# Access results
final_data = result["final_results"]
for field_name, data in final_data.items():
    print(f"{field_name}: {data['value']} (confidence: {data['confidence']})")
```

### Simplified Graph (No Retry Loop)

For faster processing without quality validation:

```python
from dataxtr.graph.builder import build_simple_extraction_graph

graph = build_simple_extraction_graph()  # Linear: load → prep → extract → validate → aggregate
result = await graph.ainvoke(state)
```

---

## Key Files Reference

### Most Important Files (Read These First)

1. **`src/dataxtr/graph/state.py`** - State schema, the contract between all nodes
2. **`src/dataxtr/models/config.py`** - Model registry, add new models here
3. **`src/dataxtr/agents/extraction.py`** - Core extraction logic with tool use
4. **`src/dataxtr/graph/builder.py`** - Graph structure, modify workflow here
5. **`examples/basic_extraction.py`** - Working example, start here for usage

### Critical Implementation Details

**Document Parser Context Management:**
```python
# Parser is stored in ContextVar, accessible to all tools
parser = DocumentParser(file_path=path, document_type=doc_type)
content, metadata = await parser.load()
parser.set_current()  # Sets context for tools
```

**Agent Tool Use Loop:**
```python
# Extraction agent runs agentic loop
messages = [HumanMessage(content=prompt)]
while iteration < max_iterations:
    response = await self.model.ainvoke(messages)
    if not response.tool_calls:
        break
    # Execute tools, append results to messages
    for tool_call in response.tool_calls:
        tool_result = await self._execute_tool(...)
        messages.append(ToolMessage(...))
```

**Quality-Driven Retry:**
```python
# Quality agent returns recommendation
if report.recommendation == "retry_different_model":
    retry_queue.append((group_id, "upgrade"))
# Supervisor picks up retry_queue and calls extraction with upgraded model
```

---

## Testing

### Test Structure

```
tests/
├── conftest.py           # Pytest fixtures (sample schemas, results)
├── test_schemas.py       # Pydantic model tests
└── test_integration.py   # Full workflow tests (TODO)
```

### Running Tests

```bash
uv run pytest                      # All tests
uv run pytest -v                   # Verbose
uv run pytest --cov=src/dataxtr    # With coverage
uv run pytest -k "test_name"       # Specific test
```

### Writing Tests

```python
def test_extraction(sample_invoice_schema):
    """Use fixtures from conftest.py"""
    assert len(sample_invoice_schema) == 2

async def test_async_extraction():
    """Async tests work with pytest-asyncio"""
    result = await some_async_function()
    assert result.passed
```

---

## Debugging Tips

### Enable Verbose Logging

```python
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
)
```

### Print LangGraph State

```python
# After each node
for step in graph.stream(state, stream_mode="values"):
    print(f"Step: {step}")
```

### Inspect Tool Calls

```python
# In extraction agent
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}, Args: {tool_call['args']}")
```

### Check Model Selection

```python
router = ModelRouter()
config = router.select_model(ExtractionComplexity.SIMPLE)
print(f"Selected: {config.model_id} ({config.provider.value})")
```

---

## Performance Optimization

### Model Selection Impact

- **FAST models**: 10-100x cheaper, good for simple fields (names, dates, IDs)
- **STANDARD models**: Best balance for most extractions
- **POWERFUL models**: Only use for complex reasoning or as retry fallback

### Parallel Processing

Current implementation processes groups **sequentially**. To parallelize:

```python
# In supervisor_node, use Send for parallel execution
from langgraph.types import Send

sends = []
for group_id in pending_groups[:3]:  # Process 3 in parallel
    sends.append(Send("extraction_agent", {"group_id": group_id}))

return Command(goto=sends)
```

### Reduce Iterations

```python
# Lower max_iterations for faster failures
state = create_initial_state(..., max_iterations=1)
```

---

## Common Errors and Solutions

### `OSError: Readme file does not exist`
**Cause:** Building package without README.md
**Fix:** Ensure README.md exists in project root

### `ValueError: No suitable model found`
**Cause:** Model registry doesn't have model for tier+requirements
**Fix:** Add model to `MODEL_REGISTRY` or adjust requirements

### `RuntimeError: No document parser in context`
**Cause:** Tools called without `parser.set_current()`
**Fix:** Ensure `document_loader_node` runs before extraction

### `Tool {tool_name} not found`
**Cause:** Tool not in agent's tool list
**Fix:** Add tool to `DOCUMENT_TOOLS` in `tools/document_tools.py`

### API Key Errors
**Cause:** Missing or incorrect environment variables
**Fix:** Verify `.env` has correct keys, run `load_dotenv()`

---

## Extension Points

### Custom Node
```python
async def custom_node(state: ExtractionState) -> dict:
    # Custom processing
    return {"custom_field": value}

# Add to graph
builder.add_node("custom", custom_node)
builder.add_edge("field_prep", "custom")
builder.add_edge("custom", "supervisor")
```

### Custom Reducer
```python
def custom_reducer(existing: list, new: list) -> list:
    # Custom merge logic
    return merged

class ExtractionState(TypedDict):
    custom_results: Annotated[list, custom_reducer]
```

### Custom Agent
```python
from dataxtr.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    SYSTEM_PROMPT = "Custom instructions..."

    async def execute(self, **kwargs):
        # Implementation
        return result
```

---

## Production Considerations

### Error Handling
- All nodes wrap operations in try/except
- Errors appended to `state["errors"]`
- Workflow continues even with partial failures

### Rate Limiting
- Not implemented (TODO)
- Add tenacity retry decorators for API calls

### Caching
- Document parsing not cached
- Consider caching parsed content for repeat extractions

### Monitoring
- Use LangSmith for trace logging (set `LANGCHAIN_API_KEY`)
- Track extraction times via `extraction_time_ms`
- Monitor confidence scores for quality drift

---

## Quick Reference

### Command Cheat Sheet
```bash
uv sync --dev                           # Setup
uv add package                          # Add dependency
uv run python examples/basic_extraction.py file.pdf  # Run
uv run pytest                           # Test
uv run ruff check .                     # Lint
```

### Import Paths
```python
from dataxtr.graph.builder import build_extraction_graph
from dataxtr.graph.state import create_initial_state, ExtractionState
from dataxtr.schemas.fields import FieldDefinition, FieldType, FieldGroup
from dataxtr.schemas.results import ExtractionResult, GroupExtractionResult
from dataxtr.schemas.quality import QualityReport, QualityIssue
from dataxtr.models.router import ModelRouter
from dataxtr.models.config import ModelProvider, ModelTier, MODEL_REGISTRY
from dataxtr.agents.field_prep import FieldPrepAgent
from dataxtr.agents.extraction import ExtractionAgent
from dataxtr.agents.quality import QualityAgent
from dataxtr.services.document_parser import DocumentParser, load_document
from dataxtr.tools.document_tools import DOCUMENT_TOOLS
```

### State Fields
```python
state["document_path"]          # Input file path
state["document_type"]          # pdf, image, docx, xlsx, csv
state["schema_fields"]          # List[FieldDefinition]
state["field_groups"]           # List[FieldGroup] (from field_prep)
state["extraction_results"]     # List[GroupExtractionResult]
state["quality_reports"]        # List[QualityReport]
state["final_results"]          # Dict[str, Any] (output)
state["workflow_status"]        # pending, in_progress, completed, failed
state["errors"]                 # List[str]
```

---

## Version Information

- **Python**: 3.11+
- **LangGraph**: >=0.2.0
- **LangChain**: >=0.3.0
- **Build System**: hatchling
- **Package Manager**: uv

---

*Last updated: 2026-01-08*
