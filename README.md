# dataxtr

Agentic data extraction system using LangGraph.

## Features

- **Supervisor-orchestrated workflow** with 3 specialized agents
- **Multi-provider LLM support** (Claude, OpenAI, Gemini)
- **Multiple document formats** (PDF, DOCX, XLSX, images)
- **Smart model routing** based on extraction complexity
- **Quality validation** with LLM-as-judge pattern

## Installation

```bash
uv sync --dev
```

## Quick Start

```bash
# Configure API keys
cp .env.example .env

# Run extraction
uv run python examples/basic_extraction.py invoice.pdf
```

## Architecture

```
Document → Loader → Field Prep Agent → Supervisor ←→ Extraction Agent(s)
                                           ↓              ↓
                                      Aggregator ← Quality Agent (retry loop)
```

## License

MIT
