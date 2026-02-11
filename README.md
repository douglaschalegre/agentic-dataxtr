# dataxtr

Agentic data extraction system using LangGraph.

## Features

- **Supervisor-orchestrated workflow** with 3 specialized agents
- **Multi-provider LLM support** (Claude, OpenAI, OpenAI Codex, Gemini, Groq, Ollama)
- **Multiple document formats** (PDF, DOCX, XLSX, images)
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
ollama pull gptoss:20b

# Configure (optional - uses localhost:11434 by default)
cp .env.example .env

# Run extraction
DEFAULT_LLM_PROVIDER=ollama uv run python examples/basic_extraction.py invoice.pdf
```

### Option 3: Using OpenAI Codex Subscription (ChatGPT Login)

```bash
# Install and login Codex CLI (creates ~/.codex/auth.json)
npm install -g @openai/codex
codex login

# Configure provider
cp .env.example .env

# Run extraction using Codex subscription auth
DEFAULT_LLM_PROVIDER=openai_codex uv run python examples/basic_extraction.py invoice.pdf

# Or use the Codex-focused example
uv run python examples/codex_extraction.py invoice.pdf
```

Optional overrides:
- Set `OPENAI_CODEX_ACCESS_TOKEN` to bypass `~/.codex/auth.json`
- Set `OPENAI_CODEX_ACCOUNT_ID` for ChatGPT workspace/account routing
- Set `OPENAI_CODEX_BASE_URL` if your Codex endpoint differs
- Set `OPENAI_CODEX_INSTRUCTIONS` to customize the required Codex responses instructions field

## Architecture

```
Document → Loader → Field Prep Agent → Supervisor ←→ Extraction Agent(s)
                                           ↓              ↓
                                      Aggregator ← Quality Agent (retry loop)
```

## License

MIT
