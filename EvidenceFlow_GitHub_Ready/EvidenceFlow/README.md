<p align="center">
  <img src="assets/banner.svg" alt="EvidenceFlow Banner" width="100%" />
</p>

# EvidenceFlow — Agentic AI Assistant

**EvidenceFlow** is an evidence-grounded Agentic AI assistant built with Python and FastAPI. It combines hybrid **BM25 + TF-IDF** local document retrieval with structured **tool calling** (document search, document reader, and safe arithmetic calculator), full source citations, and an automated evaluation suite.

## Key Features

- **Evidence-Grounded Agent**: Answers user queries strictly grounded in a local knowledge base with explicit `[source: doc_id]` citations.
- **Hybrid RAG Retrieval**: Combines BM25Okapi scoring and TF-IDF cosine similarity with zero heavy external dependencies.
- **Safe Tool Calling**: Built-in tool registry supporting `search_docs`, `read_doc`, and AST-sandboxed `calculate`.
- **FastAPI Backend**: REST endpoints (`/health`, `/documents`, `/ask`, `/evaluate`) with automatic interactive Swagger UI at `/docs`.
- **Zero-Key Offline Mode & Optional LLM Mode**: Works out-of-the-box without API keys using deterministic orchestration, or connects to an LLM provider when `OPENAI_API_KEY` is configured.
- **Automated Testing & Evaluation**: Includes 22 unit/integration tests and 5 end-to-end grounded evaluation cases.

## Architecture

```
User Query ──► FastAPI (/ask) ──► Agent Orchestrator
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
             [search_docs]       [read_doc]        [calculate]
             BM25 + TF-IDF     Full Doc Fetch    Safe AST Math
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                        Evidence Synthesis + Citations
```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Run the test suite
pytest

# 3. Start the API server
uvicorn evidenceflow.api:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser to test the interactive Swagger UI.
