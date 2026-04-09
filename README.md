# Agentic Memory Design

> **Multi-layer memory, multi-agent routing, RAG + CRM + web search, with Supabase, Qdrant, and LangFuse observability**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://python.langchain.com/)

Hospital-assistant style project: episodic, semantic, and procedural memory; short-term context; orchestrator that routes to RAG, CRM, and Tavily-backed web search.

---

## What you'll build

- **Short-term memory** — Recent turns (Supabase-backed ring buffer by default, or Redis via `USE_SB_ST=false`).
- **Long-term semantic memory** — Distilled facts from conversations, stored and queried with embeddings.
- **Episodic memory** — Conversation episodes with summaries for semantic recall.
- **Procedural memory** — Reusable “how we do things” snippets the agent can retrieve.
- **Agentic routing** — Orchestrator + router dispatch intents to **RAG** (Qdrant + knowledge base), **CRM** (Supabase patient/booking data), and **web search** (Tavily).
- **Observability** — LangFuse tracing (optional, via `.env`).
- **Ingestion** — Crawl/chunk pipeline and scripts to populate Qdrant from `data/knowledge_base/` (Markdown) and related services.

This week extends in **Week 08** (see root `README.md`); **Week 09** is *LangGraph in the House* (agentic patterns as graphs).

---

## Project structure

```
Week 07/
├── config/
│   ├── models.yaml          # LLM / embedding model config
│   ├── param.yaml           # Provider, chunking, RAG, paths
│   └── faqs.yaml            # FAQ / pre-seeded content (if used)
├── data/
│   └── knowledge_base/      # 12 hospital-themed Markdown docs (RAG corpus)
├── notebooks/
│   ├── 01_agentic_routing_engine.ipynb
│   ├── 02_memory_capture_and_distill.ipynb
│   └── 03_memory_store_and_recall.ipynb
├── sql/
│   ├── supabase_schema.sql  # Core schema (memory + CRM-related tables)
│   └── *.sql                # Seeds (specialties, locations, doctors, etc.)
├── scripts/
│   ├── init_supabase.py
│   ├── ingest_to_qdrant.py
│   ├── seed_crm_unified.py
│   └── ...
├── src/
│   ├── agents/              # Router, orchestrator, tools (rag, crm, web_search)
│   ├── memory/              # ST / LT / episodic / procedural stores + policies
│   ├── services/            # chat_service (RAG, CAG, CRAG), ingest_service, crm_service
│   └── infrastructure/      # config, db (Supabase, Qdrant, SQL), LLM, observability
├── tests/                   # e.g. test_memory_core.py, test_memory_policies.py
├── .env.example             # Copy to .env (never commit .env)
├── pyproject.toml           # Package: agentic-memory-design
├── requirements.txt         # Optional pip mirror of deps
├── Makefile                 # Common dev commands
└── README.md
```

---

## Notebooks (run in order)

| # | Notebook | Focus |
|---|----------|--------|
| 01 | `01_agentic_routing_engine.ipynb` | Multi-agent routing, tools, orchestration |
| 02 | `02_memory_capture_and_distill.ipynb` | Capture turns, distill facts, episodic storage |
| 03 | `03_memory_store_and_recall.ipynb` | Store / recall with token budgeting |

---

## Prerequisites

- Python 3.10+
- **Supabase** project (Postgres + optional pgvector for embeddings in DB paths you enable)
- **Qdrant** (cloud URL + API key recommended; see `.env.example`)
- **OpenRouter** (recommended) or direct provider keys (`OPENAI_API_KEY`, etc.)
- Optional: **Redis** if `USE_SB_ST=false` for short-term memory
- Optional: **Tavily** for web search tool; **LangFuse** for traces

---

## Quick start

```bash
cd "Week 07"

# Install (uv recommended)
uv sync
# or: pip install -e .

# Environment
cp .env.example .env
# Fill QDRANT_*, SUPABASE_*, OPENROUTER_API_KEY (and others as needed)

# Initialize DB / seeds (follow scripts in sql/ and Makefile as needed)
# e.g. apply supabase_schema.sql in Supabase SQL editor, then run seed scripts

# Jupyter
jupyter lab
# Open notebooks/01_agentic_routing_engine.ipynb first
```

---

## Configuration

- **`config/param.yaml`** — Provider (`openrouter` / `openai`), LLM defaults, embedding tier, chunking, paths.
- **`config/models.yaml`** — Model names per tier.
- **`src/infrastructure/config.py`** — Loads YAML + env; single entry for app settings.

Secrets **only** in `.env` (see `.env.example`). Do not commit `.env`.

---

## Key components

| Area | Role |
|------|------|
| `src/agents/orchestrator.py` | High-level agent flow |
| `src/agents/router.py` | Intent / route selection |
| `src/agents/tools/` | `rag_tool`, `crm_tool`, `web_search_tool` |
| `src/memory/` | `st_store`, `lt_store`, `episodic_store`, `procedural_store`, policies |
| `src/services/chat_service/` | RAG, CAG, CRAG, templates |
| `src/services/crm_service/` | CRM DB access + helpers |
| `src/services/ingest_service/` | Crawl, chunk, ingest pipeline |
| `src/infrastructure/db/` | Supabase, Qdrant, SQL client |
| `src/infrastructure/observability.py` | LangFuse wiring |

---

## Testing

```bash
# From Week 07 with venv active
pytest tests/
```

---

## Troubleshooting

- **Import errors** — Run notebooks from `Week 07` root or ensure `sys.path` includes the package root as in notebook setup cells.
- **Supabase / Qdrant** — Verify URLs, keys, and that schema matches `sql/supabase_schema.sql`.
- **Embedding dimensions** — `param.yaml` notes pgvector limits; use `embedding.tier: small` when required.

---

## Contact

Course: **hi@zuucrew.ai**

---

*Built with Zuu Crew · Agentic Memory Design*
