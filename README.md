# Week 07 - Agentic Memory Design Project Summary

This repository documents the **Week 07** guided lab work from the **AI Engineer Essentials** course by **Zuu Crew**.

The project focuses on **agentic memory architecture** for a healthcare-assistant style system. The core progression is:

- route each query to the right capability (direct, CRM, RAG, web search)
- recall short-term and long-term memory before answering
- distill important facts from conversation into semantic memory
- combine internal knowledge retrieval (RAG) with cache (CAG) and correction (CRAG)
- run everything on cloud-ready data services (Supabase + Qdrant) with observability (LangFuse)

Across this repo, the main engineering idea is:

- **Memory quality drives personalization**
- **Routing quality drives correct tool use**
- **Retrieval quality drives grounded answers**

## Important Note

This week is not just a chat demo. It is an architecture lab where memory, tools, retrieval, and orchestration are wired as a system:

- orchestrator flow is implemented in `src/agents/orchestrator.py`
- memory layers are implemented in `src/memory/`
- retrieval and cache/correction are implemented in `src/services/chat_service/`
- CRM and schema/seeding are production-style with Supabase SQL and deterministic seed files

## What Was Built

This project builds a memory-aware healthcare assistant stack:

```text
User Query
  -> Memory Recall (ST turns + LT semantic facts)
  -> Router (LLM route decision + params)
  -> Tool Dispatch (CRM / RAG / Web Search / direct)
  -> Synthesiser LLM (final response)
  -> Store new turns in ST memory
  -> Distill important facts into LT memory
```

RAG branch internals:

```text
Query
  -> CAG semantic cache (Qdrant collection: cag_cache)
  -> HIT: instant cached answer
  -> MISS: CRAG service
      -> initial retrieval (k=4)
      -> confidence check
      -> corrective retrieval (k=8) if needed
      -> grounded answer generation
  -> cache result for future semantic hits
```

Data and infra used:

- **Supabase PostgreSQL + pgvector** for memory and CRM tables
- **Qdrant Cloud** for RAG chunk vectors and CAG semantic cache
- **LangChain LCEL** for modern RAG chain composition
- **LangFuse** decorators and prompt fetch for tracing + prompt management

## What I Learned

### 1. Memory is a first-class design surface in agent systems

I learned that reliable personalization is not only about chat history. It needs explicit memory layers with clear responsibilities:

- short-term memory for recent conversational context
- long-term semantic memory for distilled facts/preferences
- episodic memory for session snapshots and semantic recall of past conversations
- procedural memory for reusable workflows (how-to steps)

### 2. Memory lifecycle needs policy, not just storage

I implemented scoring, decay, and pruning concepts in `src/memory/policies.py`:

- composite scoring from recency, repetition, explicitness
- exponential decay for stale facts
- TTL and low-score pruning rules
- semantic deduplication to reduce duplicate memory facts

This showed me that memory systems need governance logic, not only vector search.

### 3. Distillation improves long-term memory quality

In `src/memory/memory_ops.py`, the distiller:

- decides when to distill (turn count and memory-trigger keywords)
- uses an extractor LLM with structured JSON output
- tags and scores extracted facts
- dedupes then upserts into LT store

The practical lesson: writing every turn to long-term memory creates noise; distillation creates usable memory.

### 4. Routing is critical for tool correctness

In `src/agents/router.py` and `src/agents/prompts/agent_prompts.py`, I built an LLM router that outputs strict JSON:

- route: `crm | rag | web_search | direct`
- CRM sub-actions (lookup/search/create/cancel/reschedule)
- parameter extraction for each action

This taught me that route decision quality directly affects downstream tool correctness and user trust.

### 5. Retrieval works better with cache + correction, not only vector search

I implemented a layered retrieval approach:

- CAG semantic cache in Qdrant (`cag_cache`) for low-latency repeated/paraphrased queries
- CRAG confidence-gated correction for low-confidence retrieval cases
- parent-child retrieval in Qdrant so generation gets richer parent context

Main takeaway: retrieval robustness improves when cache behavior and confidence correction are explicit parts of the pipeline.

### 6. Multi-database architecture can stay clean with boundaries

I used each backend for a clear purpose:

- Supabase: transactional memory + CRM relational workflows
- Qdrant: high-speed semantic retrieval/cache for knowledge queries

The lesson is not "more databases". The lesson is: use each store for the job it is best at.

### 7. Observability should be built in from day one

Through `src/infrastructure/observability.py` and `@observe` usage, I learned to trace:

- router decisions
- tool dispatch latency/output
- distillation calls
- synthesis calls
- cache hit/miss behavior

This made debugging and iteration much clearer than print-based logging only.

## Current Repository Outputs

This repo already contains concrete assets from the workflow:

- `3` notebooks in `notebooks/`
- `12` internal KB markdown docs in `data/knowledge_base/`
- deterministic SQL seed files in `sql/` with insert statements for:
  - `10` specialties
  - `4` locations
  - `50` doctors
  - `200` patients
  - `1500` bookings
  - `6` procedures

So this is not notebook-only; it includes reusable architecture + seeded data foundations.

## Detailed Notebook Learnings

### `notebooks/01_agentic_routing_engine.ipynb`

Main topics:

- end-to-end orchestrator flow
- route demonstrations: direct, CRM, RAG, web_search
- user identification patterns in multi-turn chat
- memory continuity demo
- observability visibility with LangFuse traces

Main implementation takeaway:

- one orchestrator can combine memory recall, route selection, tool execution, and final synthesis in a deterministic flow.

### `notebooks/02_memory_capture_and_distill.ipynb`

Main topics:

- short-term memory ring buffer behavior
- semantic distillation into long-term facts
- episodic memory for session snapshots
- procedural memory for workflow retrieval

Main implementation takeaway:

- each memory type has a different role; mixing all memory into one store causes poor recall quality.

### `notebooks/03_memory_store_and_recall.ipynb`

Main topics:

- baseline response without memory
- hybrid memory recall (ST + LT)
- token budget analysis and context trimming
- side-by-side comparison with memory-enhanced responses
- progressive context building in full agent loop

Main implementation takeaway:

- token-budgeted memory retrieval can improve relevance while controlling prompt size.

## Source Code Learnings

### 1. Agent orchestration and routing

Key files:

- `src/agents/orchestrator.py`
- `src/agents/router.py`
- `src/agents/prompts/agent_prompts.py`

What is implemented:

- `AgentOrchestrator.chat()` full execution loop
- memory recall before route decision
- robust route parsing/fallback handling
- tool dispatch abstraction for CRM/RAG/web
- synthesis prompt composition with memory + tool output
- automatic ST write + conditional distillation

### 2. Memory subsystem (ST, LT, episodic, procedural)

Key files:

- `src/memory/st_store.py`
- `src/memory/lt_store.py`
- `src/memory/episodic_store.py`
- `src/memory/procedural_store.py`
- `src/memory/memory_ops.py`
- `src/memory/policies.py`

What is implemented:

- ST memory in Supabase `st_turns` with ring-buffer trimming and TTL
- LT memory in `mem_facts` with pgvector similarity + semantic merge-on-upsert
- episodic memory in `mem_episodes` using summary embeddings
- procedural memory in `mem_procedures` with semantic workflow retrieval
- distill/recall services and token budget allocation (60% ST, 40% LT target)

### 3. Retrieval stack (RAG + CAG + CRAG)

Key files:

- `src/agents/tools/rag_tool.py`
- `src/services/chat_service/rag_service.py`
- `src/services/chat_service/cag_cache.py`
- `src/services/chat_service/cag_service.py`
- `src/services/chat_service/crag_service.py`

What is implemented:

- Qdrant-backed retriever with parent-level dedupe
- LCEL RAG chain (`RunnableParallel | prompt | llm | parser`)
- semantic cache in a dedicated Qdrant collection
- CRAG confidence-based corrective retrieval
- FAQ warming flow from `config/faqs.yaml`

### 4. CRM and operational tools

Key files:

- `src/agents/tools/crm_tool.py`
- `src/agents/tools/web_search_tool.py`
- `src/services/crm_service/crm_db_client.py`

What is implemented:

- CRM actions: lookup patient, search doctors, create/cancel/reschedule bookings
- conflict checks for booking overlaps
- Tavily web search with source formatting and checked timestamp
- join-based CRM read paths across patients/doctors/locations/specialties

### 5. Ingestion and vectorization

Key files:

- `src/services/ingest_service/pipeline.py`
- `src/services/ingest_service/chunkers.py`
- `src/infrastructure/db/qdrant_client.py`
- `scripts/ingest_to_qdrant.py`

What is implemented:

- loaders for KB markdown, markdown crawl output, and JSONL
- chunking strategies including parent-child (default ingest path)
- embedding + Qdrant upsert pipeline
- auto-ingest guard (`ensure_kb_ingested`) before agent startup

### 6. Infrastructure and observability

Key files:

- `src/infrastructure/config.py`
- `src/infrastructure/llm/llm_provider.py`
- `src/infrastructure/llm/embeddings.py`
- `src/infrastructure/observability.py`

What is implemented:

- config loading from `config/param.yaml` and `config/models.yaml`
- 3-model role split (router, extractor, synthesiser)
- provider abstraction with OpenRouter/direct options
- LangFuse prompt fetch + runtime fallback templates
- trace/span metadata updates for key pipeline steps

## Config and Schema Learnings

### `config/param.yaml`

Core runtime controls include:

- provider/tier selection
- embedding tier and dimensions implications
- chunking hyperparameters
- retrieval/CAG/CRAG thresholds
- crawling and path config
- observability enable/disable switch

Current key values:

- retrieval top-k: `4`
- similarity threshold: `0.7`
- CAG similarity threshold: `0.90`
- CAG TTL: `86400` seconds
- CRAG confidence threshold: `0.6`
- CRAG expanded_k: `8`

### `config/models.yaml`

Defines model routing by provider and tier, and embedding model options (`small`/`default`) so model swaps remain declarative.

### `config/faqs.yaml`

Provides known FAQ query-answer pairs used for semantic cache warming, reducing repeated generation cost and latency.

### `sql/supabase_schema.sql`

Schema covers:

- memory tables (`st_turns`, `mem_facts`, `mem_episodes`, `mem_procedures`)
- CRM tables (`locations`, `specialties`, `doctors`, `patients`, `bookings`)
- pgvector indexes for semantic search
- RLS policies for user-scoped memory access

## Technical Deep Dive

### Supabase + pgvector (Memory + CRM)

This project uses Supabase PostgreSQL as the main transactional store and pgvector as the semantic layer for memory tables.

Core technical details:

- vector columns are explicitly typed as `vector(1536)` (aligned with embedding tier `small`)
- `mem_facts`, `mem_episodes`, and `mem_procedures` use cosine similarity with IVFFlat indexes
- index settings in schema:
  - `mem_facts`: `lists = 100`
  - `mem_episodes`: `lists = 100`
  - `mem_procedures`: `lists = 50`
- SQL helper functions are defined for semantic retrieval:
  - `search_mem_facts(...)`
  - `search_mem_episodes(...)`
  - `search_mem_procedures(...)`

Operational behavior implemented in code:

- short-term memory (`st_turns`) acts as a TTL-backed ring buffer:
  - append new turn
  - trim older turns beyond configured cap
  - read only non-expired rows
- long-term memory upsert uses semantic merge-on-insert:
  - if cosine similarity to existing fact is `>= 0.92`, update existing fact instead of inserting duplicate
- retrieval queries set `ivfflat.probes = 10` to reduce misses on smaller datasets
- row-level security is enabled for memory tables, keyed by `app.user_id` context

Design takeaway:

- Supabase is not only “storage”; it handles both relational consistency (CRM/bookings) and semantic memory retrieval (pgvector), with explicit lifecycle controls (TTL, decay, prune, soft delete).

### Qdrant (Internal KB + Semantic Cache)

Qdrant is used for fast retrieval over knowledge chunks and for semantic response caching.

Two-collection architecture:

- `nawaloka`: persistent knowledge-base vectors for RAG
- `cag_cache`: semantic query-response cache for CAG

RAG collection design details:

- points store chunk vectors + payload metadata:
  - `chunk_text`, `url`, `title`, `strategy`, `chunk_index`
  - optional `parent_id`, `parent_text` for parent-child retrieval
- parent-child ingest strategy indexes children but keeps parent text in payload for richer generation context
- retriever deduplicates hits by `parent_id` before generation to avoid repeated context
- `ensure_kb_ingested()` auto-checks collection state and triggers ingestion when empty/missing

CAG cache design details:

- cache point payload schema:
  - `query`, `answer`, `evidence_urls`, `ts`
- lookup path:
  - embed query
  - KNN-1 search (`limit=1`) with cosine threshold (default `0.90`)
  - reject stale cache entries based on TTL (`86400s` default)
- miss path:
  - run CRAG pipeline
  - write answer back into `cag_cache` for semantic reuse

Design takeaway:

- separating KB retrieval and semantic cache into two Qdrant collections keeps retrieval behavior clear and avoids mixing document vectors with response-cache vectors.

### LangFuse Tracing and Prompt Management

LangFuse is wired as a first-class observability layer, not an afterthought.

Tracing model in this repo:

- decorators via `@observe(...)` create trace spans/generations for key pipeline steps
- `update_current_trace(...)` attaches runtime identity and route metadata:
  - `user_id`, `session_id`, tags
  - route/action/confidence/latency metadata
- `update_current_observation(...)` enriches spans with:
  - model name
  - prompt/response snippets
  - token usage payloads (`input`, `output`, `total`) when available

Examples of traced steps:

- agent loop: `agent_chat`
- memory: `memory_recall`, `memory_recall_inner`, `memory_distill`
- routing/tooling: `router`, `tool_dispatch`, `crm_dispatch`, `web_search`, `rag_search`
- generation: `synthesiser`, `distill_facts`, `cag_generate`

Prompt management pattern:

- prompts are fetched at runtime from LangFuse by name
- local fallback templates are used automatically if prompt fetch fails
- this enables prompt updates without redeploying code

Prompt sets used:

- agent/router/synthesiser prompt names:
  - `nawaloka-agent-system`
  - `nawaloka-router-system`
  - `nawaloka-router-user`
  - `nawaloka-synthesiser-system`
  - `nawaloka-synthesiser-user`
- memory prompt names:
  - `nawaloka-distill-system`
  - `nawaloka-distill-user`
  - `nawaloka-recall-system`
  - `nawaloka-recall-user`

Reliability behavior:

- observability is fail-open:
  - if disabled in config or keys are missing, execution continues with no-op tracing
- scripts explicitly call `flush()` at completion to push pending events

## Scripts and Operations

Operational scripts include:

- `scripts/init_supabase.py` for schema initialization
- `scripts/test_supabase.py` for DB and pgvector checks
- `scripts/ingest_to_qdrant.py` for KB chunk ingestion
- `scripts/seed_crm_unified.py` for deterministic SQL-first CRM seeding (LLM/template fallback)
- `scripts/seed_procedures.py` for procedural memory seeding and embedding backfill
- `scripts/rebuild_cag_cache.py` for cache reset + FAQ warm-up

## Practical Engineering Lessons From This Repo

- separate memory types by responsibility
- treat distillation as a write pipeline with policy, not just logging
- make routing output structured and validated
- keep retrieval robust with cache + correction layers
- use config-first design to avoid hardcoded behavior
- include observability hooks in every critical step
- keep data seeding deterministic for reproducible demos

## Tradeoffs Highlighted by This Repo

### Memory tradeoffs

- more memory increases personalization
- but can increase stale/noisy context risk without decay/pruning

### Routing tradeoffs

- strict routing improves tool correctness
- but requires robust JSON parsing and fallbacks

### Retrieval tradeoffs

- cache hits reduce cost and latency
- but thresholds that are too low can cause wrong semantic matches

### CRAG tradeoffs

- confidence-gated correction improves robustness
- but adds retrieval/generation latency on low-confidence queries

### Architecture tradeoffs

- Supabase + Qdrant separation is clean and scalable
- but adds operational setup complexity compared to single-store demos

## Skills and Technical Patterns

This repo covers:

- multi-memory architecture (ST, LT, episodic, procedural)
- LLM-based distillation and fact scoring
- token-budgeted hybrid memory recall
- JSON-structured routing with parameter extraction
- tool orchestration across CRM, internal KB RAG, and web search
- CAG semantic cache in Qdrant
- CRAG confidence-based corrective retrieval
- parent-child chunking and Qdrant vector ingestion
- Supabase schema design with pgvector and RLS
- LangFuse tracing and prompt management fallback design

## Notebook Order

Recommended order:

1. `notebooks/01_agentic_routing_engine.ipynb`
2. `notebooks/02_memory_capture_and_distill.ipynb`
3. `notebooks/03_memory_store_and_recall.ipynb`

## Project Structure

```text
Week 07/
|-- config/
|   |-- param.yaml
|   |-- models.yaml
|   `-- faqs.yaml
|-- data/
|   `-- knowledge_base/
|       |-- 01_staff_handbook.md
|       |-- ...
|       `-- 12_training_orientation.md
|-- notebooks/
|   |-- 01_agentic_routing_engine.ipynb
|   |-- 02_memory_capture_and_distill.ipynb
|   `-- 03_memory_store_and_recall.ipynb
|-- scripts/
|   |-- init_supabase.py
|   |-- test_supabase.py
|   |-- ingest_to_qdrant.py
|   |-- seed_crm_unified.py
|   |-- seed_procedures.py
|   `-- rebuild_cag_cache.py
|-- sql/
|   |-- supabase_schema.sql
|   |-- 01_specialties.sql
|   |-- 02_locations.sql
|   |-- 03_doctors.sql
|   |-- 04_patients.sql
|   |-- 05_bookings.sql
|   `-- 06_procedures.sql
|-- src/
|   |-- agents/
|   |-- memory/
|   |-- services/
|   `-- infrastructure/
|-- tests/
|   |-- test_memory_core.py
|   `-- test_memory_policies.py
|-- Makefile
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Final Takeaways

Week 07 moved from basic agent demos into **memory-centric agent system design**.

The biggest learning is that useful long-running assistants are built from:

- explicit memory lifecycle management
- strong routing discipline
- robust retrieval with cache/correction
- clear data architecture boundaries
- operational tooling and observability

This week made the agent feel less like a single prompt and more like a real software system.

## Validation Note

Running `pytest tests -q` in this environment failed during collection due to a local interpreter/dependency mismatch (`Python 3.13` with the installed SQLAlchemy stack). The project targets `Python 3.10+` and is configured for course-managed dependencies.

## Credits

This project was completed as part of the AI Engineer Essentials course by Zuu Crew, guided by tutor-led materials and exercises.
