# Palm Mind RAG Backend

FastAPI backend with two REST APIs:

1. **Document Ingestion API**: upload `.pdf`/`.txt`, extract text, chunk (two selectable strategies), embed, store vectors in **Qdrant**, save metadata in **SQL** (SQLite by default, Postgres-ready via SQLAlchemy).
2. **Conversational RAG API**: custom RAG pipeline (no `RetrievalQAChain`, no LangChain), **Redis** chat memory, multi-turn, and LLM-driven **interview booking** (name, email, date, time) stored in SQL.

No FAISS/Chroma, no UI.

## Architecture

```
app/
  main.py                 # app factory, lifespan wiring, error handlers
  api/deps.py             # dependency injection (per-request services)
  api/routes/             # ingestion.py, chat.py (thin HTTP layer)
  core/                   # config (pydantic-settings), container, exceptions
  db/                     # SQLAlchemy models + repositories (async)
  schemas/                # Pydantic request/response models
  services/
    text_extraction.py    # pdf/txt -> text
    chunking.py           # FixedSizeChunker, SentenceChunker (Strategy pattern)
    embeddings.py         # OpenAI embeddings
    vector_store.py       # Qdrant wrapper
    memory.py             # Redis chat history + booking state
    llm.py                # chat completion wrapper
    booking.py            # LLM slot extraction + validation + persistence
    rag.py                # condense -> retrieve -> generate
    ingestion.py          # ingestion orchestration
```

### Chunking strategies (`chunking_strategy` form field)
- `fixed`: sliding character window with overlap.
- `sentence`: packs whole sentences/paragraphs up to `chunk_size`, carrying trailing sentences as overlap.

### RAG flow
1. Load history from Redis.
2. One LLM call classifies intent and extracts booking slots (JSON mode).
3. **Question:** condense follow-up into a standalone query, embed, search Qdrant (optionally filtered by `document_id`), answer from numbered context with citations.
4. **Booking:** merge extracted slots with state saved in Redis, validate (email format, date not in past, HH:MM), ask for whatever is missing, and once complete save to the `bookings` table.
5. Save the turn to Redis (capped length + TTL).

A booking can be interrupted by a normal question; the partial booking stays in Redis until finished or cancelled.

## Run

```bash
# Free local LLM and embeddings via Ollama (no paid API key required)
ollama pull llama3:latest
ollama pull nomic-embed-text

# Copy settings already configured for local Ollama:
cp .env.example .env

docker compose up -d            # Qdrant + Redis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

Run tests and strict type-checking in another terminal:
```bash
pytest -q
mypy app tests
ruff check app tests
ruff format --check app tests
```

## Examples

```bash
# Ingest
curl -F "file=@notes.pdf" -F "chunking_strategy=sentence" \
     -F "chunk_size=800" -F "chunk_overlap=150" \
     http://localhost:8000/api/v1/documents

# Chat (multi-turn: reuse session_id)
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id":"s1","message":"What does the document say about refunds?"}'
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id":"s1","message":"And for international orders?"}'

# Booking
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id":"s1","message":"I want to book an interview, I am Jordan Example"}'
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
     -d '{"session_id":"s1","message":"jordan@example.com, tomorrow at 3pm"}'

curl http://localhost:8000/api/v1/bookings
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/documents` | Ingest a document |
| GET | `/api/v1/documents`, `/api/v1/documents/{id}` | Document metadata |
| POST | `/api/v1/chat` | Conversational RAG / booking |
| GET | `/api/v1/bookings` | List stored bookings |

## Design notes
- Fully async (AsyncOpenAI, AsyncQdrantClient, redis.asyncio, async SQLAlchemy).
- If the metadata insert fails after vectors are written, vectors are deleted (compensating action).
- Typed throughout (`mypy --strict` config in `pyproject.toml`); services depend on small interfaces, wired in `main.py`/`deps.py`.
- Swap vector DB or LLM provider by replacing one service class.
