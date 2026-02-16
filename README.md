# BillBot Memory Cortex

Long-term memory management system for [BillBot](https://github.com/oogleyskr/billbot). A Python/aiohttp middleware service that ingests conversations, extracts discrete facts via a dedicated LLM, and stores them in a full-text-searchable SQLite database for instant recall.

**825 lines of Python across 4 modules.** Runs entirely on local hardware with zero cloud dependencies.

---

## Architecture

```
              ┌───────────────────────────────────┐
              │   BillBot (gpt-oss-120b on DGX)   │
              │      via OpenClaw memory skill     │
              └───────────────┬───────────────────┘
                              │ HTTP
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WSL2 (Ubuntu)                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │          Memory Cortex Middleware (port 8300)              │  │
│  │                  Python / aiohttp                          │  │
│  │                                                            │  │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐  │  │
│  │  │ server.py│  │ ingestion.py │  │     recall.py       │  │  │
│  │  │ 266 LOC  │  │   184 LOC    │  │     124 LOC         │  │  │
│  │  └────┬─────┘  └──────┬───────┘  └──────────┬──────────┘  │  │
│  │       │               │                      │             │  │
│  │       ▼               ▼                      ▼             │  │
│  │  ┌────────────────────────────────────────────────────┐    │  │
│  │  │                  db.py (251 LOC)                    │    │  │
│  │  │           SQLite + FTS5 + WAL mode                 │    │  │
│  │  │         Porter stemming, Unicode61                 │    │  │
│  │  └────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │ HTTP (172.17.96.1:8301)
                               ▼
              ┌────────────────────────────────────┐
              │        Windows 11 (Native)         │
              │                                    │
              │   llama.cpp Vulkan (llama-server)  │
              │   Qwen3-8B Q8_0  (~8.2 GB VRAM)   │
              │   AMD Radeon VII  (16 GB HBM2)     │
              │   Port 8301                        │
              └────────────────────────────────────┘
```

The middleware runs in WSL2 and reaches the Windows-native llama-server through the WSL2 NAT gateway IP (`172.17.96.1`). BillBot (or any HTTP client) calls the middleware on port 8300.

---

## How It Works

### Ingestion Pipeline (async, non-blocking)

When a conversation occurs, BillBot sends the message history to the middleware for background processing:

```
Messages ──▶ Chunk (~2K tokens, 256 overlap) ──▶ LLM Fact Extraction ──▶ SQLite + FTS5
```

1. **Receive** -- `POST /ingest` accepts conversation messages with optional session/channel/user metadata.
2. **Debounce** -- If `debounce: true` (default), the middleware waits 30 seconds after the last call before processing, preventing duplicate work during active conversations.
3. **Chunk** -- Messages are split into ~2048-token segments with 256-token overlap to preserve context across boundaries. Token count is approximated at 4 characters per token.
4. **Extract** -- Each chunk is sent to the Qwen3-8B model with a structured extraction prompt. The model returns a JSON array of discrete facts, each with `user_id`, `topic`, `fact`, and `importance` (1-10 scale).
5. **Store** -- Extracted facts are inserted into the `memories` table. SQLite triggers automatically keep the FTS5 index in sync.

### Recall Pipeline (~2-4 seconds)

When BillBot needs to remember something, it queries the middleware:

```
Query ──▶ FTS5 Search ──▶ Top-K Selection ──▶ LLM Synthesis ──▶ Natural Language Response
```

1. **Search** -- The query is tokenized and matched against the FTS5 index using OR logic for broad coverage. Results are ranked by FTS5 relevance score.
2. **Fallback** -- If FTS5 returns no matches, the system falls back to the most recent memories.
3. **Select** -- The top 8 results (configurable) are formatted with timestamps, user IDs, topics, and importance scores.
4. **Synthesize** -- The selected memories and the original query are sent to Qwen3-8B, which produces a concise natural-language summary of relevant information.
5. **Return** -- The synthesis is returned along with metadata (`memories_searched`, `memories_used`).

---

## API Reference

### `GET /health`

Health check with database statistics.

```bash
curl http://localhost:8300/health
```

```json
{
  "status": "ok",
  "timestamp": 1739587200.0,
  "stats": {
    "total_memories": 142,
    "total_summaries": 12,
    "unique_users": 3,
    "unique_topics": 18
  }
}
```

### `POST /ingest`

Ingest a conversation for background fact extraction.

```bash
curl -X POST http://localhost:8300/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "name": "oogley", "content": "I prefer Rust for systems work"},
      {"role": "assistant", "content": "Good choice! Rust is great for that."}
    ],
    "session_id": "abc123",
    "channel": "general",
    "user_id": "oogley",
    "debounce": true
  }'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | Yes | Conversation messages (`role`, `content`, optional `name`) |
| `session_id` | string | No | Session identifier (used for debounce grouping) |
| `channel` | string | No | Channel identifier |
| `user_id` | string | No | Override user_id on all extracted facts |
| `debounce` | boolean | No | Wait before processing (default: `true`) |

Response: `{"status": "debounced", "debounce_seconds": 30}` or `{"status": "accepted"}`

### `POST /recall`

Search memories and synthesize a natural-language response.

```bash
curl -X POST http://localhost:8300/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "What programming languages does oogley prefer?", "user_id": "oogley"}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `user_id` | string | No | Filter memories by user |

```json
{
  "response": "Oogley prefers Rust for systems programming and has expressed interest in Go for network services.",
  "memories_searched": 15,
  "memories_used": 4
}
```

### `POST /store`

Manually store a single memory without LLM extraction.

```bash
curl -X POST http://localhost:8300/store \
  -H "Content-Type: application/json" \
  -d '{"user_id": "oogley", "topic": "preferences", "fact": "Prefers dark themes in all editors", "importance": 7}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | string | No | User identifier |
| `topic` | string | Yes | Category (e.g., `preferences`, `projects`, `technical`) |
| `fact` | string | Yes | The fact to store, as a standalone sentence |
| `importance` | integer | No | 1-10 scale (default: 5) |

### `GET /search`

Raw FTS5 search without LLM synthesis. Useful for debugging and inspection.

```bash
curl "http://localhost:8300/search?q=rust+programming&user_id=oogley&limit=10"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | Search query |
| `user_id` | string | No | Filter by user |
| `limit` | integer | No | Max results (default: 20) |

### `GET /recent`

Retrieve most recent memories, optionally filtered by user.

```bash
curl "http://localhost:8300/recent?user_id=oogley&limit=5"
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | No | Filter by user |
| `limit` | integer | No | Max results (default: 10) |

### `GET /stats`

Database statistics.

```bash
curl http://localhost:8300/stats
```

```json
{
  "total_memories": 142,
  "total_summaries": 12,
  "unique_users": 3,
  "unique_topics": 18
}
```

---

## Getting Started

### Prerequisites

- **WSL2** with Python 3.12+
- **Windows 11** with AMD Radeon VII (Vulkan 1.3 support)
- **llama.cpp** Vulkan build (b8012 or later)
- **Qwen3-8B Q8_0 GGUF** model file

### 1. Clone the repository

```bash
git clone https://github.com/oogleyskr/billbot-memory-cortex.git
cd billbot-memory-cortex
```

### 2. Start the Memory Model (Windows)

Run the llama-server as a Windows-native process to use the Radeon VII GPU:

```batch
cd C:\Users\mferr\llama-server-vulkan
start-radeonvii.bat
```

Or start manually from WSL2:

```bash
GGML_VK_VISIBLE_DEVICES=0 /mnt/c/Users/mferr/llama-server-vulkan/llama-server.exe \
  -m "C:\Users\mferr\models\Qwen3-8B-Q8_0.gguf" \
  --port 8301 --host 0.0.0.0 -ngl 99 --ctx-size 32768 --jinja -t 4
```

### 3. Start the Middleware (WSL2)

The start script creates a Python virtual environment on first run and installs dependencies automatically:

```bash
bash /home/mferr/billbot-memory-cortex/scripts/start.sh
```

### 4. Verify

```bash
curl http://localhost:8300/health
```

### 5. Stop

```bash
bash /home/mferr/billbot-memory-cortex/scripts/stop.sh
```

---

## Configuration

All settings live in `config/config.yaml`:

```yaml
# Middleware server
server:
  host: "0.0.0.0"
  port: 8300

# Memory model (llama.cpp on Radeon VII)
model:
  base_url: "http://172.17.96.1:8301/v1"   # WSL2 NAT gateway -> Windows
  context_size: 32768
  max_tokens: 2048

# SQLite storage
database:
  path: "/home/mferr/.openclaw/memory-cortex/memories.db"

# Ingestion settings
ingestion:
  chunk_size: 2048         # Tokens per chunk (approximate)
  chunk_overlap: 256       # Overlap between chunks
  max_concurrent: 2        # Max parallel ingestion tasks
  debounce_seconds: 30     # Wait after last message before processing

# Recall settings
recall:
  max_results: 20          # Max memories from FTS5 search
  top_k: 8                 # Max memories sent to LLM for synthesis
  max_synthesis_tokens: 1024
```

### Key Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ingestion.chunk_size` | 2048 | Larger chunks give more context but slower extraction |
| `ingestion.debounce_seconds` | 30 | Lower values mean more frequent ingestion during active chats |
| `recall.top_k` | 8 | More memories = richer synthesis but slower response |
| `recall.max_synthesis_tokens` | 1024 | Controls length of synthesized response |

---

## Database Schema

### `memories` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (autoincrement) |
| `user_id` | TEXT | User identifier (nullable) |
| `topic` | TEXT | Category (e.g., preferences, projects, technical) |
| `fact` | TEXT | The extracted fact as a standalone sentence |
| `source_session` | TEXT | Originating session ID |
| `source_channel` | TEXT | Originating channel |
| `importance` | INTEGER | 1-10 scale (default: 5) |
| `created_at` | REAL | Unix timestamp of creation |
| `last_accessed_at` | REAL | Unix timestamp of last recall access |

Indexes on: `user_id`, `topic`, `importance DESC`, `created_at DESC`.

### `summaries` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (autoincrement) |
| `session_id` | TEXT | Session identifier |
| `channel` | TEXT | Channel identifier |
| `user_id` | TEXT | User identifier |
| `summary` | TEXT | Session summary text |
| `message_count` | INTEGER | Number of messages summarized |
| `created_at` | REAL | Unix timestamp |

### `memories_fts` (FTS5 virtual table)

A content-synced FTS5 virtual table over `memories` with `porter unicode61` tokenization. Columns indexed: `user_id`, `topic`, `fact`. Kept in sync via `AFTER INSERT/UPDATE/DELETE` triggers.

Database runs in **WAL mode** with `PRAGMA synchronous=NORMAL` for concurrent read performance.

---

## Project Structure

```
billbot-memory-cortex/
├── middleware/              # Python HTTP service (aiohttp)
│   ├── __init__.py          # Package marker
│   ├── server.py            # HTTP server, 7 endpoints, debounce logic (266 LOC)
│   ├── ingestion.py         # Conversation chunking + LLM fact extraction (184 LOC)
│   ├── recall.py            # FTS5 search + LLM synthesis pipeline (124 LOC)
│   └── db.py                # SQLite+FTS5 schema, WAL mode, CRUD operations (251 LOC)
├── config/
│   └── config.yaml          # Server, model, database, ingestion, recall settings
├── scripts/
│   ├── start.sh             # Start middleware (creates venv on first run)
│   └── stop.sh              # Graceful shutdown with SIGTERM, fallback SIGKILL
├── requirements.txt         # aiohttp, httpx, pyyaml
├── .gitignore
└── README.md
```

---

## Performance

| Metric | Value |
|--------|-------|
| Model inference | 45-57 tok/s (Radeon VII Vulkan) |
| Prompt processing | 43-48 tok/s |
| Recall latency | 2-4 seconds (FTS5 search + LLM synthesis) |
| Ingestion | Async, non-blocking (background tasks) |
| VRAM usage | ~8.2 GB model + ~7 GB KV cache headroom |
| Middleware RAM | Minimal (aiohttp + SQLite, <50 MB) |
| Database | WAL mode, concurrent reads, no locking issues |

---

## Hardware

### AMD Radeon VII

| Spec | Value |
|------|-------|
| GPU | AMD Radeon VII (Vega 20) |
| VRAM | 16 GB HBM2 (1 TB/s bandwidth) |
| Backend | llama.cpp Vulkan 1.3.217 (Windows inbox AMD driver) |
| Model | Qwen3-8B Q8_0 (~8.2 GB) |
| Context | 32768 tokens |
| Build | llama.cpp b8012+ |

**No AMD Adrenalin drivers required** -- the Windows inbox Vulkan driver provides full Vulkan 1.3 support for the Radeon VII. The llama-server runs as a native Windows process on port 8301, and WSL2 accesses it through the NAT gateway IP `172.17.96.1`.

---

## Related Repos

| Repository | Description |
|------------|-------------|
| [oogleyskr/billbot](https://github.com/oogleyskr/billbot) | OpenClaw fork with DGX Spark integration (primary bot) |
| [oogleyskr/billbot-android](https://github.com/oogleyskr/billbot-android) | Android companion app (Kotlin + Jetpack Compose) |
| [oogleyskr/billbot-mcpjungle](https://github.com/oogleyskr/billbot-mcpjungle) | MCPJungle gateway configs and scripts |
| [oogleyskr/billbot-service-manager](https://github.com/oogleyskr/billbot-service-manager) | Service manager MCP server |
| [oogleyskr/billbot-workspace](https://github.com/oogleyskr/billbot-workspace) | BillBot workspace files and config |
| [oogleyskr/multimodal-stack](https://github.com/oogleyskr/multimodal-stack) | Local multimodal AI services (RTX 3090) |

---

## License

MIT
