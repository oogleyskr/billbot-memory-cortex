# BillBot Memory Cortex

Long-term memory management system for BillBot, running on a dedicated Radeon VII GPU (16GB HBM2) via llama.cpp Vulkan backend.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │  BillBot (gpt-oss-120b on DGX)  │
                    │     via OpenClaw memory skill    │
                    └──────────┬──────────────────────┘
                               │ HTTP (port 8300)
                    ┌──────────▼──────────────────────┐
                    │    Memory Cortex Middleware      │
                    │    (Python/aiohttp, WSL2)        │
                    └──┬───────────────────────────┬──┘
                       │                           │
              ┌────────▼────────┐         ┌────────▼────────┐
              │   SQLite+FTS5   │         │   Qwen3-8B Q8_0 │
              │   (memories.db) │         │   Radeon VII GPU │
              └─────────────────┘         │   (Vulkan, 8301) │
                                          └──────────────────┘
```

- **Memory Model**: Qwen3-8B Q8_0 (~8.2GB) on Radeon VII via llama.cpp Vulkan
- **Storage**: SQLite with FTS5 full-text search (porter stemming + unicode)
- **Middleware**: Python HTTP service (aiohttp) on port 8300
- **Integration**: OpenClaw skill with bash CLI wrapper

## Performance

| Metric | Value |
|--------|-------|
| Model inference | ~45-57 tok/s (Radeon VII Vulkan) |
| Prompt processing | ~43-48 tok/s |
| Recall latency | ~2-4 seconds (search + synthesis) |
| Ingestion | Async, non-blocking |
| VRAM usage | ~8.2GB model + ~7GB KV cache headroom |

## How It Works

### Ingestion (async, non-blocking)
1. Conversation events POST to `/ingest` with message history
2. Messages are chunked into ~2K token segments with overlap
3. Memory model extracts discrete facts as structured JSON
4. Facts stored in SQLite with FTS5 indexing
5. Debounce prevents duplicate processing (30s default)

### Recall (tool call, ~2-4s)
1. BillBot calls the memory skill: `memory.sh recall "query"`
2. Middleware searches SQLite FTS5 for matching memories
3. Top-k results sent to memory model for synthesis
4. Natural language response returned to BillBot

### Manual Storage
Store facts directly without model processing:
```bash
memory.sh store "oogley" "preferences" "Prefers Rust for systems programming" 8
```

## Setup

### Prerequisites
- Windows 11 with Radeon VII (Vulkan 1.3 via inbox driver)
- WSL2 with Python 3.12
- llama.cpp Vulkan build (b8012+)

### 1. Start the Memory Model (Windows)
```batch
cd C:\Users\mferr\llama-server-vulkan
start-radeonvii.bat
```
Or from WSL2:
```bash
GGML_VK_VISIBLE_DEVICES=0 /mnt/c/Users/mferr/llama-server-vulkan/llama-server.exe \
  -m "C:\Users\mferr\models\Qwen3-8B-Q8_0.gguf" \
  --port 8301 --host 0.0.0.0 -ngl 99 --ctx-size 32768 --jinja -t 4
```

### 2. Start the Middleware (WSL2)
```bash
bash /home/mferr/billbot-memory-cortex/scripts/start.sh
```

### 3. Verify
```bash
curl http://localhost:8300/health
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with stats |
| `/ingest` | POST | Ingest conversation for fact extraction |
| `/recall` | POST | Search + synthesize memories |
| `/store` | POST | Manually store a single memory |
| `/search` | GET | Raw FTS5 search (no synthesis) |
| `/recent` | GET | Recent memories |
| `/stats` | GET | Database statistics |

## Project Structure

```
├── middleware/          # Python HTTP service (aiohttp)
│   ├── server.py       # HTTP server with all endpoints
│   ├── ingestion.py    # Conversation chunking + fact extraction
│   ├── recall.py       # Memory search + synthesis
│   └── db.py           # SQLite+FTS5 schema and operations
├── config/
│   └── config.yaml     # Service configuration
├── scripts/
│   ├── start.sh        # Start middleware (creates venv if needed)
│   └── stop.sh         # Stop middleware
├── requirements.txt    # Python dependencies
└── README.md
```

## Hardware

- **GPU**: AMD Radeon VII (16GB HBM2, 1TB/s bandwidth)
- **Backend**: llama.cpp b8012, Vulkan 1.3.217 (Windows inbox AMD driver)
- **Host**: Windows 11 native process, WSL2 reaches via gateway IP
- **Note**: No AMD Adrenalin drivers needed - inbox Vulkan driver works

## Networking

The Radeon VII runs llama-server as a Windows-native process (port 8301). WSL2 accesses it via the NAT gateway IP (`172.17.96.1:8301`). The middleware runs in WSL2 on port 8300, accessible from both WSL2 and Windows.
