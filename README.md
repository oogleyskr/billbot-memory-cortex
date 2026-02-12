# BillBot Memory Cortex

Long-term memory management system for BillBot, running on a dedicated Radeon VII GPU (16GB HBM2) via llama.cpp Vulkan backend.

## Architecture

- **Memory Model**: Qwen3-8B Q8_0 running on Radeon VII via llama.cpp (Vulkan)
- **Storage**: SQLite with FTS5 full-text search
- **Middleware**: Python HTTP service bridging OpenClaw <-> Memory Model <-> SQLite
- **Integration**: Exposed as an OpenClaw tool for BillBot's main model (gpt-oss-120b on DGX Spark)

## How It Works

### Ingestion (async, non-blocking)
1. Conversation events flow from OpenClaw to the middleware
2. Conversations are chunked into manageable segments
3. Memory model extracts facts, preferences, and decisions
4. Extracted knowledge stored in SQLite

### Recall (tool call)
1. BillBot's main model calls the memory tool
2. Middleware searches SQLite (FTS5) for relevant memories
3. Top matches sent to memory model for synthesis
4. Synthesized response returned to main model

## Hardware

- **GPU**: AMD Radeon VII (16GB HBM2, 1TB/s bandwidth)
- **Backend**: llama.cpp with Vulkan (using Windows inbox AMD driver)
- **Host**: Windows 11 native process, accessible from WSL2 via localhost

## Status

Under construction.

## Project Structure

```
├── middleware/          # Python HTTP service
│   ├── server.py       # Main middleware server
│   ├── ingestion.py    # Conversation chunking and fact extraction
│   ├── recall.py       # Memory search and synthesis
│   └── db.py           # SQLite schema and operations
├── config/             # Configuration files
│   └── config.yaml     # Service configuration
├── scripts/            # Management scripts
│   ├── start.sh        # Start the memory cortex
│   └── stop.sh         # Stop the memory cortex
└── README.md
```
