"""Memory Cortex HTTP server - bridges OpenClaw <-> Memory Model <-> SQLite."""

import asyncio
import logging
import os
import sys
import time

import yaml
from aiohttp import web

from . import db, ingestion, recall

logger = logging.getLogger("memory-cortex")


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# Debounce tracking for ingestion
_debounce_tasks: dict[str, asyncio.Task] = {}


async def handle_health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    config = request.app["config"]
    db_path = config["database"]["path"]
    stats = db.get_stats(db_path)
    return web.json_response({
        "status": "ok",
        "timestamp": time.time(),
        "stats": stats,
    })


async def handle_ingest(request: web.Request) -> web.Response:
    """Ingest a conversation for memory extraction.

    POST /ingest
    Body: {
        "messages": [...],       # conversation messages
        "session_id": "...",     # optional
        "channel": "...",        # optional
        "user_id": "...",        # optional
        "debounce": true         # optional, default true
    }
    """
    config = request.app["config"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    messages = body.get("messages", [])
    if not messages:
        return web.json_response({"error": "No messages provided"}, status=400)

    session_id = body.get("session_id")
    channel = body.get("channel")
    user_id = body.get("user_id")
    should_debounce = body.get("debounce", True)

    base_url = config["model"]["base_url"]
    db_path = config["database"]["path"]
    chunk_size = config["ingestion"]["chunk_size"]
    chunk_overlap = config["ingestion"]["chunk_overlap"]
    debounce_secs = config["ingestion"]["debounce_seconds"]

    async def do_ingest():
        result = await ingestion.ingest_conversation(
            base_url, db_path, messages,
            session_id=session_id, channel=channel, user_id=user_id,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        logger.info("Ingestion complete for session=%s: %s", session_id, result)

    if should_debounce and session_id:
        # Cancel previous debounce for this session
        key = f"{session_id}:{channel or ''}"
        if key in _debounce_tasks:
            _debounce_tasks[key].cancel()

        async def debounced():
            await asyncio.sleep(debounce_secs)
            await do_ingest()
            _debounce_tasks.pop(key, None)

        _debounce_tasks[key] = asyncio.create_task(debounced())
        return web.json_response({
            "status": "debounced",
            "debounce_seconds": debounce_secs,
        })
    else:
        # Run immediately in background
        asyncio.create_task(do_ingest())
        return web.json_response({"status": "accepted"})


async def handle_recall(request: web.Request) -> web.Response:
    """Recall memories relevant to a query.

    POST /recall
    Body: {
        "query": "...",          # what to search for
        "user_id": "...",        # optional, filter by user
    }
    """
    config = request.app["config"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    query = body.get("query", "").strip()
    if not query:
        return web.json_response({"error": "No query provided"}, status=400)

    user_id = body.get("user_id")

    base_url = config["model"]["base_url"]
    db_path = config["database"]["path"]
    top_k = config["recall"]["top_k"]
    max_results = config["recall"]["max_results"]
    max_synthesis_tokens = config["recall"]["max_synthesis_tokens"]

    result = await recall.recall_memories(
        base_url, db_path, query,
        user_id=user_id, top_k=top_k, max_results=max_results,
        max_synthesis_tokens=max_synthesis_tokens,
    )

    return web.json_response(result)


async def handle_search(request: web.Request) -> web.Response:
    """Raw memory search without synthesis (for debugging/inspection).

    GET /search?q=...&user_id=...&limit=20
    """
    config = request.app["config"]
    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"error": "No query (q) provided"}, status=400)

    user_id = request.query.get("user_id")
    limit = int(request.query.get("limit", "20"))

    db_path = config["database"]["path"]
    results = db.search_memories(db_path, query, user_id=user_id, limit=limit)

    return web.json_response({"results": results, "count": len(results)})


async def handle_recent(request: web.Request) -> web.Response:
    """Get recent memories.

    GET /recent?user_id=...&limit=10
    """
    config = request.app["config"]
    user_id = request.query.get("user_id")
    limit = int(request.query.get("limit", "10"))

    db_path = config["database"]["path"]
    results = db.get_recent_memories(db_path, user_id=user_id, limit=limit)

    return web.json_response({"results": results, "count": len(results)})


async def handle_store(request: web.Request) -> web.Response:
    """Manually store a single memory.

    POST /store
    Body: {
        "user_id": "...",
        "topic": "...",
        "fact": "...",
        "importance": 5
    }
    """
    config = request.app["config"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    fact = body.get("fact", "").strip()
    topic = body.get("topic", "").strip()
    if not fact or not topic:
        return web.json_response(
            {"error": "Both 'fact' and 'topic' are required"}, status=400
        )

    memory = {
        "user_id": body.get("user_id"),
        "topic": topic,
        "fact": fact,
        "importance": body.get("importance", 5),
    }

    db_path = config["database"]["path"]
    count = db.store_memories(db_path, [memory])

    return web.json_response({"status": "stored", "count": count})


async def handle_stats(request: web.Request) -> web.Response:
    """Get database statistics.

    GET /stats
    """
    config = request.app["config"]
    db_path = config["database"]["path"]
    stats = db.get_stats(db_path)
    return web.json_response(stats)


def create_app(config: dict) -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app["config"] = config

    # Initialize database
    db.init_db(config["database"]["path"])

    # Routes
    app.router.add_get("/health", handle_health)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/recall", handle_recall)
    app.router.add_get("/search", handle_search)
    app.router.add_get("/recent", handle_recent)
    app.router.add_post("/store", handle_store)
    app.router.add_get("/stats", handle_stats)

    return app


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)

    host = config["server"]["host"]
    port = config["server"]["port"]

    logger.info("Starting Memory Cortex on %s:%d", host, port)
    logger.info("Database: %s", config["database"]["path"])
    logger.info("Model endpoint: %s", config["model"]["base_url"])

    app = create_app(config)
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
