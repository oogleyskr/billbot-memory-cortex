"""Memory Cortex HTTP server - bridges OpenClaw <-> Memory Model <-> SQLite."""

import asyncio
import logging
import os
import sys
import time

import yaml
from aiohttp import web

from . import db, ingestion, recall
from .embeddings import compute_embedding, cosine_similarity, deserialize_embedding

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
            config=config,
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
    count, _ids = db.store_memories(db_path, [memory])

    return web.json_response({"status": "stored", "count": count})


async def handle_stats(request: web.Request) -> web.Response:
    """Get database statistics.

    GET /stats
    """
    config = request.app["config"]
    db_path = config["database"]["path"]
    stats = db.get_stats(db_path)
    return web.json_response(stats)


async def handle_hybrid_search(request: web.Request) -> web.Response:
    """Hybrid search combining FTS5 keyword search with vector similarity.

    POST /hybrid-search
    Body: {
        "query": "...",          # required
        "user_id": "...",        # optional, filter by user
        "limit": 10              # optional, default 10
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
    limit = body.get("limit", 10)

    db_path = config["database"]["path"]
    embed_url = config.get("embeddings", {}).get("url", "http://localhost:8105/embed")

    # 1. FTS5 keyword search
    fts_results = db.search_memories(db_path, query, user_id=user_id, limit=limit * 2)

    # Normalize FTS5 ranks to 0-1 (rank is negative, lower = better match)
    fts_scores: dict[int, float] = {}
    if fts_results:
        ranks = [r["rank"] for r in fts_results]
        min_rank = min(ranks)  # Most negative = best match
        max_rank = max(ranks)  # Least negative = worst match
        rank_range = max_rank - min_rank
        for r in fts_results:
            if rank_range != 0:
                # Invert: best match (most negative) gets score 1.0
                fts_scores[r["id"]] = 1.0 - (r["rank"] - min_rank) / rank_range
            else:
                fts_scores[r["id"]] = 1.0

    # Build lookup of FTS results by ID
    fts_by_id: dict[int, dict] = {r["id"]: r for r in fts_results}

    # 2. Vector search
    vector_scores: dict[int, float] = {}
    vector_by_id: dict[int, dict] = {}
    source_label = "fts5"

    try:
        query_embedding = await compute_embedding(query, embed_url=embed_url)
        if query_embedding is not None:
            source_label = "fts5+vector"
            mem_rows = db.get_memories_with_embeddings(
                db_path, user_id=user_id, limit=500
            )
            for row in mem_rows:
                emb = deserialize_embedding(row["embedding"])
                if emb is not None:
                    sim = cosine_similarity(query_embedding, emb)
                    # Normalize cosine similarity from [-1,1] to [0,1]
                    score = (sim + 1.0) / 2.0
                    vector_scores[row["id"]] = score
                    vector_by_id[row["id"]] = row
    except Exception as e:
        logger.warning("Vector search failed, falling back to FTS5 only: %s", e)

    # 3. Merge scores
    all_ids = set(fts_scores.keys()) | set(vector_scores.keys())
    scored: list[tuple[int, float]] = []
    for mid in all_ids:
        fts_s = fts_scores.get(mid, 0.0)
        vec_s = vector_scores.get(mid, 0.0)
        if vector_scores:
            final = 0.4 * fts_s + 0.6 * vec_s
        else:
            final = fts_s
        scored.append((mid, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:limit]

    # 4. Build response
    results = []
    for mid, score in top:
        mem = fts_by_id.get(mid) or vector_by_id.get(mid)
        if mem is None:
            continue
        results.append({
            "id": mem["id"],
            "user_id": mem.get("user_id"),
            "topic": mem["topic"],
            "fact": mem["fact"],
            "importance": mem["importance"],
            "score": round(score, 4),
            "source": source_label,
        })

    return web.json_response({"results": results, "count": len(results)})


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
    app.router.add_post("/hybrid-search", handle_hybrid_search)

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
