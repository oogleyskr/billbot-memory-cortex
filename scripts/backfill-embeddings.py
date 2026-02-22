#!/usr/bin/env python3
"""One-time script to compute embeddings for all existing memories."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from middleware.db import get_memories_without_embeddings, store_embedding, init_db
from middleware.embeddings import compute_embedding, serialize_embedding


async def backfill(
    db_path="/home/mferr/.openclaw/memory-cortex/memories.db",
    embed_url="http://localhost:8105/embed",
):
    # Ensure the embedding column exists
    init_db(db_path)

    memories = get_memories_without_embeddings(db_path)
    print(f"Found {len(memories)} memories without embeddings")

    success = 0
    failed = 0
    for mem in memories:
        text = f"{mem['topic']}: {mem['fact']}"
        embedding = await compute_embedding(text, embed_url)
        if embedding:
            blob = serialize_embedding(embedding)
            store_embedding(db_path, mem["id"], blob)
            print(f"  [{mem['id']}] Embedded: {mem['fact'][:60]}...")
            success += 1
        else:
            print(f"  [{mem['id']}] FAILED: {mem['fact'][:60]}...")
            failed += 1

    print(f"Backfill complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    asyncio.run(backfill())
