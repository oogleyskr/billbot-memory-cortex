"""Conversation ingestion: chunk conversations and extract facts via the memory model."""

import json
import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger("memory-cortex.ingestion")

EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation snippet, extract discrete facts worth remembering long-term.

For each fact, output a JSON array of objects with these fields:
- "user_id": the user identifier (if identifiable), or null
- "topic": a short category (e.g., "preferences", "projects", "personal", "technical", "decisions")
- "fact": the specific fact in a clear, standalone sentence
- "importance": 1-10 (10 = critical personal info, 1 = trivial)

Rules:
- Only extract facts that would be useful to recall in future conversations
- Do NOT extract: greetings, small talk, bot responses, temporary states, or conversation mechanics
- DO extract: user preferences, decisions, project details, personal info shared voluntarily, technical choices, opinions
- Each fact must stand alone without needing the original conversation context
- If no facts are worth extracting, return an empty array: []

Output ONLY the JSON array, nothing else. /no_think"""


def chunk_conversation(messages: list[dict], chunk_size: int = 2048,
                       overlap: int = 256) -> list[list[dict]]:
    """Split a conversation into overlapping chunks.

    Uses a simple character-based approximation (4 chars ~= 1 token).
    """
    chars_per_token = 4
    max_chars = chunk_size * chars_per_token
    overlap_chars = overlap * chars_per_token

    chunks = []
    current_chunk = []
    current_size = 0

    for msg in messages:
        content = msg.get("content", "")
        msg_size = len(content) + len(msg.get("role", "")) + 10  # overhead

        if current_size + msg_size > max_chars and current_chunk:
            chunks.append(current_chunk)
            # Keep overlap from end of current chunk
            overlap_chunk = []
            overlap_size = 0
            for m in reversed(current_chunk):
                m_size = len(m.get("content", "")) + 10
                if overlap_size + m_size > overlap_chars:
                    break
                overlap_chunk.insert(0, m)
                overlap_size += m_size
            current_chunk = overlap_chunk
            current_size = overlap_size

        current_chunk.append(msg)
        current_size += msg_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def format_chunk_for_extraction(chunk: list[dict]) -> str:
    """Format a conversation chunk as text for the extraction model."""
    lines = []
    for msg in chunk:
        role = msg.get("role", "unknown")
        name = msg.get("name", role)
        content = msg.get("content", "")
        if content:
            lines.append(f"[{name}]: {content}")
    return "\n".join(lines)


async def extract_facts(base_url: str, chunk_text: str,
                        session_id: Optional[str] = None,
                        channel: Optional[str] = None) -> list[dict]:
    """Send a conversation chunk to the memory model and extract facts."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "memory",
                    "messages": [
                        {"role": "system", "content": EXTRACTION_PROMPT},
                        {"role": "user", "content": chunk_text},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2048,
                },
            )
            response.raise_for_status()
            data = response.json()

            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""

            # Strip thinking tags if present (Qwen3 thinking mode fallback)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            facts = json.loads(content)

            if not isinstance(facts, list):
                logger.warning("Model returned non-list: %s", type(facts))
                return []

            # Attach source metadata
            for fact in facts:
                fact["source_session"] = session_id
                fact["source_channel"] = channel

            return facts

        except json.JSONDecodeError as e:
            logger.error("Failed to parse model output as JSON: %s", e)
            logger.debug("Raw output: %s", content if 'content' in dir() else "N/A")
            return []
        except httpx.HTTPError as e:
            logger.error("HTTP error calling memory model: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error during extraction: %s", e)
            return []


async def ingest_conversation(base_url: str, db_path: str,
                              messages: list[dict],
                              session_id: Optional[str] = None,
                              channel: Optional[str] = None,
                              user_id: Optional[str] = None,
                              chunk_size: int = 2048,
                              chunk_overlap: int = 256,
                              config: dict | None = None) -> dict:
    """Full ingestion pipeline: chunk, extract, store, embed."""
    from . import db
    from .embeddings import compute_embedding, serialize_embedding

    # Filter to only user messages and assistant responses with content
    relevant = [m for m in messages if m.get("content")]

    if not relevant:
        return {"chunks": 0, "facts_extracted": 0, "facts_stored": 0, "facts_embedded": 0}

    chunks = chunk_conversation(relevant, chunk_size, chunk_overlap)
    total_extracted = 0
    total_stored = 0
    total_embedded = 0

    embed_url = (config or {}).get("embeddings", {}).get("url", "http://localhost:8105/embed")

    for chunk in chunks:
        chunk_text = format_chunk_for_extraction(chunk)
        facts = await extract_facts(base_url, chunk_text, session_id, channel)

        # Override user_id if provided (for DM sessions where we know the user)
        if user_id:
            for fact in facts:
                fact["user_id"] = user_id

        stored_count, stored_ids = db.store_memories(db_path, facts)
        total_extracted += len(facts)
        total_stored += stored_count

        # Compute and store embeddings for newly inserted facts
        for fact, fid in zip(facts, stored_ids):
            text = f"{fact.get('topic', '')}: {fact.get('fact', '')}"
            try:
                embedding = await compute_embedding(text, embed_url)
                if embedding:
                    blob = serialize_embedding(embedding)
                    db.store_embedding(db_path, fid, blob)
                    total_embedded += 1
                else:
                    logger.warning("No embedding returned for memory %d", fid)
            except Exception as e:
                logger.error("Failed to embed memory %d: %s", fid, e)

    logger.info(
        "Ingested %d chunks, extracted %d facts, stored %d, embedded %d",
        len(chunks), total_extracted, total_stored, total_embedded,
    )

    return {
        "chunks": len(chunks),
        "facts_extracted": total_extracted,
        "facts_stored": total_stored,
        "facts_embedded": total_embedded,
    }
