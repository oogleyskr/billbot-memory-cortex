"""Memory recall: search stored memories and synthesize a response."""

import json
import logging
import re
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("memory-cortex.recall")

SYNTHESIS_PROMPT = """You are a memory recall assistant. You have been given a query and a set of stored memories about the user(s). Your job is to synthesize a clear, relevant response from these memories.

Rules:
- Only include information that is directly relevant to the query
- If memories contain contradictory information, prefer the most recent one
- If no memories are relevant, say so clearly
- Be concise and factual - this response will be used by another AI as context
- Do NOT make up information that isn't in the provided memories
- Include the user_id when referencing specific users' information

Output a clear, concise summary of what you found. No JSON needed - just natural text. /no_think"""


async def recall_memories(base_url: str, db_path: str, query: str,
                          user_id: Optional[str] = None,
                          top_k: int = 8,
                          max_results: int = 20,
                          max_synthesis_tokens: int = 1024) -> dict:
    """Search memories and synthesize a response.

    Returns:
        dict with 'response' (synthesized text), 'memories_searched' (count),
        'memories_used' (count of memories sent to model)
    """
    from . import db as memdb

    # Search FTS5
    results = memdb.search_memories(db_path, query, user_id=user_id, limit=max_results)

    if not results:
        # Fall back to recent memories if FTS5 returns nothing
        results = memdb.get_recent_memories(db_path, user_id=user_id, limit=top_k)

    if not results:
        return {
            "response": "No memories found for this query.",
            "memories_searched": 0,
            "memories_used": 0,
        }

    # Take top_k most relevant
    top_memories = results[:top_k]

    # Format memories for the model
    memory_text = format_memories_for_synthesis(top_memories)

    # Synthesize via memory model
    synthesis = await synthesize_response(
        base_url, query, memory_text, max_synthesis_tokens
    )

    return {
        "response": synthesis,
        "memories_searched": len(results),
        "memories_used": len(top_memories),
    }


def format_memories_for_synthesis(memories: list[dict]) -> str:
    """Format memory results as text for the synthesis model."""
    lines = []
    for i, mem in enumerate(memories, 1):
        ts = mem.get("created_at", 0)
        date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else "unknown"
        user = mem.get("user_id", "unknown")
        lines.append(
            f"{i}. [{date_str}] (user: {user}, topic: {mem['topic']}, "
            f"importance: {mem.get('importance', '?')}) {mem['fact']}"
        )
    return "\n".join(lines)


async def synthesize_response(base_url: str, query: str,
                               memory_text: str,
                               max_tokens: int = 1024) -> str:
    """Send query + memories to the model for synthesis."""
    user_message = f"""Query: {query}

Stored memories:
{memory_text}

Based on these memories, provide a relevant response to the query."""

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": "memory",
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

            msg = data["choices"][0]["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            # Strip thinking tags if present (fallback)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content

        except httpx.HTTPError as e:
            logger.error("HTTP error during synthesis: %s", e)
            return f"Error recalling memories: {e}"
        except Exception as e:
            logger.error("Unexpected error during synthesis: %s", e)
            return f"Error recalling memories: {e}"
